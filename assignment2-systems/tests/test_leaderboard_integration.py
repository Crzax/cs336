"""#7 集成层正确性冒烟测试（单卡, 小模型）。

验证 leaderboard_integration 里的两件事在数值上等价于朴素路径:
  1. flash_sdpa monkey-patch：Flash Triton attention 替换 naive SDPA 后, 模型输出应一致(bf16 宽松)。
  2. forward_hidden + fused CE：隐藏态路径的 loss/梯度 == 朴素 logits + cross_entropy。

FSDP + torch.compile 的完整 2 卡端到端跑 `python -m cs336_systems.leaderboard`, 不在单测里。

运行:
    pytest tests/test_leaderboard_integration.py -v -s
"""

import copy

import pytest
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.activation_ckpt import apply_activation_checkpointing
from cs336_systems.fused_ce import fused_linear_cross_entropy
from cs336_systems.leaderboard_integration import (
    HiddenModel,
    forward_hidden,
    use_flash_attention,
)


def _build(dtype, device="cuda", seed=0, num_layers=2):
    torch.manual_seed(seed)
    return BasicsTransformerLM(
        vocab_size=512,
        context_length=256,
        d_model=128,
        num_layers=num_layers,
        num_heads=4,
        d_ff=256,
        rope_theta=10000.0,
    ).to(device=device, dtype=dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need GPU")
def test_flash_patch_forward_matches_naive():
    """Flash attention monkey-patch 后, 整模型前向输出应与 naive SDPA 一致(bf16 宽松)。"""
    device = "cuda"
    dtype = torch.bfloat16
    model = _build(dtype)

    x = torch.randint(0, 512, (2, 64), device=device)

    logits_naive = model(x)
    with use_flash_attention():
        logits_flash = model(x)

    torch.testing.assert_close(logits_flash, logits_naive, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need GPU")
def test_fused_ce_hidden_path_matches_naive():
    """forward_hidden + fused CE 的 loss/梯度 == 朴素 model(x) logits + cross_entropy。

    两份模型 deepcopy, 分别走 naive / fused-hidden 路径, 比对 loss 和所有参数梯度。
    """
    device = "cuda"
    dtype = torch.float32  # fp32 便于逐位比较（fused CE 走朴素 matmul, 无 flash）
    model_ref = _build(dtype)
    model_fused = copy.deepcopy(model_ref)

    torch.manual_seed(1)
    x = torch.randint(0, 512, (2, 64), device=device)
    targets = torch.randint(0, 512, (2, 64), device=device)

    # ---- reference: 全量 logits + cross_entropy ----
    # nn_utils.cross_entropy 已是 mean-over-tokens; fused_linear_cross_entropy 也是 mean, 直接对齐。
    logits = model_ref(x)
    loss_ref = cross_entropy(logits, targets)
    loss_ref.backward()

    # ---- fused: forward_hidden(不出 logits) + fused_linear_cross_entropy ----
    hidden = forward_hidden(model_fused, x)
    loss_fused = fused_linear_cross_entropy(hidden, model_fused.lm_head.weight, targets)
    loss_fused.backward()

    torch.testing.assert_close(loss_fused, loss_ref, atol=1e-3, rtol=1e-3)

    ref_params = dict(model_ref.named_parameters())
    fused_params = dict(model_fused.named_parameters())
    for name, p_ref in ref_params.items():
        p_fused = fused_params[name]
        assert (p_ref.grad is None) == (p_fused.grad is None), name
        if p_ref.grad is not None:
            torch.testing.assert_close(p_fused.grad, p_ref.grad, atol=1e-3, rtol=1e-3, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need GPU")
def test_hidden_model_wrapper_equiv():
    """HiddenModel(base)(x) 应与 forward_hidden(base, x) 完全一致（只是包了一层 forward）。"""
    device = "cuda"
    dtype = torch.float32
    base = _build(dtype)
    wrapped = HiddenModel(base)

    x = torch.randint(0, 512, (2, 64), device=device)
    h_direct = forward_hidden(base, x)
    h_wrapped = wrapped(x)
    torch.testing.assert_close(h_wrapped, h_direct)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need GPU")
def test_full_stack_runs():
    """Flash + activation ckpt + forward_hidden + fused CE 全叠加能跑通并产生有限 loss/梯度。"""
    device = "cuda"
    dtype = torch.bfloat16
    model = _build(dtype, num_layers=3)
    apply_activation_checkpointing(model)

    x = torch.randint(0, 512, (2, 64), device=device)
    targets = torch.randint(0, 512, (2, 64), device=device)

    with use_flash_attention():
        hidden = forward_hidden(model, x)
        loss = fused_linear_cross_entropy(hidden, model.lm_head.weight, targets)
        loss.backward()

    assert torch.isfinite(loss), loss
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    for g in grads:
        assert torch.isfinite(g).all()

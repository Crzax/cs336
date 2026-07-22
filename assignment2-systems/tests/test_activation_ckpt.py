"""Activation checkpointing 的正确性 + 显存测试。

在 GPU 机器上运行:
    pytest tests/test_activation_ckpt.py -v -s
或直接跑显存对比:
    python tests/test_activation_ckpt.py
"""

import copy

import pytest
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.activation_ckpt import (
    CheckpointedBlock,
    apply_activation_checkpointing,
)


def _build_small_model(dtype, device="cuda", seed=0):
    torch.manual_seed(seed)
    model = BasicsTransformerLM(
        vocab_size=1024,
        context_length=256,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        rope_theta=10000.0,
    ).to(device=device, dtype=dtype)
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_numerical_equivalence(dtype):
    """checkpoint 前后 loss + 所有梯度必须逐位一致(fp32)/宽松一致(bf16)。

    原理: activation checkpointing 只影响反向存哪些张量, 数学上 forward 完全等价,
    梯度也应完全等价(前提: 无随机性副作用, 例如 dropout)。
    """
    device = "cuda"
    N_BATCH, N_SEQ = 2, 64
    atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (2e-2, 2e-2)

    # 两份完全一样的模型
    model_ref = _build_small_model(dtype)
    model_ckpt = copy.deepcopy(model_ref)
    apply_activation_checkpointing(model_ckpt)
    # 确认所有层都被包了
    assert all(isinstance(b, CheckpointedBlock) for b in model_ckpt.layers)

    # 相同输入
    torch.manual_seed(42)
    x = torch.randint(0, 1024, (N_BATCH, N_SEQ), device=device)
    targets = torch.randint(0, 1024, (N_BATCH, N_SEQ), device=device)

    # ---- reference ----
    logits_ref = model_ref(x)
    loss_ref = cross_entropy(logits_ref, targets).sum()
    loss_ref.backward()

    # ---- checkpointed ----
    logits_ckpt = model_ckpt(x)
    loss_ckpt = cross_entropy(logits_ckpt, targets).sum()
    loss_ckpt.backward()

    # loss + logits
    torch.testing.assert_close(logits_ckpt, logits_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(loss_ckpt, loss_ref, atol=atol, rtol=rtol)

    # 每个参数的梯度对齐
    # 注意 checkpointed 模型的参数名多了 "block." 前缀(因为多套了一层 CheckpointedBlock),
    # 我们按参数顺序对齐(deepcopy 保证顺序一致)。
    ref_params = list(model_ref.parameters())
    ckpt_params = list(model_ckpt.parameters())
    assert len(ref_params) == len(ckpt_params)
    for p_ref, p_ckpt in zip(ref_params, ckpt_params):
        assert (p_ref.grad is None) == (p_ckpt.grad is None)
        if p_ref.grad is not None:
            torch.testing.assert_close(p_ckpt.grad, p_ref.grad, atol=atol, rtol=rtol)


def _peak_mem(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9  # GB


def main():
    """显存对比: 不 checkpoint vs checkpoint。

    用中等规模(8 层, d_model=1024, seq=2048)让激活占比明显, 但不至于把小卡打爆。
    参数量固定, 差异全在激活。
    """
    assert torch.cuda.is_available()
    device = "cuda"
    dtype = torch.bfloat16
    N_BATCH, N_SEQ = 2, 2048
    VOCAB = 4096

    def build():
        torch.manual_seed(0)
        m = BasicsTransformerLM(
            vocab_size=VOCAB,
            context_length=N_SEQ,
            d_model=1024,
            num_layers=8,
            num_heads=16,
            d_ff=2752,
            rope_theta=10000.0,
        ).to(device=device, dtype=dtype)
        return m

    x = torch.randint(0, VOCAB, (N_BATCH, N_SEQ), device=device)
    targets = torch.randint(0, VOCAB, (N_BATCH, N_SEQ), device=device)

    def run_naive():
        m = build()
        logits = m(x)
        loss = cross_entropy(logits, targets).sum()
        loss.backward()

    def run_ckpt():
        m = build()
        apply_activation_checkpointing(m)
        logits = m(x)
        loss = cross_entropy(logits, targets).sum()
        loss.backward()

    m_naive = _peak_mem(run_naive)
    m_ckpt = _peak_mem(run_ckpt)
    print(f"[num_layers=8 d_model=1024 seq={N_SEQ} bs={N_BATCH} {dtype}]")
    print(f"  no ckpt   peak mem = {m_naive:.3f} GB")
    print(f"  activation-ckpt peak mem = {m_ckpt:.3f} GB  ({m_naive / m_ckpt:.2f}x less)")


if __name__ == "__main__":
    main()

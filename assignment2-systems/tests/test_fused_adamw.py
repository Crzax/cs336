"""Fused AdamW（Triton）的正确性与耗时测试。

在 GPU 机器上运行：
    pytest tests/test_fused_adamw.py -v -s
或直接跑耗时对比：
    python tests/test_fused_adamw.py
"""

import pytest
import torch

from cs336_basics.optimizer import AdamW
from cs336_systems.fused_adamw import FusedAdamW


def _make_params(shapes, dtype, dev, seed=0):
    """构造一组独立、内容相同的参数，供两个优化器分别更新后逐一对比。"""
    torch.manual_seed(seed)
    base = [torch.randn(s, device=dev, dtype=dtype) for s in shapes]
    ref = [b.clone().requires_grad_(True) for b in base]
    fused = [b.clone().requires_grad_(True) for b in base]
    return ref, fused


def _run_steps(params, opt_cls, n_steps, seed=0, **opt_kwargs):
    """对同一组参数跑 n 步：每步灌入确定性的梯度并 step。"""
    opt = opt_cls(params, **opt_kwargs)
    for step in range(n_steps):
        # 每步、每参数用可复现的梯度（与优化器种类无关，两边完全一致）
        g = torch.Generator(device=params[0].device).manual_seed(seed * 1000 + step)
        for p in params:
            p.grad = torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=g)
        opt.step()
    return params


def _run_case(shapes, dtype, n_steps, atol, rtol, **opt_kwargs):
    dev = "cuda"
    ref_p, fused_p = _make_params(shapes, dtype, dev)

    _run_steps(ref_p, AdamW, n_steps, **opt_kwargs)
    _run_steps(fused_p, FusedAdamW, n_steps, **opt_kwargs)

    for pr, pf in zip(ref_p, fused_p):
        torch.testing.assert_close(pf, pr, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need GPU")
@pytest.mark.parametrize("n_steps", [1, 5, 20])
def test_fp32(n_steps):
    # fp32：m/v/参数全 fp32，应与参照几乎逐位一致（跨多步累积仍对齐）
    _run_case(
        shapes=[(4096, 512), (512,), (10000, 512), (128, 64, 3)],
        dtype=torch.float32, n_steps=n_steps,
        atol=1e-5, rtol=1e-5,
        lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need GPU")
def test_bf16():
    # bf16：m/v 底层是 bf16，逐步 round-trip 有误差，容差放宽
    _run_case(
        shapes=[(4096, 512), (10000, 512)],
        dtype=torch.bfloat16, n_steps=10,
        atol=2e-2, rtol=2e-2,
        lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need GPU")
def test_no_weight_decay():
    # weight_decay=0 的分支
    _run_case(
        shapes=[(2048, 2048)],
        dtype=torch.float32, n_steps=8,
        atol=1e-5, rtol=1e-5,
        lr=3e-4, betas=(0.8, 0.99), eps=1e-6, weight_decay=0.0,
    )


def _bench(opt_cls, params, n_iter=50, warmup=10, **opt_kwargs):
    opt = opt_cls(params, **opt_kwargs)
    for p in params:
        p.grad = torch.randn_like(p)

    for _ in range(warmup):
        opt.step()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        opt.step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter  # ms/step


def main():
    """耗时对比：原生 AdamW vs FusedAdamW（模拟 8B 量级的一批大张量）。"""
    assert torch.cuda.is_available()
    dev = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)
    # 模拟很多个大矩阵参数（层数 × 每层若干权重）
    shapes = [(4096, 11008)] * 20 + [(4096, 4096)] * 20 + [(151936, 4096)]
    base = [torch.randn(s, device=dev, dtype=dtype) * 0.02 for s in shapes]

    ref_p = [b.clone().requires_grad_(True) for b in base]
    fused_p = [b.clone().requires_grad_(True) for b in base]

    t_ref = _bench(AdamW, ref_p)
    t_fused = _bench(FusedAdamW, fused_p)
    n = sum(p.numel() for p in base)
    print(f"[params={n / 1e9:.2f}B  {dtype}]")
    print(f"  AdamW      step = {t_ref:8.3f} ms")
    print(f"  FusedAdamW step = {t_fused:8.3f} ms  ({t_ref / t_fused:.2f}x faster)")


if __name__ == "__main__":
    main()

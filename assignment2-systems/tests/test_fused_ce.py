"""Fused LM-head + cross-entropy 的正确性与显存测试。

在 GPU 机器上运行：
    pytest tests/test_fused_ce.py -v -s
或直接跑显存对比：
    python tests/test_fused_ce.py
"""

import pytest
import torch

from cs336_basics.nn_utils import cross_entropy
from cs336_systems.fused_ce import fused_linear_cross_entropy


def _reference(x, weight, targets):
    """朴素路径：materialize 全量 logits 再算 cross-entropy。"""
    logits = torch.matmul(x, weight.t())          # [N, V]
    return cross_entropy(logits, targets)


def _run_case(N, D, V, dtype, chunk_size, atol, rtol, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    x = torch.randn(N, D, device=dev, dtype=dtype)
    weight = torch.randn(V, D, device=dev, dtype=dtype) * (D ** -0.5)
    targets = torch.randint(0, V, (N,), device=dev)

    # ---- fused ----
    xf = x.clone().requires_grad_(True)
    wf = weight.clone().requires_grad_(True)
    loss_f = fused_linear_cross_entropy(xf, wf, targets, chunk_size=chunk_size)
    loss_f.backward()

    # ---- reference ----
    xr = x.clone().requires_grad_(True)
    wr = weight.clone().requires_grad_(True)
    loss_r = _reference(xr, wr, targets)
    loss_r.backward()

    # fused loss 固定返回 fp32（更稳），参照在 bf16 下是 bf16，故不比较 dtype
    torch.testing.assert_close(loss_f, loss_r.float(), atol=atol, rtol=rtol, check_dtype=False)
    torch.testing.assert_close(xf.grad, xr.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(wf.grad, wr.grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need GPU")
@pytest.mark.parametrize("chunk_size", [1024, 4096, 100000])
def test_fp32(chunk_size):
    # fp32：应与参照几乎逐位一致
    _run_case(N=4096, D=512, V=10000, dtype=torch.float32,
              chunk_size=chunk_size, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need GPU")
def test_bf16():
    # bf16：允许较宽松容差（bf16 matmul 本身有误差）
    _run_case(N=4096, D=512, V=10000, dtype=torch.bfloat16,
              chunk_size=2048, atol=2e-2, rtol=2e-2)


def _peak_mem(build_and_run):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    build_and_run()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9  # GB


def main():
    """显存对比：朴素 vs fused。每个分支用独立张量，避免共享 autograd 图。"""
    assert torch.cuda.is_available()
    dev = "cuda"
    N, D, V = 8192, 1024, 50000       # 中等规模，够看出差距
    dtype = torch.bfloat16

    def make_inputs():
        torch.manual_seed(0)
        x = torch.randn(N, D, device=dev, dtype=dtype, requires_grad=True)
        weight = (torch.randn(V, D, device=dev, dtype=dtype) * (D ** -0.5)).requires_grad_(True)
        targets = torch.randint(0, V, (N,), device=dev)
        return x, weight, targets

    def naive():
        x, weight, targets = make_inputs()
        _reference(x, weight, targets).backward()

    def fused():
        x, weight, targets = make_inputs()
        fused_linear_cross_entropy(x, weight, targets, chunk_size=2048).backward()

    m_naive = _peak_mem(naive)
    m_fused = _peak_mem(fused)
    print(f"[N={N} D={D} V={V} {dtype}]")
    print(f"  naive  peak mem = {m_naive:.3f} GB")
    print(f"  fused  peak mem = {m_fused:.3f} GB  ({m_naive / m_fused:.1f}x less)")


if __name__ == "__main__":
    main()

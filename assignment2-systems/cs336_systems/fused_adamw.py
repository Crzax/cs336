"""Fused AdamW（Triton 单 kernel 逐元素更新）。

原生 AdamW 的 `step()` 对每个参数张量要发起一长串独立的 elementwise op：
`p*=…`、`m=beta1*m+…`、`v=beta2*v+…`、`sqrt`、`div`、`p-=…`。每个 op 都是
一次独立的 CUDA kernel launch，且都要把 p/m/v/grad 从 HBM 读进来、算完再写回
HBM。AdamW 是纯访存密集（memory-bound）负载——算术很少，瓶颈全在 HBM 带宽和
kernel launch 开销上。

融合思路：把「m 更新 + v 更新 + 偏置校正 + 权重衰减 + 参数更新」全部塞进**一个**
Triton kernel。每个元素的 p/m/v/grad 只从 HBM 读一次、写一次，中间量全在寄存器里
流转，kernel launch 也从 ~6 次/张量降到 1 次/张量。这正是 Triton 的主场（#1 的
fused CE 反而不该用 Triton，因为那是 GEMM = cuBLAS 主场）。

数学与参照实现 `cs336_basics.optimizer.AdamW` 严格一致：
    alpha_t = lr * sqrt(1 - beta2**t) / (1 - beta1**t)      # 偏置校正后的步长
    p  <- p - lr * weight_decay * p                          # 解耦权重衰减（用 lr，非 alpha_t）
    m  <- beta1 * m + (1 - beta1) * g
    v  <- beta2 * v + (1 - beta2) * g^2
    p  <- p - alpha_t * m / (sqrt(v) + eps)
t 从 1 开始计数。
"""

import math
from collections.abc import Callable, Iterable

import torch
import triton
import triton.language as tl


@triton.jit
def _adamw_kernel(
    p_ptr, g_ptr, m_ptr, v_ptr,
    lr, beta1, beta2, eps, weight_decay, alpha_t,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 一维网格：每个 program 处理连续的 BLOCK_SIZE 个元素（要求张量内存连续）
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # 全部升 fp32 再算：m/v 底层可能是 bf16，中间量若也用 bf16 会累积明显误差
    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask).to(tl.float32)
    v = tl.load(v_ptr + offs, mask=mask).to(tl.float32)

    p = p - lr * weight_decay * p              # 解耦权重衰减
    m = beta1 * m + (1.0 - beta1) * g          # 一阶矩
    v = beta2 * v + (1.0 - beta2) * g * g      # 二阶矩
    p = p - alpha_t * m / (tl.sqrt(v) + eps)   # 偏置校正后的参数更新

    # tl.store 会自动把 fp32 结果 cast 回指针的元素类型（p 是 param dtype，m/v 同 grad dtype）
    tl.store(p_ptr + offs, p, mask=mask)
    tl.store(m_ptr + offs, m, mask=mask)
    tl.store(v_ptr + offs, v, mask=mask)


class FusedAdamW(torch.optim.Optimizer):
    """接口与 `cs336_basics.optimizer.AdamW` 一致，step 内核换成单个 Triton kernel。"""

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        block_size: int = 1024,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.block_size = block_size

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FusedAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 1
                    # m/v 与 grad 同 dtype（沿用参照实现的 zeros_like(grad) 行为）
                    state["m"] = torch.zeros_like(grad)
                    state["v"] = torch.zeros_like(grad)

                t = state["t"]
                m, v = state["m"], state["v"]
                # 偏置校正折进单个标量步长，逐元素 kernel 里就不用再算 pow(beta, t)
                alpha_t = lr * math.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t)

                # 线性偏移 kernel 要求内存连续；参数/梯度/矩通常本就连续，非连续时兜底
                p_c = p if p.is_contiguous() else p.contiguous()
                g_c = grad if grad.is_contiguous() else grad.contiguous()

                n = p_c.numel()
                grid = (triton.cdiv(n, self.block_size),)
                _adamw_kernel[grid](
                    p_c, g_c, m, v,
                    lr, beta1, beta2, eps, weight_decay, alpha_t,
                    n,
                    BLOCK_SIZE=self.block_size,
                )
                # 兜底路径下 p_c 是副本，需写回原参数
                if p_c is not p:
                    p.copy_(p_c)

                state["t"] = t + 1

        return loss

"""Fused LM-head + cross-entropy.

朴素路径 `logits = x @ W_lm.T` 会 materialize [N, vocab] 的巨大张量
（leaderboard 配置下 fp32 约 40GB）。这里按 token 分块（chunk）遍历，
每个 chunk 只临时算出 [chunk, vocab] 的小 logits，立刻算出该 chunk 的
loss 与梯度 (dx, dW)，随即丢弃 logits——全程不 materialize 全量 logits。

梯度在 forward 里就地算好并存下（"compute backward immediately"），
backward 只需按上游标量 grad_out 线性缩放。
"""

import torch


class FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, targets, chunk_size: int = 4096):
        # x:       [..., D]  最终隐藏态
        # weight:  [V, D]    lm_head 权重（logits = x @ weight.T）
        # targets: [...]     int64 类别下标
        orig_shape = x.shape
        D = orig_shape[-1]
        x = x.reshape(-1, D)              # [N, D]
        targets = targets.reshape(-1)     # [N]
        N = x.shape[0]
        V = weight.shape[0]
        inv_N = 1.0 / N

        loss = x.new_zeros((), dtype=torch.float32)
        grad_x = torch.empty_like(x)                            # [N, D] 与 x 同 dtype
        grad_w = torch.zeros_like(weight, dtype=torch.float32)  # [V, D] fp32 累加

        row_idx = torch.arange(chunk_size, device=x.device)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            c = end - start
            xc = x[start:end]                                   # [c, D]
            tc = targets[start:end]                             # [c]

            # 分块 logits：bf16 矩阵乘走 tensor core（快），结果升 fp32 做 softmax（稳）
            logits = torch.matmul(xc, weight.t()).float()       # [c, V]
            m = logits.max(dim=-1, keepdim=True).values         # [c, 1]
            exp = torch.exp(logits - m)                         # [c, V]
            sumexp = exp.sum(dim=-1, keepdim=True)              # [c, 1]
            lse = m.squeeze(-1) + torch.log(sumexp.squeeze(-1)) # [c]
            tgt = logits.gather(-1, tc.unsqueeze(-1)).squeeze(-1) #[c]
            loss += (lse - tgt).sum()

            # d loss / d logits = (softmax - onehot(target)) / N
            g = exp / sumexp                                    # softmax [c, V] fp32
            g[row_idx[:c], tc] -= 1.0
            g *= inv_N                                          # 把 mean 的 1/N 折进梯度

            gb = g.to(x.dtype)                                  # bf16 供快速 GEMM
            grad_x[start:end] = torch.matmul(gb, weight)        # [c,V]@[V,D] -> [c,D]
            grad_w += torch.matmul(gb.t(), xc).float()          # [V,c]@[c,D] -> [V,D]

        loss *= inv_N
        ctx.save_for_backward(grad_x.reshape(orig_shape), grad_w.to(weight.dtype))
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        grad_x, grad_w = ctx.saved_tensors
        # loss 是标量、grad_out 也是标量，所有梯度对 grad_out 线性
        return grad_out * grad_x, grad_out * grad_w, None, None


def fused_linear_cross_entropy(x, weight, targets, chunk_size: int = 4096):
    """x:[...,D], weight:[V,D], targets:[...] -> 标量 loss（对 N 个 token 取均值）。

    数值等价于 `cross_entropy(x @ weight.T, targets)`，但不 materialize 全量 logits。
    """
    return FusedLinearCrossEntropy.apply(x, weight, targets, chunk_size)

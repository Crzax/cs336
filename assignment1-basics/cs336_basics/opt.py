import math
import torch
from collections.abc import Callable, Iterable
from typing import Optional

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                t, m, v = state["t"], state["m"], state["v"]
                beta1, beta2 = group["betas"]
                lr, eps, weight_decay = group["lr"], group["eps"], group["weight_decay"]
                lr_t = lr * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)
                p.detach().add_(p.data, alpha=-weight_decay * lr)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                p.detach().addcdiv_(m, v.add_(eps).sqrt(), value=-lr_t)

                state["t"] += 1

def learning_rate_schedule(t: int, lr_max: float, lr_min: float, warmup_steps: int, constant_steps: int) -> float:
    if t < warmup_steps:
        return lr_max * t / warmup_steps
    elif t > constant_steps:
        return lr_min
    else:
        return lr_min + 0.5 * (1+ math.cos(math.pi * (t - warmup_steps) / (constant_steps - warmup_steps))) * (lr_max - lr_min)

def gradient_clipping(params: Iterable[torch.Tensor], max_grad_norm: float):
    eps = 1e-6
    grads = [p.grad for p in params if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0)
    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    if total_norm > max_grad_norm:
        scale = max_grad_norm / (total_norm + eps)
        for g in grads:
            g.detach().mul_(scale)
    return total_norm
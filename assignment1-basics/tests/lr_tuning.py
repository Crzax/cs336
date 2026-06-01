from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


def run(lr: float, n_steps: int = 10, seed: int = 0):
    torch.manual_seed(seed)  # 同一个初始权重，保证不同 lr 公平比较
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)  
    losses = []
    for t in range(n_steps):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        losses.append(loss.detach().cpu().item())
        loss.backward()
        opt.step()
    return losses


if __name__ == "__main__":
    for lr in [1e1, 1e2, 1e3]:
        losses = run(lr)
        print(f"\n=== lr = {lr:g} ===")
        for i, l in enumerate(losses):
            print(f"  step {i:2d}: loss = {l:.6e}")

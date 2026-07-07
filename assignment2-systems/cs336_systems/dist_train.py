import torch
import torch.distributed as dist

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
                param.grad /= world_size
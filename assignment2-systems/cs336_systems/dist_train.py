import os   
import torch
import torch.distributed as dist

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []

        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)
    
    def _grad_hook(self, param):
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append((handle, param))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for handle, param in self.handles:
            handle.wait()
            param.grad /= world_size
        self.handles.clear()

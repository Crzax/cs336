import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, flat: bool = True):
        super().__init__()
        self.module = module
        self.flat = flat

        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self, comm_start=None, comm_end=None):
        world_size = dist.get_world_size()
        grads = [p.grad for p in self.module.parameters() if p.grad is not None]

        if self.flat:
            flat_tensor = _flatten_dense_tensors(grads)
            if comm_start is not None:
                comm_start.record()
            dist.all_reduce(flat_tensor, op=dist.ReduceOp.SUM)
            flat_tensor /= world_size
            if comm_end is not None:
                comm_end.record()
            for g, s in zip(grads, _unflatten_dense_tensors(flat_tensor, grads)):
                g.copy_(s)                                        
        else:
            if comm_start is not None:
                comm_start.record()
            for g in grads:                                      
                dist.all_reduce(g, op=dist.ReduceOp.SUM)
                g /= world_size
            if comm_end is not None:
                comm_end.record()
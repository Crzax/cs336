import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.optim import Optimizer
from typing import Type, Any, Optional, Callable

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

class AsyncDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []

        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._allreduce_hook)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _allreduce_hook(self, param):
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append((handle, param))

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for handle, param in self.handles:                                      
            handle.wait()
            param.grad /= world_size
        self.handles.clear()

class StateShardingOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self._assign_counter = 0
        self._all_params: list[torch.Tensor] = []
        self._param_owners: list[int] = []

        self._local_optimizer: Optional[Optimizer] = None

        super().__init__(params, kwargs)

    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)
        added = self.param_groups[-1]

        hyper = {k: v for k, v in added.items() if k != "params"}

        local_params: list[torch.Tensor] = []
        for p in added["params"]:
            owner = self._assign_counter % self.world_size
            self._assign_counter += 1
            self._all_params.append(p)
            self._param_owners.append(owner)
            if owner == self.rank:
                local_params.append(p)

        if not local_params:
            return

        local_group = {"params": local_params, **hyper}
        if self._local_optimizer is None:
            self._local_optimizer = self.optimizer_cls([local_group], **self.kwargs)
        else:
            self._local_optimizer.add_param_group(local_group)

    def step(self, closure: Optional[Callable[..., Any]] = None, **kwargs: Any):
        if self._local_optimizer is not None:
            loss = self._local_optimizer.step(closure, **kwargs)
        else:
            loss = closure() if closure is not None else None

        for p, owner in zip(self._all_params, self._param_owners):
            dist.broadcast(p.data, src=owner)

        return loss
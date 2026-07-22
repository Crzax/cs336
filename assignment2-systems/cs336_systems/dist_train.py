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

class FSDP(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        compute_dtype: torch.dtype | None = None,
        prefetch: int = 2,
        skip_modules: "set[torch.nn.Module] | None" = None,
    ):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.prefetch_depth = int(prefetch)
        self._sharded_modules: list[torch.nn.Module] = []
        self._fwd_pos: dict[torch.nn.Module, int] = {}
        self._inflight: dict = {}
        # 不切分这些 module 的 weight（例如 lm_head：#7 里 fused CE 手动读它的完整
        # weight, 绕过了 forward hook, 若被切分会拿到残缺分片 → 必须整份保留）。
        skip_modules = skip_modules or set()

        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

        from cs336_basics.model import Embedding, Linear

        self._sharded_params: list[torch.nn.Parameter] = []
        for m in self.module.modules():
            if m in skip_modules:
                continue
            if isinstance(m, (Linear, Embedding)) and getattr(m, "weight", None) is not None:
                self._shard_module(m)

    def _shard_module(self, m: torch.nn.Module):
        param = m.weight
        orig_shape = tuple(param.shape)
        numel = param.numel()
        ws = self.world_size

        pad = (ws - numel % ws) % ws
        shard_size = (numel + pad) // ws

        flat = param.data.reshape(-1)
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        start = self.rank * shard_size
        shard = flat[start : start + shard_size].clone()  

        param.data = shard
        param._fsdp_is_sharded = True
        param._fsdp_master = param.data
        param._fsdp_meta = (orig_shape, numel, pad, shard_size)

        m.register_forward_pre_hook(self._pre_forward)
        m.register_forward_hook(self._post_forward)
        m.register_full_backward_pre_hook(self._pre_backward)
        param.register_post_accumulate_grad_hook(self._post_grad)

        self._fwd_pos[m] = len(self._sharded_modules)
        self._sharded_modules.append(m)
        self._sharded_params.append(param)

    def _gather_full(self, param: torch.nn.Parameter) -> torch.Tensor:
        orig_shape, numel, pad, shard_size = param._fsdp_meta
        master = param._fsdp_master
        compute_shard = master if self.compute_dtype is None else master.to(self.compute_dtype)
        compute_shard = compute_shard.contiguous()

        gathered = [torch.empty_like(compute_shard) for _ in range(self.world_size)]
        with torch.cuda.nvtx.range("fsdp.all_gather_weight"):
            dist.all_gather(gathered, compute_shard)
        full_flat = torch.cat(gathered)
        return full_flat[:numel].reshape(orig_shape)

    def _async_gather(self, module):
        if module in self._inflight:
            return
        p = module.weight
        orig_shape, numel, pad, shard_size = p._fsdp_meta
        master = p._fsdp_master
        compute_shard = master if self.compute_dtype is None else master.to(self.compute_dtype)
        compute_shard = compute_shard.contiguous()
        gathered = [torch.empty_like(compute_shard) for _ in range(self.world_size)]
        with torch.cuda.nvtx.range("fsdp.prefetch_all_gather"):
            handle = dist.all_gather(gathered, compute_shard, async_op=True)
        self._inflight[module] = (handle, gathered, compute_shard, orig_shape, numel)

    def _materialize(self, module):
        p = module.weight
        if module in self._inflight:
            handle, gathered, _cs, orig_shape, numel = self._inflight.pop(module)
            with torch.cuda.nvtx.range("fsdp.wait_all_gather"):
                handle.wait()
            p.data = torch.cat(gathered)[:numel].reshape(orig_shape)
        else:
            p.data = self._gather_full(p)  

    def _prefetch_next(self, module):
        i = self._fwd_pos.get(module)
        if i is None:
            return
        end = min(i + 1 + self.prefetch_depth, len(self._sharded_modules))
        for j in range(i + 1, end):
            self._async_gather(self._sharded_modules[j])

    def _pre_forward(self, module, args):
        if not self.prefetch_depth:
            module.weight.data = self._gather_full(module.weight)
            return
        self._materialize(module)
        self._prefetch_next(module)

    def _post_forward(self, module, args, output):
        p = module.weight
        p.data = p._fsdp_master  

    def _pre_backward(self, module, grad_output):
        p = module.weight
        p.data = self._gather_full(p)

    def _post_grad(self, p):
        p.data = p._fsdp_master

    def forward(self, *inputs, **kwargs):
        if self.prefetch_depth:
            self._inflight.clear()
            for j in range(min(self.prefetch_depth + 1, len(self._sharded_modules))):
                self._async_gather(self._sharded_modules[j])
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        ws = self.world_size

        for param in self._sharded_params:
            if param.grad is None:
                continue
            _, numel, pad, shard_size = param._fsdp_meta
            flat = param.grad.reshape(-1)
            if pad:
                flat = torch.cat([flat, flat.new_zeros(pad)])
            else:
                flat = flat.contiguous()

            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat /= ws

            start = self.rank * shard_size
            shard_grad = flat[start : start + shard_size].clone().to(param._fsdp_master.dtype)
            param.grad = shard_grad
            param.data = param._fsdp_master  

        for param in self.module.parameters():
            if getattr(param, "_fsdp_is_sharded", False):
                continue
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= ws

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name, param in self.module.named_parameters():
            if getattr(param, "_fsdp_is_sharded", False):
                orig_shape, numel, _, _ = param._fsdp_meta
                master = param._fsdp_master.contiguous()
                gathered = [torch.empty_like(master) for _ in range(self.world_size)]
                dist.all_gather(gathered, master)
                full_flat = torch.cat(gathered)
                result[name] = full_flat[:numel].reshape(orig_shape)
            else:
                result[name] = param.data
        return result
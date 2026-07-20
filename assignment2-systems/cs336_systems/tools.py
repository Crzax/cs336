from typing import Optional

import torch
import torch.distributed as dist


def alternate_all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    if op != dist.ReduceOp.SUM:
        raise NotImplementedError("alternate_all_reduce only supports SUM for now")

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return

    send_to = (rank + 1) % world_size
    recv_from = (rank - 1) % world_size

    send_buf = tensor.clone()
    recv_buf = torch.empty_like(tensor)

    for _ in range(world_size - 1):
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_buf, send_to, group=group),
            dist.P2POp(dist.irecv, recv_buf, recv_from, group=group),
        ])
        for req in reqs:
            req.wait()

        tensor.add_(recv_buf)
        send_buf, recv_buf = recv_buf, send_buf


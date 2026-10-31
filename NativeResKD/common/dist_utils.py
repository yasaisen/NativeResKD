"""DDP helpers adapted from the familiar historical CLEE implementation."""

from __future__ import annotations

import builtins
import datetime
import os

import torch
import torch.distributed as dist


_ORIGINAL_PRINT = builtins.print


def setup_for_distributed(is_master: bool) -> None:
    def rank_print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            _ORIGINAL_PRINT(*args, **kwargs)

    builtins.print = rank_print


def init_distributed_mode(cfg) -> None:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "SLURM_PROCID" in os.environ and "RANK" not in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        local_rank = int(os.environ.get("SLURM_LOCALID", rank % max(1, torch.cuda.device_count())))

    cfg.rank = rank
    cfg.world_size = world_size
    cfg.local_rank = local_rank
    cfg.gpu = local_rank
    cfg.is_distributed = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        cfg.device = f"cuda:{local_rank}"
    else:
        cfg.device = "cpu"

    if not cfg.is_distributed:
        _ORIGINAL_PRINT("Not using distributed mode")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("NativeResKD DDP requires CUDA/NCCL")

    cfg.dist_backend = "nccl"
    _ORIGINAL_PRINT(
        f"| distributed init (rank {rank}, world {world_size}, local_rank {local_rank}): {cfg.dist_url}",
        flush=True,
    )
    dist.init_process_group(
        backend=cfg.dist_backend,
        init_method=cfg.dist_url,
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(days=365),
    )
    dist.barrier()
    setup_for_distributed(rank == 0)


def cleanup_distributed_mode(cfg=None) -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()
    builtins.print = _ORIGINAL_PRINT


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return value
    result = value.detach().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result / get_world_size()


def reduce_sum_count(total: float, count: int, device: torch.device) -> tuple[float, int]:
    state = torch.tensor([float(total), float(count)], dtype=torch.float64, device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(state, op=dist.ReduceOp.SUM)
    return float(state[0].item()), int(state[1].item())


def bcast_bool(flag: bool, device: torch.device | str) -> bool:
    if not is_dist_avail_and_initialized():
        return bool(flag)
    state = torch.tensor([1 if flag else 0], dtype=torch.uint8, device=device)
    dist.broadcast(state, src=0)
    return bool(state.item())


def broadcast_object(value, src: int = 0):
    if not is_dist_avail_and_initialized():
        return value
    payload = [value if get_rank() == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


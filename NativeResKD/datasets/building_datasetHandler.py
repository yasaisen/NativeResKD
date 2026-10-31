"""Public NRKD dataset handler with scalable, DDP-consistent shape planning."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch.distributed as dist

from . import _building_datasetHandler_impl as _impl
from .kd_dataset import KDDataset as _KDDataset
from .shape_bucket_sampler import DynamicShapeBucketBatchSampler as _ShapeSampler
from ..common.utils import log_print


class _ParallelPlanKDDataset(_KDDataset):
    """Precompute deterministic view plans once on rank 0, then broadcast them."""

    def _compute_all_plans(self):
        workers = max(1, int(getattr(self.cfg, "shape_plan_workers", self.cfg.qc_workers)))
        results = [None] * len(self)
        chunk_size = max(64, workers * 16)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nrkd-shape") as pool:
            for start in range(0, len(self), chunk_size):
                stop = min(len(self), start + chunk_size)
                chunk = list(pool.map(self.plan_for_index, range(start, stop)))
                results[start:stop] = chunk
                if stop % 5000 < chunk_size or stop == len(self):
                    log_print(f"Shape plans: {stop}/{len(self)} (epoch={self.epoch})")
        return results

    def set_epoch(self, epoch: int, precompute: bool = True) -> None:
        self.epoch = int(epoch)
        self._plans.clear()
        if not precompute:
            return
        distributed = dist.is_available() and dist.is_initialized()
        payload = [None]
        if not distributed or dist.get_rank() == 0:
            try:
                payload[0] = {"ok": True, "plans": self._compute_all_plans()}
            except Exception as exc:
                payload[0] = {"ok": False, "type": type(exc).__name__, "message": str(exc)}
        if distributed:
            dist.broadcast_object_list(payload, src=0)
        if not payload[0]["ok"]:
            raise RuntimeError(
                "NRKD shape-plan precompute failed: "
                f"{payload[0]['type']}: {payload[0]['message']}"
            )
        self._plans = dict(enumerate(payload[0]["plans"]))


class _PlanningShapeBucketBatchSampler(_ShapeSampler):
    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        plans_are_ready = len(getattr(self.dataset, "_plans", {})) == len(self.dataset)
        if epoch == self.epoch and plans_are_ready:
            return
        super().set_epoch(epoch)

    def _build_batches(self):
        if len(getattr(self.dataset, "_plans", {})) != len(self.dataset):
            self.dataset.set_epoch(self.epoch, precompute=True)
        return super()._build_batches()

    def __len__(self) -> int:
        # Formal NRKD uses physical batch size 1. Shape buckets cannot change
        # its batch count, so scheduler construction must not decode all 40k
        # images merely to answer len(loader).
        if self.batch_size == 1:
            total = len(self.dataset)
            if self.world_size > 1 and total % self.world_size:
                if self.pad_distributed_batches:
                    total += self.world_size - total % self.world_size
                else:
                    total -= total % self.world_size
            return total // self.world_size
        return super().__len__()


_impl.KDDataset = _ParallelPlanKDDataset
_impl.DynamicShapeBucketBatchSampler = _PlanningShapeBucketBatchSampler


class datasetHandler(_impl.datasetHandler):
    @classmethod
    def from_config(cls, cfg) -> "datasetHandler":
        handler = super().from_config(cfg)
        cfg.metadata_reports = handler.metadata_reports
        return handler


DatasetHandler = datasetHandler
prepare_kd_split = _impl.prepare_kd_split
_prepare_kd_split_local = _impl._prepare_kd_split_local

__all__ = [
    "datasetHandler",
    "DatasetHandler",
    "prepare_kd_split",
    "_prepare_kd_split_local",
]

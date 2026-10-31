from __future__ import annotations

from collections import defaultdict
from typing import Iterator

import torch
from torch.utils.data import Sampler


class DynamicShapeBucketBatchSampler(Sampler[list[int]]):
    """Epoch-aware exact-shape batches without image/token padding."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
        pad_distributed_batches: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.pad_distributed_batches = pad_distributed_batches
        self.epoch = 0
        self._batches: list[list[int]] | None = None
        self.distributed_padding_batches = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.dataset.set_epoch(epoch, precompute=True)
        self._batches = None

    def _build_batches(self) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index in range(len(self.dataset)):
            buckets[tuple(self.dataset.shape_for_index(index))].append(index)
        all_batches: list[list[int]] = []
        for shape in sorted(buckets):
            indices = buckets[shape]
            if self.shuffle:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in order]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        if self.shuffle and all_batches:
            order = torch.randperm(len(all_batches), generator=generator).tolist()
            all_batches = [all_batches[i] for i in order]
        self.distributed_padding_batches = 0
        if self.world_size > 1 and len(all_batches) % self.world_size:
            if self.pad_distributed_batches:
                needed = self.world_size - (len(all_batches) % self.world_size)
                if not all_batches:
                    raise RuntimeError("Cannot distributed-pad an empty sampler")
                all_batches.extend(all_batches[:needed])
                self.distributed_padding_batches = needed
            else:
                all_batches = all_batches[: len(all_batches) - len(all_batches) % self.world_size]
        return all_batches[self.rank :: self.world_size]

    def __iter__(self) -> Iterator[list[int]]:
        if self._batches is None:
            self._batches = self._build_batches()
        yield from self._batches

    def __len__(self) -> int:
        # The number of incomplete batches depends on the exact shape buckets;
        # ceil(dataset_size / batch_size) under-counts them and corrupts the
        # optimizer/scheduler step calculation.
        if self._batches is None:
            self._batches = self._build_batches()
        return len(self._batches)

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "distributed_padding_batches": self.distributed_padding_batches}

    def load_state_dict(self, state: dict) -> None:
        self.set_epoch(int(state.get("epoch", 0)))


"""Rank-aware TensorBoard/JSONL/CSV metrics used by NRKDTrainer."""

from __future__ import annotations

import csv
import json
import os
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .dist_utils import is_main_process
from .utils import log_print


class LossLogger:
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)
        self.losses: dict[str, list[float]] = defaultdict(list)

    def update(self, values: dict[str, float]) -> None:
        for key, value in values.items():
            numeric = float(value)
            self.sums[key] += numeric
            self.counts[key] += 1
            self.losses[key].append(numeric)

    def epoch_summary(self, reset: bool = True) -> dict[str, float]:
        result = {
            key: self.sums[key] / self.counts[key]
            for key in self.sums
            if self.counts[key]
        }
        if reset:
            self.sums.clear()
            self.counts.clear()
        return result


class MetricsTracker:
    def __init__(self, save_path: str, enabled: bool | None = None):
        self.save_path = Path(save_path)
        self.enabled = is_main_process() if enabled is None else bool(enabled)
        self.writer = None
        self.segment_file = None
        self.train_loss_logger = LossLogger()
        self.val_loss_logger = LossLogger()
        self.test_loss_logger = LossLogger()
        if not self.enabled:
            return
        self.save_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d%H%M%S_%f")
        self.segment_path = self.save_path / f"{timestamp}_{uuid.uuid4().hex[:12]}_metrics.jsonl"
        self.segment_file = self.segment_path.open("x", encoding="utf-8")
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(self.save_path))
            log_print(f"TensorBoard writer created at {self.save_path}")
        except ImportError:
            log_print("[WARN] tensorboard is unavailable; JSONL/CSV logging remains enabled")

    def _write_jsonl(self, relative_path: str, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record = dict(record)
        record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        encoded = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        target = self.save_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
        if self.segment_file is not None:
            self.segment_file.write(encoded)
            self.segment_file.flush()

    def _write_csv(self, relative_path: str, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        target = self.save_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        new_file = not target.exists()
        with target.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            if new_file:
                writer.writeheader()
            writer.writerow(record)

    def update(
        self,
        loss_dict: dict[str, float],
        stage: str,
        lr: float | None = None,
        rank: int = 0,
        global_step: int = 0,
        optimizer_step: int | None = None,
        **extra,
    ) -> None:
        logger = {
            "train": self.train_loss_logger,
            "val": self.val_loss_logger,
            "test": self.test_loss_logger,
        }[stage]
        logger.update(loss_dict)
        if not self.enabled:
            return
        record = {
            "stage": stage,
            "rank": int(rank),
            "global_step": int(global_step),
            "optimizer_step": None if optimizer_step is None else int(optimizer_step),
            **{key: float(value) for key, value in loss_dict.items()},
            **extra,
        }
        if lr is not None:
            record["lr"] = float(lr)
        self._write_jsonl("steps/training_log.jsonl", record)
        self._write_csv("steps/metrics.csv", record)
        if self.writer is not None:
            x_axis = int(optimizer_step if optimizer_step is not None else global_step)
            for key, value in loss_dict.items():
                self.writer.add_scalar(f"{stage.capitalize()}/{key}", float(value), x_axis)
            if lr is not None:
                self.writer.add_scalar("Optimization/LearningRate", float(lr), x_axis)
            for key, value in extra.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{stage.capitalize()}/{key}", float(value), x_axis)

    def epoch_summary(self, metric_dict: dict[str, Any] | None, global_step: int, epoch: int | None = None):
        train = self.train_loss_logger.epoch_summary()
        valid = self.val_loss_logger.epoch_summary()
        test = self.test_loss_logger.epoch_summary()
        record: dict[str, Any] = {
            "epoch": epoch,
            "global_step": int(global_step),
            **{f"train/{key}": value for key, value in train.items()},
            **{f"val/{key}": value for key, value in valid.items()},
            **{f"test/{key}": value for key, value in test.items()},
        }
        if metric_dict:
            record.update(metric_dict)
        self._write_jsonl("training_log.jsonl", record)
        self._write_csv("metrics.csv", record)
        if self.writer is not None:
            for key, value in record.items():
                if isinstance(value, (int, float)) and key not in {"epoch", "global_step"}:
                    self.writer.add_scalar(key, float(value), global_step)
            self.writer.flush()
        return train.get("total_loss"), valid.get("total_loss"), test.get("total_loss")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
        if self.segment_file is not None and not self.segment_file.closed:
            self.segment_file.flush()
            self.segment_file.close()
        self.segment_file = None

    @classmethod
    def from_config(cls, cfg):
        return cls(save_path=cfg.save_path)


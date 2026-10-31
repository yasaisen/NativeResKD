"""Public rank-aware metrics tracker with schema-stable per-stage CSV files."""

from __future__ import annotations

from ._metricsTracker_impl import LossLogger
from ._metricsTracker_impl import MetricsTracker as _MetricsTracker


class MetricsTracker(_MetricsTracker):
    def _write_csv(self, relative_path, record):
        if relative_path == "steps/metrics.csv" and record.get("stage"):
            relative_path = f"steps/{record['stage']}_metrics.csv"
        return super()._write_csv(relative_path, record)

    @classmethod
    def from_config(cls, cfg):
        return cls(save_path=cfg.save_path)


__all__ = ["MetricsTracker", "LossLogger"]

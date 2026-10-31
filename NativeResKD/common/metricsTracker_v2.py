"""Lightweight training monitor compatible with the CLEE/DRepGenVLM interface."""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


class TrainingMonitor:
    def __init__(self, writer, model, optimizer, log_every_n_steps=10, check_every_n_steps=500, **_):
        self.writer = writer
        self.model = model
        self.optimizer = optimizer
        self.log_period = max(1, int(log_every_n_steps))
        self.check_period = max(1, int(check_every_n_steps))
        self.last_time = time.time()

    def log_always_on(self, global_step: int, batch_size: int) -> None:
        if self.writer is None or global_step % self.log_period:
            return
        now = time.time()
        elapsed = max(now - self.last_time, 1.0e-12)
        self.writer.add_scalar(
            "System/Throughput_samples_per_sec",
            batch_size * self.log_period / elapsed,
            global_step,
        )
        self.writer.add_scalar("System/Time_per_step_sec", elapsed / self.log_period, global_step)
        self.last_time = now
        if torch.cuda.is_available():
            if dist.is_available() and dist.is_initialized():
                devices = [int(os.environ.get("LOCAL_RANK", dist.get_rank()))]
            else:
                devices = range(torch.cuda.device_count())
            for device in devices:
                self.writer.add_scalar(
                    f"VRAM_By_GPU/GPU_{device}",
                    torch.cuda.memory_reserved(device) / (1024**3),
                    global_step,
                )

    def log_periodic(self, global_step: int) -> None:
        if self.writer is None or global_step % self.check_period:
            return
        total = 0.0
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.detach().float().norm(2)) ** 2
        self.writer.add_scalar("Optimization/Global_Grad_Norm", total**0.5, global_step)


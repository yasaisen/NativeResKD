from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rng_state_dict() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def load_rng_state_dict(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # ``torch.load(..., map_location="cuda")`` also moves the serialized CPU
    # RNG byte tensor to CUDA.  The RNG restoration APIs deliberately accept
    # CPU ByteTensors only, so normalize both CPU and CUDA RNG states here.
    torch.set_rng_state(state["torch"].detach().cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [device_state.detach().cpu() for device_state in state["cuda"]]
        )


def save_checkpoint(
    path: str | Path,
    *,
    student,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    optimizer_step: int,
    config: dict,
    best_metric: float,
    sampler_state: dict | None = None,
    extra: dict | None = None,
) -> Path:
    target = Path(path)
    if target.suffix:
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target.mkdir(parents=True, exist_ok=True)
        target = target / "checkpoint.pt"
    payload = {
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "optimizer_step": int(optimizer_step),
        "config": config,
        "rng_state": rng_state_dict(),
        "sampler_state": sampler_state,
        "best_metric": float(best_metric),
        "extra": extra or {},
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    student,
    optimizer=None,
    scheduler=None,
    sampler=None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    target = Path(path)
    if target.is_dir():
        target = target / "checkpoint.pt"
    payload = torch.load(target, map_location=map_location, weights_only=False)
    student.load_state_dict(payload["student"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if sampler is not None and payload.get("sampler_state"):
        sampler.load_state_dict(payload["sampler_state"])
    if restore_rng:
        load_rng_state_dict(payload["rng_state"])
    return payload


"""Read-only student checkpoint loading for S1 evaluation."""

from __future__ import annotations

from pathlib import Path

import torch


def resolve_model_checkpoint(path: str | Path) -> Path:
    requested = Path(path).expanduser().resolve()
    if requested.name == "checkpoint.pt":
        return requested
    filename = requested.name
    if filename.startswith("[trainer]"):
        filename = filename[len("[trainer]"):]
    elif filename.startswith("[model]"):
        return requested
    return requested.parent / f"[model]{filename}"


def load_student_checkpoint(student, path: str | Path, strict: bool = True) -> Path:
    resolved = resolve_model_checkpoint(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    state = payload.get("student_state_dict", payload.get("student", payload))
    student.load_state_dict(state, strict=strict)
    return resolved


__all__ = ["load_student_checkpoint", "resolve_model_checkpoint"]

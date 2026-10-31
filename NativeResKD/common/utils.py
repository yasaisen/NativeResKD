"""Shared utilities following the CLEE/DRepGenVLM management conventions."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import random
import socket
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def log_print(text: str = "", head: bool = False, newline: bool = True, traceback: bool = False):
    from .dist_utils import is_main_process

    if not is_main_process():
        return
    frame = inspect.currentframe().f_back
    function = frame.f_code.co_name
    owner = None
    if "self" in frame.f_locals:
        owner = frame.f_locals["self"].__class__.__name__
    elif "cls" in frame.f_locals:
        owner = frame.f_locals["cls"].__name__
    prefix = f"[{datetime.now().strftime('%H:%M:%S')}] [{owner + '.' if owner else ''}{function}] "
    if head:
        print()
    print(prefix + str(text), end="\n" if newline else "", flush=True)
    if traceback:
        for item in inspect.stack()[1:]:
            print(f"  -> {item.filename}:{item.lineno} in {item.function}")


def _debug_print(variable_name: str, variable: Any) -> str:
    if torch.is_tensor(variable):
        finite = bool(torch.isfinite(variable).all()) if variable.is_floating_point() else True
        return f"[{variable_name}] Tensor(shape={tuple(variable.shape)}, dtype={variable.dtype}, device={variable.device}, finite={finite})"
    if isinstance(variable, np.ndarray):
        return f"[{variable_name}] ndarray(shape={variable.shape}, dtype={variable.dtype})"
    return f"[{variable_name}] [{type(variable).__name__}] {variable}"


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sample_seed(base_seed: int, epoch: int, index: int, stream: str = "sample") -> int:
    encoded = f"{base_seed}:{epoch}:{index}:{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**32)


def ensure_finite(name: str, value: torch.Tensor, context: dict[str, Any] | None = None) -> None:
    if not torch.isfinite(value).all():
        details = json.dumps(context or {}, ensure_ascii=False, default=str)
        raise FloatingPointError(f"Non-finite {name}; context={details}")


def variable_checker(variable_name: str, variable: torch.Tensor) -> None:
    ensure_finite(variable_name, variable)


def move_to_device(value: Any, device: torch.device | str, non_blocking: bool = True) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {key: move_to_device(item, device, non_blocking) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device, non_blocking) for item in value]
    if isinstance(value, tuple):
        converted = [move_to_device(item, device, non_blocking) for item in value]
        return type(value)(*converted) if hasattr(value, "_fields") else tuple(converted)
    return value


def environment_info() -> dict[str, Any]:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    return {
        "framework": "NativeResKD",
        "abbreviation": "NRKD",
        "version": "4.01",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "hostname": socket.gethostname(),
        "git_commit": git_commit,
        "pid": os.getpid(),
    }


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_json(path: str | Path, payload: Any) -> None:
    atomic_write_json(path, payload)


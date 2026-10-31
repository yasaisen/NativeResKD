"""CLEE/DRepGenVLM-style run configuration and resume policy for NRKD 4.01."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch.distributed as dist
import yaml

from .NRKD_baseConfig import NRKD_baseConfig
from ..common.utils import log_print


def _parse_optional_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Override must be KEY=VALUE, got {raw!r}")
    key, value = raw.split("=", 1)
    if "." in key:
        raise ValueError("NativeResKD 4.01 project configs are flat; dotted overrides are not supported")
    return key, yaml.safe_load(value)


class ConfigHandler:
    def __init__(
        self,
        cfg: NRKD_baseConfig,
        checkpoint_path: str | None = None,
        checkpoint_type: str | None = None,
        use_this_dir: bool = False,
    ):
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.checkpoint_type = checkpoint_type
        self.use_this_dir = bool(use_this_dir)
        self.cfg.nowtime = datetime.now().strftime("%y%m%d%H%M%S")

        if self.use_this_dir and checkpoint_type == "best":
            raise ValueError("--use-this-dir is supported only with --latest-checkpoint-path")
        if self.use_this_dir and checkpoint_path:
            self.cfg.save_path = self._checkpoint_root(checkpoint_path)
        elif self.cfg.save_path is None or checkpoint_path is not None:
            self.cfg.save_path = os.path.join(self.cfg.root_path, "checkpoints", self.cfg.nowtime)

    @staticmethod
    def _checkpoint_root(path: str) -> str:
        resolved = os.path.abspath(path)
        if os.path.isdir(resolved):
            return resolved
        parent = os.path.dirname(resolved)
        if os.path.basename(parent) in {"best", "latest", "checkpoint_refs"}:
            return os.path.dirname(parent)
        return parent

    @staticmethod
    def _resolve_checkpoint_request(directory_or_file: str, filename: str) -> str:
        path = Path(directory_or_file).expanduser().resolve()
        if path.is_file():
            return str(path)
        legacy = path / "checkpoint.pt"
        if legacy.is_file():
            return str(legacy)
        nested_legacy = path / filename.removesuffix("_model.pth") / "checkpoint.pt"
        if nested_legacy.is_file():
            return str(nested_legacy)
        return str(path / filename)

    def finalize_runtime(self) -> None:
        """Synchronize the run directory after process-group initialization."""
        if dist.is_available() and dist.is_initialized():
            payload = [
                {"nowtime": self.cfg.nowtime, "save_path": self.cfg.save_path}
                if dist.get_rank() == 0 else None
            ]
            dist.broadcast_object_list(payload, src=0)
            self.cfg.nowtime = payload[0]["nowtime"]
            self.cfg.save_path = payload[0]["save_path"]
        if not dist.is_initialized() or dist.get_rank() == 0:
            Path(self.cfg.save_path).mkdir(parents=True, exist_ok=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        log_print(f"Output path: {self.cfg.save_path}")
        self.cfg.print_config()

    @classmethod
    def get_cfg(cls, argv: list[str] | None = None) -> "ConfigHandler":
        parser = argparse.ArgumentParser()
        parser.add_argument("--cfg-path", default=None)
        parser.add_argument("--best-checkpoint-path", default=None)
        parser.add_argument("--latest-checkpoint-path", default=None)
        parser.add_argument("--weight-filename", default=None)
        parser.add_argument("--retrain", nargs="?", const=True, default=None, type=_parse_optional_bool)
        parser.add_argument("--use-this-dir", action="store_true")
        parser.add_argument("--set", action="append", default=[])
        args = parser.parse_args(argv)

        if args.best_checkpoint_path and args.latest_checkpoint_path:
            raise ValueError("Only one checkpoint path type can be provided")
        if not any((args.cfg_path, args.best_checkpoint_path, args.latest_checkpoint_path)):
            raise ValueError("Provide --cfg-path, --best-checkpoint-path, or --latest-checkpoint-path")
        if args.use_this_dir and not args.latest_checkpoint_path:
            raise ValueError("--use-this-dir requires --latest-checkpoint-path")

        checkpoint_dir = args.best_checkpoint_path or args.latest_checkpoint_path
        cfg_path = args.cfg_path
        if cfg_path is None and checkpoint_dir is not None:
            checkpoint_root = cls._checkpoint_root(checkpoint_dir)
            cfg_path = os.path.join(checkpoint_root, "config.json")
            log_print(f"Inferring cfg_path from checkpoint: {cfg_path}")
        cfg = NRKD_baseConfig.load(cfg_path)
        for raw in args.set:
            key, value = _parse_override(raw)
            if not hasattr(cfg, key):
                raise KeyError(f"Unknown NRKD config key: {key}")
            setattr(cfg, key, value)

        if args.retrain is not None:
            cfg.trainer_mode = "reTrain" if args.retrain else "keepTrain"
        elif checkpoint_dir is not None:
            cfg.trainer_mode = "keepTrain"
        else:
            cfg.trainer_mode = None

        checkpoint_type = None
        checkpoint_path = None
        if args.best_checkpoint_path:
            checkpoint_type = "best"
            filename = args.weight_filename or cfg.weight_filename
            checkpoint_path = cls._resolve_checkpoint_request(args.best_checkpoint_path, filename)
        elif args.latest_checkpoint_path:
            checkpoint_type = "latest"
            checkpoint_path = cls._resolve_checkpoint_request(args.latest_checkpoint_path, "latest_model.pth")

        return cls(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            checkpoint_type=checkpoint_type,
            use_this_dir=args.use_this_dir,
        )


"""Public NativeResKD ConfigHandler with unambiguous checkpoint path parsing."""

from __future__ import annotations

from pathlib import Path

from ._configHandler_impl import ConfigHandler as _ConfigHandler


class ConfigHandler(_ConfigHandler):
    @staticmethod
    def _checkpoint_root(path: str) -> str:
        resolved = Path(path).expanduser().resolve()
        if resolved.is_dir() and resolved.name in {"best", "latest", "checkpoint_refs"}:
            return str(resolved.parent)
        return _ConfigHandler._checkpoint_root(str(resolved))

    @staticmethod
    def _resolve_checkpoint_request(directory_or_file: str, filename: str) -> str:
        path = Path(directory_or_file).expanduser().resolve()
        if path.suffix.lower() in {".pt", ".pth"}:
            return str(path)
        return _ConfigHandler._resolve_checkpoint_request(str(path), filename)


__all__ = ["ConfigHandler"]

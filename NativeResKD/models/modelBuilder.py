"""Familiar framework-style model builder for NRKD."""

from __future__ import annotations

from .modeling_NRKD import NativeResolutionKnowledgeDistiller


class modelBuilder:
    @classmethod
    def from_config(cls, cfg):
        return NativeResolutionKnowledgeDistiller.from_config(cfg)


ModelBuilder = modelBuilder


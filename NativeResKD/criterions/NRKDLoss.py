"""Formal NRKD loss with projection-aware diagnostic metrics."""

from __future__ import annotations

import torch

from .representation_kd_loss import RepresentationKDLoss, gather_teacher_patches


class NRKDLoss(RepresentationKDLoss):
    def forward(self, *args, **kwargs):
        losses = super().forward(*args, **kwargs)
        projected = losses.get("projected_cosine")
        if projected is not None and not bool(torch.isfinite(projected).all()):
            # CONCH v1.5 exposes its 768-d public feature through an attentional
            # pooler and has no optional post-pool projection. This diagnostic is
            # therefore undefined, not a training-state numerical failure.
            losses.pop("projected_cosine")
        return losses


__all__ = ["NRKDLoss", "RepresentationKDLoss", "gather_teacher_patches"]

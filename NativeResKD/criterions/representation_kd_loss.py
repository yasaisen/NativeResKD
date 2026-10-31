from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..common.utils import ensure_finite


def cosine_per_item(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    a = F.normalize(a.float(), dim=-1, eps=eps)
    b = F.normalize(b.float(), dim=-1, eps=eps)
    return (a * b).sum(dim=-1)


def gather_teacher_patches(
    teacher: torch.Tensor,
    patch_mapping: torch.Tensor,
    teacher_grid_hw: tuple[int, int] | list[int] | torch.Tensor,
) -> torch.Tensor:
    """Gather teacher [B,N,D] using explicit student->teacher (y,x) coordinates."""
    if patch_mapping.ndim != 3 or patch_mapping.shape[-1] != 2:
        raise ValueError(f"patch_mapping must be [B,N,2] y/x coordinates, got {patch_mapping.shape}")
    if torch.is_tensor(teacher_grid_hw):
        widths = teacher_grid_hw[:, 1].to(patch_mapping.device)
        flat = patch_mapping[..., 0] * widths[:, None] + patch_mapping[..., 1]
    else:
        width = int(teacher_grid_hw[1])
        flat = patch_mapping[..., 0] * width + patch_mapping[..., 1]
    if flat.min() < 0 or flat.max() >= teacher.shape[1]:
        raise IndexError("Patch correspondence points outside the teacher grid")
    return teacher.gather(1, flat.long().unsqueeze(-1).expand(-1, -1, teacher.shape[-1]))


class RepresentationKDLoss(nn.Module):
    def __init__(
        self,
        global_weight: float = 1.0,
        patch_weight: float = 0.5,
        hidden_weight: float = 0.25,
        selected_layers: tuple[int, ...] = (6, 12, 18, 24),
        eps: float = 1e-8,
    ):
        super().__init__()
        self.global_weight = float(global_weight)
        self.patch_weight = float(patch_weight)
        self.hidden_weight = float(hidden_weight)
        self.selected_layers = selected_layers
        self.eps = float(eps)

    @classmethod
    def from_config(cls, cfg) -> "RepresentationKDLoss":
        return cls(
            global_weight=cfg.loss.global_weight,
            patch_weight=cfg.loss.patch_weight,
            hidden_weight=cfg.loss.hidden_weight,
            selected_layers=tuple(cfg.loss.selected_layers),
            eps=cfg.loss.cosine_eps,
        )

    def forward(
        self,
        model_outputs: dict[str, Any],
        patch_mapping: torch.Tensor,
        tissue_retention_ratio: torch.Tensor,
        view_type: list[str] | tuple[str, ...],
        teacher_grid_hw: tuple[int, int] | torch.Tensor = (32, 32),
        sample_id: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        teacher = model_outputs["teacher"]
        student = model_outputs["student"]
        global_cosine_items = cosine_per_item(
            student["global_preproj"], teacher["global_preproj"], self.eps
        )
        semantic_weight = torch.ones_like(global_cosine_items)
        lossy = torch.tensor(
            [str(item) == "lossy_trim" for item in view_type],
            device=semantic_weight.device,
            dtype=torch.bool,
        )
        ratios = tissue_retention_ratio.to(semantic_weight.device).float().clamp(0.0, 1.0)
        semantic_weight = torch.where(lossy, ratios.square(), semantic_weight)
        global_loss = ((1.0 - global_cosine_items) * semantic_weight).mean()

        patch_losses = []
        patch_cosines = []
        hidden_losses = []
        hidden_cosines = []
        context = {"sample_id": sample_id, "view_type": list(view_type)}
        for layer in self.selected_layers:
            if layer not in teacher["hidden_patch"] or layer not in student["hidden_patch"]:
                raise KeyError(f"Selected layer {layer} missing from model outputs")
            target_patch = gather_teacher_patches(
                teacher["hidden_patch"][layer], patch_mapping, teacher_grid_hw
            )
            patch_cos = cosine_per_item(student["hidden_patch"][layer], target_patch, self.eps)
            layer_patch_loss = (1.0 - patch_cos).mean()
            ensure_finite(f"patch_loss_layer_{layer}", layer_patch_loss, {**context, "layer": layer})
            patch_losses.append(layer_patch_loss)
            patch_cosines.append(patch_cos.mean())

            hidden_cos = cosine_per_item(
                student["hidden_global"][layer], teacher["hidden_global"][layer], self.eps
            )
            layer_hidden_loss = (1.0 - hidden_cos).mean()
            ensure_finite(f"hidden_loss_layer_{layer}", layer_hidden_loss, {**context, "layer": layer})
            hidden_losses.append(layer_hidden_loss)
            hidden_cosines.append(hidden_cos.mean())

        patch_loss = torch.stack(patch_losses).mean()
        hidden_loss = torch.stack(hidden_losses).mean()
        total_loss = (
            self.global_weight * global_loss
            + self.patch_weight * patch_loss
            + self.hidden_weight * hidden_loss
        )
        ensure_finite("total_loss", total_loss, context)

        projected_cosine = total_loss.new_tensor(float("nan"))
        if teacher.get("global_projected") is not None and student.get("global_projected") is not None:
            projected_cosine = cosine_per_item(
                student["global_projected"], teacher["global_projected"], self.eps
            ).mean()
        return {
            "total_loss": total_loss,
            "global_loss": global_loss,
            "patch_loss": patch_loss,
            "hidden_loss": hidden_loss,
            "semantic_weight": semantic_weight.mean(),
            "global_cosine": global_cosine_items.mean(),
            "patch_cosine": torch.stack(patch_cosines).mean(),
            "hidden_cosine": torch.stack(hidden_cosines).mean(),
            "projected_cosine": projected_cosine,
        }



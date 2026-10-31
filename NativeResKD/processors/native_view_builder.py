from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .tissue_mask import patch_tissue_ratios


FULL = "full"
BACKGROUND_TRIM = "background_trim"
LOSSY_TRIM = "lossy_trim"


@dataclass(frozen=True)
class ViewPlan:
    sampled_view_type: str
    view_type: str
    top: int
    left: int
    height: int
    width: int
    tissue_retention_ratio: float

    @property
    def crop_box_patch(self) -> tuple[int, int, int, int]:
        return self.top, self.left, self.height, self.width

    @property
    def grid_hw(self) -> tuple[int, int]:
        return self.height, self.width


def build_patch_mapping(top: int, left: int, height: int, width: int) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    return torch.stack([y + top, x + left], dim=-1).reshape(-1, 2).long()


def _retention(mask: torch.Tensor, top: int, left: int, height: int, width: int) -> float:
    total = int(mask.sum())
    if total == 0:
        raise ValueError("Teacher view contains zero tissue pixels")
    kept = int(mask[
        top * 16 : (top + height) * 16,
        left * 16 : (left + width) * 16,
    ].sum())
    return min(1.0, max(0.0, kept / total))


def background_dominant_trim(
    tissue_mask: torch.Tensor,
    *,
    patch_size: int = 16,
    patch_tissue_threshold: float = 0.05,
    removable_background_fraction: float = 0.95,
    min_grid_h: int = 24,
    min_grid_w: int = 24,
) -> tuple[int, int, int, int]:
    ratios = patch_tissue_ratios(tissue_mask, patch_size)
    background = ratios < patch_tissue_threshold
    top, left = 0, 0
    bottom, right = background.shape[0], background.shape[1]
    changed = True
    while changed:
        changed = False
        if bottom - top > min_grid_h and background[top, left:right].float().mean() >= removable_background_fraction:
            top += 1
            changed = True
        if bottom - top > min_grid_h and background[bottom - 1, left:right].float().mean() >= removable_background_fraction:
            bottom -= 1
            changed = True
        if right - left > min_grid_w and background[top:bottom, left].float().mean() >= removable_background_fraction:
            left += 1
            changed = True
        if right - left > min_grid_w and background[top:bottom, right - 1].float().mean() >= removable_background_fraction:
            right -= 1
            changed = True
    return top, left, bottom - top, right - left


def max_retained_tissue_crop(
    tissue_mask: torch.Tensor,
    target_h: int,
    target_w: int,
    rng: np.random.Generator,
    patch_size: int = 16,
) -> tuple[int, int, int, int]:
    ratios = patch_tissue_ratios(tissue_mask, patch_size)
    grid_h, grid_w = ratios.shape
    if target_h > grid_h or target_w > grid_w:
        raise ValueError("Target crop exceeds teacher grid")
    scores: list[tuple[float, int, int]] = []
    for top in range(grid_h - target_h + 1):
        for left in range(grid_w - target_w + 1):
            score = float(ratios[top : top + target_h, left : left + target_w].sum())
            scores.append((score, top, left))
    maximum = max(score for score, _, _ in scores)
    ties = [(top, left) for score, top, left in scores if abs(score - maximum) <= 1e-12]
    top, left = ties[int(rng.integers(len(ties)))]
    return top, left, target_h, target_w


class NativeViewBuilder:
    def __init__(
        self,
        *,
        p_full: float = 0.30,
        p_background_trim: float = 0.50,
        p_lossy_trim: float = 0.20,
        min_grid_h: int = 24,
        min_grid_w: int = 24,
        max_grid_h: int = 32,
        max_grid_w: int = 32,
        patch_tissue_threshold: float = 0.05,
        removable_background_fraction: float = 0.95,
        patch_size: int = 16,
    ):
        probabilities = np.asarray([p_full, p_background_trim, p_lossy_trim], dtype=np.float64)
        if not np.isclose(probabilities.sum(), 1.0) or np.any(probabilities < 0):
            raise ValueError(f"View probabilities must be nonnegative and sum to one: {probabilities}")
        self.probabilities = probabilities
        self.min_grid_h, self.min_grid_w = min_grid_h, min_grid_w
        self.max_grid_h, self.max_grid_w = max_grid_h, max_grid_w
        self.patch_tissue_threshold = patch_tissue_threshold
        self.removable_background_fraction = removable_background_fraction
        self.patch_size = patch_size

    @classmethod
    def from_config(cls, cfg) -> "NativeViewBuilder":
        return cls(
            p_full=cfg.student_view.p_full,
            p_background_trim=cfg.student_view.p_background_trim,
            p_lossy_trim=cfg.student_view.p_lossy_trim,
            min_grid_h=cfg.student_view.min_grid_h,
            min_grid_w=cfg.student_view.min_grid_w,
            max_grid_h=cfg.student_view.max_grid_h,
            max_grid_w=cfg.student_view.max_grid_w,
            patch_tissue_threshold=cfg.tissue_mask.background_patch_tissue_threshold,
            removable_background_fraction=cfg.tissue_mask.removable_border_background_fraction,
            patch_size=cfg.teacher.expected_patch_size,
        )

    def build(self, tissue_mask: torch.Tensor, rng: np.random.Generator) -> ViewPlan:
        total = int(tissue_mask.sum())
        if total == 0:
            raise ValueError("Teacher view contains zero tissue pixels")
        sampled = str(rng.choice([FULL, BACKGROUND_TRIM, LOSSY_TRIM], p=self.probabilities))
        if sampled == FULL:
            box = (0, 0, self.max_grid_h, self.max_grid_w)
            executed = FULL
        elif sampled == BACKGROUND_TRIM:
            box = background_dominant_trim(
                tissue_mask,
                patch_size=self.patch_size,
                patch_tissue_threshold=self.patch_tissue_threshold,
                removable_background_fraction=self.removable_background_fraction,
                min_grid_h=self.min_grid_h,
                min_grid_w=self.min_grid_w,
            )
            if box == (0, 0, self.max_grid_h, self.max_grid_w):
                executed = FULL
            else:
                executed = BACKGROUND_TRIM
        else:
            while True:
                target_h = int(rng.integers(self.min_grid_h, self.max_grid_h + 1))
                target_w = int(rng.integers(self.min_grid_w, self.max_grid_w + 1))
                if target_h < self.max_grid_h or target_w < self.max_grid_w:
                    break
            box = max_retained_tissue_crop(
                tissue_mask, target_h, target_w, rng, patch_size=self.patch_size
            )
            executed = LOSSY_TRIM
        top, left, height, width = box
        return ViewPlan(
            sampled_view_type=sampled,
            view_type=executed,
            top=top,
            left=left,
            height=height,
            width=width,
            tissue_retention_ratio=_retention(tissue_mask, top, left, height, width),
        )

    def deterministic_plan(
        self,
        tissue_mask: torch.Tensor,
        target_grid: tuple[int, int] | None,
        seed: int,
    ) -> ViewPlan:
        rng = np.random.default_rng(seed)
        if target_grid is None:
            box = background_dominant_trim(
                tissue_mask,
                patch_size=self.patch_size,
                patch_tissue_threshold=self.patch_tissue_threshold,
                removable_background_fraction=self.removable_background_fraction,
                min_grid_h=self.min_grid_h,
                min_grid_w=self.min_grid_w,
            )
            view = BACKGROUND_TRIM if box != (0, 0, self.max_grid_h, self.max_grid_w) else FULL
        elif tuple(target_grid) == (self.max_grid_h, self.max_grid_w):
            box, view = (0, 0, self.max_grid_h, self.max_grid_w), FULL
        else:
            box = max_retained_tissue_crop(tissue_mask, *target_grid, rng, patch_size=self.patch_size)
            view = LOSSY_TRIM
        top, left, height, width = box
        return ViewPlan(
            sampled_view_type=view,
            view_type=view,
            top=top,
            left=left,
            height=height,
            width=width,
            tissue_retention_ratio=_retention(tissue_mask, top, left, height, width),
        )



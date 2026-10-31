from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..criterions.representation_kd_loss import gather_teacher_patches
from ..datasets.kd_dataset import image_to_normalized_tensor
from ..processors.native_view_builder import NativeViewBuilder, build_patch_mapping
from ..processors.tissue_mask import build_tissue_mask


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float(), b.float(), dim=-1).mean().cpu())


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "sample_count": int(array.size),
    }


class RepresentationEvaluator:
    def __init__(self, cfg, model, output_dir: str | Path | None = None):
        self.cfg = cfg
        self.model = model
        requested = str(cfg.runtime.device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.model.to(self.device).eval()
        self.output_dir = Path(output_dir or Path(cfg.output_dir) / "validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.view_builder = NativeViewBuilder.from_config(cfg)

    @torch.no_grad()
    def evaluate(self, metadata: list[dict[str, Any]], model_id: str) -> tuple[list[dict], list[dict]]:
        records: list[dict[str, Any]] = []
        grids = [tuple(map(int, grid)) for grid in self.cfg.validation.lossy_grids]
        for index, item in enumerate(metadata):
            image = Image.open(item["image_path"]).convert("RGB")
            if image.size != (512, 512):
                raise ValueError(f"Validation image must be 512x512, got {image.size}")
            mask = build_tissue_mask(image)
            plans = [("full", self.view_builder.deterministic_plan(mask, (32, 32), index))]
            plans.append(("background_trim", self.view_builder.deterministic_plan(mask, None, index)))
            for grid in grids:
                plans.append((f"lossy_grid_{grid[0]}x{grid[1]}", self.view_builder.deterministic_plan(mask, grid, index)))
            teacher_image = image_to_normalized_tensor(
                image, self.cfg.dataset.image_mean, self.cfg.dataset.image_std
            ).unsqueeze(0).to(self.device)
            teacher = self.model.teacher(teacher_image)
            for view_label, plan in plans:
                patch = int(self.cfg.teacher.expected_patch_size)
                crop = image.crop((
                    plan.left * patch,
                    plan.top * patch,
                    (plan.left + plan.width) * patch,
                    (plan.top + plan.height) * patch,
                ))
                student_image = image_to_normalized_tensor(
                    crop, self.cfg.dataset.image_mean, self.cfg.dataset.image_std
                ).unsqueeze(0).to(self.device)
                student = self.model.student(student_image)
                mapping = build_patch_mapping(*plan.crop_box_patch).unsqueeze(0).to(self.device)
                patch_cosines, hidden_cosines = [], []
                for layer in self.cfg.loss.selected_layers:
                    target = gather_teacher_patches(teacher["hidden_patch"][layer], mapping, (32, 32))
                    patch_cosines.append(_cosine(student["hidden_patch"][layer], target))
                    hidden_cosines.append(_cosine(student["hidden_global"][layer], teacher["hidden_global"][layer]))
                projected = None
                if student["global_projected"] is not None and teacher["global_projected"] is not None:
                    projected = _cosine(student["global_projected"], teacher["global_projected"])
                records.append({
                    "sample_id": str(item.get("sample_id", index)),
                    "model_id": model_id,
                    "view_type": view_label,
                    "executed_view_type": plan.view_type,
                    "grid_h": plan.height,
                    "grid_w": plan.width,
                    "sequence_length": plan.height * plan.width,
                    "aspect_ratio": plan.width / plan.height,
                    "tissue_retention_ratio": plan.tissue_retention_ratio,
                    "global_preprojection_cosine": _cosine(student["global_preproj"], teacher["global_preproj"]),
                    "projected_embedding_cosine": projected,
                    "selected_layer_global_cosine": float(np.mean(hidden_cosines)),
                    "selected_layer_patch_cosine": float(np.mean(patch_cosines)),
                })
        summaries = self.summarize(records)
        self._save(records, summaries)
        return records, summaries

    @staticmethod
    def summarize(records: list[dict]) -> list[dict]:
        group_keys = ("model_id", "view_type", "grid_h", "grid_w", "sequence_length", "aspect_ratio")
        metric_keys = (
            "global_preprojection_cosine", "projected_embedding_cosine",
            "selected_layer_global_cosine", "selected_layer_patch_cosine",
        )
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for record in records:
            groups[tuple(record[key] for key in group_keys)].append(record)
        summaries = []
        for group, items in groups.items():
            base = dict(zip(group_keys, group))
            for metric in metric_keys:
                values = [float(item[metric]) for item in items if item[metric] is not None]
                if values:
                    summaries.append({**base, "metric": metric, **_stats(values)})
        return summaries

    def _save(self, records: list[dict], summaries: list[dict]) -> None:
        per_sample = self.output_dir / "representation_per_sample.csv"
        if records:
            with per_sample.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        (self.output_dir / "representation_summary.json").write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False) + "\n"
        )



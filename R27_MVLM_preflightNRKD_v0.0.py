"""One-batch architecture, gradient, dataset, and variable-shape preflight."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from NativeResKD.common.utils import atomic_write_json, move_to_device, set_seed
from NativeResKD.configs.configHandler import ConfigHandler
from NativeResKD.criterions.NRKDLoss import NRKDLoss
from NativeResKD.datasets.building_datasetHandler import datasetHandler
from NativeResKD.models.conch_loader import model_preflight
from NativeResKD.models.conch_rope_student import verify_surgery_initialization
from NativeResKD.models.modeling_NRKD import NativeResolutionKnowledgeDistiller


def main() -> None:
    handler = ConfigHandler.get_cfg()
    cfg = handler.cfg
    set_seed(cfg.seed, cfg.deterministic)
    handler.finalize_runtime()
    cfg.validate()

    data = datasetHandler.from_config(cfg)
    wrapper = NativeResolutionKnowledgeDistiller.from_config(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    wrapper.to(device).train()
    batch = move_to_device(next(iter(data.train_loader)), device)
    outputs = wrapper(
        batch["teacher_image"],
        batch["student_image"],
        batch["patch_mapping"],
    )
    criterion = NRKDLoss.from_config(cfg).to(device)
    losses = criterion(
        outputs,
        batch["patch_mapping"],
        batch["tissue_retention_ratio"],
        batch["view_type"],
        batch["teacher_grid_hw"],
        batch["sample_id"],
    )
    losses["total_loss"].backward()
    teacher_has_grad = any(parameter.grad is not None for parameter in wrapper.teacher.parameters())
    missing_student_gradients = [
        name
        for name, parameter in wrapper.student.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    trainable = sum(parameter.requires_grad for parameter in wrapper.student.parameters())
    gradient_fraction = (trainable - len(missing_student_gradients)) / max(1, trainable)

    variable_shapes = []
    wrapper.student.eval()
    with torch.no_grad():
        for height, width in ((512, 512), (384, 512), (512, 384), (384, 384)):
            output = wrapper.student(torch.zeros(1, 3, height, width, device=device))
            variable_shapes.append(
                {
                    "input_hw": [height, width],
                    "grid_hw": list(output["grid_hw"]),
                    "spatial_tokens": output["spatial_tokens"],
                }
            )
    report = {
        "framework": "NativeResKD",
        "abbreviation": "NRKD",
        "version": "4.01",
        **model_preflight(wrapper.teacher.vision_tower),
        "dataset_train_count": len(data.train_dataset),
        "dataset_valid_count": len(data.valid_dataset),
        "dataset_metadata_reports": data.metadata_reports,
        "one_batch_total_loss": float(losses["total_loss"].detach().float().cpu()),
        "teacher_has_grad": teacher_has_grad,
        "student_gradient_parameter_fraction": gradient_fraction,
        "missing_student_gradient_names": missing_student_gradients,
        "variable_shapes": variable_shapes,
        "surgery": verify_surgery_initialization(wrapper.teacher.vision_tower, wrapper.student),
    }
    target = Path(cfg.output_dir) / "model_preflight.json"
    atomic_write_json(target, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if teacher_has_grad or missing_student_gradients:
        raise RuntimeError("Gradient preflight failed")


if __name__ == "__main__":
    main()

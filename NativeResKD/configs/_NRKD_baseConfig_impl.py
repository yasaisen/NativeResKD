"""NativeResKD 4.01 flat, serializable project configuration."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


PROJECT_MAPPING_DICT = {
    "NativeResKD": {
        "train_ann_file": "[metadata]downstreamRepGenVLM[NAS6train]_512x512_mpp0.5_2608281427.json",
        "valid_ann_file": "[metadata]downstreamRepGenVLM[NAS6test]_512x512_mpp0.5_2608281427.json",
        "train_size_expected": 40000,
        "valid_size_expected": 4000,
        "num_epochs": 2,
        "target_effective_batch_size": 32,
    },
    "NULLMODE": {
        "train_ann_file": "",
        "valid_ann_file": "",
        "train_size_expected": 40000,
        "valid_size_expected": 4000,
        "num_epochs": 2,
        "target_effective_batch_size": 32,
    },
}


def _namespace(**kwargs):
    return SimpleNamespace(**kwargs)


class NRKD_baseConfig:
    """Flat JSON config with nested read-only views for NRKD domain modules."""

    framework_name = "NativeResKD"
    framework_abbreviation = "NRKD"
    framework_version = "4.01"

    def __init__(self, project_name: str | None = "NativeResKD", is_HPC: bool = False):
        self.project_name = project_name or "NULLMODE"
        self.project_abbreviation = self.framework_abbreviation
        self.version = self.framework_version
        self.is_HPC = bool(is_HPC)
        self.factory_timestamp = datetime.now().strftime("%y%m%d%H%M")
        project = PROJECT_MAPPING_DICT[self.project_name]

        self.setup_path(project)
        self.setup_datasetHandler(project)
        self.setup_modelBuilder(project)
        self.setup_trainer(project)
        self.setup_configHandler(project)
        self.setup_evaluation(project)

    def setup_path(self, project: dict[str, Any]) -> None:
        workspace = "/home/yasaisen/Desktop/27_research/research_main_v4/lab_01"
        self.localGPU_image_path = "/media/yasaisen/NAS8/for_research/datasets/WSI_fromNAS"
        self.localGPU_metadata_path = workspace
        self.localGPU_weight_path = "/media/yasaisen/NAS8/for_research/weights"
        self.localGPU_bracs_path = "/media/yasaisen/NAS8/for_research/datasets/BRACS/BRACS_RoI/latest_version"
        self.localGPU_root_path = "/media/yasaisen/NAS8/for_research/R27_MVLM_v4.01"

        if self.is_HPC:
            self.image_path = "/work/misaka13/datasets/WSI_fromNAS"
            self.metadata_path = "/work/misaka13/metadatas"
            self.weight_path = "/work/misaka13/weights"
            self.bracs_path = "/work/misaka13/datasets/BRACS/BRACS_RoI/latest_version"
            self.root_path = "/work/misaka13/R27_MVLM_v4.01"
        else:
            self.image_path = self.localGPU_image_path
            self.metadata_path = self.localGPU_metadata_path
            self.weight_path = self.localGPU_weight_path
            self.bracs_path = self.localGPU_bracs_path
            self.root_path = self.localGPU_root_path

        self.train_metadata_path = (
            os.path.join(self.metadata_path, project["train_ann_file"])
            if project["train_ann_file"] else None
        )
        self.valid_metadata_path = (
            os.path.join(self.metadata_path, project["valid_ann_file"])
            if project["valid_ann_file"] else None
        )
        self.roi_root = self.image_path
        self.conch_checkpoint_path = os.path.join(self.weight_path, "conchv1_5")
        self.bracs_metadata_path = "projects/bracs_metadata.csv"
        self.bracs_roi_root = self.bracs_path
        self.manifest_dir = "projects/generated_kd_manifests"

    def setup_datasetHandler(self, project: dict[str, Any]) -> None:
        self.train_size_expected = int(project["train_size_expected"])
        self.valid_size_expected = int(project["valid_size_expected"])
        self.strict_size = True
        self.magnification = 20.0
        self.expected_mpp = 0.5
        self.source_h = 512
        self.source_w = 512
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.selection_strategy = "case_aware_deterministic_round_robin"
        self.selection_seed = 42
        self.reuse_manifest = True
        self.qc_images = True
        self.qc_workers = 8
        self.reject_zero_tissue = True
        self.forbidden_roots = ["/media/yasaisen/NAS8/for_research/datasets/BRACS"]
        self.batch_size = 1
        self.per_device_batch_size = 1
        self.num_workers = 8
        self.pin_memory = True
        self.drop_last = False
        self.prefetch_factor = None
        self.num_batchs_per_epoch = None
        self.dataloader_seed = 42

        self.horizontal_flip = True
        self.vertical_flip = True
        self.rotation_90 = True
        self.tissue_mask_method = "hsv_saturation_otsu"
        self.background_patch_tissue_threshold = 0.05
        self.removable_border_background_fraction = 0.95
        self.p_full = 0.30
        self.p_background_trim = 0.50
        self.p_lossy_trim = 0.20
        self.min_grid_h = 24
        self.min_grid_w = 24
        self.max_grid_h = 32
        self.max_grid_w = 32
        self.background_trim_fallback = "full"
        self.lossy_crop_strategy = "max_retained_tissue"

    def setup_modelBuilder(self, project: dict[str, Any]) -> None:
        self.model_name = "MahmoodLab/conchv1_5"
        self.teacher_anchor_size = 512
        self.teacher_freeze = True
        self.ape_interpolation_if_needed = True
        self.expected_patch_size = 16
        self.expected_depth = 24
        self.student_init_from_teacher = True
        self.student_remove_ape = True
        self.student_full_parameter_train = True
        self.rope_type = "mixed_2d"
        self.rope_theta = 10.0
        self.rope_learnable_frequency = True
        self.rope_apply_to_q = True
        self.rope_apply_to_k = True
        self.rope_apply_to_v = False
        self.rope_exclude_prefix_tokens = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def setup_trainer(self, project: dict[str, Any]) -> None:
        self.trainer_mode = None
        self.training_mode = "DDP"
        self.num_epochs = int(project["num_epochs"])
        self.target_effective_batch_size = int(project["target_effective_batch_size"])
        self.accumulation_steps = 8 if self.is_HPC else 32
        self.gradient_clip_norm = 1.0
        self.precision = "bf16"
        self.amp = True
        self.find_unused_params = False
        self.log_every_n_steps = 10
        self.monitoring_every_n_steps = 500
        self.finite_debug = False

        self.global_loss_weight = 1.0
        self.patch_loss_weight = 0.5
        self.hidden_loss_weight = 0.25
        self.lossy_global_semantic_weight = "tissue_retention_squared"
        self.selected_layers = [6, 12, 18, 24]
        self.cosine_eps = 1.0e-8

        self.learning_rate_dict = {"backbone": 1.0e-5, "rope": 1.0e-4}
        self.weight_decay = 0.05
        self.optimizer_beta1 = 0.9
        self.optimizer_beta2 = 0.95
        self.total_steps = None
        self.warmup_steps = None
        self.warmup_ratio = 0.05
        self.max_warmup_steps = 5000

        self.weight_filename = "best_model.pth"
        self.save_freq = 1
        self.save_latest = True
        self.save_best = True
        self.checkpoint_selection_metric = "val_total_loss"
        self.early_stop_patience = None
        self.validate_every_n_epochs = 1

    def setup_configHandler(self, project: dict[str, Any]) -> None:
        self.seed = 42
        self.deterministic = True
        self.nowtime = None
        self.save_path = None
        self.world_size = 1
        self.distributed = True
        self.is_distributed = False
        self.dist_url = "env://"
        self.rank = 0
        self.gpu = 0
        self.local_rank = 0
        self.dist_backend = None

    def setup_evaluation(self, project: dict[str, Any]) -> None:
        self.validation_lossy_grids = [
            [24, 24], [24, 28], [24, 32],
            [28, 24], [28, 28], [28, 32],
            [32, 24], [32, 28], [32, 32],
        ]
        self.bracs_source_magnification = 40
        self.bracs_target_magnification = 20
        self.bracs_physical_resample = "lanczos"
        self.bracs_primary_metric = "macro_f1"
        self.bracs_max_native_tokens = None
        self.bracs_native_patch_multiple_policy = "white_pad"
        self.bracs_linear_probe_C = [0.01, 0.1, 1.0, 10.0]
        self.bracs_max_iter = 5000

    @property
    def output_dir(self) -> str:
        return self.save_path or os.path.join(self.root_path, "checkpoints", "uninitialized")

    @property
    def teacher(self):
        return _namespace(
            model_id=self.model_name,
            checkpoint_path=self.conch_checkpoint_path,
            anchor_size=self.teacher_anchor_size,
            freeze=self.teacher_freeze,
            ape_interpolation_if_needed=self.ape_interpolation_if_needed,
            expected_patch_size=self.expected_patch_size,
            expected_depth=self.expected_depth,
        )

    @property
    def student(self):
        return _namespace(
            init_from_teacher=self.student_init_from_teacher,
            remove_ape=self.student_remove_ape,
            full_parameter_train=self.student_full_parameter_train,
        )

    @property
    def rope(self):
        return _namespace(
            type=self.rope_type,
            theta=self.rope_theta,
            learnable_frequency=self.rope_learnable_frequency,
            apply_to_q=self.rope_apply_to_q,
            apply_to_k=self.rope_apply_to_k,
            apply_to_v=self.rope_apply_to_v,
            exclude_prefix_tokens=self.rope_exclude_prefix_tokens,
        )

    @property
    def dataset(self):
        return _namespace(
            train_metadata_path=self.train_metadata_path,
            valid_metadata_path=self.valid_metadata_path,
            roi_root=self.roi_root,
            train_size_expected=self.train_size_expected,
            valid_size_expected=self.valid_size_expected,
            strict_size=self.strict_size,
            magnification=self.magnification,
            expected_mpp=self.expected_mpp,
            source_h=self.source_h,
            source_w=self.source_w,
            image_mean=self.image_mean,
            image_std=self.image_std,
            selection_strategy=self.selection_strategy,
            selection_seed=self.selection_seed,
            manifest_dir=self.manifest_dir,
            reuse_manifest=self.reuse_manifest,
            qc_images=self.qc_images,
            qc_workers=self.qc_workers,
            reject_zero_tissue=self.reject_zero_tissue,
            forbidden_roots=self.forbidden_roots,
        )

    @property
    def augmentation(self):
        return _namespace(
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            rotation_90=self.rotation_90,
        )

    @property
    def tissue_mask(self):
        return _namespace(
            method=self.tissue_mask_method,
            background_patch_tissue_threshold=self.background_patch_tissue_threshold,
            removable_border_background_fraction=self.removable_border_background_fraction,
        )

    @property
    def student_view(self):
        return _namespace(
            p_full=self.p_full,
            p_background_trim=self.p_background_trim,
            p_lossy_trim=self.p_lossy_trim,
            min_grid_h=self.min_grid_h,
            min_grid_w=self.min_grid_w,
            max_grid_h=self.max_grid_h,
            max_grid_w=self.max_grid_w,
            background_trim_fallback=self.background_trim_fallback,
            lossy_crop_strategy=self.lossy_crop_strategy,
        )

    @property
    def loss(self):
        return _namespace(
            global_weight=self.global_loss_weight,
            patch_weight=self.patch_loss_weight,
            hidden_weight=self.hidden_loss_weight,
            lossy_global_semantic_weight=self.lossy_global_semantic_weight,
            selected_layers=self.selected_layers,
            cosine_eps=self.cosine_eps,
        )

    @property
    def optimizer(self):
        return _namespace(
            name="adamw",
            backbone_lr=self.learning_rate_dict["backbone"],
            rope_frequency_lr=self.learning_rate_dict["rope"],
            weight_decay=self.weight_decay,
            beta1=self.optimizer_beta1,
            beta2=self.optimizer_beta2,
        )

    @property
    def scheduler(self):
        return _namespace(name="cosine", warmup_ratio=self.warmup_ratio)

    @property
    def training(self):
        return _namespace(
            epochs=self.num_epochs,
            target_effective_batch_size=self.target_effective_batch_size,
            per_device_batch_size=self.per_device_batch_size,
            accumulation_steps=self.accumulation_steps,
            gradient_clip_norm=self.gradient_clip_norm,
            precision=self.precision,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            log_every_n_steps=self.log_every_n_steps,
            validate_every_n_epochs=self.validate_every_n_epochs,
            finite_debug=self.finite_debug,
        )

    @property
    def validation(self):
        return _namespace(lossy_grids=self.validation_lossy_grids)

    @property
    def bracs(self):
        return _namespace(
            metadata_path=self.bracs_metadata_path,
            roi_root=self.bracs_roi_root,
            source_magnification=self.bracs_source_magnification,
            target_magnification=self.bracs_target_magnification,
            physical_resample=self.bracs_physical_resample,
            primary_metric=self.bracs_primary_metric,
            max_native_tokens=self.bracs_max_native_tokens,
            native_patch_multiple_policy=self.bracs_native_patch_multiple_policy,
            linear_probe_C=self.bracs_linear_probe_C,
            max_iter=self.bracs_max_iter,
        )

    @property
    def runtime(self):
        return _namespace(
            device=self.device,
            distributed=self.distributed,
            deterministic=self.deterministic,
            rank=self.rank,
            world_size=self.world_size,
            local_rank=self.local_rank,
            is_distributed=self.is_distributed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

    def print_config(self) -> None:
        from ..common.dist_utils import is_main_process
        from ..common.utils import log_print

        log_print("Training Configuration:")
        if is_main_process():
            for key, value in self.to_dict().items():
                print(f"  {key}: {value}")

    def validate(self) -> None:
        if self.project_name != "NativeResKD":
            raise ValueError(f"Formal NRKD config requires project_name='NativeResKD', got {self.project_name!r}")
        if str(self.version) != self.framework_version:
            raise ValueError(f"NativeResKD config version must be {self.framework_version}, got {self.version}")
        if self.training_mode != "DDP":
            raise ValueError("NativeResKD 4.01 supports DDP training mode only")
        physical = int(self.per_device_batch_size) * max(1, int(self.world_size))
        if physical * int(self.accumulation_steps) != int(self.target_effective_batch_size):
            raise ValueError(
                "Effective batch mismatch: "
                f"per_device={self.per_device_batch_size}, world_size={self.world_size}, "
                f"accumulation={self.accumulation_steps}, target={self.target_effective_batch_size}"
            )

    def save(self, save_path: str | None = None, filename: str = "config.json") -> str:
        destination = save_path if save_path is not None else self.save_path
        if destination is None:
            raise ValueError("No save_path was provided or configured")
        Path(destination).mkdir(parents=True, exist_ok=True)
        path = Path(destination) / filename
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return str(path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "NRKD_baseConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls.__new__(cls)
        for key, value in data.items():
            setattr(cfg, key, value)
        defaults = cls(project_name="NativeResKD", is_HPC=bool(data.get("is_HPC", False)))
        for key, value in defaults.to_dict().items():
            if not hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    @classmethod
    def setup_configs(cls, save_path: str = "./NativeResKD/projects") -> None:
        for is_HPC in (False, True):
            cfg = cls(project_name="NativeResKD", is_HPC=is_HPC)
            suffix = "_hpc" if is_HPC else ""
            cfg.save(save_path=save_path, filename=f"NativeResKD_config{suffix}.json")


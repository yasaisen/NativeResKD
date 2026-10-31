from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..common.utils import sample_seed
from ..processors.native_view_builder import NativeViewBuilder, ViewPlan, build_patch_mapping
from ..processors.tissue_mask import build_tissue_mask


REPGENVLM_ADAPTER_VERSION = 1


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def flatten_repgenvlm_metadata(
    payload: dict[str, Any],
    *,
    roi_root: str | Path,
    source_metadata_path: str | Path | None = None,
    expected_mpp: float = 0.5,
    expected_size: tuple[int, int] = (512, 512),
    expected_magnification: float = 20.0,
    forbidden_roots: list[str] | tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Flatten the project RepGenVLM case hierarchy into unlabeled KD records."""
    cases = payload.get("case_list")
    if not isinstance(cases, list):
        raise TypeError("RepGenVLM metadata must contain a top-level case_list")
    root = Path(roi_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"KD roi_root does not exist: {root}")
    forbidden = [Path(value).expanduser().resolve() for value in forbidden_roots]
    width_expected, height_expected = map(int, expected_size)
    records: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    seen_paths: set[str] = set()

    for case_index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise TypeError(f"case_list[{case_index}] is not an object")
        if str(case.get("organ", "")).lower() != "breast":
            raise ValueError(
                f"RepGenVLM case {case.get('sample_idx', case_index)} "
                f"organ={case.get('organ')!r}, expected Breast"
            )
        case_id = case.get("case_id")
        sample_idx = case.get("sample_idx")
        if case_id in (None, "") or sample_idx is None:
            raise ValueError(f"RepGenVLM case {case_index} requires case_id and sample_idx")
        blocks = case.get("tissue_blocks", [])
        if not isinstance(blocks, list):
            raise TypeError(f"case {sample_idx} tissue_blocks is not a list")
        for block in blocks:
            if not isinstance(block, dict) or not isinstance(block.get("stains", []), list):
                raise TypeError(f"case {sample_idx} contains an invalid tissue block")
            for stain in block.get("stains", []):
                if not isinstance(stain, dict):
                    raise TypeError(f"case {sample_idx} contains a non-object stain")
                if "roi_list" not in stain:
                    continue
                rois = stain["roi_list"]
                if not isinstance(rois, list):
                    raise TypeError(f"case {sample_idx} stain roi_list is not a list")
                if stain.get("roi_num") is not None and int(stain["roi_num"]) != len(rois):
                    raise ValueError(f"case {sample_idx} stain roi_num does not match roi_list")
                if rois and str(stain.get("stain_type", "")).upper() not in {"HE", "H&E"}:
                    raise ValueError(
                        f"case {sample_idx} has ROI data under non-H&E stain "
                        f"{stain.get('stain_type')!r}"
                    )
                for roi_index, roi in enumerate(rois):
                    if not isinstance(roi, dict) or not isinstance(roi.get("main_info"), dict):
                        raise TypeError(f"case {sample_idx} roi_list[{roi_index}] is malformed")
                    main = roi["main_info"]
                    relative_value = main.get("roi_path")
                    if not isinstance(relative_value, str) or not relative_value:
                        raise ValueError(f"case {sample_idx} ROI has no main_info.roi_path")
                    relative_path = Path(relative_value)
                    if relative_path.is_absolute():
                        raise ValueError(f"RepGenVLM roi_path must be relative: {relative_value}")
                    image_path = (root / relative_path).resolve()
                    if not _is_within(image_path, root):
                        raise ValueError(f"RepGenVLM roi_path escapes roi_root: {relative_value}")
                    if any(_is_within(image_path, blocked) for blocked in forbidden):
                        raise ValueError(f"BRACS/forbidden image entered KD metadata: {image_path}")

                    mpp = main.get("mpp")
                    if isinstance(mpp, bool) or not isinstance(mpp, (int, float)):
                        raise ValueError(f"{relative_value} has invalid MPP {mpp!r}")
                    if not math.isclose(
                        float(mpp), float(expected_mpp), rel_tol=0.0, abs_tol=1e-12
                    ):
                        raise ValueError(
                            f"{relative_value} MPP={mpp}, expected {expected_mpp} "
                            "(20x physical sampling)"
                        )
                    roi_wh = main.get("roi_wh")
                    if roi_wh != [width_expected, height_expected]:
                        raise ValueError(
                            f"{relative_value} roi_wh={roi_wh}, expected "
                            f"{[width_expected, height_expected]}"
                        )
                    if main.get("area") not in (None, width_expected * height_expected):
                        raise ValueError(f"{relative_value} area disagrees with roi_wh")
                    sample_id = relative_path.stem
                    if sample_id in seen_sample_ids:
                        raise ValueError(f"Duplicate RepGenVLM sample_id: {sample_id}")
                    path_key = str(image_path)
                    if path_key in seen_paths:
                        raise ValueError(f"Duplicate RepGenVLM image path: {image_path}")
                    seen_sample_ids.add(sample_id)
                    seen_paths.add(path_key)
                    records.append(
                        {
                            "sample_id": sample_id,
                            "case_id": str(case_id),
                            "sample_idx": int(sample_idx),
                            "stain_id": str(stain.get("stain_id", "")),
                            "global_idx": int(roi["global_idx"]),
                            "image_path": path_key,
                            "relative_image_path": relative_path.as_posix(),
                            "organ": "Breast",
                            "stain": "HE",
                            "mpp": float(mpp),
                            "magnification": float(expected_magnification),
                            "width": width_expected,
                            "height": height_expected,
                            "source_metadata_path": str(
                                Path(source_metadata_path).resolve()
                                if source_metadata_path is not None
                                else ""
                            ),
                        }
                    )
    if not records:
        raise ValueError("RepGenVLM metadata contains no H&E ROI records")
    return records


def load_metadata(
    path: str | Path,
    *,
    roi_root: str | Path | None = None,
    expected_mpp: float = 0.5,
    expected_size: tuple[int, int] = (512, 512),
    expected_magnification: float = 20.0,
    forbidden_roots: list[str] | tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "case_list" in payload:
        if roi_root is None:
            raise ValueError("Hierarchical RepGenVLM metadata requires dataset.roi_root")
        return flatten_repgenvlm_metadata(
            payload,
            roi_root=roi_root,
            source_metadata_path=path,
            expected_mpp=expected_mpp,
            expected_size=expected_size,
            expected_magnification=expected_magnification,
            forbidden_roots=forbidden_roots,
        )
    if isinstance(payload, dict):
        payload = payload.get("samples", payload.get("data"))
    if not isinstance(payload, list):
        raise TypeError("Metadata must contain a list of samples")
    return payload


def _stable_hash(*parts: Any) -> bytes:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()


def case_aware_deterministic_order(
    records: list[dict[str, Any]], *, seed: int, split: str
) -> list[dict[str, Any]]:
    """Return a stable case-balanced round-robin ordering of all candidates."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        case_id = record.get("case_id")
        sample_id = record.get("sample_id")
        if case_id in (None, "") or sample_id in (None, ""):
            raise ValueError("Case-aware selection requires case_id and sample_id")
        groups[str(case_id)].append(record)
    for case_id, items in groups.items():
        items.sort(
            key=lambda item: _stable_hash(
                seed, split, "sample", case_id, item["sample_id"]
            )
        )
    case_order = sorted(
        groups, key=lambda case_id: _stable_hash(seed, split, "case", case_id)
    )
    ordered: list[dict[str, Any]] = []
    maximum = max(map(len, groups.values()), default=0)
    for position in range(maximum):
        for case_id in case_order:
            if position < len(groups[case_id]):
                ordered.append(groups[case_id][position])
    if len(ordered) != len(records):
        raise AssertionError("Case-aware ordering lost or duplicated records")
    return ordered


def validate_kd_image_record(
    record: dict[str, Any], *, expected_size: tuple[int, int], reject_zero_tissue: bool
) -> str | None:
    """Return a QC failure reason, or ``None`` when a record is usable."""
    path = Path(record["image_path"])
    if not path.is_file():
        return "missing_image"
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                return "not_png"
            if image.mode != "RGB":
                return "not_rgb"
            if image.size != tuple(map(int, expected_size)):
                return "wrong_image_size"
            image.load()
            if reject_zero_tissue and int(build_tissue_mask(image).sum()) == 0:
                return "zero_tissue"
    except (OSError, ValueError):
        return "unreadable_image"
    return None


def select_kd_records(
    records: list[dict[str, Any]],
    *,
    target_count: int,
    seed: int,
    split: str,
    expected_size: tuple[int, int] = (512, 512),
    qc_images: bool = True,
    reject_zero_tissue: bool = True,
    qc_workers: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select an exact-size deterministic split, using later records as QC reserves."""
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if len(records) < target_count:
        raise ValueError(f"KD {split} candidates={len(records)}, target={target_count}")
    if qc_workers <= 0:
        raise ValueError("qc_workers must be positive")
    ordered = case_aware_deterministic_order(records, seed=seed, split=split)
    selected: list[dict[str, Any]] = []
    skipped_counts: Counter[str] = Counter()
    skipped_samples: list[dict[str, str]] = []
    scanned = 0
    next_progress = 5000

    if qc_images:
        chunk_size = max(64, qc_workers * 16)
        with ThreadPoolExecutor(
            max_workers=qc_workers, thread_name_prefix=f"kd-{split}-qc"
        ) as pool:
            for start in range(0, len(ordered), chunk_size):
                chunk = ordered[start : start + chunk_size]
                reasons = list(
                    pool.map(
                        lambda record: validate_kd_image_record(
                            record,
                            expected_size=expected_size,
                            reject_zero_tissue=reject_zero_tissue,
                        ),
                        chunk,
                    )
                )
                for record, reason in zip(chunk, reasons):
                    scanned += 1
                    if reason is None:
                        selected.append(record)
                    else:
                        skipped_counts[reason] += 1
                        skipped_samples.append(
                            {"sample_id": str(record["sample_id"]), "reason": reason}
                        )
                    if len(selected) == target_count:
                        break
                if scanned >= next_progress:
                    print(
                        f"KD {split} QC: scanned={scanned}, selected={len(selected)}",
                        flush=True,
                    )
                    while next_progress <= scanned:
                        next_progress += 5000
                if len(selected) == target_count:
                    break
    else:
        selected = ordered[:target_count]
        scanned = target_count
    if len(selected) != target_count:
        raise RuntimeError(
            f"KD {split} has only {len(selected)} QC-passing images, target={target_count}; "
            f"skips={dict(skipped_counts)}"
        )
    sample_ids = [str(record["sample_id"]) for record in selected]
    report = {
        "adapter_version": REPGENVLM_ADAPTER_VERSION,
        "split": split,
        "strategy": "case_aware_deterministic_round_robin",
        "seed": int(seed),
        "candidate_count": len(records),
        "candidate_case_count": len({str(record["case_id"]) for record in records}),
        "scanned_count": scanned,
        "selected_count": len(selected),
        "selected_case_count": len({str(record["case_id"]) for record in selected}),
        "skipped_counts": dict(skipped_counts),
        "skipped_samples": skipped_samples,
        "selection_sha256": hashlib.sha256(
            "\n".join(sample_ids).encode("utf-8")
        ).hexdigest(),
    }
    return selected, report


def _shared_augmentation(image: Image.Image, rng: np.random.Generator, cfg) -> Image.Image:
    image = image.convert("RGB")
    if cfg.horizontal_flip and bool(rng.integers(2)):
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if cfg.vertical_flip and bool(rng.integers(2)):
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if cfg.rotation_90:
        turns = int(rng.integers(4))
        if turns:
            image = image.rotate(90 * turns, expand=False)
    return image


def image_to_normalized_tensor(image: Image.Image, mean: list[float], std: list[float]) -> torch.Tensor:
    array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


class KDDataset(Dataset):
    """Shared-augmentation teacher/student KD dataset with epoch-stable views."""

    def __init__(
        self,
        metadata: list[dict[str, Any]],
        cfg,
        *,
        training: bool,
        view_builder: NativeViewBuilder | None = None,
    ):
        self.metadata = metadata
        self.cfg = cfg
        self.training = training
        self.view_builder = view_builder or NativeViewBuilder.from_config(cfg)
        self.epoch = 0
        self._plans: dict[int, ViewPlan] = {}
        self._validate_metadata()

    @classmethod
    def from_path(cls, path: str | Path, cfg, *, training: bool) -> "KDDataset":
        return cls(
            load_metadata(
                path,
                roi_root=cfg.dataset.roi_root,
                expected_mpp=cfg.dataset.expected_mpp,
                expected_size=(cfg.dataset.source_w, cfg.dataset.source_h),
                expected_magnification=cfg.dataset.magnification,
                forbidden_roots=cfg.dataset.forbidden_roots,
            ),
            cfg,
            training=training,
        )

    def _validate_metadata(self) -> None:
        for index, item in enumerate(self.metadata):
            if "image_path" not in item:
                raise KeyError(f"metadata[{index}] has no image_path")
            item.setdefault("sample_id", Path(str(item["image_path"])).stem)
            for key, expected in (("width", self.cfg.dataset.source_w), ("height", self.cfg.dataset.source_h)):
                if item.get(key) not in (None, "") and int(item[key]) != int(expected):
                    raise ValueError(f"{item['sample_id']} metadata {key}={item[key]}, expected {expected}")
            if item.get("magnification") not in (None, "") and float(item["magnification"]) != float(self.cfg.dataset.magnification):
                raise ValueError(
                    f"{item['sample_id']} magnification={item['magnification']}, expected {self.cfg.dataset.magnification}"
                )
            if item.get("mpp") not in (None, "") and not math.isclose(
                float(item["mpp"]),
                float(self.cfg.dataset.expected_mpp),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{item['sample_id']} mpp={item['mpp']}, expected {self.cfg.dataset.expected_mpp}"
                )
            if item.get("organ") not in (None, "") and str(item["organ"]).lower() != "breast":
                raise ValueError(f"{item['sample_id']} is not Breast")
            if item.get("stain") not in (None, "") and str(item["stain"]).upper() not in {"HE", "H&E"}:
                raise ValueError(f"{item['sample_id']} is not H&E")

    def __len__(self) -> int:
        return len(self.metadata)

    def _raw_augmented(self, index: int) -> tuple[Image.Image, np.random.Generator]:
        item = self.metadata[index]
        image = Image.open(item["image_path"]).convert("RGB")
        expected = (int(self.cfg.dataset.source_w), int(self.cfg.dataset.source_h))
        if image.size != expected:
            raise ValueError(f"{item['sample_id']} image size={image.size}, expected={expected}; resize is forbidden")
        seed = sample_seed(int(self.cfg.seed), self.epoch, index, "shared_augmentation")
        rng = np.random.default_rng(seed)
        if self.training:
            image = _shared_augmentation(image, rng, self.cfg.augmentation)
        return image, rng

    def plan_for_index(self, index: int) -> ViewPlan:
        if index not in self._plans:
            image, _ = self._raw_augmented(index)
            mask = build_tissue_mask(image)
            rng = np.random.default_rng(sample_seed(int(self.cfg.seed), self.epoch, index, "view"))
            if self.training:
                plan = self.view_builder.build(mask, rng)
            else:
                plan = self.view_builder.deterministic_plan(
                    mask,
                    (self.view_builder.max_grid_h, self.view_builder.max_grid_w),
                    seed=sample_seed(int(self.cfg.seed), 0, index, "validation_full"),
                )
            self._plans[index] = plan
        return self._plans[index]

    def shape_for_index(self, index: int) -> tuple[int, int]:
        return self.plan_for_index(index).grid_hw

    def set_epoch(self, epoch: int, precompute: bool = True) -> None:
        self.epoch = int(epoch)
        self._plans.clear()
        if precompute:
            for index in range(len(self)):
                self.plan_for_index(index)

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.set_epoch(int(state.get("epoch", 0)), precompute=False)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.metadata[index]
        image, _ = self._raw_augmented(index)
        plan = self.plan_for_index(index)
        patch = int(self.cfg.teacher.expected_patch_size)
        left, top = plan.left * patch, plan.top * patch
        right = (plan.left + plan.width) * patch
        bottom = (plan.top + plan.height) * patch
        student_image = image.crop((left, top, right, bottom))
        teacher_tensor = image_to_normalized_tensor(
            image, self.cfg.dataset.image_mean, self.cfg.dataset.image_std
        )
        student_tensor = image_to_normalized_tensor(
            student_image, self.cfg.dataset.image_mean, self.cfg.dataset.image_std
        )
        return {
            "sample_id": str(item["sample_id"]),
            "teacher_image": teacher_tensor,
            "student_image": student_tensor,
            "sampled_view_type": plan.sampled_view_type,
            "view_type": plan.view_type,
            "crop_box_patch": torch.tensor(plan.crop_box_patch, dtype=torch.long),
            "patch_mapping": build_patch_mapping(*plan.crop_box_patch),
            "teacher_grid_hw": torch.tensor([32, 32], dtype=torch.long),
            "student_grid_hw": torch.tensor(plan.grid_hw, dtype=torch.long),
            "tissue_retention_ratio": torch.tensor(plan.tissue_retention_ratio, dtype=torch.float32),
        }


def kd_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    shapes = {tuple(sample["student_image"].shape[-2:]) for sample in samples}
    if len(shapes) != 1:
        raise ValueError(f"Variable-shape batch was not bucketed: {sorted(shapes)}")
    tensor_keys = (
        "teacher_image", "student_image", "crop_box_patch", "patch_mapping",
        "teacher_grid_hw", "student_grid_hw", "tissue_retention_ratio",
    )
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys}
    for key in ("sample_id", "sampled_view_type", "view_type"):
        batch[key] = [sample[key] for sample in samples]
    return batch


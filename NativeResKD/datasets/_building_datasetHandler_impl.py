"""NRKD deterministic metadata preparation and DDP-aware DataLoaders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .kd_dataset import KDDataset, kd_collate_fn, load_metadata, select_kd_records
from .shape_bucket_sampler import DynamicShapeBucketBatchSampler
from ..common.utils import atomic_write_json, log_print


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
    temporary.replace(path)


def _prepare_kd_split_local(cfg, split: str) -> tuple[list[dict], dict]:
    if split not in {"train", "valid"}:
        raise ValueError(split)
    source_value = cfg.train_metadata_path if split == "train" else cfg.valid_metadata_path
    target_count = cfg.train_size_expected if split == "train" else cfg.valid_size_expected
    source = Path(source_value).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if cfg.selection_strategy != "case_aware_deterministic_round_robin":
        raise ValueError(f"Unsupported selection strategy: {cfg.selection_strategy}")

    seed = int(cfg.selection_seed)
    manifest_dir = Path(cfg.manifest_dir).expanduser()
    stem = f"{split}_{int(target_count)}_seed{seed}"
    manifest_path = manifest_dir / f"{stem}.jsonl"
    report_path = manifest_dir / f"{stem}_report.json"
    state_path = manifest_dir / f"{stem}_state.json"
    expected_state = {
        "framework": "NativeResKD",
        "version": "4.01",
        "source_metadata_path": str(source),
        "source_metadata_sha256": _sha256_file(source),
        "roi_root": str(Path(cfg.roi_root).expanduser().resolve()),
        "target_count": int(target_count),
        "selection_seed": seed,
        "selection_strategy": cfg.selection_strategy,
        "expected_mpp": float(cfg.expected_mpp),
        "magnification": float(cfg.magnification),
        "source_wh": [int(cfg.source_w), int(cfg.source_h)],
        "forbidden_roots": sorted(
            str(Path(value).expanduser().resolve()) for value in cfg.forbidden_roots
        ),
        "qc_images": bool(cfg.qc_images),
        "reject_zero_tissue": bool(cfg.reject_zero_tissue),
    }
    if (
        cfg.reuse_manifest
        and manifest_path.is_file()
        and report_path.is_file()
        and state_path.is_file()
        and json.loads(state_path.read_text(encoding="utf-8")) == expected_state
    ):
        records = load_metadata(manifest_path)
        if len(records) != int(target_count):
            raise RuntimeError(f"Cached {split} count={len(records)}, target={target_count}")
        return records, json.loads(report_path.read_text(encoding="utf-8"))

    candidates = load_metadata(
        source,
        roi_root=cfg.roi_root,
        expected_mpp=float(cfg.expected_mpp),
        expected_size=(int(cfg.source_w), int(cfg.source_h)),
        expected_magnification=float(cfg.magnification),
        forbidden_roots=list(cfg.forbidden_roots),
    )
    records, report = select_kd_records(
        candidates,
        target_count=int(target_count),
        seed=seed,
        split=split,
        expected_size=(int(cfg.source_w), int(cfg.source_h)),
        qc_images=bool(cfg.qc_images),
        reject_zero_tissue=bool(cfg.reject_zero_tissue),
        qc_workers=int(cfg.qc_workers),
    )
    report.update(
        framework="NativeResKD",
        version="4.01",
        source_metadata_path=str(source),
        source_metadata_sha256=expected_state["source_metadata_sha256"],
        manifest_path=str(manifest_path.resolve()),
    )
    _atomic_write_jsonl(manifest_path, records)
    atomic_write_json(report_path, report)
    atomic_write_json(state_path, expected_state)
    return records, report


def prepare_kd_split(cfg, split: str) -> tuple[list[dict], dict]:
    """Build a shared manifest on rank 0 and fail every rank coherently."""
    if not (dist.is_available() and dist.is_initialized()):
        return _prepare_kd_split_local(cfg, split)

    result = None
    status = [None]
    if dist.get_rank() == 0:
        try:
            result = _prepare_kd_split_local(cfg, split)
            status[0] = {"ok": True}
        except Exception as exc:
            status[0] = {"ok": False, "type": type(exc).__name__, "message": str(exc)}
    dist.broadcast_object_list(status, src=0)
    if not status[0]["ok"]:
        raise RuntimeError(
            f"Rank-0 {split} manifest preparation failed: "
            f"{status[0]['type']}: {status[0]['message']}"
        )
    dist.barrier()
    return result if result is not None else _prepare_kd_split_local(cfg, split)


class datasetHandler:
    def __init__(
        self,
        train_dataset,
        valid_dataset,
        train_loader,
        valid_loader,
        train_sampler,
        valid_sampler,
        loader_generators,
        metadata_reports,
    ):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler
        self.loader_generators = loader_generators
        self.metadata_reports = metadata_reports

    @classmethod
    def from_config(cls, cfg) -> "datasetHandler":
        log_print("Loading NRKD DataLoaders...")
        if not cfg.train_metadata_path or not cfg.valid_metadata_path:
            raise ValueError("train_metadata_path and valid_metadata_path are required")
        train_metadata, train_report = prepare_kd_split(cfg, "train")
        valid_metadata, valid_report = prepare_kd_split(cfg, "valid")
        if cfg.strict_size:
            if len(train_metadata) != int(cfg.train_size_expected):
                raise ValueError(f"KD train count={len(train_metadata)}, expected={cfg.train_size_expected}")
            if len(valid_metadata) != int(cfg.valid_size_expected):
                raise ValueError(f"KD valid count={len(valid_metadata)}, expected={cfg.valid_size_expected}")

        train_ids = {str(item.get("sample_id", item["image_path"])) for item in train_metadata}
        valid_ids = {str(item.get("sample_id", item["image_path"])) for item in valid_metadata}
        if train_ids & valid_ids:
            raise ValueError(f"KD train/valid overlap detected ({len(train_ids & valid_ids)} samples)")
        train_cases = {str(item["case_id"]) for item in train_metadata}
        valid_cases = {str(item["case_id"]) for item in valid_metadata}
        if train_cases & valid_cases:
            raise ValueError(f"KD train/valid case leakage detected ({len(train_cases & valid_cases)} cases)")

        train_dataset = KDDataset(train_metadata, cfg, training=True)
        valid_dataset = KDDataset(valid_metadata, cfg, training=False)
        rank = int(cfg.rank)
        world_size = int(cfg.world_size)
        train_sampler = DynamicShapeBucketBatchSampler(
            train_dataset,
            int(cfg.per_device_batch_size),
            seed=int(cfg.seed),
            shuffle=True,
            drop_last=bool(cfg.drop_last),
            rank=rank,
            world_size=world_size,
            pad_distributed_batches=True,
        )
        valid_sampler = DynamicShapeBucketBatchSampler(
            valid_dataset,
            int(cfg.per_device_batch_size),
            seed=int(cfg.seed),
            shuffle=False,
            drop_last=False,
            rank=rank,
            world_size=world_size,
            pad_distributed_batches=False,
        )
        generators = {
            "train": torch.Generator().manual_seed(int(cfg.dataloader_seed)),
            "valid": torch.Generator().manual_seed(int(cfg.dataloader_seed) + 1),
        }

        def build_loader(dataset, sampler, split):
            kwargs = {
                "dataset": dataset,
                "batch_sampler": sampler,
                "num_workers": int(cfg.num_workers),
                "pin_memory": bool(cfg.pin_memory),
                "persistent_workers": False,
                "collate_fn": kd_collate_fn,
                "generator": generators[split],
            }
            if cfg.prefetch_factor is not None and int(cfg.num_workers) > 0:
                kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
            return DataLoader(**kwargs)

        train_loader = build_loader(train_dataset, train_sampler, "train")
        valid_loader = build_loader(valid_dataset, valid_sampler, "valid")
        cfg.num_batchs_per_epoch = len(train_loader)
        log_print(
            f"train={len(train_dataset)}, valid={len(valid_dataset)}, "
            f"rank_batches={len(train_loader)}, world_size={world_size}"
        )
        log_print("...Done")
        return cls(
            train_dataset,
            valid_dataset,
            train_loader,
            valid_loader,
            train_sampler,
            valid_sampler,
            generators,
            {"train": train_report, "valid": valid_report},
        )


DatasetHandler = datasetHandler


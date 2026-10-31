"""NativeResKD 4.01 DDP trainer using the CLEE/DRepGenVLM run policy."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import uuid
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ..common.dist_utils import (
    barrier,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    reduce_mean,
    reduce_sum_count,
)
from ..common.metricsTracker import MetricsTracker
from ..common.metricsTracker_v2 import TrainingMonitor
from ..common.utils import atomic_write_json, ensure_finite, log_print, move_to_device
from ..criterions.NRKDLoss import NRKDLoss
from .building_scheduler import build_cosine_annealing_with_warmup_scheduler


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


class NRKDTrainer:
    def __init__(self, cfg, model, checkpoint_path: str | None = None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        self.local_rank = int(cfg.local_rank)
        self.find_unused_params = bool(cfg.find_unused_params)
        if is_dist_avail_and_initialized():
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.find_unused_params,
                gradient_as_bucket_view=True,
                static_graph=not self.find_unused_params,
                broadcast_buffers=False,
            )

        self.criterion = NRKDLoss.from_config(cfg).to(self.device)
        self.accumulation_steps = int(cfg.accumulation_steps)
        self.optimizer = self._build_optimizer()
        self.total_steps = math.ceil(int(cfg.num_batchs_per_epoch) / self.accumulation_steps) * int(cfg.num_epochs)
        cfg.total_steps = self.total_steps
        cfg.warmup_steps = int(
            min(float(cfg.warmup_ratio) * self.total_steps, int(cfg.max_warmup_steps))
            if cfg.warmup_steps is None or int(cfg.warmup_steps) > float(cfg.warmup_ratio) * self.total_steps
            else int(cfg.warmup_steps)
        )
        self.scheduler = build_cosine_annealing_with_warmup_scheduler(
            self.optimizer,
            total_steps=max(1, self.total_steps),
            warmup_steps=int(cfg.warmup_steps),
        )
        self.metrics = MetricsTracker.from_config(cfg)
        self.monitor = TrainingMonitor(
            writer=self.metrics.writer,
            model=self.get_model_raw().student,
            optimizer=self.optimizer,
            log_every_n_steps=int(cfg.log_every_n_steps),
            check_every_n_steps=int(cfg.monitoring_every_n_steps),
        )

        self.checkpoint_path = checkpoint_path
        self.checkpoint_epoch_idx = None
        self.global_step = 0
        self.optimizer_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self._checkpoint_loaded = False
        self._is_continuation = False
        self._pending_rank_runtime_states = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.resume_signature = self._build_resume_signature()
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        log_print(
            f"NRKDTrainer home device={self.device}, world_size={get_world_size()}, "
            f"accumulation_steps={self.accumulation_steps}, total_steps={self.total_steps}, "
            f"warmup_steps={cfg.warmup_steps}"
        )

    @classmethod
    def from_config(cls, cfg, model, checkpoint_path: str | None = None):
        return cls(cfg=cfg, model=model, checkpoint_path=checkpoint_path)

    def get_model_raw(self):
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    def _build_optimizer(self):
        student = self.get_model_raw().student
        rope_ids = {id(parameter) for parameter in student.rope_parameters()}
        rope = [parameter for parameter in student.parameters() if parameter.requires_grad and id(parameter) in rope_ids]
        backbone = [parameter for parameter in student.parameters() if parameter.requires_grad and id(parameter) not in rope_ids]
        if not rope or not backbone:
            raise RuntimeError("Expected non-empty backbone and RoPE optimizer groups")
        tracked = {id(parameter) for parameter in rope + backbone}
        trainable = {id(parameter) for parameter in student.parameters() if parameter.requires_grad}
        if tracked != trainable:
            raise RuntimeError("Optimizer parameter grouping did not cover every trainable student parameter exactly once")
        return torch.optim.AdamW(
            [
                {
                    "params": backbone,
                    "lr": float(self.cfg.learning_rate_dict["backbone"]),
                    "weight_decay": float(self.cfg.weight_decay),
                    "group_name": "backbone",
                },
                {
                    "params": rope,
                    "lr": float(self.cfg.learning_rate_dict["rope"]),
                    "weight_decay": float(self.cfg.weight_decay),
                    "group_name": "rope",
                },
            ],
            betas=(float(self.cfg.optimizer_beta1), float(self.cfg.optimizer_beta2)),
        )

    def _autocast(self):
        precision = str(self.cfg.precision).lower()
        if precision == "bf16":
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        if precision == "fp16" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _forward_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        outputs = self.model(
            teacher_image=batch["teacher_image"],
            student_image=batch["student_image"],
            patch_mapping=batch["patch_mapping"],
        )
        return self.criterion(
            model_outputs=outputs,
            patch_mapping=batch["patch_mapping"],
            tissue_retention_ratio=batch["tissue_retention_ratio"],
            view_type=batch["view_type"],
            teacher_grid_hw=batch["teacher_grid_hw"],
            sample_id=batch["sample_id"],
        )

    @staticmethod
    def _block_size(batch_idx: int, num_batches: int, accumulation_steps: int) -> int:
        block_start = (int(batch_idx) // int(accumulation_steps)) * int(accumulation_steps)
        return max(1, min(int(accumulation_steps), int(num_batches) - block_start))

    def _assert_gradients(self, context: str) -> None:
        missing = []
        nonfinite = []
        has_nonzero = False
        for name, parameter in self.get_model_raw().student.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                missing.append(name)
                continue
            if not torch.isfinite(parameter.grad).all():
                nonfinite.append(name)
            if bool(torch.count_nonzero(parameter.grad).detach().cpu()):
                has_nonzero = True
        local_failure = bool(missing or nonfinite or not has_nonzero)
        failure = torch.tensor([int(local_failure)], dtype=torch.uint8, device=self.device)
        if is_dist_avail_and_initialized():
            dist.all_reduce(failure, op=dist.ReduceOp.MAX)
        if failure.item():
            raise RuntimeError(
                f"Invalid gradients at {context}; rank={get_rank()}, missing={missing[:20]}, "
                f"nonfinite={nonfinite[:20]}, has_nonzero={has_nonzero}"
            )

    def _assert_parameters_and_optimizer_finite(self, context: str) -> None:
        bad = []
        for name, parameter in self.get_model_raw().student.named_parameters():
            if parameter.requires_grad and not torch.isfinite(parameter).all():
                bad.append(f"parameter:{name}")
        for parameter, state in self.optimizer.state.items():
            for key, value in state.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    bad.append(f"optimizer:{key}")
        failure = torch.tensor([int(bool(bad))], dtype=torch.uint8, device=self.device)
        if is_dist_avail_and_initialized():
            dist.all_reduce(failure, op=dist.ReduceOp.MAX)
        if failure.item():
            raise RuntimeError(f"Non-finite training state at {context}; rank={get_rank()}, bad={bad[:20]}")

    def epoch_train(self, dataloader, epoch_idx: int) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = defaultdict(float)
        counts = defaultdict(int)
        block_totals = defaultdict(float)
        block_count = 0
        num_batches = len(dataloader)
        for batch_idx, cpu_batch in enumerate(dataloader):
            batch = move_to_device(cpu_batch, self.device)
            is_last = batch_idx + 1 == num_batches
            is_accum_step = (batch_idx + 1) % self.accumulation_steps == 0 or is_last
            no_sync = (
                self.model.no_sync()
                if isinstance(self.model, DistributedDataParallel) and not is_accum_step
                else nullcontext()
            )
            with no_sync, self._autocast():
                losses = self._forward_loss(batch)
                for key, value in losses.items():
                    ensure_finite(key, value, {"epoch": epoch_idx, "batch": batch_idx})
                divisor = self._block_size(batch_idx, num_batches, self.accumulation_steps)
                scaled_loss = losses["total_loss"] / divisor
            ensure_finite("scaled_total_loss", scaled_loss, {"epoch": epoch_idx, "batch": batch_idx})
            scaled_loss.backward()
            self.global_step += 1

            values = {key: float(value.detach().float().cpu()) for key, value in losses.items()}
            values["student_tokens"] = float(batch["student_grid_hw"].prod(dim=1).float().mean().cpu())
            values["tissue_retention"] = float(batch["tissue_retention_ratio"].float().mean().cpu())
            for key, value in values.items():
                totals[key] += value
                counts[key] += 1
                block_totals[key] += value
            block_count += 1

            if is_accum_step:
                self._assert_gradients(f"epoch={epoch_idx}, batch={batch_idx}")
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.get_model_raw().student.parameters(),
                    float(self.cfg.gradient_clip_norm),
                )
                ensure_finite("gradient_norm", grad_norm, {"epoch": epoch_idx, "batch": batch_idx})
                self.monitor.log_always_on(self.global_step, int(batch["teacher_image"].shape[0]))
                self.monitor.log_periodic(self.global_step)
                self.optimizer.step()
                self._assert_parameters_and_optimizer_finite(
                    f"after optimizer step, epoch={epoch_idx}, batch={batch_idx}"
                )
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_step += 1

                reduced = {
                    key: float(reduce_mean(torch.tensor(value / block_count, device=self.device)).cpu())
                    for key, value in block_totals.items()
                }
                reduced["grad_norm"] = float(reduce_mean(grad_norm.detach().float()).cpu())
                log_every = max(1, int(self.cfg.log_every_n_steps))
                if self.optimizer_step % log_every == 0 or is_last:
                    self.metrics.update(
                        loss_dict=reduced,
                        stage="train",
                        lr=float(self.optimizer.param_groups[0]["lr"]),
                        rank=get_rank(),
                        global_step=self.global_step,
                        optimizer_step=self.optimizer_step,
                        epoch=epoch_idx,
                        backbone_lr=float(self.optimizer.param_groups[0]["lr"]),
                        rope_lr=float(self.optimizer.param_groups[1]["lr"]),
                    )
                block_totals.clear()
                block_count = 0

        averages = {}
        for key in totals:
            total, count = reduce_sum_count(totals[key], counts[key], self.device)
            averages[key] = total / max(1, count)
        return averages

    @torch.no_grad()
    def epoch_validEval(self, dataloader, epoch_idx: int) -> dict[str, float]:
        self.model.eval()
        totals = defaultdict(float)
        counts = defaultdict(int)
        for cpu_batch in dataloader:
            batch = move_to_device(cpu_batch, self.device)
            with self._autocast():
                losses = self._forward_loss(batch)
            for key, value in losses.items():
                ensure_finite(f"validation:{key}", value, {"epoch": epoch_idx})
                totals[key] += float(value.detach().float().cpu())
                counts[key] += 1
        averages = {}
        for key in totals:
            total, count = reduce_sum_count(totals[key], counts[key], self.device)
            averages[key] = total / max(1, count)
        self.metrics.update(
            loss_dict=averages,
            stage="val",
            rank=get_rank(),
            global_step=self.global_step,
            optimizer_step=self.optimizer_step,
            epoch=epoch_idx,
        )
        return averages

    def _attach_dataloaders(self, train_dataloader=None, val_dataloader=None) -> None:
        self.train_dataloader = train_dataloader
        self.valid_dataloader = val_dataloader
        if self._pending_rank_runtime_states is None:
            return
        saved_world = len(self._pending_rank_runtime_states)
        if saved_world != get_world_size():
            raise RuntimeError(
                f"DDP topology changed during keepTrain: checkpoint world_size={saved_world}, "
                f"current={get_world_size()}. Use --retrain to change topology."
            )
        state = self._pending_rank_runtime_states[get_rank()]
        self._restore_rng_state(state["rng"])
        loaders = state.get("loader_generators", {})
        for name, dataloader in (("train", train_dataloader), ("valid", val_dataloader)):
            if dataloader is not None and name in loaders:
                if dataloader.generator is None:
                    raise RuntimeError(f"Checkpoint has {name} generator state but DataLoader has no generator")
                dataloader.generator.set_state(loaders[name].cpu())
        sampler_state = state.get("samplers", {})
        if train_dataloader is not None and "train" in sampler_state:
            train_dataloader.batch_sampler.load_state_dict(sampler_state["train"])
        if val_dataloader is not None and "valid" in sampler_state:
            val_dataloader.batch_sampler.load_state_dict(sampler_state["valid"])
        self._pending_rank_runtime_states = None
        log_print("Restored rank-local RNG, DataLoader generator, and sampler states")

    def train(self, train_dataloader, val_dataloader) -> None:
        self._attach_dataloaders(train_dataloader, val_dataloader)
        self._validate_runtime_signature()
        if not self._is_continuation:
            self.save_checkpoint(epoch_idx=-1, checkpoint_kind="initial")
        first_epoch = (self.checkpoint_epoch_idx + 1) if self.checkpoint_epoch_idx is not None else 0
        log_print(f"Training started: epochs={self.cfg.num_epochs}, first_epoch={first_epoch}")

        for epoch_idx in range(first_epoch, int(self.cfg.num_epochs)):
            train_dataloader.batch_sampler.set_epoch(epoch_idx)
            train_metrics = self.epoch_train(train_dataloader, epoch_idx)
            valid_metrics = {}
            if val_dataloader is not None and (epoch_idx + 1) % int(self.cfg.validate_every_n_epochs) == 0:
                val_dataloader.batch_sampler.set_epoch(epoch_idx)
                valid_metrics = self.epoch_validEval(val_dataloader, epoch_idx)
            val_loss = float(valid_metrics.get("total_loss", float("inf")))
            log_print(
                f"Epoch ({epoch_idx + 1}/{self.cfg.num_epochs}): "
                f"Train={train_metrics.get('total_loss', float('nan')):.6f}, Val={val_loss:.6f}"
            )

            if val_dataloader is not None and val_loss == 0.0:
                self.save_checkpoint(
                    epoch_idx=epoch_idx,
                    checkpoint_kind="emergency",
                    weight_filename=f"error_quick_checkpoint_epoch_{epoch_idx}.pth",
                )
                raise ValueError(f"Val loss is 0.0 at epoch {epoch_idx}")

            new_best = val_loss < self.best_val_loss
            if new_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.metrics.epoch_summary(
                metric_dict={
                    **{f"train/{key}": value for key, value in train_metrics.items()},
                    **{f"val/{key}": value for key, value in valid_metrics.items()},
                    "optimizer_step": self.optimizer_step,
                },
                global_step=self.global_step,
                epoch=epoch_idx,
            )
            if new_best and self.cfg.save_best:
                self.save_checkpoint(epoch_idx=epoch_idx, checkpoint_kind="best")
                log_print(f"New best saved. Val Loss: {self.best_val_loss:.6f}")
            if int(self.cfg.save_freq) > 0 and (epoch_idx + 1) % int(self.cfg.save_freq) == 0:
                self.save_checkpoint(
                    epoch_idx=epoch_idx,
                    checkpoint_kind="periodic",
                    weight_filename=f"checkpoint_epoch_{epoch_idx}.pth",
                )
            if self.cfg.save_latest:
                self.save_checkpoint(epoch_idx=epoch_idx, checkpoint_kind="latest")
            barrier()

            if self.cfg.early_stop_patience is not None and self.patience_counter >= int(self.cfg.early_stop_patience):
                log_print("Early stopping triggered")
                break
        self.metrics.close()

    def valid(self, val_dataloader) -> dict[str, float]:
        self._attach_dataloaders(None, val_dataloader)
        epoch_idx = self.checkpoint_epoch_idx if self.checkpoint_epoch_idx is not None else 0
        val_dataloader.batch_sampler.set_epoch(epoch_idx)
        metrics = self.epoch_validEval(val_dataloader, epoch_idx)
        self.metrics.epoch_summary(metrics, global_step=self.global_step, epoch=epoch_idx)
        self.metrics.close()
        return metrics

    def _capture_rank_runtime_state(self) -> dict[str, Any]:
        cuda_state = None
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state(self.device).cpu()
        loader_generators = {}
        samplers = {}
        for name, dataloader in (("train", self.train_dataloader), ("valid", self.valid_dataloader)):
            if dataloader is None:
                continue
            if dataloader.generator is not None:
                loader_generators[name] = dataloader.generator.get_state().cpu()
            if hasattr(dataloader.batch_sampler, "state_dict"):
                samplers[name] = dataloader.batch_sampler.state_dict()
        return {
            "rank": get_rank(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state().cpu(),
                "torch_cuda": cuda_state,
            },
            "loader_generators": loader_generators,
            "samplers": samplers,
        }

    def _collect_rank_runtime_states(self):
        local = self._capture_rank_runtime_state()
        if not is_dist_avail_and_initialized():
            return [local]
        gathered = [None] * get_world_size() if is_main_process() else None
        dist.gather_object(local, gathered, dst=0)
        return gathered

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].detach().cpu())
        if state.get("torch_cuda") is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("Checkpoint has CUDA RNG state but CUDA is unavailable")
            torch.cuda.set_rng_state(state["torch_cuda"].detach().cpu(), self.device)

    def _build_resume_signature(self) -> dict[str, Any]:
        reports = getattr(self.cfg, "metadata_reports", {}) or {}
        return {
            "schema_version": 1,
            "framework": "NativeResKD",
            "version": "4.01",
            "metadata": {
                split: {
                    "source_metadata_sha256": report.get("source_metadata_sha256"),
                    "selection_sha256": report.get("selection_sha256"),
                    "selected_count": report.get("selected_count"),
                }
                for split, report in reports.items()
            },
            "model": {
                "teacher": self.cfg.model_name,
                "patch_size": self.cfg.expected_patch_size,
                "depth": self.cfg.expected_depth,
                "rope_type": self.cfg.rope_type,
                "rope_theta": self.cfg.rope_theta,
                "selected_layers": list(self.cfg.selected_layers),
            },
            "loss": {
                "global": self.cfg.global_loss_weight,
                "patch": self.cfg.patch_loss_weight,
                "hidden": self.cfg.hidden_loss_weight,
            },
            "training": {
                "world_size": int(self.cfg.world_size),
                "per_device_batch_size": int(self.cfg.per_device_batch_size),
                "accumulation_steps": int(self.cfg.accumulation_steps),
                "num_batches_per_epoch": int(self.cfg.num_batchs_per_epoch),
                "num_epochs": int(self.cfg.num_epochs),
                "total_steps": int(self.total_steps),
                "warmup_steps": int(self.cfg.warmup_steps),
            },
        }

    def _validate_runtime_signature(self) -> None:
        if not self._is_continuation:
            return
        current = self._build_resume_signature()
        if current != self.resume_signature:
            raise RuntimeError(
                "keepTrain resume signature changed. Use --retrain when changing data, "
                "DDP topology, optimization, model, or loss settings.\n"
                f"checkpoint={json.dumps(self.resume_signature, sort_keys=True)}\n"
                f"current={json.dumps(current, sort_keys=True)}"
            )

    @staticmethod
    def _checkpoint_paths(directory: str | Path, weight_filename: str) -> dict[str, Path]:
        root = Path(directory)
        stem = Path(weight_filename).stem
        return {
            "trainer": root / f"[trainer]{weight_filename}",
            "model": root / f"[model]{weight_filename}",
            "manifest": root / f"[checkpoint]{stem}.json",
        }

    def save_checkpoint(
        self,
        epoch_idx: int,
        checkpoint_kind: str,
        weight_filename: str | None = None,
    ) -> None:
        valid_kinds = {"initial", "best", "latest", "periodic", "emergency"}
        if checkpoint_kind not in valid_kinds:
            raise ValueError(f"Unsupported checkpoint kind: {checkpoint_kind}")
        default_names = {
            "initial": "initial_model.pth",
            "best": self.cfg.weight_filename,
            "latest": "latest_model.pth",
        }
        filename = weight_filename or default_names.get(checkpoint_kind)
        if filename is None:
            raise ValueError(f"weight_filename is required for {checkpoint_kind}")
        self._assert_parameters_and_optimizer_finite(
            f"before {checkpoint_kind} checkpoint, epoch={epoch_idx}"
        )
        rank_states = self._collect_rank_runtime_states()
        checkpoint_id = uuid.uuid4().hex if is_main_process() else None
        if is_dist_avail_and_initialized():
            payload = [checkpoint_id]
            dist.broadcast_object_list(payload, src=0)
            checkpoint_id = payload[0]
        metadata = {
            "schema_version": "nrkd_checkpoint_pair_v1",
            "checkpoint_id": checkpoint_id,
            "checkpoint_kind": checkpoint_kind,
            "epoch_idx": int(epoch_idx),
            "global_step": int(self.global_step),
            "optimizer_step": int(self.optimizer_step),
            "weight_filename": filename,
            "framework": "NativeResKD",
            "version": "4.01",
        }
        if is_main_process():
            paths = self._checkpoint_paths(self.cfg.save_path, filename)
            model_payload = self.get_model_raw().model_checkpoint_payload(metadata)
            trainer_payload = {
                "schema_version": "nrkd_trainer_v1",
                "checkpoint_metadata": metadata,
                "epoch_idx": int(epoch_idx),
                "global_step": int(self.global_step),
                "optimizer_step": int(self.optimizer_step),
                "best_val_loss": float(self.best_val_loss),
                "patience_counter": int(self.patience_counter),
                "num_batches_per_epoch": int(self.cfg.num_batchs_per_epoch),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "rank_runtime_states": rank_states,
                "resume_signature": self._build_resume_signature(),
            }
            _atomic_torch_save(model_payload, paths["model"])
            _atomic_torch_save(trainer_payload, paths["trainer"])
            manifest = {
                "schema_version": "nrkd_checkpoint_pair_manifest_v1",
                "checkpoint_metadata": metadata,
                "artifacts": {
                    "model": {
                        "filename": paths["model"].name,
                        "size": paths["model"].stat().st_size,
                        "sha256": _sha256_file(paths["model"]),
                    },
                    "trainer": {
                        "filename": paths["trainer"].name,
                        "size": paths["trainer"].stat().st_size,
                        "sha256": _sha256_file(paths["trainer"]),
                    },
                },
            }
            atomic_write_json(paths["manifest"], manifest)
            log_print(f"{checkpoint_kind} checkpoint saved: {paths['manifest']}")
        barrier()

    def _load_pair(self, requested_path: Path) -> tuple[dict, dict]:
        filename = requested_path.name
        for prefix in ("[trainer]", "[model]"):
            if filename.startswith(prefix):
                filename = filename[len(prefix):]
        paths = self._checkpoint_paths(requested_path.parent, filename)
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        for role in ("model", "trainer"):
            expected = manifest["artifacts"][role]
            path = paths[role]
            if path.stat().st_size != int(expected["size"]) or _sha256_file(path) != expected["sha256"]:
                raise RuntimeError(f"Checkpoint integrity verification failed: {path}")
        model_payload = torch.load(paths["model"], map_location="cpu", weights_only=False)
        trainer_payload = torch.load(paths["trainer"], map_location="cpu", weights_only=False)
        identities = [
            manifest["checkpoint_metadata"].get("checkpoint_id"),
            model_payload.get("checkpoint_metadata", {}).get("checkpoint_id"),
            trainer_payload.get("checkpoint_metadata", {}).get("checkpoint_id"),
        ]
        if len(set(identities)) != 1:
            raise RuntimeError(f"Checkpoint pair ID mismatch: {identities}")
        return model_payload, trainer_payload

    def load_checkpoint(self, path: str | Path) -> int:
        requested = Path(path).expanduser().resolve()
        log_print(f"Checkpoint loading from {requested}")
        if requested.name == "checkpoint.pt":
            legacy = torch.load(requested, map_location="cpu", weights_only=False)
            self.get_model_raw().student.load_state_dict(legacy["student"], strict=True)
            if self.cfg.trainer_mode != "reTrain":
                raise RuntimeError(
                    "Legacy CONCHNativeRoPEKD checkpoint loaded its student weights, but exact "
                    "keepTrain is incompatible with the 4.01 scheduler/checkpoint policy. Re-run with --retrain."
                )
            self.checkpoint_epoch_idx = -1
            self.global_step = 0
            self.optimizer_step = 0
            log_print("reTrain mode: imported legacy student weights and reset training state")
            return -1

        model_payload, checkpoint = self._load_pair(requested)
        self.get_model_raw().load_student_payload(model_payload, strict=True)
        loaded_epoch = int(checkpoint.get("epoch_idx", -1))
        self._checkpoint_loaded = True
        if self.cfg.trainer_mode == "reTrain":
            self.checkpoint_epoch_idx = -1
            self.global_step = 0
            self.optimizer_step = 0
            self.best_val_loss = float("inf")
            self.patience_counter = 0
            self._is_continuation = False
            log_print("reTrain mode: loaded student weights only and reset training state")
            return -1

        if checkpoint.get("optimizer_state_dict") is None or checkpoint.get("scheduler_state_dict") is None:
            raise RuntimeError("Continuation checkpoint lacks optimizer/scheduler state; use --retrain")
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.checkpoint_epoch_idx = loaded_epoch
        self.global_step = int(checkpoint.get("global_step", 0))
        self.optimizer_step = int(checkpoint.get("optimizer_step", 0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.patience_counter = int(checkpoint.get("patience_counter", 0))
        self.resume_signature = checkpoint.get("resume_signature")
        if self.resume_signature is None:
            raise RuntimeError("NativeResKD keepTrain checkpoint has no resume signature; use --retrain")
        self._pending_rank_runtime_states = checkpoint.get("rank_runtime_states")
        if not isinstance(self._pending_rank_runtime_states, list):
            raise RuntimeError("NativeResKD keepTrain checkpoint has no rank-local RNG/DataLoader states")
        self._is_continuation = True
        log_print(
            f"Continuation restored epoch={loaded_epoch}, global_step={self.global_step}, "
            f"optimizer_step={self.optimizer_step}, best_val_loss={self.best_val_loss}"
        )
        return loaded_epoch


NativeResKDDDPTrainer = NRKDTrainer


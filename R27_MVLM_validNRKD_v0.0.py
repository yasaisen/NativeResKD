"""NativeResKD 4.01 validation entry point."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from NativeResKD.common.dist_utils import (  # noqa: E402
    barrier,
    cleanup_distributed_mode,
    init_distributed_mode,
    is_main_process,
)
from NativeResKD.common.utils import atomic_write_json, log_print, set_seed  # noqa: E402
from NativeResKD.configs.configHandler import ConfigHandler  # noqa: E402
from NativeResKD.datasets.building_datasetHandler import datasetHandler  # noqa: E402
from NativeResKD.models.modeling_NRKD import NativeResolutionKnowledgeDistiller  # noqa: E402
from NativeResKD.trainer.building_NRKDTrainer import NRKDTrainer  # noqa: E402


def main() -> None:
    cfg_handler = None
    trainer = None
    try:
        cfg_handler = ConfigHandler.get_cfg()
        if cfg_handler.checkpoint_path is None:
            raise ValueError("Validation requires --best-checkpoint-path or --latest-checkpoint-path")
        set_seed(cfg_handler.cfg.seed, cfg_handler.cfg.deterministic)
        init_distributed_mode(cfg_handler.cfg)
        cfg_handler.finalize_runtime()
        cfg_handler.cfg.validate()

        data_handler = datasetHandler.from_config(cfg_handler.cfg)
        model = NativeResolutionKnowledgeDistiller.from_config(cfg_handler.cfg)
        trainer = NRKDTrainer.from_config(
            cfg=cfg_handler.cfg,
            model=model,
            checkpoint_path=cfg_handler.checkpoint_path,
        )
        if is_main_process():
            cfg_handler.cfg.save()
        barrier()

        log_print("Start NativeResKD validation...")
        result = trainer.valid(data_handler.valid_loader)
        if is_main_process():
            atomic_write_json(Path(cfg_handler.cfg.save_path) / "validation_result.json", result)
        log_print("NativeResKD validation completed!")
    except KeyboardInterrupt:
        log_print("Validation interrupted by user.")
        raise
    finally:
        if trainer is not None and trainer.metrics is not None:
            trainer.metrics.close()
        cleanup_distributed_mode(cfg_handler.cfg if cfg_handler is not None else None)


if __name__ == "__main__":
    main()

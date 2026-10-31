"""Evaluate S0/S1 representation fidelity on the deterministic valid split."""

from __future__ import annotations

import argparse

from NativeResKD.common.checkpoint_io import load_student_checkpoint
from NativeResKD.configs.NRKD_baseConfig import NRKD_baseConfig
from NativeResKD.datasets.building_datasetHandler import prepare_kd_split
from NativeResKD.evaluator.representation_evaluator import RepresentationEvaluator
from NativeResKD.models.modeling_NRKD import NativeResolutionKnowledgeDistiller


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True)
    parser.add_argument("--model-id", choices=("S0", "S1"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg = NRKD_baseConfig.load(args.cfg_path)
    model = NativeResolutionKnowledgeDistiller.from_config(cfg)
    if args.model_id == "S1":
        if not args.checkpoint:
            raise ValueError("S1 evaluation requires --checkpoint")
        load_student_checkpoint(model.student, args.checkpoint)
    metadata, _ = prepare_kd_split(cfg, "valid")
    RepresentationEvaluator(cfg, model, output_dir=args.output_dir).evaluate(metadata, args.model_id)


if __name__ == "__main__":
    main()

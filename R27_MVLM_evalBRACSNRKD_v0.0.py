"""Run the matched-512 or native-ROI frozen BRACS linear-probe protocol."""

from __future__ import annotations

import argparse

from NativeResKD.common.checkpoint_io import load_student_checkpoint
from NativeResKD.configs.NRKD_baseConfig import NRKD_baseConfig
from NativeResKD.evaluator.bracs_evaluator import BRACSEvaluator
from NativeResKD.models.modeling_NRKD import NativeResolutionKnowledgeDistiller


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True)
    parser.add_argument("--model-id", choices=("T", "S0", "S1"), required=True)
    parser.add_argument("--protocol", choices=("matched_512", "native_roi"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="bracs_eval")
    args = parser.parse_args()
    cfg = NRKD_baseConfig.load(args.cfg_path)
    model = NativeResolutionKnowledgeDistiller.from_config(cfg)
    if args.model_id == "S1":
        if not args.checkpoint:
            raise ValueError("S1 evaluation requires --checkpoint")
        load_student_checkpoint(model.student, args.checkpoint)
    if args.model_id == "T":
        encoder = model.teacher
        dataset_protocol = "matched_512"
        output_model_id = "T_fixed512" if args.protocol == "native_roi" else "T"
    else:
        encoder = model.student
        dataset_protocol = args.protocol
        output_model_id = f"{args.model_id}_native" if args.protocol == "native_roi" else args.model_id
    BRACSEvaluator(
        cfg,
        encoder,
        model_id=output_model_id,
        protocol=dataset_protocol,
        output_protocol=args.protocol,
        output_dir=args.output_dir,
    ).run()


if __name__ == "__main__":
    main()

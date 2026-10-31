"""Compute within- and cross-protocol paired-bootstrap CIs for BRACS."""

from __future__ import annotations

import argparse
import json

from NativeResKD.evaluator.bracs_cross_protocol_bootstrap import (
    run_bracs_paired_bootstrap_with_cross_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="bracs_eval")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()
    result = run_bracs_paired_bootstrap_with_cross_protocol(
        args.root,
        iterations=args.iterations,
        seed=args.seed,
        confidence_level=args.confidence_level,
    )
    print(json.dumps(result["comparisons"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

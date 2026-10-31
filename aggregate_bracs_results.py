from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from NativeResKD.evaluator.bracs_cross_protocol_bootstrap import (
    run_bracs_paired_bootstrap_with_cross_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="bracs_eval")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    args = parser.parse_args()
    root = Path(args.root)
    metric_files = sorted(root.glob("*/*/metrics.json"))
    if not metric_files:
        raise FileNotFoundError(f"No {root}/*/*/metrics.json results found")
    summaries, per_class, predictions, extrapolation = [], [], [], []
    for path in metric_files:
        metric = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(
            {key: value for key, value in metric.items() if not isinstance(value, (list, dict))}
        )
        for index, value in enumerate(metric.get("per_class_f1", [])):
            per_class.append(
                {
                    "model_id": metric["model_id"],
                    "protocol": metric["protocol"],
                    "class_index": index,
                    "f1": value,
                }
            )
        for filename, destination in (
            ("predictions.csv", predictions),
            ("extrapolation_analysis.csv", extrapolation),
        ):
            source = path.with_name(filename)
            if source.exists():
                with source.open(newline="", encoding="utf-8") as handle:
                    destination.extend(csv.DictReader(handle))

    def write(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        fields = sorted({key for row in rows for key in row})
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write("summary.csv", summaries)
    write("per_class_metrics.csv", per_class)
    write("predictions.csv", predictions)
    write("extrapolation_analysis.csv", extrapolation)
    write(
        "native_coverage.csv",
        [
            {
                key: row.get(key)
                for key in (
                    "model_id",
                    "protocol",
                    "coverage",
                    "oom_count",
                    "oom_rate",
                    "coverage_adjusted_accuracy",
                )
            }
            for row in summaries
        ],
    )
    bootstrap = run_bracs_paired_bootstrap_with_cross_protocol(
        root,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
        confidence_level=args.bootstrap_confidence_level,
    )
    print(
        f"Aggregated {len(summaries)} evaluations and "
        f"{len(bootstrap['comparisons'])} paired-bootstrap comparisons under {root}"
    )


if __name__ == "__main__":
    main()

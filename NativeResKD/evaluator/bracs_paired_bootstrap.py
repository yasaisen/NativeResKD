"""Paired-bootstrap confidence intervals for BRACS Macro-F1 differences."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..common.utils import atomic_write_json


PROTOCOL_MODELS = {
    "matched_512": {"T": "T", "S0": "S0", "S1": "S1"},
    "native_roi": {"T": "T_fixed512", "S0": "S0_native", "S1": "S1_native"},
}
COMPARISONS = (("T", "S0"), ("S0", "S1"), ("T", "S1"))


def _require_f1_score():
    try:
        from sklearn.metrics import f1_score
    except ImportError as exc:
        raise RuntimeError("BRACS paired bootstrap requires scikit-learn") from exc
    return f1_score


def _read_model_result(root: Path, protocol: str, evaluated_model_id: str) -> dict[str, Any]:
    directory = root / protocol / evaluated_model_id
    prediction_path = directory / "predictions.csv"
    metric_path = directory / "metrics.json"
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    if not metric_path.is_file():
        raise FileNotFoundError(metric_path)
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    if metrics.get("model_id") != evaluated_model_id:
        raise ValueError(
            f"metrics model_id mismatch in {metric_path}: "
            f"{metrics.get('model_id')!r} != {evaluated_model_id!r}"
        )
    if metrics.get("protocol") != protocol:
        raise ValueError(
            f"metrics protocol mismatch in {metric_path}: "
            f"{metrics.get('protocol')!r} != {protocol!r}"
        )

    rows: dict[str, dict[str, str]] = {}
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "split", "label", "status", "prediction"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{prediction_path} is missing columns: {sorted(missing)}")
        for row in reader:
            if row["split"] != "test":
                continue
            sample_id = row["sample_id"]
            if not sample_id:
                raise ValueError(f"Empty test sample_id in {prediction_path}")
            if sample_id in rows:
                raise ValueError(f"Duplicate test sample_id={sample_id!r} in {prediction_path}")
            if not row["label"]:
                raise ValueError(f"Empty label for sample_id={sample_id!r} in {prediction_path}")
            rows[sample_id] = row
    if not rows:
        raise ValueError(f"No test rows in {prediction_path}")
    usable = {
        sample_id
        for sample_id, row in rows.items()
        if row["status"] == "ok" and bool(row["prediction"])
    }
    status_counts = Counter(row["status"] for row in rows.values())
    return {
        "directory": directory,
        "metrics": metrics,
        "rows": rows,
        "usable_ids": usable,
        "status_counts": dict(sorted(status_counts.items())),
    }


def _build_common_cohort(
    root: Path,
    protocol: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if protocol not in PROTOCOL_MODELS:
        raise ValueError(f"Unsupported BRACS protocol: {protocol}")
    results = {
        canonical: _read_model_result(root, protocol, evaluated)
        for canonical, evaluated in PROTOCOL_MODELS[protocol].items()
    }
    all_ids = set().union(*(set(result["rows"]) for result in results.values()))
    common_ids = set.intersection(*(result["usable_ids"] for result in results.values()))
    if not common_ids:
        raise RuntimeError(f"No jointly successful T/S0/S1 test samples for {protocol}")

    labels_by_sample: dict[str, str] = {}
    for sample_id in sorted(all_ids):
        observed = {
            result["rows"][sample_id]["label"]
            for result in results.values()
            if sample_id in result["rows"]
        }
        if len(observed) != 1:
            raise ValueError(
                f"Ground-truth label mismatch for protocol={protocol}, "
                f"sample_id={sample_id!r}: {sorted(observed)}"
            )
        labels_by_sample[sample_id] = next(iter(observed))

    all_labels = sorted(set(labels_by_sample.values()))
    common_labels = {labels_by_sample[sample_id] for sample_id in common_ids}
    missing_classes = set(all_labels) - common_labels
    if missing_classes:
        raise RuntimeError(
            f"Common cohort for {protocol} has no samples for classes: {sorted(missing_classes)}"
        )

    ordered_ids = sorted(common_ids)
    y_true = np.asarray([labels_by_sample[sample_id] for sample_id in ordered_ids], dtype=object)
    predictions = {
        canonical: np.asarray(
            [result["rows"][sample_id]["prediction"] for sample_id in ordered_ids],
            dtype=object,
        )
        for canonical, result in results.items()
    }
    cohort = {
        "protocol": protocol,
        "test_union_count": len(all_ids),
        "common_success_count": len(common_ids),
        "excluded_from_common_count": len(all_ids - common_ids),
        "labels": all_labels,
        "class_counts": {
            label: int(np.count_nonzero(y_true == label)) for label in all_labels
        },
        "models": {
            canonical: {
                "evaluated_model_id": PROTOCOL_MODELS[protocol][canonical],
                "test_row_count": len(result["rows"]),
                "usable_prediction_count": len(result["usable_ids"]),
                "missing_row_count": len(all_ids - set(result["rows"])),
                "excluded_from_common_count": len(all_ids - result["usable_ids"]),
                "status_counts": result["status_counts"],
                "reported_full_cohort_macro_f1": float(result["metrics"]["macro_f1"]),
            }
            for canonical, result in results.items()
        },
        "sample_ids": ordered_ids,
        "y_true": y_true,
        "predictions": predictions,
    }
    return cohort, results


def compute_protocol_paired_bootstrap(
    root: str | Path,
    protocol: str,
    *,
    iterations: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute stratified paired-percentile CIs on the common T/S0/S1 test cohort."""
    if int(iterations) <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    f1_score = _require_f1_score()
    cohort, _ = _build_common_cohort(Path(root), protocol)
    y_true = cohort.pop("y_true")
    predictions = cohort.pop("predictions")
    sample_ids = cohort.pop("sample_ids")
    labels = list(cohort["labels"])

    observed = {
        model_id: float(
            f1_score(y_true, prediction, labels=labels, average="macro", zero_division=0)
        )
        for model_id, prediction in predictions.items()
    }
    groups = [np.flatnonzero(y_true == label) for label in labels]
    rng = np.random.default_rng(int(seed))
    distributions = {
        pair: np.empty(int(iterations), dtype=np.float64) for pair in COMPARISONS
    }
    for bootstrap_index in range(int(iterations)):
        sampled = np.concatenate(
            [rng.choice(group, size=group.size, replace=True) for group in groups]
        )
        scores = {
            model_id: float(
                f1_score(
                    y_true[sampled],
                    prediction[sampled],
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            )
            for model_id, prediction in predictions.items()
        }
        for model_a, model_b in COMPARISONS:
            distributions[(model_a, model_b)][bootstrap_index] = (
                scores[model_b] - scores[model_a]
            )

    tail = (1.0 - float(confidence_level)) / 2.0
    comparisons = []
    for model_a, model_b in COMPARISONS:
        deltas = distributions[(model_a, model_b)]
        ci_low, ci_high = np.quantile(deltas, [tail, 1.0 - tail])
        model_a_info = cohort["models"][model_a]
        model_b_info = cohort["models"][model_b]
        comparisons.append(
            {
                "protocol": protocol,
                "comparison": f"DeltaMacro-F1({model_a},{model_b})",
                "model_a": model_a,
                "model_b": model_b,
                "evaluated_model_a": model_a_info["evaluated_model_id"],
                "evaluated_model_b": model_b_info["evaluated_model_id"],
                "delta_definition": "macro_f1_b_minus_macro_f1_a",
                "test_union_count": cohort["test_union_count"],
                "common_success_count": cohort["common_success_count"],
                "excluded_from_common_count": cohort["excluded_from_common_count"],
                "model_a_usable_count": model_a_info["usable_prediction_count"],
                "model_b_usable_count": model_b_info["usable_prediction_count"],
                "model_a_full_cohort_macro_f1": model_a_info["reported_full_cohort_macro_f1"],
                "model_b_full_cohort_macro_f1": model_b_info["reported_full_cohort_macro_f1"],
                "model_a_common_cohort_macro_f1": observed[model_a],
                "model_b_common_cohort_macro_f1": observed[model_b],
                "delta_macro_f1": observed[model_b] - observed[model_a],
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "confidence_level": float(confidence_level),
                "bootstrap_iterations": int(iterations),
                "bootstrap_seed": int(seed),
                "bootstrap_method": "true_label_stratified_paired_percentile",
                "class_labels": json.dumps(labels, ensure_ascii=False),
            }
        )
    return cohort, comparisons


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty paired-bootstrap CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def run_bracs_paired_bootstrap(
    root: str | Path = "bracs_eval",
    *,
    iterations: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    root = Path(root)
    protocol_summaries = []
    comparisons = []
    for protocol in PROTOCOL_MODELS:
        cohort, protocol_comparisons = compute_protocol_paired_bootstrap(
            root,
            protocol,
            iterations=iterations,
            seed=seed,
            confidence_level=confidence_level,
        )
        protocol_summaries.append(cohort)
        comparisons.extend(protocol_comparisons)
    payload = {
        "schema_version": "nrkd_bracs_paired_bootstrap_v1",
        "framework": "NativeResKD",
        "abbreviation": "NRKD",
        "version": "4.01",
        "delta_definition": "macro_f1_b_minus_macro_f1_a",
        "bootstrap": {
            "method": "true_label_stratified_paired_percentile",
            "iterations": int(iterations),
            "seed": int(seed),
            "confidence_level": float(confidence_level),
        },
        "protocols": protocol_summaries,
        "comparisons": comparisons,
    }
    _atomic_write_csv(root / "paired_bootstrap_macro_f1.csv", comparisons)
    atomic_write_json(root / "paired_bootstrap_macro_f1.json", payload)
    return payload


__all__ = [
    "COMPARISONS",
    "PROTOCOL_MODELS",
    "compute_protocol_paired_bootstrap",
    "run_bracs_paired_bootstrap",
]

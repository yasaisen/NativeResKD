"""Explicit cross-protocol BRACS paired-bootstrap comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..common.utils import atomic_write_json
from .bracs_paired_bootstrap import (
    _atomic_write_csv,
    _read_model_result,
    _require_f1_score,
    run_bracs_paired_bootstrap,
)


CROSS_PROTOCOL_COMPARISONS = (
    {
        "model_a": "T",
        "protocol_a": "matched_512",
        "evaluated_model_a": "T",
        "model_b": "S1",
        "protocol_b": "native_roi",
        "evaluated_model_b": "S1_native",
    },
)


def _compute_cross_protocol_comparison(
    root: Path,
    specification: dict[str, str],
    *,
    iterations: int,
    seed: int,
    confidence_level: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    f1_score = _require_f1_score()
    protocol_a = specification["protocol_a"]
    protocol_b = specification["protocol_b"]
    evaluated_a = specification["evaluated_model_a"]
    evaluated_b = specification["evaluated_model_b"]
    result_a = _read_model_result(root, protocol_a, evaluated_a)
    result_b = _read_model_result(root, protocol_b, evaluated_b)
    all_ids = set(result_a["rows"]) | set(result_b["rows"])
    common_ids = result_a["usable_ids"] & result_b["usable_ids"]
    if not common_ids:
        raise RuntimeError(
            f"No jointly successful samples for {protocol_a}:{evaluated_a} and "
            f"{protocol_b}:{evaluated_b}"
        )

    labels_by_sample = {}
    for sample_id in sorted(all_ids):
        observed = {
            result["rows"][sample_id]["label"]
            for result in (result_a, result_b)
            if sample_id in result["rows"]
        }
        if len(observed) != 1:
            raise ValueError(
                "Ground-truth label mismatch for cross-protocol comparison, "
                f"sample_id={sample_id!r}: {sorted(observed)}"
            )
        labels_by_sample[sample_id] = next(iter(observed))

    labels = sorted(set(labels_by_sample.values()))
    common_labels = {labels_by_sample[sample_id] for sample_id in common_ids}
    missing_classes = set(labels) - common_labels
    if missing_classes:
        raise RuntimeError(
            "Cross-protocol common cohort has no samples for classes: "
            f"{sorted(missing_classes)}"
        )
    ordered_ids = sorted(common_ids)
    y_true = np.asarray([labels_by_sample[sample_id] for sample_id in ordered_ids], dtype=object)
    prediction_a = np.asarray(
        [result_a["rows"][sample_id]["prediction"] for sample_id in ordered_ids],
        dtype=object,
    )
    prediction_b = np.asarray(
        [result_b["rows"][sample_id]["prediction"] for sample_id in ordered_ids],
        dtype=object,
    )
    observed_a = float(
        f1_score(y_true, prediction_a, labels=labels, average="macro", zero_division=0)
    )
    observed_b = float(
        f1_score(y_true, prediction_b, labels=labels, average="macro", zero_division=0)
    )

    groups = [np.flatnonzero(y_true == label) for label in labels]
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(iterations), dtype=np.float64)
    for bootstrap_index in range(int(iterations)):
        sampled = np.concatenate(
            [rng.choice(group, size=group.size, replace=True) for group in groups]
        )
        score_a = f1_score(
            y_true[sampled],
            prediction_a[sampled],
            labels=labels,
            average="macro",
            zero_division=0,
        )
        score_b = f1_score(
            y_true[sampled],
            prediction_b[sampled],
            labels=labels,
            average="macro",
            zero_division=0,
        )
        deltas[bootstrap_index] = float(score_b - score_a)
    tail = (1.0 - float(confidence_level)) / 2.0
    ci_low, ci_high = np.quantile(deltas, [tail, 1.0 - tail])

    cohort = {
        "comparison_scope": "cross_protocol",
        "protocol_a": protocol_a,
        "protocol_b": protocol_b,
        "evaluated_model_a": evaluated_a,
        "evaluated_model_b": evaluated_b,
        "test_union_count": len(all_ids),
        "common_success_count": len(common_ids),
        "excluded_from_common_count": len(all_ids - common_ids),
        "class_labels": labels,
        "class_counts": {
            label: int(np.count_nonzero(y_true == label)) for label in labels
        },
        "model_a_status_counts": result_a["status_counts"],
        "model_b_status_counts": result_b["status_counts"],
        "model_a_missing_row_count": len(all_ids - set(result_a["rows"])),
        "model_b_missing_row_count": len(all_ids - set(result_b["rows"])),
    }
    comparison = {
        "protocol": "cross_protocol",
        "comparison": (
            f"DeltaMacro-F1({protocol_a}:{specification['model_a']},"
            f"{protocol_b}:{specification['model_b']})"
        ),
        "model_a": specification["model_a"],
        "model_b": specification["model_b"],
        "evaluated_model_a": evaluated_a,
        "evaluated_model_b": evaluated_b,
        "delta_definition": "macro_f1_b_minus_macro_f1_a",
        "test_union_count": len(all_ids),
        "common_success_count": len(common_ids),
        "excluded_from_common_count": len(all_ids - common_ids),
        "model_a_usable_count": len(result_a["usable_ids"]),
        "model_b_usable_count": len(result_b["usable_ids"]),
        "model_a_full_cohort_macro_f1": float(result_a["metrics"]["macro_f1"]),
        "model_b_full_cohort_macro_f1": float(result_b["metrics"]["macro_f1"]),
        "model_a_common_cohort_macro_f1": observed_a,
        "model_b_common_cohort_macro_f1": observed_b,
        "delta_macro_f1": observed_b - observed_a,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "confidence_level": float(confidence_level),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(seed),
        "bootstrap_method": "true_label_stratified_paired_percentile",
        "class_labels": json.dumps(labels, ensure_ascii=False),
        "comparison_scope": "cross_protocol",
        "protocol_a": protocol_a,
        "protocol_b": protocol_b,
    }
    return cohort, comparison


def run_bracs_paired_bootstrap_with_cross_protocol(
    root: str | Path = "bracs_eval",
    *,
    iterations: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    root = Path(root)
    payload = run_bracs_paired_bootstrap(
        root,
        iterations=iterations,
        seed=seed,
        confidence_level=confidence_level,
    )
    for row in payload["comparisons"]:
        row["comparison_scope"] = "within_protocol"
        row["protocol_a"] = row["protocol"]
        row["protocol_b"] = row["protocol"]

    cross_cohorts = []
    for specification in CROSS_PROTOCOL_COMPARISONS:
        cohort, comparison = _compute_cross_protocol_comparison(
            root,
            specification,
            iterations=iterations,
            seed=seed,
            confidence_level=confidence_level,
        )
        cross_cohorts.append(cohort)
        payload["comparisons"].append(comparison)
    payload["cross_protocol_cohorts"] = cross_cohorts
    payload["comparison_count"] = len(payload["comparisons"])
    _atomic_write_csv(root / "paired_bootstrap_macro_f1.csv", payload["comparisons"])
    atomic_write_json(root / "paired_bootstrap_macro_f1.json", payload)
    return payload


__all__ = [
    "CROSS_PROTOCOL_COMPARISONS",
    "run_bracs_paired_bootstrap_with_cross_protocol",
]

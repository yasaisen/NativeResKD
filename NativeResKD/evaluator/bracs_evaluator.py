from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..datasets.bracs_dataset import BRACSDataset


def _require_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            roc_auc_score,
        )
        from sklearn.preprocessing import LabelEncoder, label_binarize
    except ImportError as exc:
        raise RuntimeError("BRACS evaluation requires: pip install -e '.[eval]'") from exc
    return locals()


class BRACSEvaluator:
    def __init__(
        self,
        cfg,
        encoder,
        *,
        model_id: str,
        protocol: str,
        output_protocol: str | None = None,
        output_dir: str | Path | None = None,
    ):
        self.cfg = cfg
        self.encoder = encoder
        self.model_id = model_id
        self.protocol = protocol
        self.output_protocol = output_protocol or protocol
        requested = str(cfg.runtime.device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.encoder.to(self.device).eval().requires_grad_(False)
        self.output_dir = Path(output_dir or "bracs_eval") / self.output_protocol / model_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def extract(self, split: str) -> list[dict[str, Any]]:
        dataset = BRACSDataset.from_config(self.cfg, split=split, protocol=self.protocol)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(self.cfg.training.num_workers))
        records = []
        for batch in loader:
            sample_id = batch["sample_id"][0]
            grid_h, grid_w = map(int, batch["grid_hw"][0])
            tokens = grid_h * grid_w
            base = {
                "sample_id": sample_id,
                "split": split,
                "label": batch["label"][0],
                "original_h": int(batch["original_hw"][0, 0]),
                "original_w": int(batch["original_hw"][0, 1]),
                "resampled_h": int(batch["resampled_hw"][0, 0]),
                "resampled_w": int(batch["resampled_hw"][0, 1]),
                "padded_h": int(batch["padded_hw"][0, 0]),
                "padded_w": int(batch["padded_hw"][0, 1]),
                "grid_h": grid_h,
                "grid_w": grid_w,
                "spatial_tokens": tokens,
                "source_magnification": float(batch["source_magnification"][0]),
                "target_magnification": float(batch["target_magnification"][0]),
                "coordinate_group": "in_coordinate_range" if grid_h <= 32 and grid_w <= 32 else "coordinate_extrapolation",
                "sequence_group": "in_sequence_range" if tokens <= 1024 else "sequence_extrapolation",
            }
            limit = self.cfg.bracs.max_native_tokens
            if limit is not None and tokens > int(limit):
                records.append({**base, "status": "token_limit", "feature": None})
                continue
            try:
                output = self.encoder(batch["image"].to(self.device))
                feature = output["global_preproj"] if isinstance(output, dict) else output
                records.append({**base, "status": "ok", "feature": feature.float().cpu().numpy()[0]})
            except torch.cuda.OutOfMemoryError:
                records.append({**base, "status": "oom", "feature": None})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return records

    def run(self) -> dict[str, Any]:
        sk = _require_sklearn()
        all_records = {split: self.extract(split) for split in ("train", "valid", "test")}
        labels = sorted({str(row["label"]) for rows in all_records.values() for row in rows})
        encoder = sk["LabelEncoder"]().fit(labels)

        def successful(split):
            rows = [row for row in all_records[split] if row["status"] == "ok"]
            if not rows:
                raise RuntimeError(f"No successful {split} features")
            return rows, np.stack([row["feature"] for row in rows]), encoder.transform([str(row["label"]) for row in rows])

        train_rows, x_train, y_train = successful("train")
        valid_rows, x_valid, y_valid = successful("valid")
        test_rows, x_test, y_test = successful("test")
        candidates = []
        for c_value in sorted(map(float, self.cfg.bracs.linear_probe_C)):
            classifier = sk["LogisticRegression"](
                C=c_value,
                max_iter=int(self.cfg.bracs.max_iter),
                solver="lbfgs",
                class_weight=None,
            ).fit(x_train, y_train)
            prediction = classifier.predict(x_valid)
            candidates.append((sk["f1_score"](y_valid, prediction, average="macro"), c_value, classifier))
        best_score = max(score for score, _, _ in candidates)
        _, selected_c, classifier = min(
            (item for item in candidates if abs(item[0] - best_score) <= 1e-12), key=lambda item: item[1]
        )
        prediction = classifier.predict(x_test)
        probabilities = classifier.predict_proba(x_test)
        metrics = self._classification_metrics(sk, y_test, prediction, probabilities, len(encoder.classes_))
        total_test = len(all_records["test"])
        metrics.update({
            "model_id": self.model_id,
            "protocol": self.output_protocol,
            "selected_C": selected_c,
            "validation_macro_f1": best_score,
            "coverage": len(test_rows) / total_test if total_test else 0.0,
            "oom_count": sum(row["status"] == "oom" for row in all_records["test"]),
            "oom_rate": sum(row["status"] == "oom" for row in all_records["test"]) / total_test if total_test else 0.0,
            "coverage_adjusted_accuracy": metrics["accuracy"] * len(test_rows) / total_test if total_test else 0.0,
        })
        for row, pred in zip(test_rows, prediction):
            row["prediction"] = encoder.inverse_transform([pred])[0]
        extrapolation = self._extrapolation(sk, all_records["test"], encoder)
        self._save(metrics, all_records, extrapolation)
        return {"metrics": metrics, "extrapolation": extrapolation}

    @staticmethod
    def _classification_metrics(sk, target, prediction, probabilities, class_count: int) -> dict[str, Any]:
        result = {
            "macro_f1": float(sk["f1_score"](target, prediction, average="macro")),
            "balanced_accuracy": float(sk["balanced_accuracy_score"](target, prediction)),
            "accuracy": float(sk["accuracy_score"](target, prediction)),
            "per_class_f1": sk["f1_score"](target, prediction, average=None, labels=np.arange(class_count)).tolist(),
            "confusion_matrix": sk["confusion_matrix"](target, prediction, labels=np.arange(class_count)).tolist(),
        }
        try:
            binary = sk["label_binarize"](target, classes=np.arange(class_count))
            result["macro_auroc_ovr"] = float(sk["roc_auc_score"](binary, probabilities, average="macro", multi_class="ovr"))
        except ValueError:
            result["macro_auroc_ovr"] = None
        return result

    @staticmethod
    def _extrapolation(sk, records: list[dict], label_encoder) -> list[dict]:
        results = []
        for field in ("coordinate_group", "sequence_group"):
            groups = defaultdict(list)
            for row in records:
                groups[row[field]].append(row)
            for group, rows in groups.items():
                ok = [row for row in rows if row["status"] == "ok" and "prediction" in row]
                entry = {
                    "group_type": field,
                    "group": group,
                    "sample_count": len(rows),
                    "coverage": len(ok) / len(rows),
                    "oom_rate": sum(row["status"] == "oom" for row in rows) / len(rows),
                    "macro_f1": None,
                    "accuracy": None,
                }
                if ok:
                    target = label_encoder.transform([str(row["label"]) for row in ok])
                    pred = label_encoder.transform([str(row["prediction"]) for row in ok])
                    entry["macro_f1"] = float(sk["f1_score"](target, pred, average="macro"))
                    entry["accuracy"] = float(sk["accuracy_score"](target, pred))
                results.append(entry)
        return results

    def _save(self, metrics: dict, records: dict[str, list[dict]], extrapolation: list[dict]) -> None:
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
        flat = []
        for rows in records.values():
            for row in rows:
                flat.append({key: value for key, value in row.items() if key != "feature"})
        if flat:
            with (self.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({key for row in flat for key in row}))
                writer.writeheader()
                writer.writerows(flat)
        if extrapolation:
            with (self.output_dir / "extrapolation_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(extrapolation[0]))
                writer.writeheader()
                writer.writerows(extrapolation)


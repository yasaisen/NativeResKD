"""NativeResKD model facade: frozen CONCH teacher plus full-train RoPE student."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .kd_model_wrapper import KDModelWrapper


class NativeResolutionKnowledgeDistiller(KDModelWrapper):
    framework_name = "NativeResKD"
    framework_abbreviation = "NRKD"
    framework_version = "4.01"

    def model_checkpoint_payload(self, checkpoint_metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "nrkd_model_v1",
            "framework": self.framework_name,
            "version": self.framework_version,
            "checkpoint_metadata": checkpoint_metadata,
            "student_state_dict": self.student.state_dict(),
        }

    def load_student_payload(self, payload: dict[str, Any], strict: bool = True) -> None:
        state = payload.get("student_state_dict", payload.get("student", payload))
        self.student.load_state_dict(state, strict=strict)

    def save_student(self, path: str | Path, checkpoint_metadata: dict[str, Any]) -> None:
        torch.save(self.model_checkpoint_payload(checkpoint_metadata), path)


NativeResKDModel = NativeResolutionKnowledgeDistiller
NRKDModel = NativeResolutionKnowledgeDistiller


from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .conch_loader import load_conch_teacher
from .conch_rope_student import (
    CONCHRoPEStudent,
    TeacherFeatureEncoder,
    surgery_from_teacher,
    verify_surgery_initialization,
)


class KDModelWrapper(nn.Module):
    def __init__(self, teacher: nn.Module, student: CONCHRoPEStudent, selected_layers: tuple[int, ...]):
        super().__init__()
        self.teacher = TeacherFeatureEncoder(teacher, selected_layers)
        self.student = student
        self.teacher.eval().requires_grad_(False)

    @classmethod
    def from_config(cls, cfg) -> "KDModelWrapper":
        teacher, _ = load_conch_teacher(cfg)
        layers = tuple(int(x) for x in cfg.loss.selected_layers)
        student = surgery_from_teacher(
            teacher,
            theta=float(cfg.rope.theta),
            learnable_frequency=bool(cfg.rope.learnable_frequency),
            selected_layers=layers,
            seed=int(cfg.seed),
        )
        verify_surgery_initialization(teacher, student)
        return cls(teacher, student, layers)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        self.student.train(mode)
        return self

    def forward(
        self,
        teacher_image: torch.Tensor,
        student_image: torch.Tensor,
        patch_mapping: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        del patch_mapping  # correspondence is consumed by the criterion, never by attention
        teacher_outputs = self.teacher(teacher_image)
        student_outputs = self.student(student_image)
        return {"teacher": teacher_outputs, "student": student_outputs}



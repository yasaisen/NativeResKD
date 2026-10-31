"""Evaluator facade retaining representation and BRACS protocols."""

from .bracs_evaluator import BRACSEvaluator
from .representation_evaluator import RepresentationEvaluator


class NRKDEvaluator:
    representation_evaluator_cls = RepresentationEvaluator
    bracs_evaluator_cls = BRACSEvaluator


__all__ = ["NRKDEvaluator", "RepresentationEvaluator", "BRACSEvaluator"]


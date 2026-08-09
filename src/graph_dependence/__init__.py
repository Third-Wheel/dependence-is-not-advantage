"""Diagnostics that keep graph dependence and graph advantage separate."""

from .api import diagnose
from .metrics import binary_auc
from .nulls import DegreePreservingNull, degree_preserving
from .types import (
    DependenceReport,
    GraphFreeScorer,
    GraphScorer,
    Metric,
    NullDraw,
    NullModel,
)

__all__ = [
    "DegreePreservingNull",
    "DependenceReport",
    "GraphFreeScorer",
    "GraphScorer",
    "Metric",
    "NullDraw",
    "NullModel",
    "binary_auc",
    "degree_preserving",
    "diagnose",
]

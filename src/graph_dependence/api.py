from __future__ import annotations

import math
import random
from collections.abc import Hashable, Sequence
from statistics import fmean
from typing import TypeVar

import networkx as nx

from .metrics import binary_auc
from .nulls import DegreePreservingNull
from .types import (
    DependenceReport,
    GraphFreeScorer,
    GraphScorer,
    Metric,
    NullDraw,
    NullModel,
    frozen_metadata,
)

ExampleT = TypeVar("ExampleT")
NodeT = TypeVar("NodeT", bound=Hashable)


def _callable_name(value: object) -> str:
    name = getattr(value, "__name__", None)
    if isinstance(name, str):
        return name
    return type(value).__name__


def _metric_score(
    predictions: Sequence[float],
    targets: Sequence[float | int | bool],
    metric: Metric,
) -> float:
    if len(predictions) != len(targets):
        raise ValueError("a scorer must return one prediction per target")
    value = float(metric(predictions, targets))
    if not math.isfinite(value):
        raise ValueError("metric must return a finite scalar")
    return value


def _resolve_null(null_model: str | NullModel[NodeT]) -> NullModel[NodeT]:
    if null_model == "degree_preserving":
        return DegreePreservingNull[NodeT]()
    if isinstance(null_model, str):
        raise ValueError("the only built-in null model is 'degree_preserving'")
    return null_model


def diagnose(
    graph: nx.Graph[NodeT],
    examples: Sequence[ExampleT],
    targets: Sequence[float | int | bool],
    graph_scorer: GraphScorer[NodeT, ExampleT],
    *,
    graph_free_scorer: GraphFreeScorer[ExampleT] | None = None,
    null_model: str | NullModel[NodeT] = "degree_preserving",
    metric: Metric = binary_auc,
    null_draws: int = 1,
    seed: int | None = 0,
) -> DependenceReport:
    """Measure graph dependence and comparator-relative graph advantage.

    ``seed`` controls null generation only. Any randomness in scorer fitting or
    evaluation remains part of the supplied scorer's procedure and should be
    controlled by the caller.
    """

    if null_draws <= 0:
        raise ValueError("null_draws must be positive")
    if len(examples) != len(targets):
        raise ValueError("examples and targets must have the same length")
    if not examples:
        raise ValueError("examples and targets must not be empty")

    frozen_examples = tuple(examples)
    frozen_targets = tuple(targets)
    observed_predictions = graph_scorer(graph, frozen_examples)
    observed_score = _metric_score(observed_predictions, frozen_targets, metric)

    model = _resolve_null(null_model)
    seed_stream = random.Random(seed)
    draw_scores: list[float] = []
    draw_metadata: list[object] = []
    draw_seeds: list[int | None] = []
    for _ in range(null_draws):
        draw_seed = None if seed is None else seed_stream.randrange(2**63)
        raw_draw = model(graph.copy(), seed=draw_seed)
        if isinstance(raw_draw, NullDraw):
            draw = raw_draw
        elif isinstance(raw_draw, nx.Graph):
            draw = NullDraw(graph=raw_draw, metadata=frozen_metadata({}))
        else:
            raise TypeError("a null model must return NullDraw or networkx.Graph")
        if draw.graph is graph:
            raise ValueError("a null model must return a fresh graph")
        null_predictions = graph_scorer(draw.graph, frozen_examples)
        draw_scores.append(_metric_score(null_predictions, frozen_targets, metric))
        draw_seeds.append(draw_seed)
        draw_metadata.append(draw.metadata)

    null_score = fmean(draw_scores)
    comparator_score: float | None = None
    if graph_free_scorer is not None:
        comparator_predictions = graph_free_scorer(frozen_examples)
        comparator_score = _metric_score(comparator_predictions, frozen_targets, metric)

    model_name = getattr(model, "name", _callable_name(model))
    if not isinstance(model_name, str):
        model_name = _callable_name(model)
    metadata = frozen_metadata(
        {
            "evaluation_examples": len(frozen_examples),
            "null_draws": null_draws,
            "null_draw_seeds": tuple(draw_seeds),
            "null_draw_metadata": tuple(draw_metadata),
            "graph_scorer": _callable_name(graph_scorer),
            "graph_free_scorer": (
                _callable_name(graph_free_scorer)
                if graph_free_scorer is not None
                else None
            ),
        }
    )
    return DependenceReport(
        dependence=observed_score - null_score,
        advantage=(
            observed_score - comparator_score if comparator_score is not None else None
        ),
        null_survival=(null_score / observed_score if observed_score > 0 else None),
        observed_score=observed_score,
        null_score=null_score,
        comparator_score=comparator_score,
        metric_name=_callable_name(metric),
        null_model_name=model_name,
        seed=seed,
        metadata=metadata,
    )

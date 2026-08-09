from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

import networkx as nx
import pytest

from graph_dependence import DegreePreservingNull, NullDraw, binary_auc, diagnose
from graph_dependence.types import frozen_metadata

Pair = tuple[int, int]


def _constant_graph_scorer(
    graph: nx.Graph[int], examples: Sequence[Pair]
) -> list[float]:
    del graph
    return [0.5 for _ in examples]


def test_graph_independent_scorer_has_zero_dependence() -> None:
    graph = nx.cycle_graph(10)
    examples = [(0, 1), (2, 3), (0, 5), (1, 6)]
    report = diagnose(
        graph,
        examples,
        [1, 1, 0, 0],
        _constant_graph_scorer,
        null_draws=3,
        seed=4,
    )
    assert report.observed_score == 0.5
    assert report.null_score == 0.5
    assert report.dependence == 0.0
    assert report.null_survival == 1.0
    assert report.advantage is None


def test_structure_reading_scorer_degrades_under_null() -> None:
    graph = nx.cycle_graph(12)
    positives = list(graph.edges())
    negatives = [(node, (node + 6) % 12) for node in range(12)]
    examples = positives + negatives
    targets = [1] * len(positives) + [0] * len(negatives)

    def reads_edges(condition: nx.Graph[int], pairs: Sequence[Pair]) -> list[float]:
        return [float(condition.has_edge(*pair)) for pair in pairs]

    report = diagnose(
        graph,
        examples,
        targets,
        reads_edges,
        null_model=DegreePreservingNull[int](
            attempts_per_edge=50, forbidden_edges=tuple(examples)
        ),
        null_draws=5,
        seed=12,
    )
    assert report.observed_score == 1.0
    assert report.null_score < report.observed_score
    assert report.dependence > 0.0


def test_advantage_uses_comparator_not_randomized_graph() -> None:
    graph = nx.gnm_random_graph(10, 20, seed=1)
    examples = [(0, 1), (2, 3)]

    def graph_score(condition: nx.Graph[int], pairs: Sequence[Pair]) -> list[float]:
        del pairs
        return [condition.number_of_edges() / 100] * 2

    def comparator(pairs: Sequence[Pair]) -> list[float]:
        return [0.05 for _ in pairs]

    def mean_metric(
        predictions: Sequence[float], targets: Sequence[float | int | bool]
    ) -> float:
        del targets
        return fmean(predictions)

    report = diagnose(
        graph,
        examples,
        [0, 1],
        graph_score,
        graph_free_scorer=comparator,
        metric=mean_metric,
        seed=2,
    )
    assert report.observed_score == pytest.approx(0.2)
    assert report.null_score == pytest.approx(0.2)
    assert report.dependence == pytest.approx(0.0)
    assert report.comparator_score == pytest.approx(0.05)
    assert report.advantage == pytest.approx(0.15)


def test_fixed_seed_reproduces_null_scores_and_draw_metadata() -> None:
    graph = nx.gnm_random_graph(20, 40, seed=5)
    examples = [(0, 1), (2, 3), (4, 5), (6, 7)]
    targets = [1, 0, 1, 0]

    def score(condition: nx.Graph[int], pairs: Sequence[Pair]) -> list[float]:
        return [float(condition.has_edge(*pair)) for pair in pairs]

    first = diagnose(graph, examples, targets, score, null_draws=4, seed=9)
    second = diagnose(graph, examples, targets, score, null_draws=4, seed=9)
    assert first.to_dict() == second.to_dict()


def test_custom_null_graph_and_nonpositive_survival() -> None:
    graph: nx.Graph[int] = nx.path_graph(4)

    class EmptyNull:
        name = "empty"

        def __call__(
            self, condition: nx.Graph[int], *, seed: int | None
        ) -> NullDraw[int]:
            del seed
            condition.remove_edges_from(list(condition.edges()))
            return NullDraw(condition, frozen_metadata({"removed": "all"}))

    def zero_score(condition: nx.Graph[int], pairs: Sequence[Pair]) -> list[float]:
        del condition
        return [0.0 for _ in pairs]

    def mean_metric(
        predictions: Sequence[float], targets: Sequence[float | int | bool]
    ) -> float:
        del targets
        return fmean(predictions)

    report = diagnose(
        graph,
        [(0, 1)],
        [1],
        zero_score,
        null_model=EmptyNull(),
        metric=mean_metric,
    )
    assert report.null_survival is None
    assert report.metadata["null_draws"] == 1


def test_diagnose_rejects_bad_shapes_and_draw_count() -> None:
    graph = nx.path_graph(4)
    with pytest.raises(ValueError, match="same length"):
        diagnose(graph, [(0, 1)], [], _constant_graph_scorer)
    with pytest.raises(ValueError, match="positive"):
        diagnose(graph, [(0, 1)], [1], _constant_graph_scorer, null_draws=0)


def test_diagnose_default_metric_is_binary_auc() -> None:
    graph = nx.path_graph(4)

    def scorer(condition: nx.Graph[int], pairs: Sequence[Pair]) -> list[float]:
        return [float(condition.has_edge(*pair)) for pair in pairs]

    report = diagnose(graph, [(0, 1), (0, 3)], [1, 0], scorer, seed=1)
    assert report.metric_name == binary_auc.__name__

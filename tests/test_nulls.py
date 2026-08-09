from __future__ import annotations

from collections.abc import Hashable
from typing import TypeVar

import networkx as nx
import pytest

from graph_dependence import DegreePreservingNull

NodeT = TypeVar("NodeT", bound=Hashable)


def _edge_set(graph: nx.Graph[NodeT]) -> set[frozenset[NodeT]]:
    return {frozenset(edge) for edge in graph.edges()}


def test_degree_preserving_null_is_reproducible_and_preserves_invariants() -> None:
    graph = nx.gnm_random_graph(30, 70, seed=4)
    intact_edges = _edge_set(graph)
    intact_degrees = dict(graph.degree())
    null = DegreePreservingNull[int](attempts_per_edge=20)

    first = null(graph, seed=19)
    second = null(graph, seed=19)

    assert _edge_set(first.graph) == _edge_set(second.graph)
    assert _edge_set(graph) == intact_edges
    assert dict(first.graph.degree()) == intact_degrees
    assert first.graph.number_of_edges() == graph.number_of_edges()
    assert all(left != right for left, right in first.graph.edges())
    assert first.metadata["labeled_degree_preserved"] is True
    assert first.metadata["edge_count_preserved"] is True
    accepted = first.metadata["accepted_swaps"]
    assert isinstance(accepted, int)
    assert accepted > 0


def test_forbidden_edges_are_not_introduced() -> None:
    graph = nx.cycle_graph(12)
    forbidden = ((0, 6), (1, 7), (2, 8), (3, 9))
    draw = DegreePreservingNull[int](attempts_per_edge=50, forbidden_edges=forbidden)(
        graph, seed=3
    )

    assert all(not draw.graph.has_edge(*edge) for edge in forbidden)
    assert draw.metadata["forbidden_edge_violations"] == 0


def test_existing_forbidden_edge_may_remain_but_is_not_reintroduced() -> None:
    graph = nx.cycle_graph(10)
    draw = DegreePreservingNull[int](attempts_per_edge=30, forbidden_edges=((0, 1),))(
        graph, seed=8
    )
    assert draw.metadata["forbidden_edge_violations"] == 0


@pytest.mark.parametrize(
    "graph",
    [nx.DiGraph([(0, 1), (1, 2)]), nx.MultiGraph([(0, 1), (0, 1)])],
)
def test_null_rejects_unsupported_graph_types(graph: nx.Graph[int]) -> None:
    with pytest.raises(TypeError):
        DegreePreservingNull[int]()(graph, seed=0)


def test_null_rejects_self_loops_and_edge_attributes() -> None:
    with pytest.raises(ValueError, match="self-loops"):
        DegreePreservingNull[int]()(nx.Graph([(0, 0)]), seed=0)

    attributed: nx.Graph[int] = nx.Graph()
    attributed.add_edge(0, 1, weight=2.0)
    with pytest.raises(ValueError, match="edge attributes"):
        DegreePreservingNull[int]()(attributed, seed=0)


def test_attempt_budget_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        DegreePreservingNull[int](attempts_per_edge=0)

"""Run a deterministic, network-free graph-dependence diagnosis."""

from __future__ import annotations

import json
from collections.abc import Sequence

import networkx as nx

from graph_dependence import DegreePreservingNull, binary_auc, diagnose

Pair = tuple[str, str]

FEATURES: dict[str, frozenset[str]] = {
    "a": frozenset({"paper", "graph", "evaluation"}),
    "b": frozenset({"paper", "graph"}),
    "c": frozenset({"graph", "null"}),
    "d": frozenset({"null", "evaluation"}),
    "e": frozenset({"evaluation", "baseline"}),
    "f": frozenset({"paper", "baseline"}),
}


def common_neighbors(graph: nx.Graph[str], examples: Sequence[Pair]) -> list[float]:
    """Score pairs by the number of common graph neighbors."""

    return [
        float(len(set(graph.neighbors(left)) & set(graph.neighbors(right))))
        for left, right in examples
    ]


def content_jaccard(examples: Sequence[Pair]) -> list[float]:
    """Score pairs using node-content overlap without reading the graph."""

    scores: list[float] = []
    for left, right in examples:
        intersection = FEATURES[left] & FEATURES[right]
        union = FEATURES[left] | FEATURES[right]
        scores.append(len(intersection) / len(union) if union else 0.0)
    return scores


def main() -> None:
    """Build a small fixed graph and print its JSON-compatible report."""

    graph: nx.Graph[str] = nx.Graph()
    graph.add_edges_from(
        [
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
            ("e", "f"),
            ("f", "a"),
            ("a", "d"),
            ("b", "e"),
            ("c", "f"),
        ]
    )
    examples: list[Pair] = [
        ("a", "c"),
        ("b", "d"),
        ("c", "e"),
        ("d", "f"),
        ("a", "e"),
        ("b", "f"),
    ]
    targets = [1, 1, 1, 1, 0, 0]

    report = diagnose(
        graph=graph,
        examples=examples,
        targets=targets,
        graph_scorer=common_neighbors,
        graph_free_scorer=content_jaccard,
        null_model=DegreePreservingNull[str](attempts_per_edge=5),
        metric=binary_auc,
        null_draws=3,
        seed=7,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

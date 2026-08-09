from __future__ import annotations

import random
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import networkx as nx

from .types import NullDraw, frozen_metadata

NodeT = TypeVar("NodeT", bound=Hashable)
Edge = tuple[NodeT, NodeT]
EdgeKey = frozenset[NodeT]


def _edge_key(edge: Edge[NodeT]) -> EdgeKey[NodeT]:
    return frozenset(edge)


@dataclass(frozen=True)
class DegreePreservingNull(Generic[NodeT]):
    """A fixed-budget double-edge-swap null for simple, unweighted graphs.

    ``forbidden_edges`` are pairs that a draw may not introduce. A forbidden
    pair already present in the intact graph may be retained, but once removed
    it is not reintroduced.
    """

    attempts_per_edge: int = 10
    forbidden_edges: tuple[Edge[NodeT], ...] = ()
    name: str = field(default="degree_preserving", init=False)

    def __post_init__(self) -> None:
        if self.attempts_per_edge <= 0:
            raise ValueError("attempts_per_edge must be positive")

    def __call__(self, graph: nx.Graph[NodeT], *, seed: int | None) -> NullDraw[NodeT]:
        if graph.is_directed():
            raise TypeError("DegreePreservingNull requires an undirected graph")
        if graph.is_multigraph():
            raise TypeError("DegreePreservingNull requires a simple graph")
        if any(left == right for left, right in graph.edges()):
            raise ValueError("DegreePreservingNull does not admit self-loops")
        if any(attributes for _, _, attributes in graph.edges(data=True)):
            raise ValueError(
                "edge attributes are not supported because swaps would require "
                "an explicit attribute-assignment rule"
            )

        work = graph.copy()
        edges: list[Edge[NodeT]] = list(work.edges())
        intact_keys = {_edge_key(edge) for edge in edges}
        forbidden = {_edge_key(edge) for edge in self.forbidden_edges} - intact_keys
        live = set(intact_keys)
        intact_degrees = dict(graph.degree())
        edge_count = len(edges)
        attempts = self.attempts_per_edge * edge_count
        accepted = 0
        rejections = {
            "self_loop": 0,
            "duplicate": 0,
            "forbidden": 0,
            "degenerate": 0,
        }
        rng = random.Random(seed)

        if edge_count >= 2:
            for _ in range(attempts):
                left_index, right_index = rng.sample(range(edge_count), 2)
                left = edges[left_index]
                right = edges[right_index]
                a, b = left if rng.getrandbits(1) else (left[1], left[0])
                c, d = right if rng.getrandbits(1) else (right[1], right[0])
                proposed_left = (a, d)
                proposed_right = (c, b)

                if a == d or c == b:
                    rejections["self_loop"] += 1
                    continue

                old_keys = {_edge_key(left), _edge_key(right)}
                new_keys = {_edge_key(proposed_left), _edge_key(proposed_right)}
                if len(new_keys) != 2:
                    rejections["duplicate"] += 1
                    continue
                if new_keys == old_keys:
                    rejections["degenerate"] += 1
                    continue
                remaining = live - old_keys
                if new_keys & remaining:
                    rejections["duplicate"] += 1
                    continue
                if new_keys & forbidden:
                    rejections["forbidden"] += 1
                    continue

                work.remove_edge(*left)
                work.remove_edge(*right)
                work.add_edge(*proposed_left)
                work.add_edge(*proposed_right)
                live.difference_update(old_keys)
                live.update(new_keys)
                edges[left_index] = proposed_left
                edges[right_index] = proposed_right
                accepted += 1

        output_keys = {_edge_key(edge) for edge in work.edges()}
        degree_preserved = dict(work.degree()) == intact_degrees
        edge_count_preserved = work.number_of_edges() == graph.number_of_edges()
        forbidden_violations = len(output_keys & forbidden)
        if not degree_preserved or not edge_count_preserved or forbidden_violations:
            raise RuntimeError("degree-preserving sampler violated an invariant")

        retained = len(output_keys & intact_keys)
        metadata = frozen_metadata(
            {
                "sampler": "fixed_budget_double_edge_swap",
                "seed": seed,
                "attempts_per_edge": self.attempts_per_edge,
                "attempted_swaps": attempts,
                "accepted_swaps": accepted,
                "rejections": frozen_metadata(rejections),
                "edge_count": edge_count,
                "edge_count_preserved": edge_count_preserved,
                "labeled_degree_preserved": degree_preserved,
                "forbidden_edge_violations": forbidden_violations,
                "edge_retention_fraction": (
                    retained / edge_count if edge_count else 1.0
                ),
                "components_observed": nx.number_connected_components(graph),
                "components_null": nx.number_connected_components(work),
            }
        )
        return NullDraw(graph=work, metadata=metadata)


def degree_preserving(
    graph: nx.Graph[NodeT],
    *,
    seed: int | None = None,
    attempts_per_edge: int = 10,
    forbidden_edges: tuple[Edge[NodeT], ...] = (),
) -> NullDraw[NodeT]:
    """Functional interface to :class:`DegreePreservingNull`."""

    return DegreePreservingNull[NodeT](
        attempts_per_edge=attempts_per_edge,
        forbidden_edges=forbidden_edges,
    )(graph, seed=seed)

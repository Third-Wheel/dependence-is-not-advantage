"""Tests for randomize_edges: degrees, forbidden pairs, uniformity."""

from __future__ import annotations

import itertools
from collections import Counter

import pytest
import torch

from graph_dependence import randomize_edges
from graph_dependence.randomize import drop_pairs


def edge_set(edge_index: torch.Tensor) -> frozenset[tuple[int, int]]:
    return frozenset((min(a, b), max(a, b)) for a, b in edge_index.t().tolist())


def random_graph(n: int, m: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    pairs = torch.combinations(torch.arange(n))
    chosen = pairs[torch.randperm(pairs.size(0), generator=g)[:m]].t()
    return torch.cat([chosen, chosen.flip(0)], dim=1)


def test_randomizing_keeps_every_degree_and_a_simple_graph():
    n = 200
    graph = random_graph(n, 600, seed=1)
    new = randomize_edges(graph, n, seed=3)
    assert torch.equal(torch.bincount(new[0], minlength=n), torch.bincount(graph[0], minlength=n))
    assert (new[0] != new[1]).all()
    assert len(edge_set(new)) == 600 and new.size(1) == 1200
    assert edge_set(new) != edge_set(graph)


def test_randomizing_never_creates_a_forbidden_pair():
    n = 60
    graph = random_graph(n, 200, seed=2)
    non_edges = [p for p in itertools.combinations(range(n), 2) if p not in edge_set(graph)]
    forbidden = torch.tensor(non_edges[::2]).t()  # half of all non-edges
    for seed in range(5):
        new = randomize_edges(graph, n, forbidden=forbidden.flip(0), seed=seed)
        assert not edge_set(new) & edge_set(forbidden)
        assert edge_set(new) != edge_set(graph)


def six_cycle_counts(swaps_per_edge: int, draws: int) -> Counter:
    start = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    start = torch.cat([start, start.flip(0)], dim=1)
    return Counter(
        edge_set(randomize_edges(start, 6, swaps_per_edge=swaps_per_edge, seed=s))
        for s in range(draws)
    )


def test_randomizing_is_uniform_over_graphs_with_the_same_degrees():
    # Every graph on six nodes in which each node has degree 2: 60 six-cycles
    # and 10 pairs of triangles.
    pairs = list(itertools.combinations(range(6), 2))
    space = {
        frozenset(edges)
        for edges in itertools.combinations(pairs, 6)
        if all(sum(v in e for e in edges) == 2 for v in range(6))
    }
    assert len(space) == 70

    draws = 7000
    counts = six_cycle_counts(swaps_per_edge=30, draws=draws)
    assert set(counts) == space
    expected = draws / len(space)
    chi2 = sum((counts[g] - expected) ** 2 / expected for g in space)
    assert chi2 < 111.1  # the 99.9th percentile of chi-square with 69 degrees of freedom

    # The same statistic rejects a chain that has barely moved, so the test can fail.
    early = six_cycle_counts(swaps_per_edge=1, draws=draws)
    assert sum((early[g] - expected) ** 2 / expected for g in space) > 111.1


def test_a_directed_graph_is_refused():
    with pytest.raises(ValueError, match="directed"):
        randomize_edges(torch.tensor([[0, 1, 2], [1, 2, 0]]), 3)


def test_hidden_targets_leave_message_passing_in_both_directions():
    graph = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    left = drop_pairs(graph, torch.tensor([[2], [1]]), 4)
    assert edge_set(left) == {(0, 1), (2, 3)} and left.size(1) == 4

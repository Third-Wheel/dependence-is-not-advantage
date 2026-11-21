"""Degree-preserving randomization of an undirected graph given as an ``edge_index``."""

from __future__ import annotations

import random

import torch
from torch import Tensor


def randomize_edges(
    edge_index: Tensor,
    num_nodes: int,
    *,
    forbidden: Tensor | None = None,
    swaps_per_edge: int = 10,
    seed: int = 0,
) -> Tensor:
    """Randomize an undirected simple graph so that every node keeps its degree.

    One double-edge swap is attempted at a time: two edges (a, b) and (c, d)
    are drawn uniformly, and (a, d), (c, b) or (a, c), (b, d) is proposed with
    equal probability. A proposal that would create a self-loop, a repeated
    edge or a ``forbidden`` pair is rejected, and the graph stays as it is for
    that step. Rejected steps count, which is what makes the chain's
    stationary distribution uniform over the simple graphs it can reach with
    these degrees (Fosdick et al., SIAM Review 2018). Applying many swaps at
    once from one pairing of the edges does not have this property.

    ``edge_index`` holds both directions of every edge, as PyTorch Geometric
    stores undirected graphs, and so does the result. ``forbidden`` is a
    [2, F] tensor of node pairs that must not become edges, for example the
    held-out pairs of a link-prediction split.

    To train your own model on a randomized graph::

        data.edge_index = randomize_edges(data.edge_index, data.num_nodes, seed=seed)
    """
    pairs, _ = undirected_pairs(edge_index, num_nodes)
    edges = pairs.t().tolist()
    m = len(edges)
    if m < 2:
        raise ValueError("the graph needs at least two edges to randomize")

    def key(a: int, b: int) -> int:
        return a * num_nodes + b if a < b else b * num_nodes + a

    present = {key(a, b) for a, b in edges}
    banned = set() if forbidden is None else {key(a, b) for a, b in forbidden.t().tolist()}
    rng = random.Random(seed)
    for _ in range(swaps_per_edge * m):
        i, j = rng.randrange(m), rng.randrange(m)
        if i == j:
            continue
        (a, b), (c, d) = edges[i], edges[j]
        if rng.random() < 0.5:
            c, d = d, c
        if a == d or c == b:
            continue
        new1, new2 = key(a, d), key(c, b)
        if new1 in present or new2 in present or new1 in banned or new2 in banned:
            continue
        present.difference_update((key(a, b), key(c, d)))
        present.update((new1, new2))
        edges[i], edges[j] = [a, d], [c, b]
    return both_directions(torch.tensor(edges, dtype=torch.long).t())


def undirected_pairs(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, int]:
    """One column (u < v) per edge of an undirected graph, and how many self-loops were dropped."""
    u, v = edge_index.cpu()
    loops = int((u == v).sum())
    keys = torch.unique((u * num_nodes + v)[u != v])
    reverse = (keys % num_nodes) * num_nodes + keys // num_nodes
    if not torch.isin(reverse, keys).all():
        raise ValueError(
            "the graph is directed: some edge (u, v) has no (v, u). Degree-preserving "
            "double-edge swaps are defined here for undirected graphs; pass "
            "torch_geometric.utils.to_undirected(edge_index) if direction can be ignored."
        )
    keys = keys[keys // num_nodes < keys % num_nodes]
    return torch.stack([keys // num_nodes, keys % num_nodes]), loops


def both_directions(pairs: Tensor) -> Tensor:
    """An ``edge_index`` holding each pair in both directions."""
    return torch.cat([pairs, pairs.flip(0)], dim=1)


def drop_pairs(edge_index: Tensor, pairs: Tensor, num_nodes: int) -> Tensor:
    """``edge_index`` without the given node pairs, in either direction."""
    keys = edge_index[0] * num_nodes + edge_index[1]
    gone = torch.cat([pairs[0] * num_nodes + pairs[1], pairs[1] * num_nodes + pairs[0]])
    return edge_index[:, ~torch.isin(keys, gone)]

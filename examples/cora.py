"""Small public-data demonstration on the Cora citation network.

This example is intentionally not the paper's Cora protocol. See README.md.
"""

from __future__ import annotations

import io
import json
import random
import tarfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import networkx as nx

from graph_dependence import DegreePreservingNull, binary_auc, diagnose

DATA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
Pair = tuple[str, str]


def _archive(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "cora.tgz"
    if not path.exists():
        print(f"Downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def load_cora(cache_dir: Path) -> tuple[nx.Graph[str], dict[str, frozenset[int]]]:
    graph: nx.Graph[str] = nx.Graph()
    features: dict[str, frozenset[int]] = {}
    with tarfile.open(_archive(cache_dir), "r:gz") as archive:
        members = {Path(member.name).name: member for member in archive.getmembers()}
        content_file = archive.extractfile(members["cora.content"])
        cites_file = archive.extractfile(members["cora.cites"])
        if content_file is None or cites_file is None:
            raise RuntimeError("Cora archive is missing an expected data file")
        with io.TextIOWrapper(content_file, encoding="utf-8") as rows:
            for row in rows:
                fields = row.split()
                node = fields[0]
                features[node] = frozenset(
                    index for index, value in enumerate(fields[1:-1]) if value == "1"
                )
                graph.add_node(node)
        with io.TextIOWrapper(cites_file, encoding="utf-8") as rows:
            for row in rows:
                source, target = row.split()
                graph.add_edge(source, target)
    return graph, features


def make_split(
    graph: nx.Graph[str], seed: int = 7
) -> tuple[nx.Graph[str], list[Pair], list[int]]:
    rng = random.Random(seed)
    training = graph.copy()
    positive_edges = list(graph.edges())
    rng.shuffle(positive_edges)
    positives = positive_edges[: max(1, len(positive_edges) // 10)]
    training.remove_edges_from(positives)

    nodes = list(graph.nodes())
    negatives: set[Pair] = set()
    while len(negatives) < len(positives):
        left, right = rng.sample(nodes, 2)
        if not graph.has_edge(left, right):
            negatives.add((left, right) if left < right else (right, left))
    negative_pairs = sorted(negatives)
    examples = positives + negative_pairs
    targets = [1] * len(positives) + [0] * len(negative_pairs)
    order = list(range(len(examples)))
    rng.shuffle(order)
    return (
        training,
        [examples[index] for index in order],
        [targets[index] for index in order],
    )


def main() -> None:
    graph, features = load_cora(Path(__file__).parent / "data")
    training, examples, targets = make_split(graph)

    def common_neighbors(
        condition: nx.Graph[str], pairs: Sequence[Pair]
    ) -> list[float]:
        return [
            float(len(set(condition[left]) & set(condition[right])))
            for left, right in pairs
        ]

    def content_jaccard(pairs: Sequence[Pair]) -> list[float]:
        scores: list[float] = []
        for left, right in pairs:
            union = features[left] | features[right]
            scores.append(
                len(features[left] & features[right]) / len(union) if union else 0.0
            )
        return scores

    report = diagnose(
        graph=training,
        examples=examples,
        targets=targets,
        graph_scorer=common_neighbors,
        graph_free_scorer=content_jaccard,
        null_model=DegreePreservingNull[str](
            attempts_per_edge=10,
            forbidden_edges=tuple(examples),
        ),
        metric=binary_auc,
        null_draws=5,
        seed=7,
    )
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()

"""Tests for gain_penalty: paired seeds, the no-edge run, link-prediction safeguards."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GCN, GraphSAGE

from graph_dependence import gain_penalty
from graph_dependence import train as train_module
from graph_dependence.randomize import undirected_pairs
from graph_dependence.train import _link_split, _rank_metric


def edge_set(edge_index: torch.Tensor) -> frozenset[tuple[int, int]]:
    return frozenset((min(a, b), max(a, b)) for a, b in edge_index.t().tolist())


def random_graph(n: int, m: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    pairs = torch.combinations(torch.arange(n))
    chosen = pairs[torch.randperm(pairs.size(0), generator=g)[:m]].t()
    return torch.cat([chosen, chosen.flip(0)], dim=1)


def test_ties_count_against_the_positive():
    constant = torch.zeros(5)
    assert _rank_metric(constant, torch.zeros(100), "hits@20") == 0.0
    assert _rank_metric(constant, torch.zeros(5, 50), "hits@20") == 0.0


def test_hits_and_mrr_on_a_small_example():
    neg = torch.tensor([0.9, 0.5, 0.1])
    pos = torch.tensor([1.0, 0.7, 0.3, 0.0])  # ranks 1, 2, 3, 4
    assert _rank_metric(pos, neg, "hits@2") == pytest.approx(0.5)
    assert _rank_metric(pos, neg, "mrr") == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)
    per_positive = neg.repeat(4, 1)
    assert _rank_metric(pos, per_positive, "hits@2") == pytest.approx(0.5)


def small_gcn(in_channels: int, out_channels: int) -> torch.nn.Module:
    return GCN(in_channels, 16, num_layers=2, out_channels=out_channels)


def planted_partition(seed: int = 0, n: int = 240) -> Data:
    """Two communities with dense links inside them and features that say little."""
    g = torch.Generator().manual_seed(seed)
    y = (torch.arange(n) >= n // 2).long()
    pairs = torch.combinations(torch.arange(n)).t()
    same = y[pairs[0]] == y[pairs[1]]
    keep = torch.rand(pairs.size(1), generator=g) < torch.where(same, 0.08, 0.004)
    pairs = pairs[:, keep]
    x = torch.randn(n, 16, generator=g)
    x[:, 0] += 0.5 * (2 * y - 1)
    order = torch.randperm(n, generator=g)
    masks = [torch.zeros(n, dtype=torch.bool) for _ in range(3)]
    for mask, part in zip(masks, (order[:60], order[60:120], order[120:]), strict=True):
        mask[part] = True
    return Data(
        x=x,
        y=y,
        edge_index=torch.cat([pairs, pairs.flip(0)], dim=1),
        train_mask=masks[0],
        val_mask=masks[1],
        test_mask=masks[2],
    )


class Recorder:
    """A model constructor that remembers each model's first weights and the graphs it was given."""

    def __init__(self, kind=GCN):
        self.kind = kind
        self.starts: list[torch.Tensor] = []
        self.edge_counts: list[set[int]] = []

    def __call__(self, in_channels: int, out_channels: int) -> torch.nn.Module:
        model = self.kind(in_channels, 16, num_layers=2, out_channels=out_channels)
        self.starts.append(torch.cat([p.detach().flatten().clone() for p in model.parameters()]))
        seen: set[int] = set()
        self.edge_counts.append(seen)
        model.register_forward_pre_hook(lambda _, args: seen.add(args[1].size(1)))
        return model


def test_node_classification_end_to_end():
    data = planted_partition()
    make = Recorder()
    report = gain_penalty(
        make,
        data,
        "node",
        seeds=3,
        lrs=(0.01, 0.003),
        max_epochs=60,
        patience=20,
        verbose=False,
    )
    real, none, rand = (report.scores[g] for g in ("real", "no_edges", "randomized"))
    assert len(real) == len(none) == len(rand) == 3

    # The identity holds seed by seed, and so for the means.
    assert report.drop.mean == pytest.approx(report.gain.mean + report.penalty.mean)
    # The graph carries the labels here and the features barely do.
    assert report.gain.mean > 10

    # Runs come in the order seed, graph (real, no edges, randomized), learning rate.
    starts = [make.starts[i : i + 6] for i in range(0, 18, 6)]
    for per_seed in starts:
        assert all(torch.equal(per_seed[0], other) for other in per_seed[1:])
    assert not torch.equal(starts[0][0], starts[1][0])

    # The no-edge runs use the same model with an empty edge list.
    m = data.edge_index.size(1)
    for i, seen in enumerate(make.edge_counts):
        assert seen == ({0} if (i // 2) % 3 == 1 else {m})

    text = str(report)
    assert "gain" in text and "penalty" in text and "drop" in text
    assert "test-time" not in text and report.test_time_drop is None


def test_training_stops_when_validation_has_stopped_improving():
    make = small_gcn
    report = gain_penalty(
        make,
        planted_partition(),
        "node",
        seeds=2,
        lrs=(0.01,),
        max_epochs=2000,
        patience=15,
        verbose=False,
    )
    for graph in ("real", "no_edges", "randomized"):
        for best, trained in zip(
            report.best_epoch[graph], report.epochs_trained[graph], strict=True
        ):
            assert trained == best + 15 < 2000
    assert not any("without stopping early" in note for note in report.notes)


def test_link_prediction_end_to_end():
    data = planted_partition(seed=1)
    n = data.num_nodes
    make = Recorder(GraphSAGE)
    report = gain_penalty(
        make,
        data,
        "link",
        seeds=2,
        lrs=(0.01,),
        max_epochs=30,
        patience=30,
        test_time_drop=True,
        embedding_dim=16,
        verbose=False,
    )
    assert report.metric.startswith("hits@20")
    assert report.drop.mean == pytest.approx(report.gain.mean + report.penalty.mean)
    assert len(report.scores["real_on_randomized"]) == 2 and report.test_time_drop is not None
    assert "test-time drop" in str(report)
    assert report.epochs_trained["real"] == [30, 30]
    assert any(
        "without stopping early" in note for note in report.notes
    )  # 30 epochs is not convergence

    # Message passing reads the training links only, with each epoch's targets hidden
    # (30% of them) while training and all of them present when scoring.
    pairs, _ = undirected_pairs(data.edge_index, n)
    split = _link_split(pairs, n)
    train = split["train"].size(1)
    hidden = round(0.3 * train)
    real_seen, reloaded_seen, none_seen, rand_seen = make.edge_counts[:4]
    assert real_seen == {2 * train, 2 * (train - hidden)}
    assert none_seen == {0}
    assert max(rand_seen) == 2 * train and min(rand_seen) >= 2 * (train - hidden)
    # The test-time drop reloads the real-graph model and shows it the whole randomized graph.
    assert reloaded_seen == {2 * train}


def test_link_prediction_randomizes_only_what_the_model_reads(monkeypatch):
    data = planted_partition(seed=2)
    n = data.num_nodes
    pairs, _ = undirected_pairs(data.edge_index, n)
    split = _link_split(pairs, n)
    drawn = []
    original = train_module.randomize_edges

    def spy(edge_index, num_nodes, **kwargs):
        drawn.append((edge_index, kwargs["forbidden"], original(edge_index, num_nodes, **kwargs)))
        return drawn[-1][2]

    monkeypatch.setattr(train_module, "randomize_edges", spy)
    make = small_gcn
    gain_penalty(
        make,
        data,
        "link",
        seeds=2,
        lrs=(0.01,),
        max_epochs=3,
        patience=3,
        verbose=False,
    )

    held_out = set()
    for key in ("valid", "test", "valid_neg", "test_neg"):
        held_out |= edge_set(split[key])
    for start, forbidden, new in drawn:
        assert edge_set(start) == edge_set(split["train"])  # drawn from the training graph
        assert edge_set(forbidden) == held_out
        assert not edge_set(new) & held_out  # no held-out pair becomes an edge
    assert edge_set(drawn[0][2]) != edge_set(drawn[1][2])  # each seed has its own draw


def test_splits_can_be_passed_with_negatives_per_positive():
    data = planted_partition(seed=3)
    n = data.num_nodes
    pairs, _ = undirected_pairs(data.edge_index, n)
    split = _link_split(pairs, n)
    g = torch.Generator().manual_seed(0)
    for part in ("valid", "test"):
        split[f"{part}_neg"] = torch.randint(n, (split[part].size(1), 25, 2), generator=g)
    make = small_gcn
    report = gain_penalty(
        make,
        data,
        "link",
        splits=split,
        metric="mrr",
        seeds=2,
        lrs=(0.01,),
        max_epochs=5,
        verbose=False,
    )
    assert report.metric.startswith("mrr") and not any(
        "built-in split" in note for note in report.notes
    )


class OneGraph(InMemoryDataset):
    def __init__(self, graphs: list[Data]) -> None:
        super().__init__(None)
        self.data, self.slices = self.collate(graphs)


def test_a_dataset_holding_one_graph_is_accepted():
    data = planted_partition()
    make = small_gcn
    report = gain_penalty(
        make,
        OneGraph([data]),
        "node",
        seeds=2,
        lrs=(0.01,),
        max_epochs=3,
        verbose=False,
    )
    assert len(report.scores["real"]) == 2
    with pytest.raises(ValueError, match="several graphs"):
        gain_penalty(make, OneGraph([data, data]), "node", seeds=2)


def test_bad_inputs_are_refused():
    data = planted_partition()
    make = small_gcn
    with pytest.raises(ValueError, match="two seeds"):
        gain_penalty(make, data, "node", seeds=1)
    with pytest.raises(ValueError, match="'node' or 'link'"):
        gain_penalty(make, data, "graph")
    looped = data.clone()
    looped.edge_index = torch.cat([data.edge_index, torch.tensor([[0, 5], [0, 5]])], dim=1)
    report = gain_penalty(make, looped, "node", seeds=2, lrs=(0.01,), max_epochs=2, verbose=False)
    assert "2 self-loops were removed" in " ".join(str(report).split())
    directed = data.clone()
    directed.edge_index = data.edge_index[:, : data.edge_index.size(1) // 2]
    with pytest.raises(ValueError, match="directed"):
        gain_penalty(make, directed, "node", seeds=2)

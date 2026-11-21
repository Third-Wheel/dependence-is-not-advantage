"""Train one model on the real graph, a randomized graph and no edges, and split its drop."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .randomize import both_directions, drop_pairs, randomize_edges, undirected_pairs
from .report import ARMS, LABELS, Report, seed_note

RUNS = {"real": "real-graph", "no_edges": "no-edge", "randomized": "randomized-graph"}

# The built-in link-prediction split, used when no ``splits`` are passed.
VALID_SHARE = 0.05
TEST_SHARE = 0.10
EVAL_NEGATIVES = 1000  # shared non-links that every validation (or test) link is ranked against
SPLIT_SEED = 0


def _random_non_edges(
    pairs: Tensor, num_nodes: int, count: int, generator: torch.Generator
) -> Tensor:
    """``count`` distinct node pairs that are not edges, drawn uniformly."""
    taken = set((pairs[0] * num_nodes + pairs[1]).tolist())
    if count > num_nodes * (num_nodes - 1) // 2 - len(taken):
        raise ValueError("the graph has too few non-edges to draw evaluation negatives from")
    found: list[int] = []
    while len(found) < count:
        u, v = torch.randint(num_nodes, (2, 2 * count), generator=generator)
        lo, hi = torch.minimum(u, v), torch.maximum(u, v)
        for k in (lo * num_nodes + hi)[lo != hi].tolist():
            if k not in taken and len(found) < count:
                taken.add(k)
                found.append(k)
    keys = torch.tensor(found)
    return torch.stack([keys // num_nodes, keys % num_nodes])


def _link_split(pairs: Tensor, num_nodes: int) -> dict[str, Tensor]:
    """A fixed random split of the edges, with shared random non-links to rank against."""
    g = torch.Generator().manual_seed(SPLIT_SEED)
    order = torch.randperm(pairs.size(1), generator=g)
    n_valid = max(1, round(VALID_SHARE * pairs.size(1)))
    n_test = max(1, round(TEST_SHARE * pairs.size(1)))
    negatives = _random_non_edges(pairs, num_nodes, 2 * EVAL_NEGATIVES, g)
    return {
        "valid": pairs[:, order[:n_valid]],
        "test": pairs[:, order[n_valid : n_valid + n_test]],
        "train": pairs[:, order[n_valid + n_test :]],
        "valid_neg": negatives[:, :EVAL_NEGATIVES],
        "test_neg": negatives[:, EVAL_NEGATIVES:],
    }


# ----------------------------------------------------------------- metrics


def _accuracy(logits: Tensor, y: Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean())


def _rank_metric(pos: Tensor, neg: Tensor, metric: str) -> float:
    """Hits@K or MRR of positive scores against negative scores.

    ``neg`` is one shared set of scores, [M], or one row per positive, [E, K].
    A negative that ties with a positive is ranked above it, so a model that
    gives every pair the same score gets no credit.
    """
    if neg.dim() == 1:
        ordered = torch.sort(neg).values
        rank = 1 + neg.numel() - torch.searchsorted(ordered, pos, right=False)
    else:
        rank = 1 + (neg >= pos.unsqueeze(1)).sum(dim=1)
    if metric == "mrr":
        return float((1.0 / rank).mean())
    return float((rank <= int(metric.removeprefix("hits@"))).float().mean())


# ------------------------------------------------------------------- tasks


@dataclass
class _Task:
    """What differs between node classification and link prediction."""

    name: str
    metric: str
    graph: Tensor  # the real message-passing graph, both directions
    forbidden: Tensor | None  # pairs the randomized graph must not contain
    build: Callable[[], torch.nn.Module]
    loss: Callable[[Any, Tensor, torch.Generator], Tensor]
    score: Callable[[Any, Tensor, str], float]
    clip: bool
    notes: list[str]


def _node_task(
    make_model: Callable, data: Any, metric: str | Callable | None, device: torch.device
) -> _Task:
    for name in ("x", "y", "train_mask", "val_mask", "test_mask"):
        if getattr(data, name, None) is None:
            raise ValueError(f"node classification needs data.{name}")
    masks = {"train": data.train_mask, "valid": data.val_mask, "test": data.test_mask}
    if any(m.dim() != 1 for m in masks.values()):
        raise ValueError(
            "the masks hold several splits; pick one, e.g. data.train_mask = data.train_mask[:, 0]"
        )
    pairs, loops = undirected_pairs(data.edge_index, data.num_nodes)
    x, y = data.x.to(device), data.y.to(device)
    masks = {k: m.to(device) for k, m in masks.items()}
    measure = metric if callable(metric) else _accuracy
    if not (metric is None or callable(metric) or metric == "accuracy"):
        raise ValueError(
            "node classification: metric is 'accuracy' or a function (logits, labels) -> float"
        )
    classes = int(y.max()) + 1

    def loss(model: torch.nn.Module, graph: Tensor, _: torch.Generator) -> Tensor:
        return F.cross_entropy(model(x, graph)[masks["train"]], y[masks["train"]])

    def score(model: torch.nn.Module, graph: Tensor, split: str) -> float:
        return float(measure(model(x, graph)[masks[split]], y[masks[split]]))

    return _Task(
        name="node classification",
        metric=getattr(metric, "__name__", "accuracy"),
        graph=both_directions(pairs),
        forbidden=None,
        build=lambda: make_model(x.size(1), classes),
        loss=loss,
        score=score,
        clip=False,
        notes=[f"{loops} self-loops were removed from all three graphs."] if loops else [],
    )


class _LinkModel(torch.nn.Module):
    """The user's encoder, and an MLP scoring a node pair from the product of its embeddings."""

    def __init__(self, encoder: torch.nn.Module, dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, 1),
        )

    def pair_scores(self, z: Tensor, pairs: Tensor) -> Tensor:
        """Scores for pairs given as [2, M], or as [E, K, 2] with K pairs per row."""
        if pairs.dim() == 3:
            return self.head(z[pairs[..., 0]] * z[pairs[..., 1]]).squeeze(-1)
        return self.head(z[pairs[0]] * z[pairs[1]]).squeeze(-1)


def _link_task(
    make_model: Callable,
    data: Any,
    metric: str | None,
    splits: dict[str, Tensor] | None,
    hide_targets: float,
    embedding_dim: int,
    device: torch.device,
) -> _Task:
    if getattr(data, "x", None) is None:
        raise ValueError("link prediction needs node features in data.x")
    metric = metric or "hits@20"
    if not (metric == "mrr" or (metric.startswith("hits@") and metric[5:].isdigit())):
        raise ValueError("link prediction: metric is 'hits@K' or 'mrr'")
    if not 0 <= hide_targets < 1:
        raise ValueError("hide_targets is the share of training links hidden each epoch, in [0, 1)")
    n = data.num_nodes
    notes = []
    loops = 0
    if splits is None:
        pairs, loops = undirected_pairs(data.edge_index, n)
        splits = _link_split(pairs, n)
        notes.append(
            f"built-in split: {VALID_SHARE:.0%} of links for validation and {TEST_SHARE:.0%}"
            f" for test, each ranked against {EVAL_NEGATIVES} shared random non-links."
            " Pass splits= to use a benchmark's own."
        )
    missing = {"train", "valid", "test", "valid_neg", "test_neg"} - set(splits)
    if missing:
        raise ValueError(f"splits is missing {sorted(missing)}")
    loops += int((splits["train"][0] == splits["train"][1]).sum())
    if loops:
        notes.append(f"{loops} self-loops were removed from all three graphs.")
    train, _ = undirected_pairs(both_directions(splits["train"]), n)
    held_out = [splits[k] for k in ("valid", "test")]
    # Negatives are [2, M] when shared, or [E, K, 2] with K per positive.
    for k in ("valid_neg", "test_neg"):
        shared = splits[k].dim() == 2
        held_out.append(splits[k] if shared else splits[k].reshape(-1, 2).t())
    x = data.x.to(device)
    train_dev = train.to(device)
    held = {k: v.to(device) for k, v in splits.items() if k != "train"}
    n_hidden = max(1, round(hide_targets * train.size(1))) if hide_targets else 0

    def build() -> torch.nn.Module:
        return _LinkModel(make_model(x.size(1), embedding_dim), embedding_dim)

    def loss(model: _LinkModel, graph: Tensor, generator: torch.Generator) -> Tensor:
        # The generator belongs to the seed, so all three graphs see the same pairs.
        if n_hidden:
            order = torch.randperm(train.size(1), generator=generator)
            pos = train_dev[:, order[:n_hidden].to(device)]
            graph = drop_pairs(graph, pos, n)
        else:
            pos = train_dev
        neg = torch.randint(n, (2, pos.size(1)), generator=generator).to(device)
        z = model.encoder(x, graph)
        out = torch.cat([model.pair_scores(z, pos), model.pair_scores(z, neg)])
        target = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))])
        return F.binary_cross_entropy_with_logits(out, target.to(device))

    def score(model: _LinkModel, graph: Tensor, split: str) -> float:
        z = model.encoder(x, graph)
        return _rank_metric(
            model.pair_scores(z, held[split]),
            model.pair_scores(z, held[f"{split}_neg"]),
            metric,
        )

    return _Task(
        name="link prediction",
        metric=metric,
        graph=both_directions(train),
        forbidden=torch.cat([p.cpu() for p in held_out], dim=1),
        build=build,
        loss=loss,
        score=score,
        clip=True,
        notes=notes,
    )


# ---------------------------------------------------------------- training


class _Run(NamedTuple):
    valid: float
    test: float
    epoch: int  # the epoch with the best validation score
    trained: int  # how many epochs the run lasted
    converged: bool
    state: dict[str, Tensor] | None


def _fit(
    task: _Task,
    graph: Tensor,
    seed: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    keep_state: bool,
    device: torch.device,
) -> _Run:
    """Train one model on one graph; the test score is read once, at the best validation epoch."""
    torch.manual_seed(seed)  # the same initialization on all three graphs
    model = task.build().to(device)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = _Run(-math.inf, math.nan, 0, 0, False, None)
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = task.loss(model, graph, generator)
        if not torch.isfinite(loss):
            if best.epoch == 0:
                raise RuntimeError(
                    f"the loss is not finite at the first step (learning rate {lr:g})"
                )
            # diverged: an earlier epoch, or another learning rate, is kept
            return best._replace(trained=epoch, converged=True)
        loss.backward()
        if task.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            valid = task.score(model, graph, "valid")
            if valid > best.valid:
                state = copy.deepcopy(model.state_dict()) if keep_state else None
                test = task.score(model, graph, "test")
                best = _Run(valid, test, epoch, epoch, False, state)
        if epoch - best.epoch >= patience:
            return best._replace(trained=epoch, converged=True)
    return best._replace(trained=max_epochs)


def gain_penalty(
    make_model: Callable[[int, int], torch.nn.Module],
    data: Any,
    task: str,
    *,
    seeds: int | Iterable[int] = 5,
    lrs: Iterable[float] = (0.01, 0.001),
    max_epochs: int = 10_000,
    patience: int = 1000,
    metric: str | Callable[[Tensor, Tensor], float] | None = None,
    test_time_drop: bool = False,
    splits: dict[str, Tensor] | None = None,
    hide_targets: float = 0.3,
    embedding_dim: int = 64,
    swaps_per_edge: int = 10,
    weight_decay: float = 0.0,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> Report:
    """Train one model on the real graph, a randomized graph and no edges, and split the drop.

    make_model
        ``make_model(in_channels, out_channels)`` returns a fresh module whose
        ``forward(x, edge_index)`` gives one row per node, for example
        ``lambda i, o: torch_geometric.nn.GCN(i, 64, num_layers=2, out_channels=o)``.
        For node classification the rows are class logits. For link prediction
        they are embeddings of size ``embedding_dim``, and a pair is scored by
        an MLP on the elementwise product of its two embeddings.
    data
        A PyTorch Geometric ``Data`` object, or a dataset holding one. The graph
        must be undirected. Node classification needs ``x``, ``y`` and
        one-dimensional ``train_mask``, ``val_mask`` and ``test_mask``.
    task
        ``"node"`` or ``"link"``.
    seeds
        How many seeds, or the seeds themselves. A seed fixes the model's
        initialization on all three graphs and has its own randomized graph.
    lrs
        Learning rates tried for every run; each run keeps the one with the
        best validation score, so the three graphs get the same tuning effort.
    max_epochs, patience
        Training stops when validation has not improved for ``patience``
        epochs. The report names any run still improving at ``max_epochs``.
        Stopping early is not proof of convergence: a randomized-graph run
        can sit on a plateau and improve much later, so if its best epoch is
        small, double ``patience`` and see whether the penalty moves.
    metric
        Node classification: ``"accuracy"`` (default) or a function
        ``(logits, labels) -> float`` in [0, 1], higher better. Link
        prediction: ``"hits@K"`` (default ``"hits@20"``) or ``"mrr"``.
    test_time_drop
        Also evaluate each real-graph model on the randomized graph without
        retraining, and report that fall separately.
    splits
        Link prediction only. A dict of node-pair tensors: ``train``, ``valid``
        and ``test`` of shape [2, E], and ``valid_neg`` and ``test_neg`` of
        shape [2, M] (shared negatives) or [E, K, 2] (K per positive). Without
        it the edges of ``data`` are split at random, the same way every time.
    hide_targets
        Link prediction only. The share of training links drawn as each
        epoch's positives and removed from message passing for that epoch.
        0 trains on all links while the model reads them.
    swaps_per_edge
        Attempted double-edge swaps per edge when randomizing the graph.
    """
    if task not in ("node", "link"):
        raise ValueError("task is 'node' or 'link'")
    seeds = list(range(seeds)) if isinstance(seeds, int) else [int(s) for s in seeds]
    if len(seeds) < 2:
        raise ValueError("at least two seeds are needed for a paired interval")
    lrs = tuple(float(lr) for lr in lrs)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if isinstance(data, torch.utils.data.Dataset):  # take the dataset's one graph
        graphs: Any = data
        if len(graphs) != 1:
            raise ValueError("the dataset holds several graphs; pass the one graph to train on")
        data = graphs[0]
    if task == "node":
        spec = _node_task(make_model, data, metric, device)
    else:
        if callable(metric):
            raise ValueError("link prediction: metric is 'hits@K' or 'mrr'")
        spec = _link_task(make_model, data, metric, splits, hide_targets, embedding_dim, device)

    keys = [*ARMS, "real_on_randomized"] if test_time_drop else list(ARMS)
    scores: dict[str, list[float]] = {k: [] for k in keys}
    best_epoch: dict[str, list[int]] = {g: [] for g in ARMS}
    epochs_trained: dict[str, list[int]] = {g: [] for g in ARMS}
    chosen_lr: dict[str, list[float]] = {g: [] for g in ARMS}
    unfinished = dict.fromkeys(ARMS, 0)
    for seed in seeds:
        graphs = {
            "real": spec.graph,
            "no_edges": spec.graph[:, :0],
            "randomized": randomize_edges(
                spec.graph,
                data.num_nodes,
                forbidden=spec.forbidden,
                swaps_per_edge=swaps_per_edge,
                seed=seed,
            ),
        }
        graphs = {g: e.to(device) for g, e in graphs.items()}
        for g in ARMS:
            keep = test_time_drop and g == "real"
            fits = [
                _fit(
                    spec,
                    graphs[g],
                    seed,
                    lr,
                    weight_decay,
                    max_epochs,
                    patience,
                    keep,
                    device,
                )
                for lr in lrs
            ]
            pick = max(range(len(lrs)), key=lambda i: fits[i].valid)
            run = fits[pick]
            scores[g].append(100 * run.test)
            best_epoch[g].append(run.epoch)
            epochs_trained[g].append(run.trained)
            chosen_lr[g].append(lrs[pick])
            unfinished[g] += not run.converged
            if verbose:
                print(
                    f"seed {seed}  {LABELS[g]:<11} lr {lrs[pick]:<7g} best epoch {run.epoch:>5}  "
                    f"validation {100 * run.valid:5.1f}  test {100 * run.test:5.1f}",
                    flush=True,
                )
            if keep and run.state is not None:
                model = spec.build().to(device)
                model.load_state_dict(run.state)
                model.eval()
                with torch.no_grad():
                    scores["real_on_randomized"].append(
                        100 * spec.score(model, graphs["randomized"], "test")
                    )
    if verbose:
        print(flush=True)
    notes = list(spec.notes)
    if len(seeds) < 5:
        notes.append(seed_note(len(seeds)))
    for g, count in unfinished.items():
        if count:
            notes.append(
                f"{count} of {len(seeds)} {RUNS[g]} runs reached max_epochs={max_epochs} without "
                "stopping early. Raise max_epochs: the drop depends on how long these runs train."
            )
    return Report(
        scores=scores,
        metric=f"{spec.metric} on a 0-100 scale",
        task=spec.name,
        seeds=seeds,
        lrs=lrs,
        best_epoch=best_epoch,
        epochs_trained=epochs_trained,
        chosen_lr=chosen_lr,
        notes=notes,
    )

# Graph Dependence

`graph_dependence` reports two comparisons that graph-ablation studies often
conflate. **Dependence** compares one graph procedure on the observed graph and
under a stated graph null. **Advantage** compares that procedure with a named
graph-free scorer. A large ablation drop does not, by itself, establish an
advantage over information available without the graph.

```python
from graph_dependence import DegreePreservingNull, binary_auc, diagnose

report = diagnose(
    graph=G,
    examples=pairs,
    targets=labels,
    graph_scorer=score_with_graph,
    graph_free_scorer=score_without_graph,
    null_model=DegreePreservingNull(attempts_per_edge=10),
    metric=binary_auc,
    null_draws=10,
    seed=7,
)
print(report.to_dict())
```

The call evaluates three arms and keeps their contrasts separate:

| Quantity | Definition | Question answered |
| --- | --- | --- |
| `dependence` | `observed_score - null_score` | How much does this procedure rely on the observed edge arrangement under this null? |
| `advantage` | `observed_score - comparator_score` | Does this procedure outperform this named graph-free scorer? |
| `null_survival` | `null_score / observed_score` | What fraction of the observed score survives randomization? |

`advantage` is `None` when no graph-free scorer is supplied. `null_survival` is
`None` when the observed score is not positive, matching the ratio's domain in
the paper. The intact, null, and comparator scores are always reported beside
the contrasts.

## Install

The package requires Python 3.10 or newer and NetworkX:

```bash
python -m pip install .
```

For local development:

```bash
python -m pip install -e '.[dev]'
pytest
mypy
ruff check .
```

## Scorers and metrics

A graph scorer receives a NetworkX graph and the evaluation examples. A
graph-free scorer receives the same examples but no graph. Both return one score
per example. A metric receives those scores and the targets and returns one
scalar, where larger values mean better performance.

```python
def score_with_graph(graph, examples):
    return [len(list(nx.common_neighbors(graph, u, v))) for u, v in examples]


def score_without_graph(examples):
    return [content_similarity[u, v] for u, v in examples]
```

The built-in `binary_auc` is deliberately small and has no numerical dependency.
Custom scorers and metrics are ordinary callables. If a scorer trains a model or
uses randomness, condition-specific fitting and seed control remain the caller's
responsibility; `seed` controls null generation only.

## Degree-preserving null

`DegreePreservingNull` applies a fixed budget of double-edge-swap attempts to an
undirected, simple, unweighted graph. It rejects self-loops, duplicate edges, and
new edges listed in `forbidden_edges`. Each draw reports accepted swaps, edge
retention, component counts, and explicit degree and edge-count checks.

Degree preservation names a constraint, not a unique probability distribution.
The implementation therefore records its sampler and budget and does not claim
uniform sampling over all graphs with the same degree sequence. Edge attributes
are rejected because assigning them to newly formed edges would silently define
another intervention.

## Public example

[`examples/cora.py`](examples/cora.py) downloads the public Cora citation graph,
constructs a deterministic link-prediction split, and compares common-neighbor
scores with a graph-free bag-of-words Jaccard scorer:

```bash
python examples/cora.py
```

This is a compact API example, not a reproduction of the paper's Cora row. The
paper uses the published HeaRT split, its hard negatives, and Hits@20; the example
uses a small deterministic split and AUC so it can run without a training
framework. Tests provide the v0.1 validation contract: graph-independent scorers
have zero dependence, structure-reading scorers degrade when the null removes
their signal, advantage uses the graph-free comparator, and fixed seeds reproduce
null draws.

## Scope

This repository is a reference implementation of the diagnostic, not a full
paper-reproduction system. It does not train GraphSAGE, recreate manuscript
tables, prescribe a universally valid null model, or turn predictive improvement
into evidence of downstream benefit. The API is intentionally small enough to
audit in one sitting.

See the paper, [*Dependence Is Not Advantage: What Randomizing the Graph Does Not
Measure*](https://www.alphaxiv.org/abs/2608.dependence-is-not-advantage), for the
formal definitions, empirical study, and claim boundaries. Citation guidance and
the software citation metadata are in [`CITATION.cff`](CITATION.cff).

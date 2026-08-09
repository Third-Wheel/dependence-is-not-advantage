# Graph Dependence

This repository contains the reference implementation for [_Dependence Is Not
Advantage: What Randomizing the Graph Does Not
Measure_](https://www.alphaxiv.org/abs/2608.dependence-is-not-advantage). The
paper is by Prahas Duggireddy and the package is maintained by Third Wheel.

The package reports two comparisons that are easy to mix up in graph-ablation
studies:

- **Dependence:** how a graph procedure changes when the observed graph is
  replaced by a stated graph null.
- **Advantage:** how that procedure compares with a named scorer that does not
  receive the graph.

Randomization answers the first question. It does not answer the second one by
itself. The implementation is intentionally small, typed, and close to these
definitions. Version 0.1.0 is an early reference implementation, not a general
graph-learning framework or a full reproduction of the paper.

Paper and project links:

- [Preprint](https://www.alphaxiv.org/abs/2608.dependence-is-not-advantage)
- [Project page](https://third-wheel.github.io/dependence-is-not-advantage/)
- [Source on GitHub](https://github.com/Third-Wheel/dependence-is-not-advantage)
- [Citation metadata](CITATION.cff)

## Quick start

The smallest complete example needs no network access. From a checkout:

```bash
python -m pip install .
python examples/minimal.py
```

It uses a fixed graph, fixed node-content features, a degree-preserving null,
and seed 7. The command prints a JSON-compatible report that is small enough
to read alongside the source.

For an evaluation of your own, the central call looks like this:

```python
from graph_dependence import DegreePreservingNull, binary_auc, diagnose

# Provide these objects from your evaluation protocol.
report = diagnose(
    graph=graph,
    examples=examples,
    targets=targets,
    graph_scorer=score_with_graph,
    graph_free_scorer=score_without_graph,
    null_model=DegreePreservingNull(attempts_per_edge=10),
    metric=binary_auc,
    null_draws=10,
    seed=7,
)
print(report.to_dict())
```

The call keeps the intact, null, and graph-free comparator scores together.
The resulting report can be serialized with `to_dict()` without a custom
encoder.

## What the report means

| Field           | Comparison                          | Interpretation                                                   |
| --------------- | ----------------------------------- | ---------------------------------------------------------------- |
| `dependence`    | `observed_score - null_score`       | Change under the declared graph intervention.                    |
| `advantage`     | `observed_score - comparator_score` | Difference from the named graph-free scorer.                     |
| `null_survival` | `null_score / observed_score`       | Fraction of the observed score that remains after randomization. |

`advantage` is `None` when no graph-free scorer is supplied. `null_survival` is
`None` when the observed score is not positive, because the ratio is undefined
there. The report also records the three reference scores, the seed, the null
model name, and sampler diagnostics.

## The public API

| Symbol                                                  | Use                                                                                      |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `diagnose`                                              | Evaluate the intact, null, and optional graph-free arms and return one report.           |
| `DegreePreservingNull`                                  | Generate fixed-budget double-edge-swap null graphs and record the checks performed.      |
| `DependenceReport`                                      | Store scores, contrasts, seed, and metadata; `to_dict()` returns JSON-compatible values. |
| `binary_auc`                                            | Provide a small dependency-free binary ranking metric for examples and tests.            |
| `GraphScorer`, `GraphFreeScorer`, `Metric`, `NullModel` | Define the typed callable boundaries supplied by the caller.                             |

The formal definitions, empirical results, and claim boundaries remain in the
paper. The package implements the diagnostic around them; it does not recreate
the paper's training runs or manuscript tables.

## Supplying scorers and metrics

A graph scorer receives a NetworkX graph and the evaluation examples. A
graph-free scorer receives the same examples without the graph. Both return one
score per example. A metric receives those scores and the targets and returns a
scalar, with larger values meaning better performance.

```python
def score_with_graph(graph, examples):
    return [len(list(nx.common_neighbors(graph, u, v))) for u, v in examples]


def score_without_graph(examples):
    return [content_similarity[u, v] for u, v in examples]
```

Custom scorers and metrics are ordinary callables. If a scorer trains a model
or uses randomness, condition-specific fitting and seed control are the
caller's responsibility. The `seed` argument controls null generation only.

## The degree-preserving null

`DegreePreservingNull` applies a fixed budget of double-edge-swap attempts to
an undirected, simple, unweighted graph. It rejects self-loops, duplicate
edges, and new edges listed in `forbidden_edges`. Each draw records accepted
swaps, edge retention, component counts, and explicit degree and edge-count
checks.

Degree preservation describes a constraint, not a unique probability
distribution. The implementation records its sampler and budget and does not
claim uniform sampling over all graphs with the same degree sequence. Edge
attributes are rejected because assigning them to newly formed edges would
silently define a different intervention.

## Public-data example

[`examples/cora.py`](examples/cora.py) downloads the public Cora citation graph,
constructs a deterministic link-prediction split, and compares
common-neighbor scores with a graph-free bag-of-words Jaccard scorer:

```bash
python examples/cora.py
```

This is an API example, not a reproduction of the paper's Cora row. The paper
uses the published HeaRT split, its hard negatives, and Hits@20. The example
uses a small deterministic split and AUC so that it can remain independent of
a training framework.

## Repository layout

```text
src/graph_dependence/  package implementation and py.typed marker
examples/minimal.py    deterministic, network-free example
examples/cora.py       optional public-data example
tests/                 package and scientific-boundary tests
site/                  paper page, searchable PDF, figure, and citations
```

The paper and visible landing page are authoritative for scholarly claims.
`pyproject.toml` is authoritative for package version and support metadata;
`CITATION.cff` is authoritative for software citation metadata.

## Scope

This repository does not train GraphSAGE, recreate manuscript tables, prescribe
a universally valid null model, or turn predictive improvement into evidence of
downstream recommendation benefit. It is a reference implementation for
separating dependence from advantage in an evaluation protocol.

For the formal definitions and empirical study, see the [paper](https://www.alphaxiv.org/abs/2608.dependence-is-not-advantage). For software reuse,
cite the version used and see [`CITATION.cff`](CITATION.cff).

## Development

The package supports Python 3.11 and newer. To run the local checks after an
editable install:

```bash
python -m pip install -e '.[dev]'
ruff check .
mypy --strict .
pytest --cov=graph_dependence --cov-branch
```

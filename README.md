# graph-dependence

Applied papers often test whether a graph neural network needs its graph by randomizing the
edges, keeping every node's degree, and reporting how far the score drops. The drop is the sum
of two things:

```
drop    = score(real graph) - score(randomized graph)  =  gain + penalty
gain    = score(real graph) - score(no edges)          what the real graph adds to the model
penalty = score(no edges)   - score(randomized graph)  what wrong edges cost it
```

Only the gain says whether the graph helps, and measuring it takes one more run: the same model,
layers unchanged, trained with no edges. This package runs all three and reports each part with a
95% interval.

The two parts can be very different sizes. Here are GAT and GraphSAGE on CiteSeer link
prediction (Hits@20, 400 epochs, mean of five seeds):

| model     | real graph | no edges | randomized | drop | gain |
|-----------|-----------:|---------:|-----------:|-----:|-----:|
| GAT       | 51.6       | 55.9     | 4.4        | 91%  | −4.3 |
| GraphSAGE | 55.3       | 56.5     | 43.5       | 21%  | −1.2 |

Judged by the drop, GAT depends on the graph four times as much as GraphSAGE does. Judged by
the gain, neither model does better with the real graph than with no edges at all. All of GAT's
drop is penalty. Its layers have no separate weight for a node's own features, so every layer mixes
in whatever neighbours the graph supplies, and random ones wreck it.

The paper, [*Dependence Is Not Advantage: What a Graph-Randomization Drop
Measures*](https://third-wheel.github.io/dependence-is-not-advantage/), has more cases, among them a
human–virus protein-interaction model, and explains why the drop varies so much from one layer
type to another.

## Which comparison answers your question

| If you want to know | compare |
|---|---|
| whether the graph is worth building | the model with the graph against the best model without it |
| whether this model uses the graph | the same model trained with the real graph and with no edges (the **gain**) |
| whether the wiring matters beyond the degrees | the real graph against a degree-preserving randomized graph (the **drop**) |
| whether a trained model survives errors in its graph | the trained model, tested on a perturbed graph without retraining (the **test-time drop**) |

The package measures the last three. For the first, tune a strong graph-free model yourself.

## Install

```bash
pip install graph-dependence
```

This installs PyTorch and PyTorch Geometric if they are missing. For a GPU build of PyTorch,
install it first by following [pytorch.org](https://pytorch.org/get-started/locally/). The
Planetoid datasets in the example below also need `pip install scipy`.

## Train your model three ways: `gain_penalty`

Give `gain_penalty` a function that builds your model, a PyTorch Geometric graph and the task,
`"node"` (node classification) or `"link"` (link prediction):

```python
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCN

from graph_dependence import gain_penalty


def make_model(in_channels, out_channels):
    return GCN(in_channels, 64, num_layers=2, out_channels=out_channels, dropout=0.5)


data = Planetoid("data", "Cora")[0]
report = gain_penalty(make_model, data, "node", seeds=5)
print(report)
```

This takes about six minutes on a laptop CPU (an M1 Pro). After a line for each run, it prints:

```
node classification, accuracy on a 0-100 scale, 5 seeds
learning rate chosen for each run on validation from (0.01, 0.001)

                   real graph     no edges   randomized
score                    79.7         55.0         27.7
best epoch                 12          319           45
epochs trained           1012         1319         1045

                                            mean   95% interval
gain            real - no edges            +24.7   [+22.9, +26.4]
penalty         no edges - randomized      +27.3   [+24.5, +30.2]
drop            real - randomized          +52.0   [+48.5, +55.4]

The drop is 65% of the real-graph score. Across seeds the randomized-graph
scores range from 25.7 to 31.0.

Reading: the gain interval is above zero, so the graph helps this model. Gain is
47% of the drop and penalty is the rest.
```

The graph is worth about 25 points to GCN on Cora, and even so more than half of the 52-point
drop is penalty.

For each seed, `gain_penalty` draws a degree-preserving randomized graph and trains the model on
the real graph, the randomized graph and no edges, all from the same initialization. Every run
tries the same learning rates (`lrs=(0.01, 0.001)` by default), keeps the one with the best
validation score and stops early on validation. Pass `test_time_drop=True` to also score each
real-graph model on the randomized graph without retraining.

For link prediction the model reads only the training links. The randomized graph is drawn from
those links, never creates a held-out pair, and the training targets stay the real links. A model
trained on shuffled targets falls to chance whatever graph it reads, which says nothing about the
graph. Pass a benchmark's own `splits=` to use its negatives; otherwise the package makes a
fixed random split and says so in the report.

The report's fields (`report.gain`, `report.penalty`, `report.drop`, `report.scores`,
`report.best_epoch`, ...) are there for your own tables. `help(gain_penalty)` lists every option.

## Use your own training code: `split_drop`

If you already train your model yourself, make the randomized graph with `randomize_edges`, train
the three versions over several seeds, and pass the test scores in the same seed order:

```python
from graph_dependence import randomize_edges, split_drop

randomized = randomize_edges(data.edge_index, data.num_nodes, seed=seed)  # one draw per seed

report = split_drop(
    real=[...],  # one test score per seed, higher is better
    no_edges=[...],  # the same model and layers, trained with an empty edge_index
    randomized=[...],  # trained on that seed's randomized graph
    metric="AUPRC",
)
print(report)
```

`randomize_edges` takes an undirected `edge_index` holding both directions of every edge, as
PyTorch Geometric stores it, and returns one in the same form. It applies double-edge swaps one at a
time and counts rejected swaps, which makes every graph it can reach with the same degrees equally
likely in the long run. Its `forbidden` argument keeps given node pairs, such as held-out links, from
becoming edges.

## Reading the report

- **A positive drop does not show that the graph helps.** If the gain's interval includes zero or
  lies below it, the drop shows only that real edges beat random ones.
- **The penalty depends on the layer.** A layer with a separate weight for the node's own features,
  such as GraphSAGE or GraphConv, can learn to rely less on its neighbours and pays a much smaller
  penalty than GCN or GAT. In the paper, adding that weight to GCN and GAT cut the penalty by 17 to
  42 points and raised the gain by only 2 to 8. Compare drops across architectures with this in mind.
- **The penalty depends on how long the models train.** On CiteSeer, training ten times longer cut
  GCN's penalty from 45 to 33 points. The report flags runs that hit `max_epochs`, but a run that
  stopped early may still be on a plateau. If the randomized-graph runs stop very early, double
  `patience` and see whether the penalty moves.
- **A test-time drop answers a different question.** It measures how a trained model copes with a
  graph unlike its training graph, and it can be far larger than the retrained drop. For DeepGNHV on
  viruses seen in training, the paper measures 44.3 AUPRC points against 1.3. Report it separately.

## Citation

```bibtex
@misc{duggireddy2026dependence,
  title  = {Dependence Is Not Advantage: What a Graph-Randomization Drop Measures},
  author = {Duggireddy, Prahas},
  year   = {2026},
  note   = {Preprint},
  url    = {https://www.alphaxiv.org/abs/2608.dependence-is-not-advantage}
}
```

Version 0.2.0 replaces the 0.1 API (`diagnose` and its graph-free comparator) with the
gain-and-penalty split described here.

Apache-2.0 licensed. Written by [Prahas Duggireddy](https://orcid.org/0009-0008-6106-0759) at
[Third Wheel](https://thirdwheel.ai).

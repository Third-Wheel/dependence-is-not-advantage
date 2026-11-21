"""Split a graph-randomization drop into what the graph adds and what wrong edges cost.

A randomization ablation trains a graph neural network on the real graph and on a
degree-preserving randomized graph, and reports the fall in score, the drop. The
drop is a sum of two parts::

    drop    = score(real graph) - score(randomized graph) = gain + penalty
    gain    = score(real graph) - score(no edges)           what the graph adds to this model
    penalty = score(no edges)   - score(randomized graph)   what wrong edges cost it

Only the gain measures whether the graph helps the model, and it takes a third
run, the same model with no edges.

``gain_penalty`` trains a PyTorch Geometric model on all three graphs for node
classification or link prediction. ``split_drop`` takes scores you measured with
your own training code. Both return a ``Report``; ``print(report)`` shows the
gain, the penalty and the drop with 95% intervals paired by seed.
"""

from .randomize import randomize_edges
from .report import Estimate, Report, split_drop
from .train import gain_penalty

__all__ = ["Estimate", "Report", "gain_penalty", "randomize_edges", "split_drop"]
__version__ = "0.2.0"

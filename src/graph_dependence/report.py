"""Gain, penalty and drop with intervals paired by seed, and a plain reading of them."""

from __future__ import annotations

import math
import statistics
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

ARMS = ("real", "no_edges", "randomized")
WIDTH = 80  # the printed report fits a standard terminal
LABELS = {"real": "real graph", "no_edges": "no edges", "randomized": "randomized"}

# Two-sided 95% t quantiles for 1 to 30 degrees of freedom.
_T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


class Estimate(NamedTuple):
    """A mean of per-seed differences with its two-sided 95% t-interval."""

    mean: float
    low: float
    high: float


@dataclass
class Report:
    """The scores of one model on three graphs, and what its drop is made of.

    ``scores`` maps ``"real"``, ``"no_edges"`` and ``"randomized"`` to test
    scores, one per seed and in the same seed order. ``"real_on_randomized"``
    is present when a test-time drop was measured: the model trained on the
    real graph, scored on the randomized one without retraining.
    ``print(report)`` gives a summary table.

    The training fields are filled in by ``gain_penalty`` and left as ``None``
    by ``split_drop``.
    """

    scores: dict[str, list[float]]
    metric: str = "score"
    task: str | None = None
    seeds: list[int] | None = None
    lrs: tuple[float, ...] | None = None
    best_epoch: dict[str, list[int]] | None = None
    epochs_trained: dict[str, list[int]] | None = None
    chosen_lr: dict[str, list[float]] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def gain(self) -> Estimate:
        """Real graph minus no edges: what the real graph adds to this model."""
        return paired(self.scores["real"], self.scores["no_edges"])

    @property
    def penalty(self) -> Estimate:
        """No edges minus randomized graph: what wrong edges cost this model."""
        return paired(self.scores["no_edges"], self.scores["randomized"])

    @property
    def drop(self) -> Estimate:
        """Real graph minus randomized graph, both trained: gain plus penalty."""
        return paired(self.scores["real"], self.scores["randomized"])

    @property
    def test_time_drop(self) -> Estimate | None:
        """The real-graph model on the real graph minus the same model on the randomized graph."""
        if "real_on_randomized" not in self.scores:
            return None
        return paired(self.scores["real"], self.scores["real_on_randomized"])

    def __str__(self) -> str:
        n = len(self.scores["real"])
        everything = [s for values in self.scores.values() for s in values]
        digits = 1 if max(abs(s) for s in everything) > 1.5 else 3
        mean = {a: statistics.fmean(self.scores[a]) for a in ARMS}
        gain, penalty, drop = self.gain, self.penalty, self.drop

        lines = [", ".join(p for p in (self.task, self.metric, f"{n} seeds") if p)]
        if self.lrs:
            lines.append(f"learning rate chosen for each run on validation from {self.lrs}")
        lines += ["", f"{'':<16}" + "".join(f"{LABELS[a]:>13}" for a in ARMS)]
        lines.append(f"{'score':<16}" + "".join(f"{mean[a]:>13.{digits}f}" for a in ARMS))
        for name, values in (
            ("best epoch", self.best_epoch),
            ("epochs trained", self.epochs_trained),
        ):
            if values:
                lines.append(
                    f"{name:<16}" + "".join(f"{statistics.fmean(values[a]):>13.0f}" for a in ARMS)
                )
        lines += [
            "",
            f"{'':<40}{'mean':>8}   95% interval",
            _line("gain", "real - no edges", gain, digits),
            _line("penalty", "no edges - randomized", penalty, digits),
            _line("drop", "real - randomized", drop, digits),
        ]
        if self.test_time_drop is not None:
            lines.append(_line("test-time drop", "no retraining", self.test_time_drop, digits))

        low, high = min(self.scores["randomized"]), max(self.scores["randomized"])
        summary = (
            f"The drop is {100 * drop.mean / mean['real']:.0f}% of the real-graph score. "
            if mean["real"] > 0
            else ""
        )
        summary += (
            f"Across seeds the randomized-graph scores range from {low:.{digits}f}"
            f" to {high:.{digits}f}."
        )
        paragraphs = [summary]
        if self.test_time_drop is not None:
            paragraphs.append(
                "The test-time drop scores each real-graph model on the randomized graph"
                " without retraining. It measures how a trained model copes with an"
                " unfamiliar graph, not the drop above."
            )
        paragraphs.append("Reading: " + reading(gain, penalty, drop))
        paragraphs += [f"Note: {note}" for note in self.notes]
        for text in paragraphs:
            lines += ["", textwrap.fill(text, WIDTH)]
        return "\n".join(lines)


def split_drop(
    real: Sequence[float],
    no_edges: Sequence[float],
    randomized: Sequence[float],
    *,
    real_on_randomized: Sequence[float] | None = None,
    metric: str = "score",
) -> Report:
    """Split a drop you measured with your own training code into gain and penalty.

    Train your model three times for each seed: on the real graph, on a
    degree-preserving randomized graph (``randomize_edges`` makes one) and
    with no edges, keeping the model's layers. Pass the three lists of test
    scores, one score per seed, in the same seed order, with higher meaning
    better (negate a loss). ``real_on_randomized`` is optional: the
    real-graph model scored on the randomized graph without retraining.

        report = split_drop(real=[r1, r2, r3], no_edges=[n1, n2, n3],
                            randomized=[s1, s2, s3], metric="AUPRC")
        print(report)
    """
    scores = {"real": real, "no_edges": no_edges, "randomized": randomized}
    if real_on_randomized is not None:
        scores["real_on_randomized"] = real_on_randomized
    clean = {k: [float(s) for s in v] for k, v in scores.items()}
    sizes = {len(v) for v in clean.values()}
    if len(sizes) != 1:
        raise ValueError("pass one score per seed for every graph, in the same seed order")
    n = sizes.pop()
    if n < 2:
        raise ValueError("at least two seeds are needed for a paired interval")
    if not all(math.isfinite(s) for v in clean.values() for s in v):
        raise ValueError("every score must be a finite number")
    notes = [seed_note(n)] if n < 5 else []
    return Report(scores=clean, metric=metric, notes=notes)


def reading(gain: Estimate, penalty: Estimate, drop: Estimate) -> str:
    """One or two sentences that claim no more than the intervals show."""
    if gain.low > 0:
        text = "the gain interval is above zero, so the graph helps this model."
        if penalty.mean > 0:
            text += (
                f" Gain is {100 * gain.mean / drop.mean:.0f}% of the drop and penalty is the rest."
            )
        return text
    if gain.high < 0:
        text = (
            "the gain interval is below zero: this model scores higher with no edges"
            " than with the real graph."
        )
        if drop.low > 0:
            text += " The drop is positive only because random edges cost it more than real ones."
        return text
    text = (
        "the gain interval includes zero, so these runs do not show that the graph"
        " helps this model."
    )
    if drop.low > 0:
        text += " A positive drop here means only that real edges beat random ones."
    return text


def seed_note(n: int) -> str:
    return (
        f"with {n} seeds a 95% interval is {t975(n - 1):.1f} standard errors either side of "
        "the mean; five or more seeds give much tighter intervals."
    )


def t975(df: int) -> float:
    """The two-sided 95% quantile of Student's t with ``df`` degrees of freedom."""
    if df <= len(_T975):
        return _T975[df - 1]
    z = 1.959964  # Cornish-Fisher expansion around the normal quantile; exact to 3 decimals here
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def paired(a: list[float], b: list[float]) -> Estimate:
    """The mean of a - b over seeds, with a t-interval on the per-seed differences."""
    d = [x - y for x, y in zip(a, b, strict=True)]
    m = statistics.fmean(d)
    half = t975(len(d) - 1) * statistics.stdev(d) / math.sqrt(len(d))
    return Estimate(m, m - half, m + half)


def _line(name: str, what: str, e: Estimate, digits: int) -> str:
    f = f"+.{digits}f"
    return f"{name:<16}{what:<24}{e.mean:>+8.{digits}f}   [{e.low:{f}}, {e.high:{f}}]"

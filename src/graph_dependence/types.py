from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar

import networkx as nx

ExampleT_contra = TypeVar("ExampleT_contra", contravariant=True)
NodeT = TypeVar("NodeT", bound=Hashable)
NodeT_scorer = TypeVar("NodeT_scorer", bound=Hashable)


class GraphScorer(Protocol[NodeT_scorer, ExampleT_contra]):
    """Score evaluation examples using a graph condition."""

    def __call__(
        self,
        graph: nx.Graph[NodeT_scorer],
        examples: Sequence[ExampleT_contra],
        /,
    ) -> Sequence[float]: ...


class GraphFreeScorer(Protocol[ExampleT_contra]):
    """Score the same evaluation examples without access to a graph."""

    def __call__(self, examples: Sequence[ExampleT_contra], /) -> Sequence[float]: ...


class Metric(Protocol):
    """Reduce predictions and targets to a scalar where larger is better."""

    def __call__(
        self,
        predictions: Sequence[float],
        targets: Sequence[float | int | bool],
        /,
    ) -> float: ...


@dataclass(frozen=True)
class NullDraw(Generic[NodeT]):
    """One graph sampled from a declared null model."""

    graph: nx.Graph[NodeT]
    metadata: Mapping[str, object]


class NullModel(Protocol[NodeT]):
    """Generate a fresh null graph and describe the implemented draw."""

    @property
    def name(self) -> str:
        """Stable, human-readable null-model name."""
        ...

    def __call__(
        self, graph: nx.Graph[NodeT], /, *, seed: int | None
    ) -> NullDraw[NodeT] | nx.Graph[NodeT]: ...


def frozen_metadata(values: Mapping[str, object]) -> Mapping[str, object]:
    """Return a shallow immutable copy suitable for a frozen result object."""

    return MappingProxyType(dict(values))


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class DependenceReport:
    """The two contrasts, their three reference scores, and null diagnostics."""

    dependence: float
    advantage: float | None
    null_survival: float | None
    observed_score: float
    null_score: float
    comparator_score: float | None
    metric_name: str
    null_model_name: str
    seed: int | None
    metadata: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "dependence": self.dependence,
            "advantage": self.advantage,
            "null_survival": self.null_survival,
            "observed_score": self.observed_score,
            "null_score": self.null_score,
            "comparator_score": self.comparator_score,
            "metric_name": self.metric_name,
            "null_model_name": self.null_model_name,
            "seed": self.seed,
            "metadata": _plain(self.metadata),
        }

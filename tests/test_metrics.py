from __future__ import annotations

import pytest

from graph_dependence import binary_auc


def test_binary_auc_perfect_and_reversed() -> None:
    targets = [0, 0, 1, 1]
    assert binary_auc([0.1, 0.2, 0.8, 0.9], targets) == 1.0
    assert binary_auc([0.9, 0.8, 0.2, 0.1], targets) == 0.0


def test_binary_auc_ties_are_average_ranked() -> None:
    assert binary_auc([1.0, 1.0, 1.0, 1.0], [0, 1, 0, 1]) == 0.5


@pytest.mark.parametrize(
    ("predictions", "targets", "message"),
    [
        ([], [], "must not be empty"),
        ([0.1], [0, 1], "same length"),
        ([0.1, 0.2], [0, 0], "both positive and negative"),
        ([0.1, 0.2], [0, 2], "zero or one"),
        ([0.1, float("nan")], [0, 1], "finite"),
    ],
)
def test_binary_auc_rejects_invalid_inputs(
    predictions: list[float], targets: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        binary_auc(predictions, targets)

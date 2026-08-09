from __future__ import annotations

import math
from collections.abc import Sequence


def binary_auc(
    predictions: Sequence[float],
    targets: Sequence[float | int | bool],
) -> float:
    """Return binary ROC AUC, assigning tied scores their average rank.

    The implementation is the normalized Mann--Whitney statistic. Targets must
    be exactly zero or one, and both classes must be present.
    """

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not predictions:
        raise ValueError("predictions and targets must not be empty")

    pairs: list[tuple[float, int]] = []
    for prediction, target in zip(predictions, targets, strict=True):
        score = float(prediction)
        label_value = float(target)
        if not math.isfinite(score):
            raise ValueError("predictions must be finite")
        if label_value not in (0.0, 1.0):
            raise ValueError("binary_auc targets must be zero or one")
        pairs.append((score, int(label_value)))

    pairs.sort(key=lambda item: item[0])
    positive_count = sum(label for _, label in pairs)
    negative_count = len(pairs) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("binary_auc requires both positive and negative targets")

    positive_rank_sum = 0.0
    start = 0
    while start < len(pairs):
        stop = start + 1
        while stop < len(pairs) and pairs[stop][0] == pairs[start][0]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in pairs[start:stop])
        start = stop

    mann_whitney = positive_rank_sum - positive_count * (positive_count + 1) / 2
    return mann_whitney / (positive_count * negative_count)

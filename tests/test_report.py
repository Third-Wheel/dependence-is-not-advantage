"""Tests for the report: intervals, the reading, and split_drop."""

from __future__ import annotations

import pytest

from graph_dependence import Estimate, split_drop
from graph_dependence.report import paired, reading, t975


def test_t_quantiles():
    for df, value in {
        4: 2.776,
        9: 2.262,
        30: 2.042,
        40: 2.021,
        60: 2.000,
        120: 1.980,
    }.items():
        assert t975(df) == pytest.approx(value, abs=5e-4)


def test_paired_interval():
    e = paired([3.0, 5.0, 4.0], [1.0, 2.0, 3.0])  # differences 2, 3, 1
    assert e.mean == pytest.approx(2.0)
    assert e.high - e.mean == pytest.approx(4.303 * 1.0 / 3**0.5)


def test_the_reading_claims_no_more_than_the_intervals_show():
    above, below, across = (
        Estimate(5, 3, 7),
        Estimate(-5, -7, -3),
        Estimate(5, -3, 13),
    )
    assert "the graph helps this model" in reading(above, above, Estimate(10, 8, 12))
    # A positive drop beside a gain whose interval includes zero is not called "all penalty".
    text = reading(across, across, above)
    assert "do not show that the graph helps" in text and "only because" not in text
    assert "only because random edges cost it more" in reading(below, above, above)


def test_split_drop_on_your_own_scores():
    report = split_drop(
        real=[0.750, 0.752, 0.749],
        no_edges=[0.748, 0.751, 0.743],
        randomized=[0.735, 0.741, 0.732],
        real_on_randomized=[0.31, 0.30, 0.32],
        metric="AUPRC",
    )
    assert report.drop.mean == pytest.approx(report.gain.mean + report.penalty.mean)
    assert report.test_time_drop is not None and report.test_time_drop.mean > 0.4
    text = str(report)
    assert text.startswith("AUPRC, 3 seeds")
    assert "0.750" in text  # scores on a 0-1 scale keep three decimals
    assert "best epoch" not in text  # no training details to show


def test_split_drop_refuses_scores_it_cannot_pair():
    with pytest.raises(ValueError, match="same seed order"):
        split_drop(real=[1.0, 2.0], no_edges=[1.0], randomized=[1.0, 2.0])
    with pytest.raises(ValueError, match="two seeds"):
        split_drop(real=[1.0], no_edges=[1.0], randomized=[1.0])
    with pytest.raises(ValueError, match="finite"):
        split_drop(real=[1.0, float("nan")], no_edges=[1.0, 2.0], randomized=[1.0, 2.0])

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from nessie.metrics import (
    precision_recall_at_k_percent_average_precision,
    precision_recall_f1_flagged,
)


def test_flagger_metrics():
    gold = np.array([True, False, True, True, True], dtype=bool)
    predicted_flags = np.array([True, True, True, False, False], dtype=bool)

    scores = precision_recall_f1_flagged(gold, predicted_flags)

    assert scores.precision == 2 / 3
    assert scores.recall == 2 / 4
    assert scores.f1 == 2 * (2 / 3 * 2 / 4) / (2 / 3 + 2 / 4)


@pytest.mark.parametrize(
    "k,expected_precision,expected_recall",
    [
        (0.2, 0 / 1, 0 / 1),
        (0.4, 1 / 2, 1 / 2),
        (0.6, 1 / 3, 1 / 3),
        (0.8, 2 / 4, 2 / 4),
        (1.0, 3 / 5, 3 / 5),
    ],
)
def test_scorer_metrics(k, expected_precision, expected_recall):
    # Values taken from
    # https://machinelearninginterview.com/topics/machine-learning/mapatk_evaluation_metric_for_ranking/
    relevancies = np.array([True, False, True, True, False], dtype=bool)
    scores = np.array([0.8, 0.9, 0.6, 0.5, 0.7], dtype=float)

    expected_ap = average_precision_score(relevancies, scores)

    result = precision_recall_at_k_percent_average_precision(relevancies, scores, k=k)
    ap = result.average_precision
    p = result.precision_at_k_percent
    r = result.precision_at_k_percent

    assert np.isclose(ap, expected_ap), f"{ap} != {expected_ap}"
    assert np.isclose(p, expected_precision), f"{p} != {expected_precision}"
    assert np.isclose(r, expected_recall), f"{r} != {expected_recall}"

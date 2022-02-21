import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytrec_eval
from sklearn.metrics import precision_recall_fscore_support


@dataclass
class FlaggerScore:
    precision: float
    recall: float
    f1: float
    percent_flagged: float
    name: str = None


@dataclass
class ScorerScore:
    average_precision: float
    precision_at_k_percent: float
    recall_at_k_percent: float


def precision_recall_f1_flagged(gold: npt.NDArray[bool], predictions: npt.NDArray[bool]) -> FlaggerScore:
    precision, recall, fscore, support = precision_recall_fscore_support(gold, predictions, average="binary")
    percent_flagged = sum(gold) / len(predictions)

    return FlaggerScore(precision=precision, recall=recall, f1=fscore, percent_flagged=percent_flagged)


def precision_recall_at_k_percent_average_precision(
    targets: npt.NDArray[bool], scores: npt.NDArray[float], k: float = 0.1
) -> ScorerScore:
    binary_judgements = np.asarray(targets)
    relevancy_scores = np.asarray(scores)

    unique_y = np.unique(binary_judgements)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    assert binary_judgements.shape == relevancy_scores.shape
    assert binary_judgements.ndim == 1

    n = len(binary_judgements)

    qrel = {
        "q1": {f"d{i + 1}": int(binary_judgements[i]) for i in range(n)},
    }

    run = {
        "q1": {f"d{i + 1}": float(relevancy_scores[i]) for i in range(n)},
    }

    k_percent = math.ceil(k * n)
    precision_label = f"P_{k_percent}"
    recall_label = f"recall_{k_percent}"

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {"map", precision_label, recall_label})
    scores = evaluator.evaluate(run)["q1"]

    ap = scores["map"]
    precision = scores[precision_label]
    recall = scores[recall_label]

    return ScorerScore(average_precision=ap, precision_at_k_percent=precision, recall_at_k_percent=recall)

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import iobes
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from seqeval.metrics.sequence_labeling import get_entities
from sklearn.preprocessing import LabelEncoder, normalize

from nessie.helper import RaggedResult
from nessie.types import (
    RaggedFloatArray2D,
    RaggedFloatArray3D,
    RaggedStringArray,
)

RaggedArray = npt.NDArray[Union[npt.NDArray, List[Any]]]

UNALIGNED_LABEL = "___NESSIE_NO_ALIGNMENT___"


# Aggregation


@dataclass
class SpanId:
    sentence: int
    start: int
    end: int


@dataclass
class AlignmentResult:
    labels: npt.NDArray[str]
    predictions: npt.NDArray[str]
    probabilities: npt.NDArray[float]  # 2D array of shape (num_entities, num_classes)
    repeated_probabilities: Optional[
        npt.NDArray[float]
    ]  # 2D array of shape (num_entities, num_repetitions, num_classes)
    span_ids: List[SpanId]
    le: LabelEncoder


def span_matching(
    tagging_A: List[Tuple[int, int]], tagging_B: List[Tuple[int, int]], keep_A: bool = False
) -> Dict[int, int]:
    """
    Assume we have a list of tokens which was tagged with spans by two different approaches A and B.
    This method tries to find the best 1:1 assignment of spans from B to spans from A. If there are more spans in A than
    in B, then spans from B will go unused and vice versa. The quality of an assignment between two spans depends on
    their overlap in tokens. This method removes entirely disjunct pairs of spans.
    Note: In case A contains two (or more) spans of the same length which are a single span in B (or vice versa), either
    of the spans from A may be mapped to the span in B. Which exact span from A is mapped is undefined.
    :param tagging_A: list of spans, defined by (start, end) token offsets (exclusive!), must be non-overlapping!
    :param tagging_B: a second list of spans over the same sequence in the same format as tagging_A
    :param keep_A: include unmatched spans from A as [idx_A, None] in the returned value
    :return: Dict[int,int] where keys are indices from A and values are indices from B
    """
    if not tagging_A:
        return {}
    elif not tagging_B:
        if keep_A:
            return {i: None for i in range(len(tagging_A))}
        else:
            return {}

    # Our cost function is span overlap:
    # (1) the basis: min(end indices) - max(start indices)
    # (2) If two spans are entirely disjunct, the result of (1) will be negative. Use max(0, ...) to set those
    #     cases to 0.
    # (3) High overlap should result in low costs, therefore multiply by -1
    overlap = lambda idx_a, idx_b: -1 * max(
        0, (min([tagging_A[idx_a][1], tagging_B[idx_b][1]]) - max([tagging_A[idx_a][0], tagging_B[idx_b][0]]))
    )
    cost_matrix: np.ndarray = np.fromfunction(np.vectorize(overlap), (len(tagging_A), len(tagging_B)), dtype=np.int)
    a_indices, b_indices = linear_sum_assignment(cost_matrix)

    # throw away mappings which have no token overlap at all (i.e. costs == 0)
    assignment_costs = cost_matrix[a_indices, b_indices]
    valid_assignments = [i for i in range(len(a_indices)) if assignment_costs[i] < 0]

    # dropped_assignments = len(a_indices) - len(valid_assignments)
    # if dropped_assignments:
    #     self.logger.debug(f"Threw away {dropped_assignments} assignment without token overlap")

    # collect valid assignments
    assignments = {a_idx: b_idx for i, (a_idx, b_idx) in enumerate(zip(a_indices, b_indices)) if i in valid_assignments}

    if keep_A:
        a_to_none = {i: None for i in range(len(tagging_A))}
        a_to_none.update(assignments)
        assignments = a_to_none
    return assignments


def align_span_labeling_result(noisy_labels: RaggedStringArray, result: RaggedResult):
    return align_for_span_labeling(
        noisy_labels, result.predictions, result.probabilities, result.repeated_probabilities, result.le
    )


def align_for_span_labeling(
    noisy_labels: RaggedStringArray,
    predictions: RaggedStringArray,
    probabilities: RaggedFloatArray2D,
    repeated_probabilities: RaggedFloatArray3D,
    le: LabelEncoder,
    span_aggregator: Optional[Callable[[List[np.ndarray]], np.ndarray]] = None,
    function_aggregator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> AlignmentResult:
    """The goal of this function is to align existing and predicted sequence labeling annotations
    and their respective probabilities.

    1. Original and predicted sequence labeling are aligned in a way that maximizes the overlap between them (see `span_matching`)
    2. BIO tagged sequences are reduced to a list of spans with their type, e.g. [O, B-PER, I-PER, O, B-LOC] becomes [PER, LOC]
    3. Probabilities for spans are aggregated
    4. Spans that exist in the original sequence and have no counterpart in the predicted one get a special label and
       are assigned the probability of the O label

    Args:
        noisy_labels:
        predictions:
        probabilities:
        repeated_probabilities:
        le:
        span_aggregator:
        function_aggregator:

    Returns:

    """

    assert len(noisy_labels) == len(predictions) == len(probabilities)
    if repeated_probabilities is not None:
        assert len(noisy_labels) == len(repeated_probabilities)

    aligned_span_ids = []
    aligned_labels = []
    aligned_predictions = []
    aligned_probabilities = []
    aligned_repeated_probabilities = [] if repeated_probabilities is not None else None

    label_map = _build_label_map(le.classes_)

    new_le = LabelEncoder()
    new_le.classes_ = np.array(list(label_map.keys()))

    if span_aggregator is None:
        span_aggregator = lambda x: np.mean(x, axis=0)

    if function_aggregator is None:
        function_aggregator = np.mean

    for sentence_id, (o, p) in enumerate(zip(noisy_labels, predictions)):
        assert len(o) == len(p)
        original_entities = get_entities(o)
        predicted_entities = get_entities(p)

        original_spans = [(s[1], s[2] + 1) for s in original_entities]
        predicted_spans = [(s[1], s[2] + 1) for s in predicted_entities]

        matches = span_matching(original_spans, predicted_spans, keep_A=True)

        assert len(matches) == len(original_entities)

        # Get the info
        cur_probabilities = probabilities[sentence_id]

        for o_idx, p_idx in matches.items():
            o_label, o_start, o_end = original_entities[o_idx]
            is_aligned = p_idx is not None

            if is_aligned:
                p_label, p_start, p_end = predicted_entities[p_idx]
            else:
                p_label = UNALIGNED_LABEL
                p_start = o_start
                p_end = o_end

            # seqeval uses inclusive indices, we like to use exclusive
            o_end += 1
            p_end += 1

            aligned_labels.append(o_label)
            aligned_predictions.append(p_label)
            aligned_span_ids.append(SpanId(sentence_id, o_start, o_end))

            cur_probability = np.asarray(cur_probabilities[p_start:p_end])
            if len(cur_probabilities[p_start:p_end]) > 0:
                cur_probability = span_aggregator(cur_probabilities[p_start:p_end])

            aggregated_cur_probability = _aggregate_class_probabilities(
                cur_probability, label_map, new_le.classes_, function_aggregator
            )

            assert aggregated_cur_probability.shape == (len(label_map),)

            aligned_probabilities.append(aggregated_cur_probability)

            if repeated_probabilities is not None:
                cur_repeated_probabilities = repeated_probabilities[sentence_id]
                cur = cur_repeated_probabilities[p_start:p_end]
                T = cur[0].shape[0]
                if len(cur) > 1:
                    cur_aligned_repeated_probability = span_aggregator(cur)
                else:
                    cur_aligned_repeated_probability = cur[0]

                cur_aligned_repeated_probability = np.vstack(
                    [
                        _aggregate_class_probabilities(x, label_map, new_le.classes_, function_aggregator)
                        for x in cur_aligned_repeated_probability
                    ]
                )

                assert cur_aligned_repeated_probability.shape == (T, len(label_map))

                cur_aligned_repeated_probability = normalize(cur_aligned_repeated_probability, norm="l1", axis=1)

                aligned_repeated_probabilities.append(cur_aligned_repeated_probability)

    aligned_labels = np.asarray(aligned_labels, dtype=object)
    aligned_predictions = np.asarray(aligned_predictions)
    aligned_probabilities = np.asarray(aligned_probabilities)

    if repeated_probabilities is not None:
        aligned_repeated_probabilities = np.asarray(aligned_repeated_probabilities)

    assert len(aligned_labels) == len(aligned_predictions) == len(aligned_probabilities) == len(aligned_span_ids)

    aligned_probabilities = normalize(aligned_probabilities, norm="l1", axis=1)

    result = AlignmentResult(
        labels=aligned_labels,
        predictions=aligned_predictions,
        probabilities=aligned_probabilities,
        repeated_probabilities=aligned_repeated_probabilities,
        span_ids=aligned_span_ids,
        le=new_le,
    )

    return result


def _aggregate_class_probabilities(
    probabilities: np.ndarray,
    label_map: Dict[str, List[int]],
    new_classes: npt.NDArray[str],
    function_aggregator: Callable[[np.ndarray], float],
) -> np.ndarray:
    # The original probability matrix contains probabilities for the separate BIO tags, e.g. B-PER and I-PER .
    # We reduce it so that it only contains the type, e.g. only PER and LOC
    aggregated_probability = np.empty(len(new_classes))
    for cls_idx, cls in enumerate(new_classes):
        cls_indices = label_map[cls]
        cls_probs = probabilities[cls_indices]
        aggregated_probability[cls_idx] = function_aggregator(cls_probs)

    return aggregated_probability


def _build_label_map(classes: List[str]) -> Dict[str, List[int]]:
    """Returns a mapping from entity type to BIO tag indices e.g. PER -> [idx(B-PER), idx(I-PER)]"""
    label_map = {}

    for idx, cls in enumerate(classes):
        if cls == "O":
            continue
        else:
            typ = iobes.utils.extract_type(cls)
            if typ not in label_map:
                label_map[typ] = [idx]
            else:
                label_map[typ].append(idx)

    return label_map
import abc
from abc import ABC
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import numpy.typing as npt
from faiss import IndexFlatL2
from scipy.special import logsumexp
from scipy.stats import entropy
from tqdm import trange

from nessie.detectors.error_detector import Detector, DetectorKind


class KnnErrorDetector(Detector, ABC):
    """Based on

    Not a cute stroke: Analysis of Rule- and Neural Network-based Information
    Extraction Systems for Brain Radiology Reports
    Andreas Grivas, Beatrice Alex, Claire Grover, Richard Tobin, William Whiteley
    In: Proceedings of the 11th International Workshop on Health Text Mining and Information Analysis
    """

    # https://arxiv.org/pdf/1911.00172.pdf
    # https://www.aclweb.org/anthology/2020.louhi-1.4.pdf
    # https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
    # https://github.com/Edinburgh-LTG/edieviz

    def __init__(self, k: float = 10):
        self._k = k

    def score(self, labels: List[str], embedded_instances: npt.NDArray[float], **kwargs) -> npt.NDArray[float]:
        """Finds neighbours of each instance in the embedding space, computes a distribution over
        labels based on their distance, the resulting score is the entropy over this distribution.

        Args:
            labels: a (num_samples, ) numpy array
            embedded_instances: 2d numpy array of shape (num_items, encoding_dim)
        Returns:
            a (num_samples, ) numpy array containing the scores for each instance
        """

        num_items, encoding_dim = embedded_instances.shape
        assert len(labels) == num_items

        encoded_input = embedded_instances.astype(np.float32)

        index = IndexFlatL2(encoding_dim)
        index.add(encoded_input)

        kind = self.error_detector_kind()
        if kind == DetectorKind.SCORER:
            scores = np.empty_like(labels, dtype=np.float64)
        elif kind == DetectorKind.FLAGGER:
            scores = np.empty_like(labels, dtype=bool)
        else:
            raise NotImplementedError(kind)

        # We do not batch the index.search because of
        # https://github.com/facebookresearch/faiss/issues/2126
        # https://github.com/facebookresearch/faiss/issues/2158
        # When that is fixed, then one could do  index.search(encoded_input, self._k)
        for i in trange(num_items):
            # Labels should include the instances' own label, as its encoding is closest to itself
            distances_to_neighbours, neighbour_indices = index.search(encoded_input[i : i + 1], self._k)

            label = labels[i]
            neighbour_labels = [labels[idx] for idx in neighbour_indices.squeeze()]
            probs = _k_nearest_interpolation(distances_to_neighbours.squeeze(), neighbour_labels)

            # Instances with higher entropy corresponding to choices that are more uncertain,
            # therefore hopefully more likely to be an annotation error
            scores[i] = self._score_item(label, probs)

        return scores

    def _score_item(self, label: str, probs: Dict[str, float]) -> Union[float, bool]:
        raise NotImplementedError()


class KnnEntropy(KnnErrorDetector):
    def error_detector_kind(self) -> DetectorKind:
        return DetectorKind.SCORER

    def _score_item(self, label: str, probs: Dict[str, float]) -> float:
        return entropy(list(probs.values()))


class KnnFlagger(KnnErrorDetector):
    def error_detector_kind(self) -> DetectorKind:
        return DetectorKind.FLAGGER

    def _score_item(self, label: str, probs: Dict[str, float]) -> bool:
        most_likely_label = max(probs, key=probs.get)
        return label != most_likely_label


def _k_nearest_interpolation(distances: List[float], labels: List[str]) -> Dict[str, float]:
    # Compute the probability distribution  over labels for a single item and its k neighbours.
    # This implements Equation 2 - P_{knn} from
    # Generalization through Memorization: Nearest Neighbor Language Models
    # Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, Mike Lewis
    # ICLR 2020 - https://arxiv.org/abs/1911.00172
    #
    # Code adapted from:
    # Not a cute stroke: Analysis of Rule- and Neural Network-Based Information Extraction
    # Systems for Brain Radiology Reports
    # Andreas Grivas, Beatrice Alex, Claire Grover, Richard Tobin, William Whiteley
    # LOUHI 2020 - https://aclanthology.org/2020.louhi-1.4.pdf
    # https://github.com/Edinburgh-LTG/edieviz

    assert len(distances) == len(labels)

    scores = defaultdict(list)

    # We use logsumexp to avoid overflow
    # http://gregorygundersen.com/blog/2020/02/09/log-sum-exp/

    # Softmax defn for label y_i and scores s_i (here neg dist):
    # P(y_i) = e^{s_i} / (\sum_j e^{s_j})

    # However in our case, the score for a single label i
    # may be a sum of terms if multiple returned results have this label
    # e.g. may be P(y_a) = e^{s_1} + e^{s_2} / (e^{s_1} + e^{s_2} + e^{s_3})

    # To ameliorate overflow issues, use logs + logsumexp
    # log(P(y_i)) = log(numerator / denom)
    #             = log(numerator) - log(denom)
    #             = logsumexp(numerator_scores) - logsumexp(all_scores)

    # So to get probs we need only exp the result above.

    all_scores = []
    for d, l in zip(distances, labels):
        # Parse to float64 for increased accuracy
        d = np.float64(d)
        scores[l].append(-d)
        all_scores.append(-d)

    # Compute logsumexp of denominator (sum of all class activations)
    denom = logsumexp(all_scores)

    # NOTE: as mentioned above,
    # we use logsumexp for the numerator as well, since we are
    # aggregating across labels. E.g. if label: B-Tumour occurs
    # multiple times, we need to sum across all activations for that label.
    # As in Equation 2 of the KNN Language Models.
    probs = {l: np.exp(logsumexp(scores[l]) - denom) for l in set(labels)}
    return probs

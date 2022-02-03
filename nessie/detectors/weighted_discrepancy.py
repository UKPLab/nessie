from collections import defaultdict

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import minmax_scale

from nessie.detectors.error_detector import Detector, DetectorKind
from nessie.types import StringArray


class WeightedDiscrepancy(Detector):
    """Inconsistency detection in semantic annotation
    Nora Hollenstein, Nathan Schneider, Bonnie Webber
    In: Proceedings of LREC 2016
    """

    def score(self, texts: StringArray, labels: StringArray, **kwargs) -> npt.NDArray[float]:
        """Label entropy is computed by collecting labels for instances with the same surface form
        and then computing the the weighted discrepancy. The lower, the more likely is it that
        the majority label is correct and that instances with minority labels are wrong. We assign
        instances with the majority label for its surface form a score of 0.0, as it is likely not wrong.

        Args:
            texts: a (num_instances, ) string sequence containing the text/surface form of each instance
            labels: a (num_instances, ) string sequence containing the noisy label for each instance
        Returns:
            a (num_samples, ) numpy array containing the scores for each instance
        """
        assert len(texts) == len(labels)
        counts = defaultdict(lambda: defaultdict(int))

        # Count labels seen per surface form
        for surface_form, label in zip(texts, labels):
            counts[surface_form.lower()][label] += 1

        # Compute the scores
        scores = np.zeros(len(texts))

        for i, (surface_form, label) in enumerate(zip(texts, labels)):
            surface_form = surface_form.lower()
            counts_for_token = counts[surface_form]

            if len(counts_for_token) < 2:
                continue

            most_common_tag_for_token = max(counts_for_token, key=counts_for_token.get)
            least_common_tag_for_token = min(counts_for_token, key=counts_for_token.get)
            n_s = sum(counts_for_token.values())

            # If the label is already the most common one, then we do not want
            # to give it a bad score
            if label == most_common_tag_for_token:
                continue

            c_max = counts_for_token[most_common_tag_for_token]
            c_min = counts_for_token[least_common_tag_for_token]

            score = (c_max - c_min) / len(counts_for_token) * n_s

            scores[i] = score

        scores = minmax_scale(scores)
        return scores

    def error_detector_kind(self) -> DetectorKind:
        return DetectorKind.SCORER

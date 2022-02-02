from collections import defaultdict

import numpy as np
import numpy.typing as npt
from scipy.stats import entropy
from sklearn.preprocessing import minmax_scale

from nessie.detectors.error_detector import DetectorKind, ModelBasedDetector
from nessie.types import StringArray


class LabelEntropy(ModelBasedDetector):
    """Hollenstein, N, Schneider, N & Webber, B 2016, Innessie detection in semantic annotation. in 10th
    edition of the Language Resources and Evaluation Conference. pp. 3986-3990, 10th edition of the
    Language Resources and Evaluation Conference, Portorož , Slovenia,
    """

    def error_detector_kind(self) -> DetectorKind:
        return DetectorKind.SCORER

    def score(self, texts: StringArray, labels: StringArray, **kwargs) -> npt.NDArray[float]:
        """Label entropy is computed by collecting labels for instances with the same surface form
        and then computing the entropy over this distribution. The lower, the more likely is it that
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
        scores = np.ones(len(labels), dtype=np.float64)
        for i, (surface_form, label) in enumerate(zip(texts, labels)):
            surface_form = surface_form.lower()
            counts_for_token = counts[surface_form]

            most_common_label = max(counts_for_token, key=counts_for_token.get)

            if len(counts_for_token) < 2:
                continue

            # If the label is already the most common one, then we do not want
            # to give it a bad score
            if label == most_common_label:
                continue

            probs = []
            n_s = sum(counts_for_token.values())
            for count in counts_for_token.values():
                prob = count / n_s
                probs.append(prob)

            scores[i] = entropy(probs, base=2)

        # Low entropy means high score, so we need to flip the scores
        scores = 1.0 - minmax_scale(scores)

        return scores

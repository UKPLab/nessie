import numpy as np
import numpy.typing as npt

from nessie.detectors.error_detector import DetectorKind, ModelBasedDetector


class PredictionMargin(ModelBasedDetector):
    """Dmitriy Dligach and Martha Palmer. 2011. Reducing the need for double annotation.
    In Proceedings of the 5th Linguistic Annotation Workshop (LAW V '11).
    Association for Computational Linguistics, USA, 65–73.
    """

    def error_detector_kind(self):
        return DetectorKind.SCORER

    def score(self, probabilities: npt.NDArray[float], **kwargs) -> npt.NDArray[float]:
        """The prediction margin for an instance is the absolute difference between the two largest probabilities
        predicted by a machine learning model. A smaller margin indicates a larger uncertainty.

        Args:
            probabilities: a (num_instances, num_classes) numpy array obtained from a machine learning model

        Returns:
            scores: a (num_instances,) numpy array containing the resulting scores
        """

        # Order probabilities for each single prediction from highest to lowest
        ordered = np.sort(probabilities, axis=-1)

        # Compute the absolute difference between most highest and second highest
        # probability per instance
        p1 = ordered[:, -1]
        p2 = ordered[:, -2]

        # We want to invert the score, because a small prediction margin means high chance
        # of annotation error.
        result = 1.0 - np.abs(p2 - p1)

        return result

    def uses_probabilities(self) -> bool:
        return True

import numpy.typing as npt
from scipy.stats import entropy

from nessie.detectors.error_detector import DetectorKind, ModelBasedDetector


class ClassificationEntropy(ModelBasedDetector):
    """Given a distribution over labels for each instance in form of a (num_instances, num_labels)
    numpy array, the resulting score for each instance is the entropy of each instances label distribution.
    If the entropy is larger, then this means more uncertainty and higher chance of being an annotation error.

    Active Learning Book
    Synthesis Lectures on Artificial Intelligence and Machine Learning
    Morgan & Claypool Publishers, June 2012
    Section 2.3

    See also https://modal-python.readthedocs.io/en/latest/content/query_strategies/uncertainty_sampling.html
    """

    def score(self, probabilities: npt.NDArray[float], **kwargs) -> npt.NDArray[float]:
        """Scores the input according to their class distribution entropy.

        Args:
            probabilities: a (num_instances, num_classes) numpy array obtained from a machine learning model

        Returns:
            scores: a (num_instances,) numpy array containing the resulting scores
        """

        scores = entropy(probabilities.T).T
        return scores

    def error_detector_kind(self):
        return DetectorKind.SCORER

    def uses_probabilities(self) -> bool:
        return True

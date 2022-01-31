import numpy as np
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

    def error_detector_kind(self):
        return DetectorKind.SCORER

    def score(self, probabilities: np.ndarray, **kwargs) -> np.ndarray:
        scores = entropy(probabilities.T).T
        return scores

    def uses_probabilities(self) -> bool:
        return True

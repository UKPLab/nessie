from abc import ABC
from enum import Enum, auto

import numpy as np
from sklearn.preprocessing import LabelEncoder


class DetectorType(Enum):
    BORDA_COUNT = auto()
    CLASSIFICATION_ENTROPY = auto()
    CLASSIFICATION_UNCERTAINTY = auto()
    CONFIDENT_LEARNING = auto()
    CROSS_WEIGH = auto()
    CURRICULUM_SPOTTER = auto()
    DATAMAP_CONFIDENCE = auto()
    DATAMAP_CONFIDENCE_SEQUENCE = auto()
    DIVERSE_ENSEMBLE = auto()
    DROPOUT_UNCERTAINTY = auto()
    ITEM_RESPONSE_THEORY = auto()
    LABEL_AGGREGATION = auto()
    LABEL_ENTROPY = auto()
    LEITNER_SPOTTER = auto()
    MEAN_DISTANCE = auto()
    PREDICTION_MARGIN = auto()
    RETAG = auto()
    UNIFORM_ENSEMBLE = auto()
    VARIATION_PRINCIPLE = auto()
    VARIATION_PRINCIPLE_SPAN = auto()
    WEIGHTED_DISCREPANCY = auto()
    KNN_FLAGGER = auto()
    KNN_ENTROPY = auto()
    PROJECTION_ENSEMBLE = auto()

    # Baselines
    BASELINE_MAJORITY_LABEL = auto()
    BASELINE_MAJORITY_LABEL_PER_SURFACE_FORM = auto()
    BASELINE_RANDOM_FLAGGER = auto()
    BASELINE_RANDOM_SCORER = auto()


class DetectorKind(Enum):
    FLAGGER = auto()
    SCORER = auto()


class Detector:
    def error_detector_kind(self) -> DetectorKind:
        raise NotImplementedError()

    def score(self, *args, **kwargs):
        raise NotImplementedError()

    def __str__(self):
        return str(self.__class__.__name__)

    def __repr__(self):
        return str(self.__class__.__name__)

    def uses_probabilities(self) -> bool:
        return False

    def needs_multiple_probabilities(self) -> bool:
        return False

    def supports_correction(self) -> bool:
        return False

    def correct(self, *args, **kwargs):
        assert self.supports_correction()


class ModelBasedDetector(Detector, ABC):
    def score(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        repeated_probabilities: np.ndarray,
        confidences_over_time: np.ndarray,
        le: LabelEncoder,
        **kwargs
    ) -> np.ndarray:
        """

        Args:
            texts:
            labels: Array of strings, these are *not* encoded
            predictions: Array of strings, these are *not* encoded
            probabilities: ndarray of shape (n_predictions, n_classes)
            repeated_probabilities: ndarray of shape (n_predictions, T, n_classes) repeatedly sampled probabilities
            confidences_over_time: ndarray of shape (n_predictions, T, n_classes)
            le: label encoder that can be used to map the numeric labels in `probabilities` to string labels
            **kwargs:

        Returns:

        """
        raise NotImplementedError()

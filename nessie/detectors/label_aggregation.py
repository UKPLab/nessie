import numpy as np
import numpy.typing as npt
import pandas as pd
from crowdkit.aggregation import DawidSkene
from sklearn.preprocessing import LabelEncoder

from nessie.detectors.error_detector import DetectorKind, ModelBasedDetector
from nessie.types import FloatArray2D, StringArray


class LabelAggregation(ModelBasedDetector):
    """Uses crowdsourcing aggregation tools to adjudicate labels obtained via Monte-Carlo dropout.

    Spotting Spurious Data with Neural Networks
    Hadi Amiri, Timothy A. Miller, Guergana Savova
    https://aclanthology.org/N18-1182.pdf
    """

    def error_detector_kind(self):
        return DetectorKind.FLAGGER

    def score(
        self, labels: StringArray, repeated_probabilities: FloatArray2D, le: LabelEncoder, **kwargs
    ) -> npt.NDArray[bool]:
        """Uses crowdsourcing aggregation tools to adjudicate labels obtained via Monte-Carlo dropout. Flags
        instances that then disagree with the adjudicated predicions.

        Args:
            labels: a (num_instances, ) string sequence containing the noisy label for each instance
            repeated_probabilities: A float array of (num_instances, T, num_classes) which has T label distributions
            per instance, e.g. as obtained by using different dropout for each prediction run during inference.
            le: the label encoder that allows converting the probabilities back to labels

        Returns:
            a (num_instances,) numpy array of bools containing the flags
        """
        labels = np.asarray(labels)
        repeated_probabilities = np.asarray(repeated_probabilities)

        predictions = self._aggregate_predictions(repeated_probabilities, le)

        result: np.ndarray = labels != predictions

        return result.astype(bool)

    def uses_probabilities(self) -> bool:
        return False

    def needs_multiple_probabilities(self) -> bool:
        return True

    def supports_correction(self) -> bool:
        return True

    def correct(
        self, labels: StringArray, repeated_probabilities: FloatArray2D, le: LabelEncoder, **kwargs
    ) -> npt.NDArray[str]:
        return self._aggregate_predictions(repeated_probabilities, le)

    def _aggregate_predictions(self, repeated_probabilities: np.ndarray, le: LabelEncoder) -> np.ndarray:
        probs = repeated_probabilities.argmax(axis=2)
        n, t = probs.shape
        num_items = n * t

        tasks = np.repeat(np.arange(n), t).astype(object)
        performers = np.tile(np.arange(t), n).astype(object)
        annotator_labels = probs.flatten().astype(object)

        assert len(tasks) == num_items
        assert len(performers) == num_items
        assert len(annotator_labels) == num_items

        data = {"task": tasks, "performer": performers, "label": annotator_labels}

        df = pd.DataFrame(data)

        encoded_predictions = DawidSkene(n_iter=100).fit_predict(df)
        predictions: np.ndarray = le.inverse_transform(encoded_predictions)

        return predictions

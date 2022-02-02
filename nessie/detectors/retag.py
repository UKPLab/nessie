import numpy as np
import numpy.typing as npt

from nessie.detectors.error_detector import DetectorKind, ModelBasedDetector
from nessie.types import StringArray


class Retag(ModelBasedDetector):
    """Halteren, Hans van. “The Detection of Innessie in Manually Tagged Text.”
    In Proceedings of the COLING-2000 Workshop on Linguistically Interpreted Corpora, 48–55.
    Centre Universitaire, Luxembourg: International Committee on Computational Linguistics, 2000.
    https://www.aclweb.org/anthology/W00-1907.
    """

    def error_detector_kind(self):
        return DetectorKind.FLAGGER

    def score(self, labels: StringArray, predictions: StringArray, **kwargs) -> npt.NDArray[bool]:
        """Flags the input if the noisy labels disagree with the predictions of a ML model.

        Args:
            labels: a (num_instances, ) string sequence containing the noisy label for each instance
            predictions: a (num_instances, ) numpy array obtained from a machine learning model
        Returns:
            a (num_instances,) numpy array of bools containing the flags after using retag
        """
        assert len(labels) == len(predictions)

        labels = np.asarray(labels)
        predictions = np.asarray(predictions)

        result: np.ndarray = labels != predictions

        return result.astype(bool)

    def supports_correction(self) -> bool:
        return True

    def correct(self, predictions: np.ndarray, **kwargs) -> np.ndarray:
        return predictions

import numpy as np
import numpy.typing as npt
from scipy.stats import mode

from nessie.detectors.error_detector import Detector, DetectorKind
from nessie.types import StringArray, StringArray2D


class MajorityVotingEnsemble(Detector):
    def score(self, labels: StringArray, ensemble_predictions: StringArray2D, **kwargs) -> npt.NDArray[bool]:
        """Flag instances where majority predictions disagree with given labels.

        Args:
            labels: a (num_samples, ) numpy array
            ensemble_predictions: a (num_models, num_samples) numpy array containing predictions for each model

        Returns:
            a (num_samples, ) numpy array containing where items are flagged that disagree with the majority vote
        """
        # For every instance, select the label that was predicted most often

        labels = np.asarray(labels)
        ensemble_predictions = np.asarray(ensemble_predictions)

        predictions = self._aggregate_predictions(ensemble_predictions)

        assert len(labels) == len(predictions)
        result: np.ndarray = labels != predictions

        return result.astype(bool)

    def error_detector_kind(self) -> DetectorKind:
        return DetectorKind.FLAGGER

    def supports_correction(self) -> bool:
        return True

    def correct(self, ensemble_predictions: np.ndarray, **kwargs) -> np.ndarray:
        return self._aggregate_predictions(ensemble_predictions)

    def _aggregate_predictions(self, ensemble_predictions: np.ndarray) -> np.ndarray:
        predictions = mode(ensemble_predictions, axis=0)[0].squeeze()
        return predictions

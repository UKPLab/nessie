from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from netcal import AbstractCalibration
from netcal.metrics import ECE

from nessie.helper import Callback, State
from nessie.types import FloatArray2D, IntArray


class CalibratorWrapper:
    def __init__(self, calibrator: AbstractCalibration):
        self._calibrator = calibrator

    def fit(self, probas: FloatArray2D, gold_labels: IntArray):
        self._calibrator.fit(probas, gold_labels)

    def transform(self, probas: FloatArray2D) -> FloatArray2D:
        calibrated_probas = self._calibrator.transform(probas)

        if self._calibrator._is_binary_classification():
            # netcal returns 1-d probabilities for binary classification,
            # but we expect (k, num_classes). Therefore, we just re-add them
            calibrated_probas = np.column_stack((1 - calibrated_probas, calibrated_probas))

        return calibrated_probas


class CalibrationCallback(Callback):
    def __init__(self, calibrator: CalibratorWrapper):
        self._calibrator = calibrator
        self._labels: Optional[npt.NDArray[int]] = None
        self._uncalibrated_probabilities: Optional[npt.NDArray[float]] = None
        self._calibrated_probabilities: Optional[npt.NDArray[float]] = None
        self._calibrated_repeated_probabilities: Optional[npt.NDArray[float]] = None

    def on_begin(self, state: State):
        self._labels = np.empty(state.num_samples, dtype=float)
        self._uncalibrated_probabilities = np.empty((state.num_samples, state.num_labels), dtype=float)
        self._calibrated_probabilities = np.empty((state.num_samples, state.num_labels), dtype=float)

        if state.should_compute_repeated_probabilities:
            self._calibrated_repeated_probabilities = np.empty(
                (state.num_samples, state.num_repetitions, state.num_labels), dtype=object
            )

    def on_after_predicting(self, state: State):
        assert state.probas_eval is not None
        assert state.labels_eval is not None

        calibrator = self._calibrator
        calibrator.fit(state.probas_eval, state.labels_eval)

        probas_eval_calibrated = calibrator.transform(state.probas_eval)

        assert state.probas_eval.shape == probas_eval_calibrated.shape

        if state.should_compute_repeated_probabilities:
            repeated_probabilities_calib = np.swapaxes(state.repeated_probabilities, 0, 1)
            repeated_probabilities_calib = np.array([calibrator.transform(p) for p in repeated_probabilities_calib])
            repeated_probabilities_calib = np.swapaxes(repeated_probabilities_calib, 0, 1)
            assert repeated_probabilities_calib.shape == state.repeated_probabilities.shape
            self._calibrated_repeated_probabilities[state.eval_indices] = repeated_probabilities_calib

        self._labels[state.eval_indices] = state.labels_eval
        self._uncalibrated_probabilities[state.eval_indices] = state.probas_eval
        self._calibrated_probabilities[state.eval_indices] = probas_eval_calibrated

    @property
    def calibrated_probabilities(self) -> npt.NDArray[float]:
        """

        Returns: float array of shape (num_instances, num_classes) containing calibrated probabilities

        """
        result = np.asarray(self._calibrated_probabilities)
        return result

    @property
    def calibrated_repeated_probabilities(self) -> npt.NDArray[float]:
        """

        Returns: float array of shape (num_instances, num_repetitions, num_classes)
                 containing calibrated repeated probabilities

        """
        result = np.asarray(self._calibrated_repeated_probabilities)
        return result

    @property
    def calibration_error(self) -> Tuple[float, float]:
        """

        Returns: tuple containing Expected Calibration Errors (ECE) before and after calibration

        """
        ece = ECE(10)
        ece_uncalibrated = ece.measure(self._uncalibrated_probabilities, self._labels)
        ece_calibrated = ece.measure(self._calibrated_probabilities, self._labels)

        return ece_uncalibrated, ece_calibrated


class CalibrationOnHoldoutCallback(Callback):
    pass

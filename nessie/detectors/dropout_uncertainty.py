import sys

import numpy as np
import numpy.typing as npt

from nessie.detectors.error_detector import Detector, DetectorKind
from nessie.types import FloatArray2D


class DropoutUncertainty(Detector):
    """Compute Uncertainty via Monte Carlo Dropout.
    This first has been proposed in:

    Spotting Spurious Data with Neural Networks
    MHadi Amiri, Timothy A. Miller, Guergana Savova
    https://aclanthology.org/N18-1182.pdf
    https://arxiv.org/abs/1703.00410

    Also see:
    How Certain is Your Transformer?
    Artem Shelmanov, Evgenii Tsymbalov, Dmitri Puzyrev, Kirill Fedyanin, Alexander Panchenko, Maxim Panov
    EACL 2021
    """

    def error_detector_kind(self):
        return DetectorKind.SCORER

    def score(self, repeated_probabilities: FloatArray2D, **kwargs) -> npt.NDArray[float]:
        """Given probabilities obtained via Monte Carlo Dropout, compute the score via
        the  entropy of the model predictions.

        Args:
            repeated_probabilities: A float array of (num_instances, T, num_classes) which has T label distributions
            per instance, e.g. as obtained by using different dropout for each prediction run during inference.

        Returns:
            scores: a (num_instances,) numpy array of bools containing the scores after running DU
        """

        # repeated_probabilities has shape (num_instances, T, num_classes)
        # result = _original_formulation(repeated_probabilities)
        # result = _variance_formulation(repeated_probabilities)
        result = _entropy_formulation(repeated_probabilities)

        return result

    def uses_probabilities(self) -> bool:
        return True

    def needs_multiple_probabilities(self) -> bool:
        return True


def _original_formulation(repeated_probabilities: npt.NDArray[float]) -> npt.NDArray[float]:
    n, T, c = repeated_probabilities.shape

    result = np.empty(n)
    for i, Y in enumerate(repeated_probabilities):
        first_term = 0
        for y_t in Y:
            first_term += y_t.dot(y_t)

        y_mean = Y.mean(axis=0)
        second_term = y_mean.dot(y_mean)

        result[i] = first_term / T - second_term

    return result


def _variance_formulation(repeated_probabilities: npt.NDArray[float]) -> npt.NDArray[float]:
    # The uncertainty is the variance over our T predictions
    # https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch
    repeated_predictions = np.argmax(repeated_probabilities, axis=2)

    return np.var(repeated_predictions, axis=1)


def _entropy_formulation(repeated_probabilities: npt.NDArray[float]) -> npt.NDArray[float]:
    # https://towardsdatascience.com/bayesian-deep-learning-with-fastai-how-not-to-be-uncertain-about-your-uncertainty-6a99d1aa686e
    # https://github.com/dhuynh95/fastai_bayesian/blob/master/fastai_bayesian/metrics.py
    # https://arxiv.org/pdf/1906.08158.pdf
    # https://aclanthology.org/2021.eacl-main.157.pdf

    dropout_predictions = np.swapaxes(repeated_probabilities, 0, 1)
    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)
    epsilon = sys.float_info.min

    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    return entropy

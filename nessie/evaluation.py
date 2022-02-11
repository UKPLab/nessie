import logging
import typing
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from nessie.detectors import Detector
from nessie.models import Model
from nessie.types import StringArray
from nessie.util import RANDOM_STATE, set_my_seed

logger = logging.getLogger("nessie")

T = typing.TypeVar("T")


@dataclass
class FlatResult:
    predictions: npt.NDArray[str]
    probabilities: npt.NDArray[float]  # 2D array of shape (num_instances, num_classes)
    repeated_probabilities: Optional[
        npt.NDArray[float]
    ]  # 2D array of shape (num_instances, num_repetitions, num_classes)
    le: LabelEncoder


class CrossValidationHelper:
    def __init__(self, n_splits: int = 10, num_repetitions: Optional[int] = 50):
        """

        Args:
            n_splits : Number of folds to use for cross-validation. If 1, then train and test on all and the same data
            num_repetitions: Number of repeated predictions used for methods that require it, e.g.
                *Dropout Uncertainty*. Set it to `None` or `0` to not obtain repeated probabilities
        """

        assert n_splits >= 1

        self._n_splits = n_splits
        self._num_repetitions = num_repetitions

    def run(self, X: StringArray, y_noisy: StringArray, model: Model) -> FlatResult:
        """Uses cross-validation to obtain predictions and probabilities from the given model on the given data.

        Args:
            X: The training data for training the model
            y_noisy: The labels for training the model
            model: The model that is trained during cross-validation and whose outputs are used for the detectors

        Returns:
            Model results evaluated via cross-validation. The shape `T` depends on the model used, e.g. flat for
            text classification or ragged lists for sequence labeling.
        """

        X = np.asarray(X)
        y_noisy = np.asarray(y_noisy)
        num_samples = len(X)

        num_labels = len(np.unique(y_noisy))

        should_compute_repeated_probabilities = (
            self._num_repetitions is not None and self._num_repetitions > 0 and model.has_dropout()
        )

        # Collect
        predictions = np.empty(num_samples, dtype=object)
        probabilities = np.empty((num_samples, num_labels))

        if should_compute_repeated_probabilities:
            repeated_probabilities = np.empty((num_samples, self._num_repetitions, num_labels), dtype=object)
        else:
            repeated_probabilities = None

        # Cross validation loop

        kf = get_cross_validator(self._n_splits)

        for i, (train_indices, eval_indices) in enumerate(kf.split(X, y_noisy)):
            logger.info(f"Model: [{model.name}], Fold {i + 1}/{self._n_splits}")

            X_train, X_eval = X[train_indices], X[eval_indices]
            y_train, y_eval = y_noisy[train_indices], y_noisy[eval_indices]

            assert len(X_train) == len(y_train)
            assert len(X_eval) == len(y_eval)
            assert len(eval_indices) == len(y_eval)

            logger.info(f"Fitting model: [{model.name()}]")
            start_time = timer()
            model.fit(X_train.tolist(), y_train.tolist())
            end_time = timer()
            training_time = end_time - start_time
            logger.info(f"Done fitting: [{model.name()}] in {training_time:.2f} seconds")

            # Predict
            logger.info(f"Predicting: [{model.name()}]")
            pred_eval = model.predict(X_eval)
            probas_eval = model.predict_proba(X_eval)
            logger.info(f"Done predicting: [{model.name()}]")

            num_samples_eval = len(pred_eval)
            num_classes = len(model.label_encoder().classes_)

            assert len(pred_eval) == num_samples_eval
            assert probas_eval.shape == (num_samples_eval, num_classes)

            # If we should compute several varying predictions, e.g. for Bayesian Uncertainty Estimation,
            # then we collect them here
            if should_compute_repeated_probabilities:
                logger.info("Obtaining multiple predictions")
                repeated_probas = obtain_repeated_probabilities(model, X_eval, self._num_repetitions)
                repeated_probabilities[eval_indices] = repeated_probas
            else:
                logger.info("Will not obtain multiple predictions")

            predictions[eval_indices] = pred_eval
            probabilities[eval_indices] = probas_eval

        return FlatResult(
            predictions=predictions,
            probabilities=probabilities,
            repeated_probabilities=repeated_probabilities,
            le=model.label_encoder(),
        )

    def run_for_nested(self, X: StringArray, y_noisy: StringArray, model: Model) -> FlatResult:
        pass


class SingeSplitCV:
    def split(self, X, *args, **kwargs):
        indices = np.arange(len(X))
        yield indices, indices


def get_cross_validator(n_splits: int) -> Union[StratifiedKFold, SingeSplitCV]:
    if n_splits > 1:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        return SingeSplitCV()


def obtain_repeated_probabilities(model: Model, X: StringArray, num_repetitions: int) -> npt.NDArray[float]:
    """

    Args:
        model:
        X:
        num_repetitions:

    Returns: A ndarray of shape (|X|, T, |classes|)

    """
    repeated_probabilities = []

    saved_seed = RANDOM_STATE
    with torch.no_grad():
        for t in range(num_repetitions):
            logging.info(f"Obtaining multiple probabilities for {model.name()}: {t + 1}/{num_repetitions}")
            set_my_seed(t + 23)
            model.use_dropout(True)

            y_probs_eval_again = model.predict_proba(X)
            repeated_probabilities.append(y_probs_eval_again)

            model.use_dropout(False)

    set_my_seed(saved_seed)

    # Check whether the dropout sampling really gave us different samples
    for idx_a in range(num_repetitions):
        for idx_b in range(num_repetitions):
            if idx_a == idx_b:
                continue

            a = repeated_probabilities[idx_a]
            b = repeated_probabilities[idx_b]
            msg = "Dropout promised different outputs per run, but got some equal"
            if np.allclose(a, b):
                warnings.warn(msg)

    repeated_probabilities = np.asarray(repeated_probabilities)
    repeated_probabilities = np.swapaxes(repeated_probabilities, 0, 1)

    return repeated_probabilities

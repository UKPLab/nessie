import logging
import warnings
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import List, Optional, Union

import awkward as ak
import numpy as np
import numpy.typing as npt
import torch
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from nessie.models import Model, SequenceTagger
from nessie.types import IntArray, RaggedStringArray, StringArray, StringArray2D
from nessie.util import RANDOM_STATE, set_my_seed

logger = logging.getLogger("nessie")


@dataclass
class Result:
    predictions: npt.NDArray[str]
    probabilities: npt.NDArray[float]  # 2D array of shape (num_instances, num_classes)
    repeated_probabilities: Optional[
        npt.NDArray[float]
    ]  # 2D array of shape (num_instances, num_repetitions, num_classes)
    le: LabelEncoder

    def unflatten(self, sizes: IntArray) -> "RaggedResult":
        predictions_ragged = ak.unflatten(self.predictions.tolist(), sizes)
        probabilities_ragged = ak.unflatten(self.probabilities, sizes)

        if self.repeated_probabilities is not None:
            repeated_probabilities_ragged = ak.unflatten(self.repeated_probabilities, sizes)
        else:
            repeated_probabilities_ragged = None

        ragged_result = RaggedResult(
            predictions=predictions_ragged,
            probabilities=probabilities_ragged,
            repeated_probabilities=repeated_probabilities_ragged,
            le=self.le,
        )
        return ragged_result


@dataclass
class RaggedResult:
    predictions: ak.Array  # ragged 3D string array of shape (num_sentences, [num_tokens, num_classes])
    probabilities: ak.Array  # ragged 3D float array of shape (num_sentences, [num_tokens, num_classes])
    repeated_probabilities: ak.Array  # ragged 3D float array of shape (num_sentences, [num_tokens, num_repetitions, num_classes])
    le: LabelEncoder

    def flatten(self) -> Result:
        predictions_flat = ak.flatten(self.predictions).to_numpy()
        probabilities_flat = ak.flatten(self.probabilities).to_numpy()

        if self.repeated_probabilities is not None:
            repeated_probabilities_flat = ak.flatten(self.repeated_probabilities).to_numpy()
        else:
            repeated_probabilities_flat = None

        result = Result(
            predictions=predictions_flat,
            probabilities=probabilities_flat,
            repeated_probabilities=repeated_probabilities_flat,
            le=self.le,
        )

        return result

    @property
    def sizes(self) -> npt.NDArray[int]:
        return ak.num(self.predictions).to_numpy()


@dataclass
class State:
    num_samples: int = None
    num_labels: int = None
    num_repetitions: int = None
    should_compute_repeated_probabilities: bool = None
    eval_indices: npt.NDArray[int] = None
    probas_eval: npt.NDArray[float] = None  # 2D array of shape (num_instances, num_classes)
    labels_eval: npt.NDArray[int] = None  # 1D array of shape (num_instances, )
    repeated_probabilities: npt.NDArray[float] = None  # 2D array of shape (num_instances, num_repetitions, num_classes)


class Callback:
    def on_begin(self, state: State):
        pass

    def on_before_fitting(self, state: State):
        pass

    def on_after_fitting(self, state: State):
        pass

    def on_before_predicting(self, state: State):
        pass

    def on_after_predicting(self, state: State):
        pass


class CallbackList(Callback):
    def __init__(self):
        self._callbacks: List[Callback] = []

    def add_callback(self, cb: Callback):
        self._callbacks.append(cb)

    def add_callbacks(self, cb: List[Callback]):
        self._callbacks.extend(cb)

    def on_begin(self, state: State):
        for cb in self._callbacks:
            cb.on_begin(state)

    def on_before_fitting(self, state: State):
        for cb in self._callbacks:
            cb.on_before_fitting(state)

    def on_after_fitting(self, state: State):
        for cb in self._callbacks:
            cb.on_after_fitting(state)

    def on_before_predicting(self, state: State):
        for cb in self._callbacks:
            cb.on_before_predicting(state)

    def on_after_predicting(self, state: State):
        for cb in self._callbacks:
            cb.on_after_predicting(state)


class CrossValidationHelper:
    def __init__(self, n_splits: int = 10, num_repetitions: Optional[int] = 50):
        """Helper class that performs `n`-fold cross validation for you.

        Args:
            n_splits : Number of folds to use for cross-validation. If 1, then train and test on all and the same data
            num_repetitions: Number of repeated predictions used for methods that require it, e.g.
                *Dropout Uncertainty*. Set it to `None` or `0` to not obtain repeated probabilities
        """

        assert n_splits >= 1

        self._n_splits = n_splits
        self._num_repetitions = num_repetitions

        self._callbacks: CallbackList = CallbackList()

    def run(self, X: StringArray, y_noisy: StringArray, model: Model) -> Result:
        """Uses cross-validation to obtain predictions and probabilities from the given model on the given data.

        Args:
            X: The training data for training the model
            y_noisy: The labels for training the model
            model: The model that is trained during cross-validation and whose outputs are used for the detectors

        Returns:
            Model results evaluated via cross-validation.
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

        state = State()
        state.num_samples = num_samples
        state.num_labels = num_labels
        state.should_compute_repeated_probabilities = should_compute_repeated_probabilities
        state.num_repetitions = self._num_repetitions

        self._callbacks.on_begin(state)

        for i, (train_indices, eval_indices) in enumerate(kf.split(X, y_noisy)):
            logger.info(f"Model: [{model.name}], Fold {i + 1}/{self._n_splits}")

            X_train, X_eval = X[train_indices], X[eval_indices]
            y_train, y_eval = y_noisy[train_indices], y_noisy[eval_indices]

            assert len(X_train) == len(y_train)
            assert len(X_eval) == len(y_eval)
            assert len(eval_indices) == len(y_eval)

            state.eval_indices = eval_indices

            # Fit
            self._callbacks.on_before_fitting(state)
            logger.info(f"Fitting model: [{model.name()}]")
            start_time = timer()
            model.fit(X_train.tolist(), y_train.tolist())
            end_time = timer()
            training_time = end_time - start_time
            logger.info(f"Done fitting: [{model.name()}] in {training_time:.2f} seconds")

            state.labels_eval = model.label_encoder().transform(y_eval)
            self._callbacks.on_after_fitting(state)

            # Predict
            self._callbacks.on_before_predicting(state)
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
                repeated_probas = obtain_repeated_probabilities_flat(model, X_eval, self._num_repetitions)
                repeated_probabilities[eval_indices] = repeated_probas
                state.repeated_probabilities = repeated_probas
            else:
                logger.info("Will not obtain multiple predictions")

            predictions[eval_indices] = pred_eval
            probabilities[eval_indices] = probas_eval

            state.probas_eval = probas_eval
            self._callbacks.on_after_predicting(state)

        return Result(
            predictions=predictions,
            probabilities=probabilities,
            repeated_probabilities=repeated_probabilities,
            le=model.label_encoder(),
        )

    def run_for_ragged(self, X: RaggedStringArray, y_noisy: RaggedStringArray, model: Model) -> RaggedResult:
        """Uses cross-validation to obtain predictions and probabilities from the given model on the given data.
        This is used for tasks with ragged inputs and outputs like sequence labeling.

        Args:
            X: The training data for training the model
            y_noisy: The labels for training the model
            model: The model that is trained during cross-validation and whose outputs are used for the detectors

        Returns:
            Model results evaluated via cross-validation.
        """

        X = ak.Array(X)
        y_noisy = ak.Array(y_noisy)

        sizes = ak.num(X)
        num_samples = ak.sum(sizes)

        num_labels = len(np.unique(ak.flatten(y_noisy).to_numpy()))

        should_compute_repeated_probabilities = (
            self._num_repetitions is not None and self._num_repetitions > 0 and model.has_dropout()
        )

        # Collect
        predictions_flat = np.empty(num_samples, dtype=object)
        probabilities_flat = np.empty((num_samples, num_labels))

        if should_compute_repeated_probabilities:
            repeated_probabilities_flat = np.empty((num_samples, self._num_repetitions, num_labels))
        else:
            repeated_probabilities_flat = None

        # Cross validation loop

        indices = np.arange(0, num_samples)
        grouped_indices = ak.unflatten(indices, sizes)

        kf = get_cross_validator(self._n_splits, stratified=False)

        for i, (train_indices, eval_indices) in enumerate(kf.split(X, y_noisy)):
            logger.info(f"Model: [{model.name}], Fold {i + 1}/{self._n_splits}")

            score_indices = ak.flatten(grouped_indices[eval_indices]).to_numpy()

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
            assert ak.all(ak.flatten(ak.count(probas_eval, axis=-1)) == num_classes)

            # If we should compute several varying predictions, e.g. for Bayesian Uncertainty Estimation,
            # then we collect them here
            if should_compute_repeated_probabilities:
                logger.info("Obtaining multiple predictions")
                repeated_probas_flat = obtain_repeated_probabilities_ragged_flattened(
                    model, X_eval, self._num_repetitions
                )
                repeated_probabilities_flat[score_indices] = repeated_probas_flat
            else:
                logger.info("Will not obtain multiple predictions")

            predictions_flat[score_indices] = ak.flatten(pred_eval).to_numpy()
            probabilities_flat[score_indices] = ak.flatten(probas_eval).to_numpy()

        result = Result(
            predictions=predictions_flat,
            probabilities=probabilities_flat,
            repeated_probabilities=repeated_probabilities_flat,
            le=model.label_encoder(),
        )

        return result.unflatten(sizes)

    def add_callback(self, cb: Callback):
        self._callbacks.add_callback(cb)


class SingeSplitCV:
    def split(self, X, *args, **kwargs):
        indices = np.arange(len(X))
        yield indices, indices


def get_cross_validator(n_splits: int, stratified: bool = True) -> Union[BaseCrossValidator, SingeSplitCV]:
    if n_splits > 1:
        if stratified:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        else:
            return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        return SingeSplitCV()


def obtain_repeated_probabilities_flat(model: Model, X: StringArray, num_repetitions: int) -> npt.NDArray[float]:
    """Uses Monte-Carlo dropout to obtain several probability estimates per instance.

    Args:
        model: The model to use
        X: The input
        num_repetitions: number of repetitions

    Returns: A ndarray of shape `(|X|, num_repetitions, |classes|)`

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


def obtain_repeated_probabilities_ragged_flattened(
    model: SequenceTagger, X: StringArray2D, num_repetitions: int
) -> npt.NDArray[float]:
    """Uses Monte-Carlo dropout to obtain several probability estimates per instance.

    Args:
        model: The model to use
        X: The inputs (need to be ragged, e.g. for token labeling)
        num_repetitions: Number of repetitions

    Returns: A ndarray of shape `(|X|, num_repetitions, |classes|)`

    """
    repeated_probabilities_flat = []

    saved_seed = RANDOM_STATE
    with torch.no_grad():
        for t in range(num_repetitions):
            set_my_seed(t + 23)
            model.use_dropout(True)

            y_probs_eval_again = model.predict_proba(X)
            repeated_probabilities_flat.append(ak.flatten(y_probs_eval_again))

            model.use_dropout(False)

    set_my_seed(saved_seed)

    # Check whether the dropout sampling really gave us different samples
    for idx_a in range(num_repetitions):
        for idx_b in range(num_repetitions):
            if idx_a == idx_b:
                continue

            a = repeated_probabilities_flat[idx_a]
            b = repeated_probabilities_flat[idx_b]
            msg = "Dropout promised different outputs per run, but got some equal"
            if np.allclose(a, b):
                warnings.warn(msg)

    repeated_probabilities_flat = np.asarray(repeated_probabilities_flat)
    repeated_probabilities_flat = np.swapaxes(repeated_probabilities_flat, 0, 1)

    return repeated_probabilities_flat

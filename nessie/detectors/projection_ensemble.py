import contextlib
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from scipy.stats import mode
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection

from nessie.detectors.error_detector import Detector, DetectorKind
from nessie.types import FloatArray2D


class MaxEntProjectionEnsemble(Detector):
    """Identifying Incorrect Labels in the CoNLL-2003 Corpus
    Frederick Reiss, Hong Xu, Bryan Cutler, Karthik Muthuraman, Zachary Eichenberger
    Proceedings of the 24th Conference on Computational Natural Language Learning - 2020
    https://aclanthology.org/2020.conll-1.16/
    """

    def __init__(
        self, n_components: List[int] = None, seeds: List[int] = None, num_jobs: int = 4, max_iter: int = 10_000
    ):
        """Two lists are given of model sizes and seeds, and the combinations
        of the two is the complete set of parameters used to train the models

        Args:
            n_components:
            seeds:
        """
        if n_components is None:
            n_components = [32, 64, 128, 256]

        if seeds is None:
            seeds = [1, 2, 3, 4]

        self._n_components = n_components
        self._seeds = seeds
        self._num_jobs = num_jobs
        self._max_iter = max_iter

    def error_detector_kind(self) -> DetectorKind:
        return DetectorKind.FLAGGER

    def score(
        self,
        X_train_embedded: FloatArray2D,
        y_train_encoded: npt.NDArray[int],
        X_eval_embedded: FloatArray2D,
        y_eval_encoded: npt.NDArray[int],
        **kwargs,
    ) -> Tuple[List[str], List[List[str]], npt.NDArray[bool]]:
        """Uses an ensemble of logistic regression models that use different Gaussian projections of
        dense embeddings as input. These are aggregated via majority vote and instances are flagged whose
        label disagree.

        Args:
            X_train_embedded: shape (n_instances, encoding_dim)
            y_train_encoded: shape (n_instances)
            X_eval_embedded: shape (n_instances, encoding_dim)
            y_eval_encoded: shape (n_instances)
        Returns:
            predictions: A string list of the predictions for every instance after majority vote
            ensemble:predictions: A list of string lists containing the predictions for every instance before majority vote
            flags: A boolean sequence containing the flags

        """
        assert len(X_train_embedded) == len(y_train_encoded)
        assert len(X_eval_embedded) == len(y_eval_encoded)

        ensemble_predictions = []

        combinations = [(n, seed) for n in self._n_components for seed in self._seeds]

        models = Parallel(n_jobs=self._num_jobs, verbose=50, prefer="processes")(
            delayed(_train_single_model)(X_train_embedded, y_train_encoded, n, seed, self._max_iter)
            for n, seed in combinations
        )

        models.append(LogisticRegression(max_iter=self._max_iter).fit(X_train_embedded, y_train_encoded))

        assert len(models) == self.ensemble_size

        for model in models:
            predictions = model.predict(X_eval_embedded)
            ensemble_predictions.append(predictions)

        ensemble_predictions = np.array(ensemble_predictions)

        predictions = mode(ensemble_predictions, axis=0).mode.ravel()

        return predictions, ensemble_predictions, y_eval_encoded != predictions

    @property
    def ensemble_size(self) -> int:
        return len(self._n_components) * len(self._seeds) + 1


def _train_single_model(
    X_encoded: np.ndarray, y_encoded: np.ndarray, n_components: int, seed: int, max_iter: int
) -> BaseEstimator:
    """

    Args:
        X_encoded: shape (n_instances, encoding_dim)
        y_encoded: shape (n_instances)
        n_components:
        seed:

    Returns:

    """

    reduce_pipeline = Pipeline(
        [
            (
                "dimred",
                GaussianRandomProjection(n_components=n_components, random_state=seed),
            ),
            (
                "mlogreg",
                LogisticRegression(max_iter=max_iter),
            ),
        ]
    )

    model = reduce_pipeline.fit(X_encoded, y_encoded)
    return model


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    # https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = Parallel.print_progress
    Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        Parallel.print_progress = original_print_progress
        tqdm_object.close()

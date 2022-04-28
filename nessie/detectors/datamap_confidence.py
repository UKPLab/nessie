"""Swayamdipta, Swabha, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith, Yejin Choi.
“Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics.”
Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020.

https://doi.org/10.18653/v1/2020.emnlp-main.746
"""

from typing import List, Union

import awkward as ak
import numpy as np
import numpy.typing as npt
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from nessie.detectors.error_detector import Detector, DetectorKind
from nessie.models.model import CallbackableModel, Model
from nessie.types import StringArray, StringArray2D


class DataMapConfidence(Detector):
    def __init__(self, model: CallbackableModel, needs_flattening: bool = False):
        self._model = model
        self._needs_flattening = needs_flattening

    def error_detector_kind(self):
        return DetectorKind.SCORER

    def score(
        self, X: Union[StringArray, StringArray2D], y: Union[StringArray, StringArray2D], **kwargs
    ) -> Union[npt.NDArray[float], ak.Array]:
        # Model setup
        model = self._model
        callback = DataMapConfidenceCallback(model, X)
        model.add_callback("datamap_callback", callback)

        model.fit(X, y)

        if not self._needs_flattening:
            labels = y
            confidences_over_time = np.array(callback.probas)
        else:
            labels = ak.flatten(y).to_numpy()
            confidences_over_time = ak.flatten(callback.probas, 2).to_numpy()

        labels_encoded = model.label_encoder().transform(labels)

        confidences_over_time = confidences_over_time.swapaxes(0, 1)
        n, t, c = confidences_over_time.shape

        assert len(labels) == n

        confidence = np.empty(n)

        for idx in range(n):
            probs_i = confidences_over_time[idx]
            label_i = labels_encoded[idx]
            confidence[idx] = probs_i[:, label_i].mean()

        uncertainty = 1.0 - confidence

        if not self._needs_flattening:
            return uncertainty
        else:
            counts = ak.num(y)
            return ak.unflatten(uncertainty, counts)


class DataMapConfidenceCallback(TrainerCallback):
    """Callback that is called during certain events (starting/ending training, ...)"""

    def __init__(self, model: Model, X: List[List[str]]):
        super().__init__()

        self._model = model
        self._X = X

        self.probas = []

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.probas = []

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.probas.append(self._model.predict_proba(self._X))

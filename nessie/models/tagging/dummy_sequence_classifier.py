from typing import Optional

import awkward as ak
import numpy as np
from sklearn.preprocessing import LabelEncoder

from nessie.models import SequenceTagger
from nessie.types import RaggedStringArray
from nessie.util import get_random_probabilities


class DummySequenceTagger(SequenceTagger):
    def __init__(self):
        self._le: Optional[LabelEncoder] = None

    def fit(self, X: RaggedStringArray, y: RaggedStringArray):
        self._le = LabelEncoder().fit(ak.flatten(y).to_numpy())

    def predict(self, X: RaggedStringArray) -> ak.Array:
        probas = self.predict_proba(X)
        probas_flat = ak.flatten(probas).to_numpy()
        indices_flat = np.argmax(probas_flat, axis=1)
        labels_flat = self._le.inverse_transform(indices_flat)
        return ak.unflatten(labels_flat, ak.num(probas))

    def score(self, X: RaggedStringArray) -> ak.Array:
        counts = ak.num(ak.Array(X))
        num_samples = ak.sum(counts)
        probas_flat = get_random_probabilities(num_samples, 1, seed=None).squeeze()
        return ak.unflatten(probas_flat, counts)

    def predict_proba(self, X: RaggedStringArray) -> ak.Array:
        counts = ak.num(ak.Array(X))
        num_samples = ak.sum(counts)
        probas_flat = get_random_probabilities(num_samples, len(self._le.classes_), seed=None)
        return ak.unflatten(probas_flat, counts)

    def label_encoder(self) -> LabelEncoder:
        return self._le

    def has_dropout(self) -> bool:
        return True

    def use_dropout(self, is_activated: bool):
        pass

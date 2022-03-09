from typing import Optional

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import LabelEncoder

from nessie.models import TextClassifier
from nessie.types import StringArray
from nessie.util import get_random_probabilities


class DummyTextClassifier(TextClassifier):
    def __init__(self):
        self._le: Optional[LabelEncoder] = None

    def fit(self, X: StringArray, y: StringArray):
        self._le = LabelEncoder().fit(y)

    def predict(self, X: StringArray) -> npt.NDArray[str]:
        probas = self.predict_proba(X)
        indices = np.argmax(probas, axis=1)
        return self._le.inverse_transform(indices)

    def score(self, X: StringArray) -> npt.NDArray[float]:
        return get_random_probabilities(len(X), 1, seed=None).squeeze()

    def predict_proba(self, X: StringArray) -> npt.NDArray[float]:
        return get_random_probabilities(len(X), len(self._le.classes_), seed=None)

    def label_encoder(self) -> LabelEncoder:
        return self._le

    def has_dropout(self) -> bool:
        return True

    def use_dropout(self, is_activated: bool):
        pass

from typing import Callable, Generic, Optional, TypeVar

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import LabelEncoder

from nessie.models import TextClassifier
from nessie.models.featurizer import SentenceEmbedder
from nessie.types import StringArray

T = TypeVar("T")


class SklearnTextClassifier(TextClassifier, Generic[T]):
    """This model uses a sentence embedder like S-BERT to embed sentences, these are inputs to
    train a model with scikit learn API."""

    def __init__(self, model_builder: Callable[[], T], embedder: SentenceEmbedder):
        self._embedder = embedder
        self._model_builder = model_builder
        self._model: Optional[T] = None
        self._label_encoder: Optional[LabelEncoder] = None

    def fit(self, X: StringArray, y: StringArray):
        # Embed the sentences with our sentence embedder, e.g. S-BERT
        self._embedder.train()
        X_embedded = self._embedder.embed(X)

        # Encode the labels so that they are numeric instead of textual
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        model = self._model_builder()
        model.fit(X_embedded, y_encoded)

        self._model = model
        self._label_encoder = le

    def predict(self, X: StringArray) -> npt.NDArray[str]:
        assert self._model, "Model not set for predicting, train first"

        # Embed the sentences with our sentence embedder, e.g. S-BERT
        self._embedder.eval()
        X_embedded = self._embedder.embed(X)

        # Get the predictions
        y_pred_encoded = self._model.predict(X_embedded)

        y_pred = self._label_encoder.inverse_transform(y_pred_encoded)

        return np.array(y_pred)

    def score(self, X: StringArray) -> npt.NDArray[float]:
        assert self._model, "Model not set for predicting, train first"
        # Embed the sentences with our sentence embedder, e.g. S-BERT

        self._embedder.eval()
        X_embedded = self._embedder.embed(X)

        # Get the prediction probabilities over all classes
        probs = self._model.predict_proba(X_embedded)

        result = probs.max(axis=1)

        return np.array(result)

    def predict_proba(self, X: StringArray) -> npt.NDArray[float]:
        assert self._model, "Model not set for predicting, train first"
        # Embed the sentences with our sentence embedder, e.g. S-BERT

        self._embedder.eval()
        X_embedded = self._embedder.embed(X)

        # Get the prediction probabilities over all classes
        probs = self._model.predict_proba(X_embedded)

        return np.array(probs)

    def label_encoder(self) -> LabelEncoder:
        assert self._label_encoder, "Label encoder not set for predicting, train first"
        return self._label_encoder

    def name(self) -> str:
        return f"{self.__class__.__name__} {self._embedder.__class__.__name__}"

    def __str__(self):
        return self.name()

    def __repr__(self):
        return self.name()

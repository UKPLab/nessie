from abc import ABC

import awkward as ak
import numpy.typing as npt
from sklearn.preprocessing import LabelEncoder
from transformers import TrainerCallback

from nessie.types import RaggedStringArray, StringArray


class Model(ABC):
    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def score(self, X):
        """Returns the best score for each item"""
        raise NotImplementedError()

    def predict_proba(self, X):
        """Returns a distribution over all labels for each item"""
        raise NotImplementedError()

    def label_encoder(self) -> LabelEncoder:
        """Returns a label encoder that can be used to map labels to ints and vice versa"""
        raise NotImplementedError()

    def name(self) -> str:
        return self.__class__.__name__

    def has_dropout(self) -> bool:
        return False

    def use_dropout(self, is_activated: bool):
        assert self.has_dropout()

    def __str__(self):
        return str(self.__class__.__name__)

    def __repr__(self):
        return str(self.__class__.__name__)


class TextClassifier(Model, ABC):
    def fit(self, X: StringArray, y: StringArray):
        raise NotImplementedError()

    def predict(self, X: StringArray) -> npt.NDArray[str]:
        raise NotImplementedError()

    def score(self, X: StringArray) -> npt.NDArray[float]:
        raise NotImplementedError()

    def predict_proba(self, X: StringArray) -> npt.NDArray[float]:
        """Returns a distribution over labels for each instance.

        Args:
            X: The texts to predict on
        Returns:
            A (num_instances, num_labels) numpy array
        """
        raise NotImplementedError()


class SequenceTagger(Model, ABC):
    def fit(self, X: RaggedStringArray, y: RaggedStringArray):
        raise NotImplementedError()

    def predict(self, X: RaggedStringArray) -> ak.Array:
        raise NotImplementedError()

    def score(self, X: RaggedStringArray) -> ak.Array:
        raise NotImplementedError()

    def predict_proba(self, X: RaggedStringArray) -> ak.Array:
        """Returns a distribution over labels for each instance.

        Args:
            X: The token sequences to predict on
        Returns:
            A (num_sentences, num_tokens, num_labels) ragged array
        """
        raise NotImplementedError()


class Callbackable(ABC):
    def add_callback(self, name: str, callback: TrainerCallback):
        raise NotImplementedError()

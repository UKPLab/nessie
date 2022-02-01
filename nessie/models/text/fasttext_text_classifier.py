from typing import Optional

import numpy as np
import numpy.typing as npt
from fasttext.FastText import _FastText, train_supervised
from more_itertools import flatten
from sklearn.preprocessing import LabelEncoder, normalize

from nessie.models import TextClassifier
from nessie.types import StringArray
from nessie.util import RANDOM_STATE, tempinput


class FastTextTextClassifier(TextClassifier):
    def __init__(self, verbose: bool = False):
        self.model_: Optional[_FastText] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._verbose = verbose

    def fit(self, X: StringArray, y: StringArray):
        s = self._build_corpus(X, y)

        # Fasttext cannot train from in memory strings, so we dump them to a temporary file
        with tempinput(s) as filename:
            self.model_ = train_supervised(
                input=str(filename),
                lr=1.0,
                wordNgrams=2,
                bucket=200000,
                dim=50,
                loss="hs",
                verbose=self._verbose,
                seed=RANDOM_STATE,
            )

    def _build_corpus(self, X: StringArray, y: StringArray) -> str:
        # fasttext can only load corpora from files, so we need to hack things up

        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        lines = []
        for text, label in zip(X, y_encoded):
            lines.append(f"__label__{label}\t{text}")

        s = "\n".join(lines)
        return s

    def predict(self, X: StringArray) -> npt.NDArray[str]:
        assert self.model_, "Model not set for predicting, train first"

        # Strip the __label__
        X = self._clean(X)
        predictions = self.model_.predict(X)[0]

        preds = [int(label[9:]) for label in flatten(predictions)]
        return self._label_encoder.inverse_transform(preds)

    def score(self, X: StringArray) -> npt.NDArray[float]:
        assert self.model_, "Model not set for predicting, train first"

        X = self._clean(X)

        scores = list(flatten(self.model_.predict(X)[1]))
        return np.array(scores)

    def predict_proba(self, X: StringArray) -> npt.NDArray[float]:
        assert self.model_, "Model not set for predicting, train first"

        k = len(self.label_encoder().classes_)
        result = np.zeros((len(X), k))

        # FastText returns two lists (of lists), one for the labels and one for the predictions
        # For each sentence given, there is now a list of predictions and a list of respective scores.
        # These can be shorter than the overall number of classes, e.g.
        # labels = [["__label__1", "__label__2"], ["__label__0"]]
        # preds = [[0.42, 0.13], [0.95]]
        labels, preds = self.model_.predict(self._clean(X), k=k)

        for i, (labels_for_item, preds_for_item) in enumerate(zip(labels, preds)):
            for label, pred in zip(labels_for_item, preds_for_item):
                idx = int(label[9:])
                result[i, idx] = pred

        # Fasttext probabilities sometimes sum to a little bit above 1 so we normalize it to 0..1
        # result = result / np.expand_dims(np.linalg.norm(result, axis=1), axis=-1)
        result = normalize(result, norm="l1", axis=1)
        return result

    def label_encoder(self) -> LabelEncoder:
        return self._label_encoder

    def _clean(self, sentences: StringArray) -> StringArray:
        return [s for s in sentences]

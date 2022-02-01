from typing import Optional

import awkward as ak
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from nessie.models import SequenceTagger
from nessie.models.tagging.util import featurize_sentence
from nessie.types import RaggedStringArray


class MaxEntSequenceTagger(SequenceTagger):
    def __init__(self, max_iter=10000):
        self._max_iter = max_iter

        self._model: Optional[LogisticRegression] = None
        self._vectorizer: Optional[DictVectorizer] = None
        self._label_encoder: Optional[LabelEncoder] = None

    def fit(self, X: RaggedStringArray, y: RaggedStringArray):
        le = LabelEncoder()

        X_feat = [featurize_sentence(list(sentence)) for sentence in X]
        X_feat_flattened = ak.flatten(X_feat).to_list()
        y_flattened = ak.flatten(y).to_list()
        y_flattened_encoded = le.fit_transform(y_flattened)

        v = DictVectorizer(sparse=False)

        X_feat_flattened_encoded = v.fit_transform(X_feat_flattened)
        X_feat_flattened_encoded = np.nan_to_num(X_feat_flattened_encoded)

        model = LogisticRegression(max_iter=self._max_iter).fit(X_feat_flattened_encoded, y_flattened_encoded)

        self._model = model
        self._vectorizer = v
        self._label_encoder = le

    def predict(self, X: RaggedStringArray) -> ak.Array:
        assert self._model, "Model not set for predicting, train first"
        assert self._label_encoder, "Encoder not set for predicting, train first"

        counts = ak.num(X)

        X_feat = [featurize_sentence(sentence) for sentence in X]
        X_feat_flattened = ak.flatten(X_feat).to_list()
        X_feat_flattened_encoded = self._vectorizer.transform(X_feat_flattened)
        X_feat_flattened_encoded = np.nan_to_num(X_feat_flattened_encoded)

        y_pred_flattened_encoded = self._model.predict(X_feat_flattened_encoded)
        y_pred_flattened = self._label_encoder.inverse_transform(y_pred_flattened_encoded)
        y_pred = ak.unflatten(y_pred_flattened, counts)

        return y_pred

    def score(self, X: RaggedStringArray) -> ak.Array:
        assert self._model, "Model not set for predicting, train first"
        assert self._label_encoder, "Encoder not set for predicting, train first"

        counts = ak.num(X)

        X_feat = [featurize_sentence(sentence) for sentence in X]
        X_feat_flattened = ak.flatten(X_feat).to_list()
        X_feat_flattened_encoded = self._vectorizer.transform(X_feat_flattened)
        X_feat_flattened_encoded = np.nan_to_num(X_feat_flattened_encoded)

        y_proba_flattened = self._model.predict_proba(X_feat_flattened_encoded)
        y_scores_flattened = np.max(y_proba_flattened, axis=1)
        y_scores = ak.unflatten(y_scores_flattened, counts)

        assert len(X_feat_flattened) == len(y_scores_flattened)
        assert len(y_scores) == len(X)

        return y_scores

    def predict_proba(self, X: RaggedStringArray) -> ak.Array:
        assert self._model, "Model not set for predicting, train first"
        assert self._label_encoder, "Encoder not set for predicting, train first"

        counts = ak.num(X)

        X_feat = [featurize_sentence(sentence) for sentence in X]
        X_feat_flattened = ak.flatten(X_feat).to_list()
        X_feat_flattened_encoded = self._vectorizer.transform(X_feat_flattened)
        X_feat_flattened_encoded = np.nan_to_num(X_feat_flattened_encoded)

        y_proba_flattened = self._model.predict_proba(X_feat_flattened_encoded)
        y_proba = ak.unflatten(y_proba_flattened, counts)

        return y_proba

    def label_encoder(self) -> LabelEncoder:
        return self._label_encoder

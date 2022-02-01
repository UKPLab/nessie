from typing import Dict, List, Optional

import awkward as ak
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn_crfsuite import CRF

from nessie.models import SequenceTagger
from nessie.models.tagging.util import featurize_sentence
from nessie.types import RaggedStringArray


class CrfSequenceTagger(SequenceTagger):
    def __init__(self, all_possible_transitions: bool = False):
        self._model: Optional[CRF] = None
        self._all_possible_transitions = all_possible_transitions

    def fit(self, X: RaggedStringArray, y: RaggedStringArray):
        crf = CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=self._all_possible_transitions,
        )

        X_feat = [featurize_sentence(sentence) for sentence in X]

        crf.fit(X_feat, y)

        self._model = crf

    def predict(self, X: RaggedStringArray) -> ak.Array:
        assert self._model, "Model not set for predicting, train first"
        X_feat = [featurize_sentence(sentence) for sentence in X]
        return ak.Array(self._model.predict(X_feat))

    def score(self, X: RaggedStringArray) -> ak.Array:
        assert self._model, "Model not set for scoring, train first"
        X_feat = [featurize_sentence(sentence) for sentence in X]

        predictions = self._model.predict(X_feat)
        marginals = self._model.predict_marginals(X_feat)

        assert len(predictions) == len(marginals)

        result = np.empty(len(predictions), dtype=object)
        for i, (predictions_in_sentence, marginals_in_sentence) in enumerate(zip(predictions, marginals)):
            scores = []
            assert len(predictions_in_sentence) == len(marginals_in_sentence)
            for p, m in zip(predictions_in_sentence, marginals_in_sentence):

                scores.append(m[p])

            result[i] = scores

        return ak.Array(result)

    def predict_proba(self, X: RaggedStringArray) -> ak.Array:
        assert self._model, "Model not set for scoring, train first"
        X_feat = [featurize_sentence(sentence) for sentence in X]

        probs: List[List[Dict[str, float]]] = self._model.predict_marginals(X_feat)
        le = self.label_encoder()

        assert len(X) == len(probs)

        result = np.empty(len(probs), dtype=object)
        for i, sentence in enumerate(probs):
            current = []
            for token in sentence:
                scores_for_token = np.empty(len(le.classes_))
                for label, score in token.items():
                    idx = le.transform((label,))
                    scores_for_token[idx] = score
                current.append(scores_for_token)
            result[i] = current

        return ak.Array(result)

    def label_encoder(self) -> LabelEncoder:
        assert self._model, "Model not set, train first"

        le = LabelEncoder()
        le.classes_ = np.array(self._model.classes_)
        return le

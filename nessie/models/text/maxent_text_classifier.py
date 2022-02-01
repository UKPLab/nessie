from sklearn.linear_model import LogisticRegression

from nessie.models.featurizer import SentenceEmbedder
from nessie.models.text.sklean_text_classifier import SklearnTextClassifier
from nessie.util import RANDOM_STATE


class MaxEntTextClassifier(SklearnTextClassifier):
    def __init__(self, embedder: SentenceEmbedder, max_iter=10000):
        super().__init__(lambda: LogisticRegression(max_iter=max_iter, random_state=RANDOM_STATE), embedder)

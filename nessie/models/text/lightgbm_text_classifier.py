from lightgbm import LGBMClassifier

from nessie.models.featurizer import SentenceEmbedder
from nessie.models.text.sklean_text_classifier import SklearnTextClassifier
from nessie.util import RANDOM_STATE


class LgbmTextClassifier(SklearnTextClassifier):
    def __init__(self, embedder: SentenceEmbedder):
        super().__init__(lambda: LGBMClassifier(random_state=RANDOM_STATE), embedder)

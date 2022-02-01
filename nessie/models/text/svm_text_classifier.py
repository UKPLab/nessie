from sklearn.svm import SVC

from nessie.models.featurizer import SentenceEmbedder
from nessie.models.text.sklean_text_classifier import SklearnTextClassifier


class SvmTextClassifier(SklearnTextClassifier):
    def __init__(self, embedder: SentenceEmbedder):
        super().__init__(lambda: SVC(probability=True), embedder)

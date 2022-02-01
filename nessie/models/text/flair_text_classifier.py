import logging
import tempfile
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from flair.data import Dictionary, Sentence
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import BytePairEmbeddings, DocumentRNNEmbeddings, WordEmbeddings
from flair.models import TextClassifier as FTextClassifier
from flair.trainers import ModelTrainer
from sklearn.preprocessing import LabelEncoder, normalize

from nessie.models import TextClassifier
from nessie.types import StringArray
from nessie.util import tempinput


class FlairTextClassifier(TextClassifier):
    LABEL_TYPE = "class"

    def __init__(self, verbose: bool = True, max_epochs: int = 150, batch_size: int = 64):
        self._model: Optional[FTextClassifier] = None
        self._verbose = verbose
        self._max_epochs = max_epochs
        self._batch_size = batch_size

        if not self._verbose:
            logger = logging.getLogger("flair")
            # logger.setLevel(logging.WARN)

    def fit(self, X: StringArray, y: StringArray):
        corpus = self._build_corpus(X, y)
        label_dict = corpus.make_label_dictionary(label_type=self.LABEL_TYPE)

        model = self._build_model(label_dict)
        trainer = ModelTrainer(model, corpus)

        with tempfile.TemporaryDirectory() as tmpdirname:
            trainer.train(
                tmpdirname,
                learning_rate=0.1,
                mini_batch_size=self._batch_size,
                anneal_factor=0.5,
                patience=5,
                max_epochs=self._max_epochs,
                train_with_dev=True,
                num_workers=0,
                embeddings_storage_mode="gpu",
            )

        self._model = model

    def predict(self, X: StringArray) -> npt.NDArray[str]:
        assert self._model, "Model not set for predicting, train first"

        sentences = [Sentence(s) for s in X]
        self._model.predict(sentences, verbose=self._verbose, mini_batch_size=self._batch_size * 2)

        return np.array([s.labels[0].value for s in sentences])

    def score(self, X: StringArray) -> npt.NDArray[float]:
        assert self._model, "Model not set for predicting, train first"

        sentences = [Sentence(s) for s in X]
        self._model.predict(sentences, mini_batch_size=self._batch_size * 2)

        return np.array([s.labels[0].score for s in sentences])

    def predict_proba(self, X: StringArray) -> npt.NDArray[float]:
        assert self._model, "Model not set for predicting, train first"

        sentences = [Sentence(s) for s in X]
        self._model.predict(sentences, return_probabilities_for_all_classes=True, mini_batch_size=self._batch_size * 2)

        result = np.array([[label.score for label in sentence.labels[1:]] for sentence in sentences])
        result = normalize(result, norm="l1", axis=1)
        return result

    def _build_corpus(self, X: StringArray, y: StringArray) -> CSVClassificationCorpus:
        # flair can only load corpora from files, so we need to hack things up

        df = pd.DataFrame({"text": X, "label": y})

        s = ""
        for sentence, label in zip(X, y):
            s += f"{sentence}\t{label}\n"
        s = s.strip()

        with tempinput(s) as f:
            column_name_map = {0: "text", 1: f"label_{self.LABEL_TYPE}"}

            corpus = CSVClassificationCorpus(
                f.parent,
                column_name_map,
                train_file=f.name,
                skip_header=False,
                max_tokens_per_doc=512,
                delimiter="\t",
                in_memory=True,
                label_type=self.LABEL_TYPE,
            )

        return corpus

    def _build_model(self, label_dictionary: Dictionary) -> FTextClassifier:
        embeddings = [WordEmbeddings("glove"), BytePairEmbeddings("en")]

        document_embeddings = DocumentRNNEmbeddings(embeddings, hidden_size=256)

        model = FTextClassifier(document_embeddings, label_dictionary=label_dictionary, label_type="class")

        return model

    def label_encoder(self) -> LabelEncoder:
        assert self._model, "Model not set, train first"

        le = LabelEncoder()

        # The [1:-2] is there to remove <unk>
        le.classes_ = np.array([label.decode("utf-8") for label in self._model.label_dictionary.idx2item[1:]])
        return le

    def has_dropout(self) -> bool:
        return False

    def use_dropout(self, is_activated: bool):
        assert self.has_dropout()

        if is_activated:
            self._model.train()
        else:
            self._model.eval()

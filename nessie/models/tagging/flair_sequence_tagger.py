import logging
import tempfile
from typing import Optional

import awkward as ak
import numpy as np
from flair.data import Dictionary, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import BytePairEmbeddings, StackedEmbeddings, WordEmbeddings
from flair.models import SequenceTagger as FSequenceTagger
from flair.trainers import ModelTrainer
from sklearn.preprocessing import LabelEncoder, normalize

from nessie.models import SequenceTagger
from nessie.types import RaggedStringArray
from nessie.util import tempinput


class FlairSequenceTagger(SequenceTagger):
    TAG_TYPE = "tag"

    def __init__(self, verbose: bool = True, max_epochs: int = 150, batch_size: int = 64):
        self._model: Optional[FSequenceTagger] = None
        self._verbose = verbose
        self._max_epochs = max_epochs
        self._batch_size = batch_size

        if not self._verbose:
            logger = logging.getLogger("flair")
            logger.setLevel(logging.WARN)

    def fit(self, X: RaggedStringArray, y: RaggedStringArray):
        corpus = self._build_corpus(X, y)
        tag_dictionary = corpus.make_label_dictionary(label_type=self.TAG_TYPE)

        model = self._build_model(tag_dictionary)
        trainer = ModelTrainer(model, corpus)

        with tempfile.TemporaryDirectory() as tmpdirname:
            trainer.train(
                tmpdirname,
                learning_rate=0.1,
                mini_batch_size=self._batch_size,
                max_epochs=self._max_epochs,
                train_with_dev=True,
            )

        self._model = model

    def predict(self, X: RaggedStringArray) -> ak.Array:
        assert self._model, "Model not set for predicting, train first"

        sentences = [Sentence(list(s)) for s in X]

        self._model.predict(sentences, verbose=self._verbose, mini_batch_size=self._batch_size * 2)

        return ak.Array([[t.get_tag(self.TAG_TYPE).value for t in s] for s in sentences])

    def score(self, X: RaggedStringArray) -> ak.Array:
        assert self._model, "Model not set for predicting, train first"

        sentences = [Sentence(list(s)) for s in X]

        self._model.predict(sentences, verbose=self._verbose, mini_batch_size=self._batch_size * 2)

        return ak.Array([[t.get_tag(self.TAG_TYPE).score for t in s] for s in sentences])

    def predict_proba(self, X: RaggedStringArray) -> ak.Array:
        assert self._model, "Model not set for predicting, train first"

        sentences = [Sentence(list(s)) for s in X]

        self._model.predict(sentences, verbose=self._verbose, all_tag_prob=True, mini_batch_size=self._batch_size * 2)

        get_score = lambda token: [label.score for label in token.get_tags_proba_dist(self.TAG_TYPE)[1:-2]]

        # The [1:-2] is there to remove <unk>, <START> and <STOP> tokens
        result = []
        for s in sentences:
            probs = np.array([get_score(t) for t in s])
            probs = normalize(probs, norm="l1", axis=1)
            result.append(probs)

        return ak.Array(result)

    def label_encoder(self) -> LabelEncoder:
        assert self._model, "Model not set, train first"

        le = LabelEncoder()
        # The [1:-2] is there to remove <unk>, <START> and <STOP> tokens

        le.classes_ = np.array([label.decode("utf-8") for label in self._model.tag_dictionary.idx2item[1:-2]])
        return le

    def _build_corpus(self, X: RaggedStringArray, y: RaggedStringArray) -> ColumnCorpus:
        # Flair can only load corpora from files, so we need to hack things up

        lines = []
        for tokens_in_sentence, tags_in_sentence in zip(X, y):
            assert len(tokens_in_sentence) == len(tags_in_sentence)

            for token, tag in zip(tokens_in_sentence, tags_in_sentence):
                lines.append(f"{token.strip()}\t{tag.strip()}")

            # Space between two sentence blocks
            lines.append("")

        s = "\n".join(lines)

        with tempinput(s) as f:
            columns = {0: "tokens", 1: self.TAG_TYPE}
            corpus = ColumnCorpus(
                f.parent, columns, train_file=f.name, sample_missing_splits="only_dev", in_memory=True
            )

        return corpus

    def _build_model(self, tag_dictionary: Dictionary) -> FSequenceTagger:
        embedding_types = [WordEmbeddings("glove"), BytePairEmbeddings("en")]
        embeddings = StackedEmbeddings(embeddings=embedding_types)
        return FSequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=self.TAG_TYPE,
            use_crf=True,
        )

    def has_dropout(self) -> bool:
        return True

    def use_dropout(self, is_activated: bool):
        assert self.has_dropout()

        if is_activated:
            self._model.train()
        else:
            self._model.eval()

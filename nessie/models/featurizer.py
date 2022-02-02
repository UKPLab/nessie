from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

import awkward as ak
import numpy as np
import numpy.typing as npt
import torch
from cleantext import clean
from diskcache import Cache
from flair.data import Sentence
from flair.embeddings import TokenEmbeddings
from more_itertools import chunked
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from slurmee import slurmee

from nessie.config import SBERT_MODEL_NAME
from nessie.types import RaggedStringArray, StringArray
from nessie.util import my_backoff


class SentenceEmbedder:
    def embed(self, sentences: StringArray) -> npt.NDArray[str]:
        raise NotImplementedError()

    def get_dimension(self) -> int:
        raise NotImplementedError()

    def train(self):
        pass

    def eval(self):
        pass


class CachedSentenceTransformer(SentenceEmbedder):
    _caches = defaultdict(dict)

    @my_backoff()
    def __init__(self, model_name: str = SBERT_MODEL_NAME, cache_dir: Optional[Path] = None):
        super().__init__()
        self._model = SentenceTransformer(model_name)

        # Do not use disk cache when running in slurm
        if slurmee.get_job_id() or cache_dir is None:
            self._cache = CachedSentenceTransformer._caches[model_name]
        else:
            self._cache = Cache(cache_dir / model_name)

    def embed(self, sentences: StringArray) -> npt.NDArray[str]:
        not_in_cache = [s for s in sentences if s not in self._cache]
        vecs_not_in_cache = list(self._model.encode(not_in_cache))

        assert len(not_in_cache) == len(vecs_not_in_cache)

        for s, vec in zip(not_in_cache, vecs_not_in_cache):
            self._cache[s] = vec

        return np.array([self._cache[s].squeeze() for s in sentences])

    def get_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


class TfIdfSentenceEmbedder(SentenceEmbedder):
    def __init__(self):
        self._train = True
        self._vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words="english")

    def embed(self, sentences: StringArray) -> npt.NDArray[str]:
        sentences = self._clean(sentences)
        if self._train:
            return self._vectorizer.fit_transform(sentences)
        else:
            return self._vectorizer.transform(sentences)

    def get_dimension(self) -> int:
        return len(self._vectorizer.vocabulary)

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def _clean(self, sentences: List[str]) -> List[str]:
        return [clean(s, clean_all=True) for s in sentences]


class FlairTokenEmbeddingsWrapper:
    def __init__(self, embedder: TokenEmbeddings, batch_size: int = 8):
        self._embedder = embedder
        self._batch_size = batch_size

    def embed(self, sentences: RaggedStringArray, flat: bool = False) -> Union[npt.NDArray[float], ak.Array]:
        flair_sentences = [Sentence(list(x)) for x in sentences]

        embedded = []
        for sentence_batch in chunked(flair_sentences, self._batch_size):
            self._embedder.embed(sentence_batch)

            embedded.extend(token.embedding.detach().cpu() for s in sentence_batch for token in s)

            del sentence_batch
            torch.cuda.empty_cache()

        embedded = np.vstack(embedded)
        sentences = ak.Array(sentences)
        counts = ak.num(sentences)
        num_instances = ak.sum(counts)

        assert embedded.shape == (num_instances, self._embedder.embedding_length)

        if flat:
            return np.asarray(embedded)
        else:
            return ak.unflatten(embedded, counts)

    @property
    def embedding_dim(self) -> int:
        return self._embedder.embedding_length

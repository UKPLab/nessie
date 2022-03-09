from pathlib import Path
from typing import Set

import awkward as ak
import numpy as np
import numpy.typing as npt
import pytest
from flair.embeddings import TransformerWordEmbeddings
from numpy.random import default_rng

from nessie.dataloader import SequenceLabelingDataset, TextClassificationDataset
from nessie.models.featurizer import (
    CachedSentenceTransformer,
    FlairTokenEmbeddingsWrapper,
    TfIdfSentenceEmbedder,
)
from nessie.models.tagging import (
    CrfSequenceTagger,
    DummySequenceTagger,
    FlairSequenceTagger,
    MaxEntSequenceTagger,
    TransformerSequenceTagger,
)
from nessie.models.text import (
    DummyTextClassifier,
    FastTextTextClassifier,
    FlairTextClassifier,
    LgbmTextClassifier,
    MaxEntTextClassifier,
    TransformerTextClassifier,
)
from nessie.noise import flipped_label_noise
from nessie.util import RANDOM_STATE

PATH_ROOT: Path = Path(__file__).resolve().parents[1]

# Default params

NUM_INSTANCES = 32
NUM_LABELS = 4
NUM_MODELS = 3
T = 10

# Example data
PATH_EXAMPLE_DATA: Path = PATH_ROOT / "example_data"
PATH_EXAMPLE_DATA_TEXT: Path = PATH_EXAMPLE_DATA / "easy_text.tsv"
PATH_EXAMPLE_DATA_TOKEN: Path = PATH_EXAMPLE_DATA / "easy_token.conll"
PATH_EXAMPLE_DATA_SPAN: Path = PATH_EXAMPLE_DATA / "easy_span.conll"

# Constants

BERT_BASE = "google/bert_uncased_L-2_H-128_A-2"
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


# Sequence Tagger


@pytest.fixture
def crf_sequence_tagger_fixture():
    return CrfSequenceTagger()


@pytest.fixture
def dummy_sequence_tagger_fixture():
    return DummySequenceTagger()


@pytest.fixture
def flair_sequence_tagger_fixture():
    max_epochs = 1
    batch_size = 32
    return FlairSequenceTagger(max_epochs=max_epochs, batch_size=batch_size)


@pytest.fixture
def maxent_sequence_tagger_fixture():
    return MaxEntSequenceTagger(max_iter=100)


@pytest.fixture
def transformer_sequence_tagger_fixture():
    max_epochs = 2
    return TransformerSequenceTagger(max_epochs=max_epochs, batch_size=BATCH_SIZE, model_name=BERT_BASE)


# Text classifier


@pytest.fixture
def dummy_text_classifier_fixture():
    return DummyTextClassifier()


@pytest.fixture
def fasttext_text_classifier_fixture():
    return FastTextTextClassifier()


@pytest.fixture
def flair_text_classifier_fixture():
    max_epochs = 2
    return FlairTextClassifier(max_epochs=max_epochs, batch_size=BATCH_SIZE)


@pytest.fixture
def lightgbm_tfidf_text_classifier_fixture():
    return LgbmTextClassifier(TfIdfSentenceEmbedder())


@pytest.fixture
def lightgbm_sbert_text_classifier_fixture(sentence_embedder_fixture):
    return LgbmTextClassifier(sentence_embedder_fixture)


@pytest.fixture
def maxent_tfidf_text_classifier_fixture():
    return MaxEntTextClassifier(TfIdfSentenceEmbedder(), max_iter=100)


@pytest.fixture
def maxent_sbert_text_classifier_fixture(sentence_embedder_fixture):
    return MaxEntTextClassifier(sentence_embedder_fixture, max_iter=100)


@pytest.fixture
def transformer_text_classifier_fixture():
    max_epochs = 2
    return TransformerTextClassifier(max_epochs=max_epochs, batch_size=BATCH_SIZE, model_name=BERT_BASE)


# Embedder


@pytest.fixture(scope="session")
def sentence_embedder_fixture() -> CachedSentenceTransformer:
    return CachedSentenceTransformer(SBERT_MODEL_NAME)


@pytest.fixture(scope="session")
def token_embedder_fixture() -> FlairTokenEmbeddingsWrapper:
    return FlairTokenEmbeddingsWrapper(TransformerWordEmbeddings(BERT_BASE))


# Datasets


def generate_random_text_classification_dataset(num_instances: int, num_labels: int) -> TextClassificationDataset:
    rng = default_rng(seed=RANDOM_STATE)

    possible_texts = [
        "I like cookies.",
        "I like reindeer.",
        "He likes sunsets and long strolls on the beach.",
        "He does not like Mondays.",
    ]

    texts = rng.choice(possible_texts, num_instances)

    gold_labels_encoded = rng.integers(0, num_labels, (num_instances,))
    f = lambda x: chr(ord("A") + x)
    gold_labels = np.vectorize(f)(gold_labels_encoded)

    noisy_labels = flipped_label_noise(gold_labels, 0.05)

    return TextClassificationDataset(texts, gold_labels, noisy_labels)


def generate_random_pos_tagging_dataset(num_instances: int, num_labels: int) -> SequenceLabelingDataset:
    rng = default_rng(seed=RANDOM_STATE)

    possible_sentences = [
        ["I", "love", "cats", "very", "much", "."],
        ["I", "cherish", "cats", "very", "much", "."],
        ["The", "cats", "I", "like", "are", "very", "much", "different", "in", "their", "appearance", "."],
    ]

    sentences = ak.Array(rng.choice(np.asarray(possible_sentences, dtype=object), num_instances))
    counts = ak.num(sentences)

    gold_labels_encoded_flat = rng.integers(0, num_labels, (ak.sum(counts),))
    f = lambda x: chr(ord("A") + x)
    gold_labels_flat = np.vectorize(f)(gold_labels_encoded_flat)

    noisy_labels_flat = flipped_label_noise(gold_labels_flat, 0.05)

    return SequenceLabelingDataset(
        sentences, ak.unflatten(gold_labels_flat, counts), ak.unflatten(noisy_labels_flat, counts)
    )


def get_random_ensemble_predictions(num_instances: int, tagset: Set[str], num_models: int) -> npt.NDArray[str]:
    rng = default_rng(seed=RANDOM_STATE)

    tagset = list(tagset)

    encoded_labels = rng.integers(0, len(tagset), (num_models, num_instances))
    f = lambda x: tagset[x]
    labels = np.vectorize(f)(encoded_labels)

    return labels

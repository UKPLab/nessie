from pathlib import Path
from typing import Set

import awkward as ak
import numpy as np
import numpy.typing as npt
from numpy.random import default_rng
from sklearn.preprocessing import normalize

from nessie.dataloader import SequenceLabelingDataset, TextClassificationDataset
from nessie.noise import flipped_label_noise
from nessie.util import RANDOM_STATE

PATH_ROOT: Path = Path(__file__).resolve().parents[1]

# Example data
PATH_EXAMPLE_DATA: Path = PATH_ROOT / "example_data"
PATH_EXAMPLE_DATA_TEXT: Path = PATH_EXAMPLE_DATA / "easy_text.tsv"
PATH_EXAMPLE_DATA_TOKEN: Path = PATH_EXAMPLE_DATA / "easy_token.conll"
PATH_EXAMPLE_DATA_SPAN: Path = PATH_EXAMPLE_DATA / "easy_span.conll"


# Constants

BERT_BASE = "google/bert_uncased_L-2_H-128_A-2"
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


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

    sentences = ak.Array(rng.choice(possible_sentences, num_instances))
    counts = ak.num(sentences)

    gold_labels_encoded_flat = rng.integers(0, num_labels, (ak.sum(counts),))
    f = lambda x: chr(ord("A") + x)
    gold_labels_flat = np.vectorize(f)(gold_labels_encoded_flat)

    noisy_labels_flat = flipped_label_noise(gold_labels_flat, 0.05)

    return SequenceLabelingDataset(
        sentences, ak.unflatten(gold_labels_flat, counts), ak.unflatten(noisy_labels_flat, counts)
    )


def get_random_probabilities(num_instances: int, num_labels: int, seed: int = RANDOM_STATE) -> npt.NDArray[float]:
    rng = default_rng(seed=seed)
    probabilities = rng.random((num_instances, num_labels))
    probabilities = normalize(probabilities, norm="l1", axis=1)

    return probabilities


def get_random_repeated_probabilities(num_instances: int, num_labels: int, T: int) -> npt.NDArray[float]:
    result = []

    for i in range(T):
        probabilities = get_random_probabilities(num_instances, num_labels, seed=i + 1)
        probabilities = normalize(probabilities, norm="l1", axis=1)
        result.append(probabilities)

    result = np.asarray(result).swapaxes(0, 1)

    assert result.shape == (num_instances, T, num_labels)

    return result


def get_random_ensemble_predictions(num_instances: int, tagset: Set[str], num_models: int) -> npt.NDArray[str]:
    rng = default_rng(seed=RANDOM_STATE)

    tagset = list(tagset)

    encoded_labels = rng.integers(0, len(tagset), (num_models, num_instances))
    f = lambda x: tagset[x]
    labels = np.vectorize(f)(encoded_labels)

    return labels

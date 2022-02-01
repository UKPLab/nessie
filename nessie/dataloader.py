import re
from dataclasses import dataclass
from pathlib import Path
from typing import Set

import awkward as ak
import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class TextClassificationDataset:
    """Dataset containing texts, gold and noisy labels for tasks like sentiment analysis or topic classification.

    Args:
        texts: String sequence like List, numpy array, ...
        gold_labels: String sequence like List, numpy array, ...
        gold_labels: String sequence like List, numpy array, ...
    """

    texts: npt.NDArray[str]
    gold_labels: npt.NDArray[str]
    noisy_labels: npt.NDArray[str]

    def __post_init__(self):
        self.texts = np.asarray(self.texts, dtype=object)
        self.gold_labels = np.asarray(self.gold_labels, dtype=object)
        self.noisy_labels = np.asarray(self.noisy_labels, dtype=object)

        assert len(self.texts) == len(self.gold_labels) == len(self.noisy_labels)

    @property
    def tagset_noisy(self) -> Set[str]:
        return set(self.noisy_labels)

    @property
    def num_instances(self) -> int:
        return len(self.texts)

    def subset(self, n: int) -> "TextClassificationDataset":
        if n > self.num_instances:
            raise IndexError(f"Dataset only contains [{self.num_instances}] instances, but asked were [{n}")

        return TextClassificationDataset(
            self.texts[:n],
            self.gold_labels[:n],
            self.noisy_labels[:n],
        )


@dataclass
class SequenceLabelingDataset:
    """Dataset containing tokens, gold and noisy labels for tasks like POS tagging or NER.

    Args:
        sentences: List of list of strings
        gold_labels: List of list of strings
        noisy_labels: List of list of strings
    """

    sentences: ak.Array
    gold_labels: ak.Array
    noisy_labels: ak.Array

    def __post_init__(self):
        self.sentences = ak.Array(self.sentences)
        self.gold_labels = ak.Array(self.gold_labels)
        self.noisy_labels = ak.Array(self.noisy_labels)

        # Check that the nested dimensions have the same sizes
        assert ak.all(ak.num(self.sentences, axis=1) == ak.num(self.gold_labels, axis=1))
        assert ak.all(ak.num(self.sentences, axis=1) == ak.num(self.noisy_labels, axis=1))

    @property
    def tagset_noisy(self) -> Set[str]:
        return set(ak.flatten(self.noisy_labels))

    @property
    def num_sentences(self) -> int:
        return len(self.sentences)

    @property
    def num_instances(self) -> int:
        return len(ak.flatten(self.noisy_labels))

    def subset(self, n: int) -> "SequenceLabelingDataset":
        if n > self.num_sentences:
            raise IndexError(f"Dataset only contains [{self.num_sentences}] sentences, but asked were [{n}")

        return SequenceLabelingDataset(
            self.sentences[:n],
            self.gold_labels[:n],
            self.noisy_labels[:n],
        )


def load_text_classification_tsv(path: Path) -> TextClassificationDataset:
    df = pd.read_csv(path, sep="\t", names=["texts", "gold", "noisy"])
    result = TextClassificationDataset(texts=df["texts"], gold_labels=df["gold"], noisy_labels=df["noisy"])
    return result


def load_sequence_labeling_dataset(path: Path) -> SequenceLabelingDataset:
    with path.open() as f:
        data = f.read().strip()

    sentences = []
    gold_labels = []
    noisy_labels = []

    for sentence_id, block in enumerate(re.split(r"\n\s*?\n", data)):
        tokens = []
        gold = []
        noisy = []

        for line in block.strip().split("\n"):
            token, gold_label, noisy_label = line.rstrip().split("\t")
            tokens.append(token)
            gold.append(gold_label)
            noisy.append(noisy_label)

        sentences.append(tokens)
        gold_labels.append(gold)
        noisy_labels.append(noisy)

    dataset = SequenceLabelingDataset(sentences=sentences, gold_labels=gold_labels, noisy_labels=noisy_labels)

    return dataset

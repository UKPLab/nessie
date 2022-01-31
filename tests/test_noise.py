import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score

from nessie.noise import flipped_label_noise


def test_flipped_label_noise():
    dataset = load_dataset("trec")["train"]

    labels = dataset["label-coarse"]

    p = 0.1
    noisy_labels = flipped_label_noise(labels, p)

    score = accuracy_score(labels, noisy_labels)

    assert np.isclose(np.sum(score), 1.0 - p, 0.02)

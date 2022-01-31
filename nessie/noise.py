# https://www.microsoft.com/en-us/research/uploads/prod/2020/12/mlc_aaai21_zheng_camera.pdf
# Datasets and Setup
from typing import List

import numpy as np
from sklearn import preprocessing

from nessie.util import RANDOM_STATE


def uniform_label_noise(dataset, p: float):
    """For a dataset with C classes,
    a clean example with true label y is randomly corrupted to
    all possible classes y' with probability ρ/C
    and stays in its  original label with probability 1 − ρ.
     (Note the corrupted label might also happen to be the original label, hence the
    label has probability of 1 − ρ +  ρ / C to stay uncorrupted.)"""
    assert 0 <= p <= 1

    raise NotImplementedError()


def flipped_label_noise(labels: List[str], p: float, seed: int = RANDOM_STATE):
    """
     For a dataset with C classes,
    a clean example with true label y is randomly flipped to one
    of the rest C − 1 classes with probability ρ and stays in its
    original label with probability 1 − ρ.

    """
    assert 0 <= p <= 1

    rng = np.random.default_rng(seed)

    le = preprocessing.LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    N = len(labels)
    C = len(le.classes_)

    # We compute the offsets by which we want to flip.
    # We assign a 1-p probability to have offset 0, thereby
    # not flipping and a probability of p / (C -1) to all other
    # labels.
    probs = np.repeat(p / (C - 1), C)
    probs[0] = 1 - p

    assert np.isclose(np.sum(probs), 1.0)

    flips = rng.choice(C, N, p=probs)
    noisy_classes = (encoded_labels + flips) % C

    return le.inverse_transform(noisy_classes)

from typing import Dict

import numpy as np
import numpy.typing as npt
from scipy.spatial import distance
from sklearn.neighbors import LocalOutlierFactor

from nessie.detectors.error_detector import Detector, DetectorKind
from nessie.types import FloatArray2D, StringArray


class MeanDistance(Detector):
    """Stefan Larson, Anish Mahendran, Andrew Lee, Jonathan K. Kummerfeld, Parker Hill, Michael A.
    Laurenzano, Johann Hauswald, Lingjia Tang, Jason Mars
    Outlier Detection for Improved Data Quality and Diversity in Dialog Systems
    In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics
    Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 517–527).

    https://www.aclweb.org/anthology/N19-1051.pdf
    """

    def __init__(self, metric: str = "euclidean"):
        self._metric = metric

    def score(self, labels: StringArray, embedded_instances: FloatArray2D, **kwargs) -> npt.NDArray[float]:
        """Compute the mean vector for each instance grouped by label, the outlier score for each instance is the
        distance to the mean vector for its label.

        For each class, do

        1. Generate a vector representation of each instance.
        2. Average vectors to get a mean representation.
        3. Calculate the distance of each instance from the mean.
        4. Rank by distance in ascending order.
        5. (Cut off the list, keeping only the top k% as outliers.

        Args:
            labels: a (num_instances, ) string sequence containing the noisy label for each instance
            embedded_instances: 2d numpy array of shape (num_items, encoding_dim)
        Returns:
            a (num_samples, ) numpy array containing the scores for each instance
        """

        num_instances, embedding_dim = embedded_instances.shape
        assert num_instances == len(labels)

        # Compute mean vectors for each class
        mean_vectors = {}

        labels = np.asarray(labels)
        unique_labels = set(labels)

        for label in unique_labels:
            vectors_with_label = embedded_instances[labels == label]
            mean_vector = np.mean(vectors_with_label, axis=0)

            assert mean_vector.shape == (embedding_dim,)
            mean_vectors[label] = mean_vector

        # Construct scores from each vector to its mean label vector

        if self._metric == "euclidean" or self._metric is None:
            scores = self._compute_distance_to_mean(embedded_instances, labels, mean_vectors, distance.euclidean)
        elif self._metric == "cosine":
            scores = self._compute_distance_to_mean(embedded_instances, labels, mean_vectors, distance.cosine)
        elif self._metric == "dot":
            scores = self._compute_distance_to_mean(
                embedded_instances, labels, mean_vectors, lambda u, v: -np.dot(u, v)
            )

        elif self._metric == "lof":
            scores = self._compute_lof(embedded_instances, labels)
        else:
            raise ValueError(f"Unknown distance metric [{self._metric}]")

        assert len(scores) == len(embedded_instances) == len(labels)

        return scores

    def _compute_distance_to_mean(
        self, vectors: np.ndarray, labels: np.ndarray, mean_vectors: Dict[int, np.ndarray], f
    ) -> np.ndarray:
        n = len(vectors)
        distances = np.empty(n)
        for i in range(n):
            vector = vectors[i]
            label = labels[i]
            mean_vector = mean_vectors[label]
            distances[i] = f(vector, mean_vector)

        return distances

    def _compute_lof(self, vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
        lof_scores = np.empty_like(labels, dtype=np.float)

        for label in np.unique(labels):
            vectors_with_label = vectors[labels == label]
            clf = LocalOutlierFactor()
            clf.fit(vectors_with_label)

            lof_scores[labels == label] = -clf.negative_outlier_factor_

        return lof_scores

    def error_detector_kind(self) -> DetectorKind:
        return DetectorKind.SCORER


def _debug_mean_distance():
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from scipy.spatial import distance
    from sklearn.datasets import make_blobs
    from sklearn.neighbors import LocalOutlierFactor

    N = 1000
    vectors, encoded_labels = make_blobs(n_samples=N, centers=2, n_features=2, random_state=0)

    mean_vectors = {}

    for label in np.unique(encoded_labels):
        vectors_with_label = vectors[encoded_labels == label]
        mean_vector = np.mean(vectors_with_label, axis=0)

        assert mean_vector.shape == (2,)
        mean_vectors[label] = mean_vector

    x = vectors[:, 0]
    y = vectors[:, 1]

    n = N
    euclidean_distances = np.empty(n)

    for i in range(n):
        vector = vectors[i]
        label = encoded_labels[i]
        mean_vector = mean_vectors[label]

        euclidean_distances[i] = distance.euclidean(vector, mean_vector)

    lof_scores = np.empty(n)

    for label in np.unique(encoded_labels):
        vectors_with_label = vectors[encoded_labels == label]
        clf = LocalOutlierFactor()
        clf.fit(vectors_with_label)

        lof_scores[encoded_labels == label] = clf.negative_outlier_factor_

    plt.figure()
    sns.scatterplot(x=x, y=y, hue=euclidean_distances)

    for mv in mean_vectors.values():
        print(mv)
        plt.scatter(mv[0], mv[1], c="black")

    plt.figure()
    sns.scatterplot(x=x, y=y, hue=lof_scores, style=encoded_labels)

    plt.show()


if __name__ == "__main__":
    _debug_mean_distance()

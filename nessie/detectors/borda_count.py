import numpy as np
from scipy.stats import rankdata

from nessie.detectors.error_detector import Detector, DetectorKind


class BordaCount(Detector):
    """Aggregate ranking scores via Borda count. Given a matrix of kxn instances,
    where k is the number of scorers and n the number of instances, for each scorer,
    assign the highest rank a score of n, the second largest n-1 and so on. Then
    sum up the scores and rank for the newly computed scores.

    This has been described first in

    Inconsistencies in Crowdsourced Slot-Filling Annotations: A Typology and Identification Methods
    Stefan Larson, Adrian Cheung, Anish Mahendran, Kevin Leach, Jonathan K. Kummerfeld
    COLING 2020
    """

    def score(self, ensemble_scores: np.ndarray) -> np.ndarray:
        """Aggregates the given ensemble scores obtained previously from several scorers into
        one by means of Borda count.

        Args:
            ensemble_scores: a (num_scorers, num_samples) numpy array

        Returns:
            scores: a (num_samples,) numpy array containing the aggregated scores
        """
        ranks = rankdata(ensemble_scores, method="ordinal", axis=1)
        assert ranks.shape == ensemble_scores.shape

        scores = np.sum(ranks, axis=0)
        assert len(scores) == ensemble_scores.shape[1]

        return scores

    def error_detector_kind(self) -> DetectorKind:
        return DetectorKind.SCORER

    def uses_probabilities(self) -> bool:
        return True

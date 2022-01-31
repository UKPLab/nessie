import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import LabelEncoder

from nessie.detectors.error_detector import DetectorKind, ModelBasedDetector
from nessie.types import StringArray


class ClassificationUncertainty(ModelBasedDetector):
    """Given a distribution over labels for each instance in form of a (num_instances, num_labels)
    numpy array, the resulting score is just 1 - the probability of the noisy label specified.

    Halteren, Hans van. “The Detection of Innessie in Manually Tagged Text.”
    In: Proceedings of the COLING-2000 Workshop on Linguistically Interpreted Corpora, 48–55.
    Centre Universitaire, Luxembourg: International Committee on Computational Linguistics, 2000.
    https://www.aclweb.org/anthology/W00-1907.

    See also

    Dan Hendrycks and Kevin Gimpel
    A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks
    In: Proceedings of International Conference on Learning Representations
    """

    def score(
        self, labels: StringArray, probabilities: npt.NDArray[float], le: LabelEncoder, **kwargs
    ) -> npt.NDArray[float]:
        """Scores the input according to their classification uncertainty.

        Args:
            labels: a (num_instances, ) string sequence containing the noisy label for each instance
            probabilities: a (num_instances, num_classes) numpy array obtained from a machine learning model
            le: the label encoder that allows converting the probabilities back to labels

        Returns:
            scores: a (num_instances,) numpy array containing the resulting scores
        """

        labels_encoded = le.transform(labels)
        scores = probabilities[np.arange(len(probabilities)), labels_encoded]

        return 1.0 - scores

    def error_detector_kind(self):
        return DetectorKind.SCORER

    def uses_probabilities(self) -> bool:
        return True

import numpy as np
import numpy.typing as npt
from cleanlab.pruning import get_noise_indices
from sklearn.preprocessing import LabelEncoder

from nessie.detectors.error_detector import DetectorKind, ModelBasedDetector
from nessie.types import StringArray


class ConfidentLearning(ModelBasedDetector):
    """{Confident Learning} estimates the joint distribution of noisy and true labels.
     A threshold is then learnt (the average self-confidence), instances whose computed
     probability of having the correct label is below the respective threshold are flagged
     as erroneous.

    Curtis G. Northcutt, Lu Jiang, & Isaac L. Chuang (2021).
    Confident Learning: Estimating Uncertainty in Dataset Labels.
    Journal of Artificial Intelligence Research (JAIR), 70, 1373–1411.
    https://github.com/cgnorthcutt/cleanlab
    """

    def score(
        self, labels: StringArray, probabilities: npt.NDArray[float], le: LabelEncoder, **kwargs
    ) -> npt.NDArray[bool]:
        """Flags the input via confident learning.

        Args:
            labels: a (num_instances, ) string sequence containing the noisy label for each instance
            probabilities: a (num_instances, num_classes) numpy array obtained from a machine learning model
            le: the label encoder that allows converting the probabilities back to labels
        Returns:
            scores: a (num_instances,) numpy array of bools containing the flags after using CL
        """

        s = le.transform(labels)
        K = len(le.classes_)

        assert len(s) == len(labels)
        assert probabilities.shape == (len(labels), K)

        # Internally, the code of cleanlab counts unique labels, if you e.g. have 20 potential labels
        # and only 19 are actually assigned in `s`, then it would break.
        # See https://github.com/cleanlab/cleanlab/issues/85
        # If that happens, we remap the labels again to 0..len(seen_labels) with a temporary label encoder
        # and then select only the rows of probabilities with seen label.
        seen_labels = np.unique(s)

        if K != len(seen_labels):
            temp_probabilities = probabilities[:, np.unique(seen_labels)]

            temp_le = LabelEncoder()
            temp_le.classes_ = np.array(list(sorted(seen_labels)))

            temp_s = temp_le.transform(s)

            scores = get_noise_indices(s=temp_s, psx=temp_probabilities)
        else:
            scores = get_noise_indices(s=s, psx=probabilities)

        return scores.astype(bool)

    def error_detector_kind(self):
        return DetectorKind.FLAGGER

    def uses_probabilities(self) -> bool:
        return True

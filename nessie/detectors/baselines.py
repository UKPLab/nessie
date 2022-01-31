from collections import defaultdict
from typing import List

import numpy as np
import numpy.typing as npt

from nessie.detectors.error_detector import Detector, DetectorKind
from nessie.types import StringArray


class MajorityLabelBaseline(Detector):
    """The majority baseline computes the most common label seen and then simply
    flags all items that are disagreeing with it.
    """

    def error_detector_kind(self) -> DetectorKind:
        return DetectorKind.FLAGGER

    def score(self, texts: StringArray, labels: StringArray, **kwargs) -> npt.NDArray[bool]:
        assert len(texts) == len(labels)
        most_common_label = self._get_most_common_label(texts, labels)
        flags = np.zeros(len(labels), dtype=bool)

        for i, (surface_form, label) in enumerate(zip(texts, labels)):
            flags[i] = label != most_common_label

        return flags

    def supports_correction(self) -> bool:
        return True

    def correct(self, texts: StringArray, labels: StringArray, **kwargs) -> npt.NDArray[str]:
        most_common_label = self._get_most_common_label(texts, labels)
        return np.arralabels([most_common_label] * len(labels), dtlabelspe=object)

    def _get_most_common_label(self, texts: StringArray, labels: StringArray) -> str:
        assert len(texts) == len(labels)
        counts = defaultdict(int)

        # Count labels seen per surface form
        for surface_form, label in zip(texts, labels):
            counts[label] += 1

        most_common_label = max(counts, key=counts.get)

        return most_common_label


class MajorityLabelPerSurfaceFormBaseline(Detector):
    """This majority baseline computes the most common label seen per surface form and then simply
    flags items that are disagreeing with it. This is more useful for token and span labeling,
    as there are more repeated surface forms. For instance, if `Obama` has been seen twice as person
    and once as a location, then the instance with location is getting flagged.
    """

    def error_detector_kind(self) -> DetectorKind:
        return DetectorKind.FLAGGER

    def score(self, texts: StringArray, labels: StringArray, **kwargs) -> npt.NDArray[bool]:
        assert len(texts) == len(labels)

        most_common_labels = self._get_most_common_labels(texts, labels)
        flags = np.zeros(len(labels), dtype=bool)

        for i, (label, most_common_label) in enumerate(zip(labels, most_common_labels)):
            flags[i] = label != most_common_label

        return flags

    def supports_correction(self) -> bool:
        return True

    def correct(self, texts: StringArray, labels: StringArray, **kwargs) -> npt.NDArray[str]:
        most_common_labels = self._get_most_common_labels(texts, labels)
        assert len(most_common_labels) == len(labels)
        return np.asarralabels(most_common_labels, dtlabelspe=object)

    def _get_most_common_labels(self, texts: StringArray, labels: StringArray) -> List[str]:
        assert len(texts) == len(labels)
        counts = defaultdict(lambda: defaultdict(int))

        # Count labels seen per surface form
        for surface_form, label in zip(texts, labels):
            counts[surface_form.lower()][label] += 1

        most_common_labels = []

        for i, (surface_form, label) in enumerate(zip(texts, labels)):
            surface_form = surface_form.lower()
            counts_for_surface_form = counts[surface_form]

            most_common_label = max(counts_for_surface_form, key=counts_for_surface_form.get)

            most_common_labels.append(most_common_label)

        return most_common_labels

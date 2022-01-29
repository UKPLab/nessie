import pytest

import awkward as ak


from nessie.dataloader import load_text_classification_tsv, load_sequence_labeling_dataset
from nessie.detectors import MajorityLabelBaseline, MajorityLabelPerSurfaceFormBaseline
from nessie.detectors import Detector
from tests.fixtures import PATH_EXAMPLE_DATA_TEXT, PATH_EXAMPLE_DATA_TOKEN, PATH_EXAMPLE_DATA_SPAN


@pytest.fixture
def majority_label_baseline_fixture() -> MajorityLabelBaseline:
    return MajorityLabelBaseline()


@pytest.fixture
def majority_label_per_surface_form_baseline_fixture() -> MajorityLabelPerSurfaceFormBaseline:
    return MajorityLabelPerSurfaceFormBaseline()


@pytest.mark.parametrize("detector_fixture", [
    'majority_label_baseline_fixture',
])
def test_detectors_for_text_classification(detector_fixture, request):
    detector: Detector = request.getfixturevalue(detector_fixture)
    ds = load_text_classification_tsv(PATH_EXAMPLE_DATA_TEXT)

    params = {
        'texts': ds.texts,
        'labels': ds.noisy_labels
    }

    detector.score(**params)


@pytest.mark.parametrize("detector_fixture", [
    'majority_label_per_surface_form_baseline_fixture',
])
def test_detectors_for_text_classification_flat(detector_fixture, request):
    detector: Detector = request.getfixturevalue(detector_fixture)
    ds = load_sequence_labeling_dataset(PATH_EXAMPLE_DATA_TOKEN)

    params = {
        'texts': ak.flatten(ds.sentences),
        'labels': ak.flatten(ds.noisy_labels)
    }

    detector.score(**params)


@pytest.mark.parametrize("detector_fixture", [
    'majority_label_per_surface_form_baseline_fixture',
])
def test_detectors_for_span_labeling_flat(detector_fixture, request):
    detector: Detector = request.getfixturevalue(detector_fixture)
    ds = load_sequence_labeling_dataset(PATH_EXAMPLE_DATA_SPAN)

    params = {
        'texts': ak.flatten(ds.sentences),
        'labels': ak.flatten(ds.noisy_labels)
    }

    detector.score(**params)

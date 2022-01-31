import awkward as ak
import numpy as np
import pytest
from scipy.stats import rankdata

from nessie.dataloader import (
    load_sequence_labeling_dataset,
    load_text_classification_tsv,
)
from nessie.detectors import (
    BordaCount,
    Detector,
    MajorityLabelBaseline,
    MajorityLabelPerSurfaceFormBaseline,
)
from tests.fixtures import (
    PATH_EXAMPLE_DATA_SPAN,
    PATH_EXAMPLE_DATA_TEXT,
    PATH_EXAMPLE_DATA_TOKEN,
)

# Smoke tests


@pytest.fixture
def majority_label_baseline_fixture() -> MajorityLabelBaseline:
    return MajorityLabelBaseline()


@pytest.fixture
def majority_label_per_surface_form_baseline_fixture() -> MajorityLabelPerSurfaceFormBaseline:
    return MajorityLabelPerSurfaceFormBaseline()


@pytest.mark.parametrize(
    "detector_fixture",
    [
        "majority_label_baseline_fixture",
    ],
)
def test_detectors_for_text_classification(detector_fixture, request):
    detector: Detector = request.getfixturevalue(detector_fixture)
    ds = load_text_classification_tsv(PATH_EXAMPLE_DATA_TEXT)

    params = {"texts": ds.texts, "labels": ds.noisy_labels}

    detector.score(**params)


@pytest.mark.parametrize(
    "detector_fixture",
    [
        "majority_label_per_surface_form_baseline_fixture",
    ],
)
def test_detectors_for_text_classification_flat(detector_fixture, request):
    detector: Detector = request.getfixturevalue(detector_fixture)
    ds = load_sequence_labeling_dataset(PATH_EXAMPLE_DATA_TOKEN)

    params = {"texts": ak.flatten(ds.sentences), "labels": ak.flatten(ds.noisy_labels)}

    detector.score(**params)


@pytest.mark.parametrize(
    "detector_fixture",
    [
        "majority_label_per_surface_form_baseline_fixture",
    ],
)
def test_detectors_for_span_labeling_flat(detector_fixture, request):
    detector: Detector = request.getfixturevalue(detector_fixture)
    ds = load_sequence_labeling_dataset(PATH_EXAMPLE_DATA_SPAN)

    params = {"texts": ak.flatten(ds.sentences), "labels": ak.flatten(ds.noisy_labels)}

    detector.score(**params)


# Method specific tests


def test_majority_label_baseline(majority_label_baseline_fixture: MajorityLabelBaseline):
    texts = [
        "I like cookies.",
        "I like reindeer.",
        "He likes sunsets and long strolls on the beach.",
        "He does not like Mondays.",
    ]

    labels = ["pos", "pos", "pos", "neg"]

    flags = majority_label_baseline_fixture.score(texts, labels)

    assert list(flags) == [False, False, False, True]


def test_majority_label_per_surface_form_baseline(
    majority_label_per_surface_form_baseline_fixture: MajorityLabelPerSurfaceFormBaseline,
):
    sentences = [
        ["Obama", "Harvard"],
        ["Harvard"],
        ["Harvard", "Boston"],
    ]

    labels = [
        ["PER", "LOC"],
        ["LOC"],
        ["MISC", "LOC"],
    ]

    sentences = ak.flatten(ak.Array(sentences))
    labels = ak.flatten(ak.Array(labels))

    flags = majority_label_per_surface_form_baseline_fixture.score(sentences, labels)

    assert len(sentences) == len(labels) == len(flags)
    assert list(flags) == [False, False, False, True, False]


def test_borda_count():
    votes = np.array(
        [
            [4, 3, 2, 1],
            [4, 3, 2, 1],
            [1, 4, 3, 2],
        ]
    )

    method = BordaCount()
    scores = method.score(votes)

    # We invert scores so that ranks are computed from largest to lowest
    actual_ranks = rankdata(-scores, method="ordinal")

    assert np.array_equal(actual_ranks, np.array([2, 1, 3, 4]))

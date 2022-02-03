from typing import List, Tuple

import awkward as ak
import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest
from numpy.random import default_rng
from scipy.stats import rankdata
from sklearn.preprocessing import LabelEncoder

from nessie.detectors import (
    BordaCount,
    ClassificationEntropy,
    ClassificationUncertainty,
    ConfidentLearning,
    Detector,
    DropoutUncertainty,
    ItemResponseTheoryFlagger,
    LabelAggregation,
    LabelEntropy,
    MajorityLabelBaseline,
    MajorityLabelPerSurfaceFormBaseline,
    MajorityVotingEnsemble,
    MeanDistance,
    PredictionMargin,
    Retag,
    VariationNGrams,
    WeightedDiscrepancy,
)
from nessie.detectors.knn_entropy import KnnEntropy
from nessie.detectors.variational_principle import VariationNGramsSpan
from nessie.models.featurizer import (
    CachedSentenceTransformer,
    FlairTokenEmbeddingsWrapper,
)
from tests.conftest import (
    generate_random_pos_tagging_dataset,
    generate_random_text_classification_dataset,
    get_random_ensemble_predictions,
    get_random_probabilities,
    get_random_repeated_probabilities,
)

NUM_INSTANCES = 100
NUM_LABELS = 4
NUM_MODELS = 3
T = 10

# Method fixtures


@pytest.fixture
def majority_label_baseline_fixture() -> MajorityLabelBaseline:
    return MajorityLabelBaseline()


@pytest.fixture
def majority_label_per_surface_form_baseline_fixture() -> MajorityLabelPerSurfaceFormBaseline:
    return MajorityLabelPerSurfaceFormBaseline()


@pytest.fixture
def classification_entropy_fixture() -> ClassificationEntropy:
    return ClassificationEntropy()


@pytest.fixture
def classification_uncertainty_fixture() -> ClassificationUncertainty:
    return ClassificationUncertainty()


@pytest.fixture
def confident_learning_fixture() -> ConfidentLearning:
    return ConfidentLearning()


@pytest.fixture
def dropout_uncertainty_fixture() -> DropoutUncertainty:
    return DropoutUncertainty()


@pytest.fixture
def ensemble_fixture() -> MajorityVotingEnsemble:
    return MajorityVotingEnsemble()


@pytest.fixture
def irt_fixture() -> ItemResponseTheoryFlagger:
    return ItemResponseTheoryFlagger(num_iters=5)


@pytest.fixture
def knn_entropy_fixture() -> KnnEntropy:
    return KnnEntropy()


@pytest.fixture
def label_aggregation_fixture() -> LabelAggregation():
    return LabelAggregation()


@pytest.fixture
def label_entropy_fixture() -> LabelEntropy():
    return LabelEntropy()


@pytest.fixture
def mean_distance_fixture() -> MeanDistance():
    return MeanDistance()


@pytest.fixture
def prediction_margin_fixture() -> PredictionMargin():
    return PredictionMargin()


@pytest.fixture
def retag_fixture() -> Retag():
    return Retag()


@pytest.fixture
def variation_ngrams_fixture() -> VariationNGrams():
    return VariationNGrams()


@pytest.fixture
def variation_ngrams_span_fixture() -> VariationNGramsSpan():
    return VariationNGramsSpan()


@pytest.fixture
def weighted_discrepancy_fixture() -> WeightedDiscrepancy():
    return WeightedDiscrepancy()


# Smoke tests


@pytest.mark.parametrize(
    "detector_fixture",
    [
        "majority_label_baseline_fixture",
        "classification_entropy_fixture",
        "classification_uncertainty_fixture",
        "confident_learning_fixture",
        "dropout_uncertainty_fixture",
        "ensemble_fixture",
        "irt_fixture",
        "knn_entropy_fixture",
        "label_aggregation_fixture",
        "mean_distance_fixture",
        "prediction_margin_fixture",
        "retag_fixture",
        "weighted_discrepancy_fixture",
    ],
)
def test_detectors_for_text_classification(
    detector_fixture, request: FixtureRequest, sentence_embedder_fixture: CachedSentenceTransformer
):
    detector: Detector = request.getfixturevalue(detector_fixture)
    ds = generate_random_text_classification_dataset(NUM_INSTANCES, NUM_LABELS)

    probabilities = get_random_probabilities(ds.num_instances, len(ds.tagset_noisy))
    repeated_probabilities = get_random_repeated_probabilities(ds.num_instances, len(ds.tagset_noisy), T)
    ensemble_predictions = get_random_ensemble_predictions(ds.num_instances, ds.tagset_noisy, NUM_MODELS)

    embedded_sentences = request.config.cache.get("methods/text/embedded_sentences", None)
    if embedded_sentences is None:
        embedded_sentences = sentence_embedder_fixture.embed(ds.texts)
        request.config.cache.set("methods/text/embedded_sentences", embedded_sentences.tolist())

    le = LabelEncoder().fit(ds.noisy_labels)

    params = {
        "texts": ds.texts,
        "labels": ds.noisy_labels,
        "probabilities": probabilities,
        "predictions": ensemble_predictions[0],
        "repeated_probabilities": repeated_probabilities,
        "ensemble_predictions": ensemble_predictions,
        "embedded_instances": np.asarray(embedded_sentences),
        "le": le,
    }

    detector.score(**params)


@pytest.mark.parametrize(
    "detector_fixture",
    [
        "majority_label_per_surface_form_baseline_fixture",
        "classification_entropy_fixture",
        "classification_uncertainty_fixture",
        "confident_learning_fixture",
        "dropout_uncertainty_fixture",
        "ensemble_fixture",
        "irt_fixture",
        "knn_entropy_fixture",
        "label_aggregation_fixture",
        "label_entropy_fixture",
        "mean_distance_fixture",
        "prediction_margin_fixture",
        "retag_fixture",
        "variation_ngrams_fixture",
        "weighted_discrepancy_fixture",
    ],
)
def test_detectors_for_token_classification_flat(
    detector_fixture, request: FixtureRequest, token_embedder_fixture: FlairTokenEmbeddingsWrapper
):
    detector: Detector = request.getfixturevalue(detector_fixture)
    ds = generate_random_pos_tagging_dataset(NUM_INSTANCES, NUM_LABELS)

    probabilities_flat = get_random_probabilities(ds.num_instances, len(ds.tagset_noisy))
    repeated_probabilities_flat = get_random_repeated_probabilities(ds.num_instances, len(ds.tagset_noisy), T)
    ensemble_predictions = get_random_ensemble_predictions(ds.num_instances, ds.tagset_noisy, NUM_MODELS)

    embedded_tokens = request.config.cache.get("methods/token/embedded_tokens", None)
    if embedded_tokens is None:
        embedded_tokens = token_embedder_fixture.embed(ds.sentences, flat=True)
        request.config.cache.set("methods/token/embedded_tokens", embedded_tokens.tolist())

    le = LabelEncoder().fit(ak.flatten(ds.noisy_labels))

    params = {
        "texts": ak.flatten(ds.sentences),
        "sentences": ds.sentences,
        "labels": ak.flatten(ds.noisy_labels),
        "tags": ds.noisy_labels,
        "probabilities": probabilities_flat,
        "predictions": ensemble_predictions[0],
        "repeated_probabilities": repeated_probabilities_flat,
        "ensemble_predictions": ensemble_predictions,
        "embedded_instances": np.asarray(embedded_tokens),
        "le": le,
    }

    detector.score(**params)


@pytest.mark.parametrize(
    "detector_fixture",
    [
        "majority_label_per_surface_form_baseline_fixture",
        "classification_entropy_fixture",
        "classification_uncertainty_fixture",
        "confident_learning_fixture",
        "dropout_uncertainty_fixture",
        "ensemble_fixture",
        "irt_fixture",
        "knn_entropy_fixture",
        "label_aggregation_fixture",
        "mean_distance_fixture",
        "prediction_margin_fixture",
        "retag_fixture",
        "variation_ngrams_span_fixture",
    ],
)
def test_detectors_for_span_labeling_flat(detector_fixture, request: FixtureRequest):
    pass


# Method specific tests


def test_majority_label_baseline():
    detector = MajorityLabelBaseline()

    texts = [
        "I like cookies.",
        "I like reindeer.",
        "He likes sunsets and long strolls on the beach.",
        "He does not like Mondays.",
    ]

    labels = ["pos", "pos", "pos", "neg"]

    flags = detector.score(texts, labels)

    assert list(flags) == [False, False, False, True]


def test_majority_label_per_surface_form_baseline():
    detector = MajorityLabelPerSurfaceFormBaseline()

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

    flags = detector.score(sentences, labels)

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


@pytest.mark.parametrize(
    "proba,expected", [([[0.1, 0.85, 0.05], [0.6, 0.3, 0.1], [0.39, 0.61, 0.0]], [0.51818621, 0.89794572, 0.66874809])]
)
def test_classification_entropy(proba, expected):
    # https://modal-python.readthedocs.io/en/latest/content/query_strategies/uncertainty_sampling.html

    probabilities = np.array(proba)

    detector = ClassificationEntropy()
    scores = detector.score(probabilities)

    assert np.allclose(scores, expected)


def test_classification_uncertainty():
    n = 100
    k = 4

    le = LabelEncoder()
    le.classes_ = np.array(["A", "B", "C", "D"], dtype=object)

    probabilities = get_random_probabilities(n, k)
    encoded_labels = np.random.randint(0, k, n)
    labels = le.inverse_transform(encoded_labels)
    expected_scores = 1 - np.array([probabilities[i, encoded_labels[i]] for i in range(n)])

    detector = ClassificationUncertainty()
    scores = detector.score(labels, probabilities, le)

    assert np.array_equal(scores, expected_scores)


def test_ensemble():
    labels = ["A", "B", "B", "A"]

    ensemble_predictions = [
        ["A", "A", "B", "A"],
        ["A", "A", "A", "B"],
        ["B", "B", "B", "B"],
    ]

    expected_flags = [False, True, False, True]

    detector = MajorityVotingEnsemble()
    flags = list(detector.score(labels, ensemble_predictions))

    assert flags == expected_flags


@pytest.mark.parametrize(
    "X,y,expected",
    [
        [
            (
                ["I", "love", "cats", "very", "much", "."],
                ["I", "cherish", "cats", "very", "much", "."],
                ["The", "cats", "I", "like", "are", "very", "much", "different", "in", "their", "appearance", "."],
            ),
            (
                ["PRON", "VERB", "NOUN", "ADV", "ADV", "."],
                ["PRON", "VERB", "NOUN", "ADV", "ADV", "."],
                ["DET", "NOUN", "PRON", "VERB", "VERB", "ADV", "ADJ", "ADJ", "PREP", "PRON", "NOUN", "."],
            ),
            (
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False, True, False, False, False, False, False],
            ),
        ]
    ],
)
def test_variation_principle(X: List[List[str]], y: List[List[str]], expected: List[List[bool]]):
    detector = VariationNGrams()

    scores = detector.score(X, y)

    assert len(expected) == len(scores)

    for e, s in zip(expected, scores):
        assert list(e) == list(s)


@pytest.mark.parametrize(
    "haystack,needle,expected",
    [
        (["I", "love", "cats", "very", "much", "."], ["very", "much"], [(3, 5)]),
        ([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], [1, 2, 3], [(1, 4), (7, 10)]),
        ([1, 2, 3], [1, 2, 3], [(0, 3)]),
    ],
)
def test_variation_principle_find_sublists(haystack: List, needle: List, expected: Tuple[int, int]):
    matches = list(VariationNGrams._find_sublist_indices_in_list(haystack, needle))

    assert len(matches) == len(expected)
    for match, e in zip(matches, expected):
        assert e == match


@pytest.mark.parametrize(
    "X,y,k,corrected,flags",
    [
        (
            [
                ["from", "my", "bank", "account", "to", "her"],
                ["from", "my", "bank", "account", "to", "him"],
                ["with", "my", "bank", "account", "to", "provide"],
                ["use", "my", "bank", "account", "to", "send"],
                ["out", "of", "my", "bank", "account", "to", "my"],
            ],
            [
                ["O", "O", "B-SOURCE", "I-SOURCE", "O", "O"],
                ["O", "O", "B-TARGET", "I-TARGET", "O", "O"],
                ["O", "O", "B-SOURCE", "I-SOURCE", "O", "O"],
                ["O", "O", "B-SOURCE", "I-SOURCE", "O", "O"],
                ["O", "O", "O", "B-TARGET", "I-TARGET", "O", "O"],
            ],
            1,
            [["SOURCE"], ["SOURCE"], ["SOURCE"], ["SOURCE"], ["SOURCE"]],
            [[False], [True], [False], [False], [True]],
        ),
        (
            [
                ["Obama", "is", "an", "alumni", "of", "Harvard"],
                ["Harvard", "is", "a", "university"],
                ["Harvard", "is", "in", "Boston"],
                ["Obama", "is", "a", "former", "president"],
                ["Obama", "is", "a", "law", "scholar"],
                ["Harvard", "is", "a", "good", "university"],
            ],
            [
                ["B-PER", "O", "O", "O", "O", "B-LOC"],
                ["B-LOC", "O", "O", "O"],
                ["B-MISC", "O", "O", "B-LOC"],
                ["B-ORG", "O", "O", "O", "O"],
                ["B-PER", "O", "O", "O", "O"],
                ["B-LOC", "O", "O", "O", "O"],
            ],
            1,
            [["PER", "LOC"], ["LOC"], ["LOC", "LOC"], ["PER"], ["PER"], ["LOC"]],
            [[False, False], [False], [True, False], [True], [False], [False]],
        ),
    ],
)
def test_variation_principle_span(
    X: List[List[str]], y: List[List[str]], k: int, corrected: List[List[str]], flags: List[List[bool]]
):
    detector = VariationNGramsSpan(k=k)

    actual_flags = detector.score(X, y).to_list()
    actual_corrected = detector.correct(X, y).to_list()

    assert actual_corrected == corrected
    assert actual_flags == flags

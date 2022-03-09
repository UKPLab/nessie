import awkward as ak
import numpy as np

from nessie.helper import CrossValidationHelper
from nessie.models.tagging.dummy_sequence_classifier import DummySequenceTagger
from nessie.models.text import DummyTextClassifier
from tests.conftest import (
    generate_random_pos_tagging_dataset,
    generate_random_text_classification_dataset,
)


def test_cv_helper_text_classification():
    ds = generate_random_text_classification_dataset(256, 4)

    model = DummyTextClassifier()

    cv = CrossValidationHelper(n_splits=3)
    result = cv.run(ds.texts, ds.noisy_labels, model)

    assert result.predictions.shape == (ds.num_instances,)
    assert result.probabilities.shape == (ds.num_instances, ds.num_labels)
    assert result.repeated_probabilities.shape == (ds.num_instances, cv._num_repetitions, ds.num_labels)
    assert result.le is not None


def test_cv_helper_token_labeling():
    ds = generate_random_pos_tagging_dataset(256, 4)

    model = DummySequenceTagger()

    cv = CrossValidationHelper(n_splits=3)
    result = cv.run_for_ragged(ds.sentences, ds.noisy_labels, model)
    result_flat = result.flatten()

    # Check that the nested dimensions have the same sizes
    assert np.all(ak.num(result.predictions, axis=1) == ds.sizes)
    assert np.all(ak.num(result.probabilities, axis=1) == ds.sizes)
    assert np.all(ak.num(result.repeated_probabilities, axis=1) == ds.sizes)

    # Check that dimensions after flattening fit
    assert result_flat.predictions.shape == (ds.num_instances,)
    assert result_flat.probabilities.shape == (ds.num_instances, ds.num_labels)
    assert result_flat.repeated_probabilities.shape == (ds.num_instances, cv._num_repetitions, ds.num_labels)
    assert result_flat.le is not None

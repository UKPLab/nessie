from nessie.evaluation import CrossValidationHelper
from tests.conftest import (
    DummySequenceTagger,
    DummyTextClassifier,
    generate_random_pos_tagging_dataset,
    generate_random_text_classification_dataset,
)


def test_cv_helper_text_classification():
    ds = generate_random_text_classification_dataset(256, 4)

    model = DummyTextClassifier()

    cv = CrossValidationHelper()
    result = cv.run(ds.texts, ds.noisy_labels, model)

    assert result.predictions.shape == (ds.num_instances,)
    assert result.probabilities.shape == (ds.num_instances, ds.num_labels)
    assert result.repeated_probabilities.shape == (ds.num_instances, cv._num_repetitions, ds.num_labels)
    assert result.le is not None


def test_cv_helper_token_labeling():
    ds = generate_random_pos_tagging_dataset(256, 4)

    model = DummySequenceTagger()

    cv = CrossValidationHelper()
    result = cv.run_for_ragged(ds.sentences, ds.noisy_labels, model)

    assert result.predictions.shape == (ds.num_instances,)
    assert result.probabilities.shape == (ds.num_instances, ds.num_labels)
    assert result.repeated_probabilities.shape == (ds.num_instances, cv._num_repetitions, ds.num_labels)
    assert result.le is not None

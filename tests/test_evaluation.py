from nessie.evaluation import CrossValidationHelper
from tests.conftest import (
    NUM_INSTANCES,
    NUM_LABELS,
    DummyTextClassifier,
    generate_random_text_classification_dataset,
)


def test_cv_helper():
    ds = generate_random_text_classification_dataset(256, 4)

    model = DummyTextClassifier()

    cv = CrossValidationHelper()
    result = cv.run(ds.texts, ds.noisy_labels, model)

    assert result.predictions.shape == (ds.num_instances,)
    assert result.probabilities.shape == (ds.num_instances, ds.num_labels)
    assert result.repeated_probabilities.shape == (ds.num_instances, cv._num_repetitions, ds.num_labels)
    assert result.le is not None

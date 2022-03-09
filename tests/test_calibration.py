from netcal.scaling import LogisticCalibration

from nessie.calibration import CalibrationCallback, CalibratorWrapper
from nessie.helper import CrossValidationHelper
from nessie.models.text import DummyTextClassifier
from tests.conftest import generate_random_text_classification_dataset


def test_calibration_text_classification():
    ds = generate_random_text_classification_dataset(256, 4)

    model = DummyTextClassifier()

    calibrator = CalibratorWrapper(LogisticCalibration())
    calibration_callback = CalibrationCallback(calibrator)

    cv = CrossValidationHelper()
    cv.add_callback(calibration_callback)

    result = cv.run(ds.texts, ds.noisy_labels, model)

    assert result.predictions.shape == (ds.num_instances,)
    assert result.probabilities.shape == (ds.num_instances, ds.num_labels)
    assert result.repeated_probabilities.shape == (ds.num_instances, cv._num_repetitions, ds.num_labels)
    assert result.le is not None

    assert calibration_callback.calibrated_probabilities.shape == (ds.num_instances, ds.num_labels)
    assert calibration_callback.calibrated_repeated_probabilities.shape == (
        ds.num_instances,
        cv._num_repetitions,
        ds.num_labels,
    )
    assert len(calibration_callback.calibration_error) == 2

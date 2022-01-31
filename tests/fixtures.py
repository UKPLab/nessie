from pathlib import Path

import numpy.typing as npt
from numpy.random import default_rng
from sklearn.preprocessing import normalize

PATH_ROOT: Path = Path(__file__).resolve().parents[1]

# Example data
PATH_EXAMPLE_DATA: Path = PATH_ROOT / "example_data"
PATH_EXAMPLE_DATA_TEXT: Path = PATH_EXAMPLE_DATA / "easy_text.tsv"
PATH_EXAMPLE_DATA_TOKEN: Path = PATH_EXAMPLE_DATA / "easy_token.conll"
PATH_EXAMPLE_DATA_SPAN: Path = PATH_EXAMPLE_DATA / "easy_span.conll"


def get_random_probabilities(num_instances: int, num_labels: int) -> npt.NDArray[float]:
    rng = default_rng()
    probabilities = rng.random((num_instances, num_labels))
    probabilities = normalize(probabilities, norm="l1", axis=1)

    return probabilities

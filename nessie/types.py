from typing import List, Union

import awkward as ak
import numpy.typing as npt

StringArray = Union[List[str], npt.NDArray[str], ak.Array]
StringArray2D = Union[List[List[str]], npt.NDArray[str]]
FloatArray2D = Union[List[List[float]], npt.NDArray[float]]
IntArray = Union[List[int], npt.NDArray[int], ak.Array]

RaggedStringArray = Union[List[List[str]], npt.NDArray[StringArray], ak.Array]

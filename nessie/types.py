from typing import List, Union

import awkward as ak
import numpy.typing as npt

StringArray = Union[List[str], npt.NDArray[str], ak.Array]
StringArray2D = Union[List[List[str]], npt.NDArray[str]]
FloatArray2D = Union[List[List[float]], npt.NDArray[float]]
IntArray = Union[List[int], npt.NDArray[int], ak.Array]
BoolArray = Union[List[bool], npt.NDArray[bool]]
FloatArray = Union[List[float], npt.NDArray[float], ak.Array]

RaggedStringArray = Union[List[List[str]], npt.NDArray[StringArray], ak.Array]
RaggedFloatArray = Union[List[List[float]], npt.NDArray[FloatArray], ak.Array]
RaggedFloatArray2D = Union[List[List[List[float]]], npt.NDArray[float], ak.Array]
RaggedFloatArray3D = Union[List[List[List[List[float]]]], npt.NDArray[float], ak.Array]

from typing import List, Union

import awkward as ak
import numpy.typing as npt

StringArray = Union[List[str], npt.NDArray[str], ak.Array]
RaggedStringArray = Union[List[List[str]], npt.NDArray[StringArray], ak.Array]

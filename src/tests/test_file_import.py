from typing import Any

import numpy as np
import numpy.typing as npt
from data_clusterer.file_import import load_numpy_array


def test_load_numpy_array(
    file_with_numpy_data: str, test_data: npt.NDArray[Any]
):
    test_data_from_file = load_numpy_array(file_with_numpy_data)
    assert test_data_from_file is not None
    assert np.array_equal(test_data_from_file, test_data)


def test_load_numpy_no_file():
    test_data_from_file = load_numpy_array("not_existing_file.npy")
    assert test_data_from_file is None


def test_load_numpy_invalid_file(file_with_json_data: str):
    test_data_from_file = load_numpy_array(file_with_json_data)
    assert test_data_from_file is None

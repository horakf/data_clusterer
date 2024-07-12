import os
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from data_clusterer.file_export import create_directories, save_numpy_array


@pytest.mark.parametrize(
    "given_path, expected_directory",
    [
        ("test_dir1/test_file.npy", "test_dir1"),
        ("test_dir3/test_dir2/test_file.json", "test_dir3/test_dir2"),
        ("./test_file.npy", "."),
    ],
)
def test_create_directories(
    temporary_directory: str, given_path: str, expected_directory: str
):
    test_instance = os.path.join(temporary_directory, given_path)
    expected_full_path = os.path.join(temporary_directory, expected_directory)
    create_directories(test_instance)
    assert os.path.isdir(expected_full_path)


def test_save_numpy_file_creation(
    temporary_numpy_file: str, test_data: npt.NDArray[Any]
):
    save_numpy_array(temporary_numpy_file, test_data)
    assert os.path.isfile(temporary_numpy_file)
    loaded_file = np.load(temporary_numpy_file)
    assert loaded_file is not None
    assert np.array_equal(loaded_file, test_data)

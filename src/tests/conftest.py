import json
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture
def test_data():
    return np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0],
            [1.0, 0.6],
            [9.0, 11.0],
            [8.0, 2.0],
            [10.0, 2.0],
            [9.0, 3.0],
        ]
    )


@pytest.fixture
def temporary_directory(tmp_path_factory: pytest.TempPathFactory):
    test_path = tmp_path_factory.mktemp("test_data")
    yield str(test_path)


@pytest.fixture
def temporary_numpy_file(temporary_directory: str):
    yield str(os.path.join(temporary_directory, "numpy_test.npy"))


@pytest.fixture
def temporary_json_file(temporary_directory: str):
    yield str(os.path.join(temporary_directory, "json_test.json"))


@pytest.fixture
def file_with_numpy_data(
    test_data: npt.NDArray[Any], temporary_numpy_file: str
):
    np.save(temporary_numpy_file, test_data)
    yield temporary_numpy_file


@pytest.fixture
def file_with_json_data(test_data: npt.NDArray[Any], temporary_json_file: str):
    test_data = test_data.tolist()
    with open(temporary_json_file, "w") as f:
        json.dump(test_data, f)
    yield str(temporary_json_file)

import json
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
def file_with_numpy_data(
    test_data: npt.NDArray[Any], tmp_path_factory: pytest.TempPathFactory
):
    test_path = tmp_path_factory.mktemp("test_data") / "numpy_test.npy"
    np.save(test_path, test_data)
    yield str(test_path)


@pytest.fixture
def file_with_json_data(
    test_data: npt.NDArray[Any], tmp_path_factory: pytest.TempPathFactory
):
    test_path = tmp_path_factory.mktemp("test_data") / "json_test.json"
    test_data = test_data.tolist()
    with open(test_path, "w") as f:
        json.dump(test_data, f)
    yield str(test_path)

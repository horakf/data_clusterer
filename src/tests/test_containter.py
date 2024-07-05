from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from data_clusterer.algorithms import KMeansAlgorithm
from data_clusterer.container import Container


@pytest.fixture
def container(file_with_numpy_data: str, file_with_json_data: str):
    container = Container(
        config={
            "algorithm": {
                "name": "kmeans",
                "hyperparameters": {
                    "n_clusters": 3,
                    "random_state": 42,
                },
            },
            "input_data": {
                "format": "numpy",
                "file_path": file_with_numpy_data,
            },
            "target_data": {
                "format": "numpy",
                "file_path": file_with_json_data,
            },
        }
    )
    yield container


def test_kmeans_factory(container: Container):
    test_instance = container.algorithm()
    assert isinstance(test_instance, KMeansAlgorithm)
    assert test_instance.model.n_clusters == 3
    assert test_instance.model.random_state == 42
    assert test_instance.model.max_iter == 300


def test_load_numpy_file(container: Container, test_data: npt.NDArray[Any]):
    test_instance = container.load_data()
    assert test_instance is not None
    assert np.array_equal(test_instance, test_data)

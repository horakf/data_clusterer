from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from data_clusterer.algorithms import KMeansAlgorithm
from data_clusterer.container import Container


@pytest.fixture
def container(file_with_numpy_data: str, temporary_numpy_file: str):
    container = Container(
        config={
            "algorithm": {
                "name": "kmeans",
                "hyperparameters": {
                    "n_clusters": 3,
                    "init": "k-means++",
                    "random_state": 42,
                    "max_iter": 300,
                },
            },
            "input_data": {
                "format": "numpy",
                "file_path": file_with_numpy_data,
            },
            "target_data": {
                "format": "numpy",
                "file_path": temporary_numpy_file,
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
    assert test_instance.model.init == "k-means++"


def test_load_numpy_file(container: Container, test_data: npt.NDArray[Any]):
    test_instance = container.load_data()
    assert test_instance is not None
    assert np.array_equal(test_instance, test_data)


def test_save_numpy_file(
    container: Container,
    test_data: npt.NDArray[Any],
    temporary_numpy_file: str,
):
    container.save_data(data=test_data)
    loaded_test_data = np.load(temporary_numpy_file)
    assert loaded_test_data is not None
    assert np.array_equal(loaded_test_data, test_data)

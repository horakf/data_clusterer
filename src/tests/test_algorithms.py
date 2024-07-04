from typing import Any, Dict

import numpy as np
import numpy.typing as npt
import pytest
from data_clusterer.algorithms import KMeansAlgorithm


@pytest.fixture
def kmeans_object():
    return KMeansAlgorithm(n_clusters=3, random_state=42)


@pytest.mark.parametrize(
    "given_kwargs, expected_n_clusters, expected_random_state",
    [
        ({"n_clusters": 3, "random_state": 42}, 3, 42),
        ({"n_clusters": None, "random_state": 50}, 8, 50),
        ({"n_clusters": None, "random_state": None}, 8, None),
    ],
)
def test_kmeans_init(
    given_kwargs: Dict[str, int | None],
    expected_n_clusters: int,
    expected_random_state: int | None,
):
    test_instance = KMeansAlgorithm(**given_kwargs)
    assert test_instance.model.n_clusters == expected_n_clusters
    assert test_instance.model.random_state == expected_random_state


def test_kmeans_fit_labels_length(
    kmeans_object: KMeansAlgorithm, test_data: npt.NDArray[Any]
):
    assert len(kmeans_object.fit(test_data)) == len(test_data)


@pytest.mark.parametrize(
    "n_clusters, expected_clusters", [(3, [0, 1, 2]), (5, [0, 1, 2, 3, 4])]
)
def test_kmeans_fit_number_of_clusters(
    test_data: npt.NDArray[Any], n_clusters: int, expected_clusters: list[int]
):
    test_instance = KMeansAlgorithm(n_clusters=n_clusters)
    assert all(
        label in expected_clusters for label in test_instance.fit(test_data)
    )


def test_kmeans_fit_empty_data(kmeans_object: KMeansAlgorithm):
    with pytest.raises(ValueError):
        kmeans_object.fit(np.array([]).reshape(0, 2))

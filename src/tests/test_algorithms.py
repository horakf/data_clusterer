from typing import Any, Dict

import numpy as np
import numpy.typing as npt
import pytest
from data_clusterer.algorithms import (
    AglomerativeAlgorithm,
    DBSCANAlgorithm,
    KMeansAlgorithm,
)


@pytest.fixture
def kmeans_object():
    return KMeansAlgorithm(n_clusters=3, random_state=42)


@pytest.fixture
def dbscan_object():
    return DBSCANAlgorithm(eps=4, min_samples=3)


@pytest.fixture
def aglomerative_object():
    return AglomerativeAlgorithm(n_clusters=3, linkage="ward")


@pytest.fixture
def aglomerative_hyperparameters():
    return {
        "n_clusters": None,
        "distance_threshold": 5,
    }


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


@pytest.mark.parametrize(
    "given_kwargs, expected_eps, expected_min_samples",
    [
        ({"eps": 4, "min_samples": 3}, 4, 3),
        ({"eps": None, "min_samples": 2}, 0.5, 2),
        ({"eps": None, "min_samples": None}, 0.5, 5),
    ],
)
def test_dbscan_init(
    given_kwargs: Dict[str, int | None],
    expected_eps: int,
    expected_min_samples: int | None,
):
    test_instance = DBSCANAlgorithm(**given_kwargs)
    assert test_instance.model.eps == expected_eps
    assert test_instance.model.min_samples == expected_min_samples


def test_dbscan_fit_labels_length(
    dbscan_object: DBSCANAlgorithm, test_data: npt.NDArray[Any]
):
    assert len(dbscan_object.fit(test_data)) == len(test_data)


def test_dbscan_fit_empty_data(dbscan_object: DBSCANAlgorithm):
    with pytest.raises(ValueError):
        dbscan_object.fit(np.array([]).reshape(0, 2))


@pytest.mark.parametrize(
    "given_kwargs, expected_n_clusters, expected_linkage",
    [
        ({"n_clusters": 3, "linkage": "ward"}, 3, "ward"),
        ({"n_clusters": None, "linkage": "single"}, None, "single"),
        ({"n_clusters": None, "linkage": None}, None, "ward"),
    ],
)
def test_aglomerative_init(
    given_kwargs: Dict[str, int | None],
    expected_n_clusters: int,
    expected_linkage: int | None,
):
    test_instance = AglomerativeAlgorithm(**given_kwargs)
    assert test_instance.model.n_clusters == expected_n_clusters
    assert test_instance.model.linkage == expected_linkage


def test_aglomerative_fit_labels_length(
    aglomerative_object: AglomerativeAlgorithm, test_data: npt.NDArray[Any]
):
    assert len(aglomerative_object.fit(test_data)) == len(test_data)


@pytest.mark.parametrize(
    "n_clusters, expected_clusters", [(3, [0, 1, 2]), (5, [0, 1, 2, 3, 4])]
)
def test_aglomerative_fit_number_of_clusters(
    test_data: npt.NDArray[Any], n_clusters: int, expected_clusters: list[int]
):
    test_instance = AglomerativeAlgorithm(n_clusters=n_clusters)
    assert all(
        label in expected_clusters for label in test_instance.fit(test_data)
    )


def test_aglomerative_fit_empty_data(
    aglomerative_object: AglomerativeAlgorithm,
):
    with pytest.raises(ValueError):
        aglomerative_object.fit(np.array([]).reshape(0, 2))


def test_aglomerative_none_clusters(
    aglomerative_hyperparameters: Dict[str, Any], test_data: npt.NDArray[Any]
):
    test_instance = AglomerativeAlgorithm(**aglomerative_hyperparameters)
    assert test_instance.model.n_clusters is None
    assert test_instance.model.distance_threshold == 5
    assert len(test_instance.fit(test_data)) == len(test_data)

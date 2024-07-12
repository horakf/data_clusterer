from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest
from data_clusterer.algorithms import KMeansAlgorithm
from sklearn.cluster import KMeans


@pytest.fixture
def kmeans_object():
    return KMeansAlgorithm(
        n_clusters=3, init="k-means++", random_state=42, max_iter=300
    )


@pytest.fixture
def kmeans_expected_labels():
    return [1, 1, 0, 0, 1, 0, 2, 2, 2]


@pytest.mark.parametrize(
    "n_clusters, init, random_state, max_iter",
    [
        (3, "k-means++", 42, 300),
        (2, np.array([[1.0, 2.0], [5.0, 8.0]]), None, 200),
        (3, "random", 0, 100),
    ],
)
def test_kmeans_init(
    n_clusters: int,
    init: str | npt.NDArray[Any],
    random_state: int | None,
    max_iter: int,
):
    test_instance = KMeansAlgorithm(n_clusters, init, random_state, max_iter)
    assert test_instance.model.n_clusters == n_clusters
    if isinstance(init, str):
        assert test_instance.model.init == init
    else:
        assert np.array_equal(test_instance.model.init, init)
    assert test_instance.model.random_state == random_state
    assert test_instance.model.max_iter == max_iter


@patch.object(
    KMeans, "fit_predict", return_value=np.array([1, 1, 0, 0, 1, 0, 2, 2, 2])
)
def test_kmeans_fit(
    mock_fit_predict: MagicMock,
    kmeans_object: KMeansAlgorithm,
    test_data: npt.NDArray[Any],
    kmeans_expected_labels: List[int],
):
    labels = kmeans_object.fit(test_data)
    mock_fit_predict.assert_called_once_with(test_data)
    assert labels.tolist() == kmeans_expected_labels


def test_kmeans_fit_empty_data(kmeans_object: KMeansAlgorithm):
    with pytest.raises(ValueError):
        kmeans_object.fit(np.array([]).reshape(0, 2))

from typing import Any

import numpy.typing as npt
from sklearn.cluster import KMeans


class KMeansAlgorithm:
    """Class for setting up and running KMeans clustering algorithm.

    Attributes:
        model (KMeans): An instance of a KMeans object.
    """

    def __init__(
        self,
        n_clusters: int,
        init: str | npt.NDArray[Any],
        random_state: int | None,
        max_iter: int,
    ) -> None:
        """Initialization of the model and hyperparameters' attributes.

        Model parameter is initialized with KMeans object and given
        hyperparameters.

        Args:
            n_clusters (int): The number of clusters to form as well as
                the number of centroids to generate.
            init (str | npt.NDArray[Any]): Method for initialization of
                the algorithm.
            random_state (int | RandomState | None): Determines random
                number generation for centroid initialization.
            max_iter (int): Maximum number of iterations of the k-means
                algorithm for a single run.
        Returns:
            None
        """
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            random_state=random_state,
            max_iter=max_iter,
        )

    def fit(self, data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Function used for running KMeans clustering algorithm on given data.

        Compute cluster centers and predict cluster index for each sample in
        the input data.

        Args:
            data (npt.NDArray): Numpy array containing input data
        Returns:
            npt.NDArray: Array containing cluster label for each feature (array
            column) of the input data.
        """
        return self.model.fit_predict(data)


class DBSCANAlgorithm: ...


class AglomerativeAlgorithm: ...

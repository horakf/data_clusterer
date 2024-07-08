from typing import Any

import numpy.typing as npt
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans


class KMeansAlgorithm:
    """Class for setting up and running KMeans clustering algorithm.

    Attributes:
        model (KMeans): An instance of a KMeans object.

    """

    def __init__(self, **kwargs) -> None:
        """Initialization of the model and hyperparameters' attributes.

        First model parameter is initialized with KMeans object. Then,
        non-None values passed as keywords arguments are assigned to the
        corresponding KMeans class attributes. If any attribute is not set,
        default settings of KMeans class is used for the attribute.

        Args:
            **kwargs (dict): Keyword arguments containing name - value pairs
            for setting KMeans hyperparameters.
        Returns:
            None

        """
        self.model = KMeans()
        for key, value in kwargs.items():
            if not value:
                continue
            setattr(self.model, key, value)

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


class DBSCANAlgorithm:
    """Class for setting up and running DBSCAN clustering algorithm.

    Attributes:
        model (DBSCAN): An instance of a DBSCAN object.

    """

    def __init__(self, **kwargs) -> None:
        """Initialization of the model and hyperparameters' attributes.

        First model parameter is initialized with DBSCAN object. Then,
        non-None values passed as keywords arguments are assigned to the
        corresponding DBSCAN class attributes. If any attribute is not set,
        default settings of DBSCAN class is used for the attribute.

        Args:
            **kwargs (dict): Keyword arguments containing name - value pairs
            for setting DBSCAN hyperparameters.
        Returns:
            None

        """
        self.model = DBSCAN()
        for key, value in kwargs.items():
            if not value:
                continue
            setattr(self.model, key, value)

    def fit(self, data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Function used for running DBSCAN clustering algorithm on given data.

        Compute cluster centers and predict cluster index for each sample in
        the input data.

        Args:
            data (npt.NDArray): Numpy array containing input data
        Returns:
            npt.NDArray: Array containing cluster label for each feature (array
            column) of the input data.
        """
        return self.model.fit_predict(data)


class AglomerativeAlgorithm:
    """Class for setting up and running AgglomerativeClustering algorithm.

    Attributes:
        model (AgglomerativeClustering): An instance of an
        AgglomerativeClustering object.

    """

    def __init__(self, **kwargs) -> None:
        """Initialization of the model and hyperparameters' attributes.

        First model parameter is initialized with AgglomerativeClustering
        object. Then, non-None values passed as keywords arguments are
        assigned to the corresponding AgglomerativeClustering class
        attributes. If any attribute is not set, default settings of
        AgglomerativeClustering class is used for the attribute.

        Args:
            **kwargs (dict): Keyword arguments containing name - value pairs
            for setting AgglomerativeClustering hyperparameters.
        Returns:
            None

        """
        self.model = AgglomerativeClustering()
        for key, value in kwargs.items():
            if not value and key != "n_clusters":
                continue
            setattr(self.model, key, value)

    def fit(self, data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Function used for running AgglomerativeClustering algorithm on
        given data.

        Compute cluster centers and predict cluster index for each sample in
        the input data.

        Args:
            data (npt.NDArray): Numpy array containing input data
        Returns:
            npt.NDArray: Array containing cluster label for each feature (array
            column) of the input data.
        """
        return self.model.fit_predict(data)

from dependency_injector import containers, providers

from data_clusterer.algorithms import (
    AglomerativeAlgorithm,
    DBSCANAlgorithm,
    KMeansAlgorithm,
)


class Container(containers.DeclarativeContainer):
    """A container class providing instantiation and lifecycle of application
    dependencies.

    Provides creation of instances of clustering algorithms, import/export
    of files and model selection, managing hyperparameters and data formats
    using configuration file.

    Attributes:
        config: Manages loading of configuration file and provides access to
            settings inside the file.
        kmeans: Creates instance of K-Means algorithm class and provides it
            with given hyperparameters.
        dbscan: Creates instance of DBSCAN algorithm class and provides it
            with given hyperparameters.
        aglomerative: Creates instance of Aglomerative Clustering algorithm
            class and provides it with given hyperparameters.
        algorithm: Selects and provides correct algorithm depending on config.
    """

    config = providers.Configuration()

    kmeans = providers.Factory(
        KMeansAlgorithm,
        n_clusters=config.algorithm.hyperparameters.n_clusters,
        init=config.algorithm.hyperparameters.init,
        max_iter=config.algorithm.hyperparameters.max_iter,
        random_state=config.algorithm.hyperparameters.random_state,
    )

    dbscan = providers.Factory(
        DBSCANAlgorithm,
    )

    aglomerative = providers.Factory(
        AglomerativeAlgorithm,
    )

    algorithm = providers.Selector(
        config.algorithm.name,
        kmeans=kmeans,
        dbscan=dbscan,
        aglomerative=aglomerative,
    )

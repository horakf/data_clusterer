from dependency_injector import containers, providers

from data_clusterer.algorithms import (
    AglomerativeAlgorithm,
    DBSCANAlgorithm,
    KMeansAlgorithm,
)
from data_clusterer.file_export import save_json_file, save_numpy_array
from data_clusterer.file_import import load_json_file, load_numpy_array


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
        load_numpy: Creates instance of function for loading numpy file and
            provides it with given file path.
        load_json: Creates instance of function for loading json file and
            provides it with given file path.
        load_data: Selects and provides corresponding file loading provider
            for the given file format.
        save_numpy: Creates instance of function for exporting provided array
            into numpy file and provides it with given file path.
        save_json: Creates instance of function for exporting provided array
            into json file and provides it with given file path.
        save_data: Selects and provides corresponding file saving provider
            for the given file format.
    """

    config = providers.Configuration()

    kmeans = providers.Factory(
        KMeansAlgorithm,
        n_clusters=config.algorithm.hyperparameters.n_clusters,
        init=config.algorithm.hyperparameters.init,
        n_init=config.algorithm.hyperparameters.n_init,
        max_iter=config.algorithm.hyperparameters.max_iter,
        tol=config.algorithm.hyperparameters.tol,
        verbose=config.algorithm.hyperparameters.verbose,
        random_state=config.algorithm.hyperparameters.random_state,
        copy_x=config.algorithm.hyperparameters.copy_x,
        algorithm=config.algorithm.hyperparameters.algorithm,
    )

    dbscan = providers.Factory(
        DBSCANAlgorithm,
        eps=config.algorithm.hyperparameters.eps,
        min_samples=config.algorithm.hyperparameters.min_samples,
        metric=config.algorithm.hyperparameters.metric,
        metric_params=config.algorithm.hyperparameters.metric_params,
        algorithm=config.algorithm.hyperparameters.algorithm,
        leaf_size=config.algorithm.hyperparameters.leaf_size,
        p=config.algorithm.hyperparameters.p,
        n_jobs=config.algorithm.hyperparameters.n_jobs,
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

    load_numpy = providers.Factory(
        load_numpy_array, file_path=config.input_data.file_path
    )

    load_json = providers.Factory(
        load_json_file, file_path=config.input_data.file_path
    )

    load_data = providers.Selector(
        config.input_data.format, numpy=load_numpy, json=load_json
    )

    save_numpy = providers.Factory(
        save_numpy_array, file_path=config.target_data.file_path
    )

    save_json = providers.Factory(
        save_json_file, file_path=config.target_data.file_path
    )

    save_data = providers.Selector(
        config.target_data.format, numpy=save_numpy, json=save_json
    )

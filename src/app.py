import argparse

import numpy.typing as npt
from data_clusterer.container import Container
from dependency_injector.wiring import Provide, inject


def main(config_file: str) -> None:
    container = Container()
    container.config.from_yaml(config_file)
    data = container.load_data()
    if data is not None:
        container.wire(modules=[__name__])
        labels = run_clustering(data)
        container.save_data(data=labels)


@inject
def run_clustering(data: npt.NDArray, algorithm=Provide[Container.algorithm]):
    model = algorithm
    labels = model.fit(data)
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data clustering application."
    )
    parser.add_argument(
        "config_file", type=str, help="Path to the configuration file."
    )
    args = parser.parse_args()
    main(args.config_file)

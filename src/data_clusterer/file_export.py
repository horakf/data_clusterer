import json
import os
from typing import Any

import numpy as np
import numpy.typing as npt


def create_directories(file_path: str) -> None:
    """Creates missing directories on given path to file.

    Args:
        file_path (str): Path to the target file.
    Returns:
        None
    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)


def save_numpy_array(file_path: str, data: npt.NDArray[Any]) -> None:
    """Saves numpy array data into numpy file.

    Args:
        file_path (str): Path to the target file.
        data (npt.NDArray): Data to be exported.
    Returns:
        None
    """
    create_directories(file_path)
    np.save(file_path, data)


def save_json_file(file_path: str, data: npt.NDArray[Any]) -> None:
    """Saves numpy array data into json file.

    Args:
        file_path (str): Path to the target file.
        data (npt.NDArray): Data to be exported.
    Returns:
        None
    """
    data_list = data.tolist()
    create_directories(file_path)
    with open(file_path, "w") as f:
        json.dump(data_list, f)

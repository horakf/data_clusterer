from typing import Any

import numpy as np
import numpy.typing as npt


def catch_file_errors(file_type: str):
    """Decorator function for catching file loading errors.

    Args:
        file_type (str): File format that is loaded.
    Returns:
        Result of the decorated function, usually npt.NDArray or None
        if error occurs.
    """

    def inner(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError:
                print(
                    "File not Found! "
                    "Please provide valid path in the configuration."
                )
                return None
            except Exception as e:
                print(
                    f"Error loading file: {e}. "
                    f"Please provide valid file in {file_type} format."
                )
                return None

        return wrapper

    return inner


@catch_file_errors("numpy (.npy)")
def load_numpy_array(file_path: str) -> npt.NDArray[Any]:
    """Loads data from numpy file format into numpy array.

    Args:
        file_path (str): Path to the file with input data.
    Returns:
        npt.NDArray: Array containing input data used for further clustering.
    """
    return np.load(file_path)


def load_json_file(): ...

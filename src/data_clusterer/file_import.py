import numpy as np
import numpy.typing as npt


def load_numpy_array(file_path: str) -> npt.NDArray | None:
    """Loads data from numpy file format into numpy array.

    Args:
        file_path (str): Path to the file with input data.
    Returns:
        npt.NDArray: Array containing input data used for further clustering.
    """
    try:
        data = np.load(file_path)

    except FileNotFoundError:
        print(
            "File not Found! Please provide valid path in the configuration."
        )
        return None

    except Exception as e:
        print(
            f"Error loading file: {e}. "
            "Please provide valid file in numpy (.npy) format."
        )
        return None

    return data


def load_json_file(): ...

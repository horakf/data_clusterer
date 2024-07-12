### data_clusterer

---

![Tests](https://github.com/horakf/data_clusterer/actions/workflows/tests.yml/badge.svg)

An application that provides implementation of selected algorithms for clustering user-input data.

#### Installation

---

I. Clone the repository,

II. create a virtual enviroment:
```
python -m venv env
```

III. activate the virt. enviroment:
```
source env/bin/activate   # Linux
env\Scripts\activate.bat  # Windows
```

IV. install the package:
```
$ pip install .
```

#### Usage

I. Prepare configurations file in .yaml format with desirable clustering algorithm, hyperparameters, input/output data format and paths to source and target files,

supported hyperparameters:
- K-Means: n_clusters, init, max_iter, random_state

config file example:
```
algorithm:
  name: kmeans
  hyperparameters:
    n_clusters: 3
    random_state: 42
    init: random
    max_iter: 300

input_data:
  format: numpy
  file_path: ./src/data/input/numpy_data.npy

target_data:
  format: numpy
  file_path: ./src/data/output/output.npy
```

II. provide source data on a path entered in the configurations file in either numpy or json file format,

III. run the script with configurations file path:
```
python src/app.py path/to/config.yaml
```

The result is stored in the output file according to the format and path specified in the configuration file.

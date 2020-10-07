![python-testing](https://github.com/Urban-Analytics/RAMP-UA/workflows/python-testing/badge.svg)
[![codecov](https://codecov.io/gh/Urban-Analytics/RAMP-UA/branch/master/graph/badge.svg)](https://codecov.io/gh/Urban-Analytics/RAMP-UA)
# RAMP-UA

This is the code repository for the RAMP Urban Analytics project.

This project contains two implementations of a microsim model which runs on a synthetic population:
1. Python / R implementation, found in [microsim/microsim_model.py](./microsim/microsim_model.py)
2. High performance OpenCL implementation, which can run on both CPU and CPU, 
which is found in the [microsim/opencl](./microsim/opencl) folder. 

Further documentation on the OpenCL model can be found [microsim/opencl/doc](./microsim/opencl/doc)

Both models should be logically equivalent (with some minor differences). 

## Environment setup

**NB:** The OpenCL model requires following additional installation instructions located in the 
[OpenCL Readme](./microsim/opencl/README.md)

This project currently supports running on Linux and macOS.

To start working with this repository you need to clone it onto your local machine:

```bash
$ git clone https://github.com/Urban-Analytics/RAMP-UA.git
$ cd RAMP-UA
```

This project requires a specific conda environment in order to run so you will need the [conda package manager system](https://docs.anaconda.com/anaconda/install/) installed. Once conda has been installed you can create an environment for this project using the provided environment file.

```bash
$ conda env create -f environment.yml
```

To retrieve data to run the mode you will need to use [Git Large File Storage](https://git-lfs.github.com/) to download the input data. Git-lfs is installed within the conda environment (you may need to run `git lfs install` on your first use of git lfs). To retrieve the data you run the following commands within the root of the project repository:

```bash
$ git lfs fetch
$ git lfs checkout
``` 

Next we install the RAMP-UA package into the environment using `setup.py`:

```bash
# if developing the code base use:
$ python setup.py develop
# for using the code base use
$ python setup.py install
```

### Running the models
Both models can be run from the [microsim/main.py](./microsim/main.py) script, which can be configured with various arguments
to choose which model implementation to run.

#### Python / R model

The Python / R model runs by default, so simply run the main script with no argument.s

```bash
$ python microsim/main.py 
```

#### OpenCL model
To run the OpenCL model pass the `--opencl` flag to the main script, as below.

The OpenCL model runs in "headless" mode by default, however it can also be run with an interactive GUI and visualisation,
to run with the GUI pass the `--opencl-gui` flag, as below.

Run Headless
```bash
$ python microsim/main.py --opencl
```

Run with GUI
```bash
$ python microsim/main.py --opencl --opencl-gui
```

#### Caching of population initialisation
The population initialisation step runs before either of the models and can be time consuming (~10 minutes). In order to run
the models using a cache of previous results simply pass the `--use-cache` flag.

### Output Dashboards
Outputs are currently written to the [devon_data/output](./devon_data/output) directory.

Interactive HTML dashboards can be created using the Bokeh library.
 
Run the command below to generate the full dashboard for the Python / R model output, which should automatically open
the HTML file when it finishes.
 ```bash
$ python microsim/dashboard.py
```
Configuration YAML files for the dashboard are located in the [model_parameters](./model_parameters) folder.

The OpenCL model has a more limited dashboard (this may be extended soon), which can be run as follows:
 ```bash
$ python microsim/opencl/ramp/opencl_dashboard.py
```

## Creating releases
This repository takes advantage of a GitHub action for [creating tagged releases](https://github.com/marvinpinto/action-automatic-releases) using [semantic versioning](https://semver.org/).

To initiate the GitHub action and create a release:

```bash
$ git checkout branch

$ git tag -a v0.1.2 -m 'tag comment about release'

$ git push --tags
```
Once pushed the action will initiate and attempt to create a release.
![python-testing](https://github.com/Urban-Analytics/RAMP-UA/workflows/python-testing/badge.svg)
[![codecov](https://codecov.io/gh/Urban-Analytics/RAMP-UA/branch/master/graph/badge.svg)](https://codecov.io/gh/Urban-Analytics/RAMP-UA)
# RAMP-UA

This is the code repository for the RAMP Urban Analytics project.

The microsimulation model that ties the different components together is in the [microsim](./microsim) folder.

## Environment setup

This project currently supports running on Linux and macOS.

To start working with this repository you need to clone it onto your local machine:

```bash
$ git clone https://github.com/Urban-Analytics/RAMP-UA.git
$ cd RAMP-UA
```

This project requires a specific conda environment in order to run so you will need the [conda package manager system](https://docs.anaconda.com/anaconda/install/) installed. Once conda has been installed you can create an environment for this project using the provided environment file.

```bash
$ conda create env -f environment.yml
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

Once the above is complete you can run the model using the following line:

```bash
$ python microsim/microsim_model.py
```

Outputs are written to the [microsim/data/outputs](./microsim/data/outputs) directory.

For more details, see the full project repository on OSF.IO: https://osf.io/qzw6f/ (currently this is private, sorry, while we work out which data sources can be shared and which can't be, but the whole project will become public asap).

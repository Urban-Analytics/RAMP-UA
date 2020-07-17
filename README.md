![python-testing](https://github.com/Urban-Analytics/RAMP-UA/workflows/python-testing/badge.svg)
[![codecov](https://codecov.io/gh/Urban-Analytics/RAMP-UA/branch/master/graph/badge.svg)](https://codecov.io/gh/Urban-Analytics/RAMP-UA)
# RAMP-UA

This is the code repository for the RAMP Urban Analytics project.

The microsimulation model that ties the different components together is in the [microsim](./microsim) folder.

After cloning the repository, you will need to use [Git Large File Storage](https://git-lfs.github.com/) to download the input data. First install git-lfs, then run: 
```
git lfs fetch
git lfs checkout
``` 
to download the data.

Once you have the required libraries (see [microsim/README](./microsim/README.md)) you can run the model with:

```
python microsim/microsim_model.py
```

Outputs are written to the [microsim/data/outputs](./microsim/data/outputs) directory.

For more details, see the full project repository on OSF.IO: https://osf.io/qzw6f/ (currently this is private, sorry, while we work out which data sources can be shared and which can't be, but the whole project will become public asap).

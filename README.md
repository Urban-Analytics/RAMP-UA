![python-testing](https://github.com/Urban-Analytics/RAMP-UA/workflows/python-testing/badge.svg)
[![codecov](https://codecov.io/gh/Urban-Analytics/RAMP-UA/branch/master/graph/badge.svg)](https://codecov.io/gh/Urban-Analytics/RAMP-UA)
# RAMP-UA and EcoTwins integration

This branch is a first draft of the scaling up of the RAMP Urban Analytics project to a national level. It takes the project workflow as it is now and aims at making it run on other UK regions (in the current project only Devon is implemented, see issue https://github.com/Urban-Analytics/RAMP-UA/issues/254).
To do so, few steps are to be taken:
- reshuffling of parts of the data and files (folders structure)
- removing some hard-coded parts
- some probable re-organizing the project workflow, ie separating the data download from the rest of the model, which itself shall be separated into a preparatory phase and a actual running phase [more on this to come, @HSalat responsible for this part].

In this version, the model currently has been checked only against the OpenCL implementation of the RAMP-UA project (see #TO_DO list), but potentially could run in two ways:
1. Python / R implementation, found in [microsim/microsim_model.py](./microsim/microsim_model.py)
2. High performance OpenCL implementation, which can run on both CPU and GPU, which is found in the [microsim/opencl](./microsim/opencl) folder. 

Further documentation on the OpenCL model can be found at [microsim/opencl/doc](./microsim/opencl/doc)

Both models should be logically equivalent (with some minor differences). 



## Main differences with RAMP-UA
- The model doesn't use the same options in command line as offered in RAMP-UA, namely the only option is the `--parameters_file` that calls the `yml` where all the options are stored. This means that if you want to change them you have to edit this file only.
The "default" config file is still `default.yml` in `model_parameters` but this needs to be edited according to each user (at first use at least, see next bullet).
- Given some issues with the `get_cwd` command, this has been substituted all over the code with the new parameter `project-dir-absolute-path` that must be defined within `default.yml`. This choice is of course open to discussion and more ideas on this are welcome.
- The folders structure (see below)
- The file `microsim/column_names.py` has been included in the file `microsim/constants.py` (see class `ColumnNames`), which is meant to store all constant variables within the project. This can be further implemented.

Conclusive folders structure:
![EcoTwins folders structure](https://github.com/Urban-Analytics/RAMP-UA/tree/EcoTwins/img/folders_structure.jpg?raw=true)

RampUA model diagram
![RampUA model](https://github.com/Urban-Analytics/RAMP-UA/tree/EcoTwins/img/fmodel_diagram_EcoTwins.jpg?raw=true)

EcoTwins model diagram
![EcoTwins model](https://github.com/Urban-Analytics/RAMP-UA/tree/EcoTwins/img/model_diagram_rampUA.jpg?raw=true)


## Assumptions/How it is working now
- Currently only the OpenCL has been checked against this version.
- The current folders structure has the regional files located within each main folder (ie multiple folders within `data/regional_data`, `cache`, `output`), while we probably want to move this structure to a folder for each region and within them a folder for each of `data`, `cache`, `output` (see the document at https://hackmd.io/szqgmlYVTA-E5Pf4OSytZQ?view for some explicative diagrams).
- The re-structuring at the moment only involved the input (ie the parameters and data mainly), so at the moment the cache and the outputs ARE NOT by region yet. This means that if you want to run the model like it is now (11 March 2021) on different regions, the easiest (and lame) way to do it would be to create a copy of the existing project and change the parameters accordingly (see in fact that in the default parameters right now the project is called 'EcoTwins2', where I have stored the WestYorkshire data to try the model on that area).

## Environment setup
Please follow the instructions for the RAMP-UA project in general.
Though the command line options are not available anymore (as specified above), so now all the parameters are set directly within `model_parameters/default.yml`

### Caching of population initialisation
The population initialisation step runs before either of the models and can be time consuming (up to fa couple of hours for Devon data, a first trial for West Yorkshire stopped at 3+ hours).
This step is gonna generated in a separate phase of the model which runs only the 'initialisation' process (as mentioned at the beginning).


## Documentation
Documentation currently follows the one for RAMP-UA, plus some README files here and there (see for example in the data folder).


## TODO list:
Features that currently are not available, but are to be implemented on this version as well.
- [X] reorganise folders structure depending on the agreed option (discussed separately in the HackMD mentioned above)
- [X] implement the connection with the 'initialisation' process (TBD)
- [ ] implement R model compatibility
- [ ] run tests
- [X] understand where/when the `msoa_building_coordinates.json` is created (see `load_msoa_locations.py`)
- [ ] correct the hard-coded coordinates and MSOA code within `snapshot.py` add this to the input data
- [ ] fix the `project-dir-absolute-path` variable (eliminate if possible)

![python-testing](https://github.com/Urban-Analytics/RAMP-UA/workflows/python-testing/badge.svg)
[![codecov](https://codecov.io/gh/Urban-Analytics/RAMP-UA/branch/master/graph/badge.svg)](https://codecov.io/gh/Urban-Analytics/RAMP-UA)
# RAMP-UA and EcoTwins integration

## Notes for first time users (temporarily needed to have the model run)
The model is in a trial phase still, you can see the description and the main differences with RAMP-UA model hereafter. You can have it run though taking into account a few assumptions:
[spoiler: some steps are temporary, at least until all the data dependencies are checked]
1. Edit in `coding/constants.py` the variable `abspath` = "path_to_where_the_project_folder_is_located_in_your_computer"
2. inside `parameters_file/default.yml` edit:
    - `study-area` the name for the folder in `processed_data` and `output` folders that pertains to your chosen study area, the name you assign here will be used throughout the whole process;
    - `list-of-msoas` the name of the file containing the list of the MSOAs IDs for your study area
   
Once edited all the above, you have to run the model using the two `main` modules:
1. Run the initialisation process via `python coding/main_initialisation.py -p model_parameters/default.yml`, this module will download the data (to `data/raw_data/`), process them and prepare the table/data for the model that will be stored in `data/processed_data/your-study-area-folder/`
2. in this folder TEMPORARILY you shall add also these 4 files, that have not yet been completely implemented:
    1. `google_mobility_lockdown_daily_14_day_moving_average.csv`
    2. `initial_cases.csv`
    3. `msoas_risk.csv`
    4. `msoa_building_coordinates.json`
    - (these files are available for Devon and West Yorksire)
3. finally you can run the OpenCL model using `python coding/main_model.py -p model_parameters/default.yml`.


NOTE: the model now runs with only one parameter in input, check 'Main difference with RAMP-UA' below, so be aware of this (IE if you want to choose whether to run the OpenCL or not, you change the parameter directly in `model_parameters/default.yml`, not by command line);
The current version only runs the OpenCL model and the GUI is on its way (see the todo list for the dashboard map).

## Description
This branch is a first draft of the scaling up of the RAMP Urban Analytics project to a national level. It takes the project workflow as it is now and aims at making it run on other UK regions (in the current project only Devon is implemented, see issue https://github.com/Urban-Analytics/RAMP-UA/issues/254).
To do so, few steps are to be taken:
- reshuffling of parts of the data and files (folders structure)
- removing some hard-coded parts
- some probable re-organizing the project workflow, ie separating the data download from the rest of the model, which itself shall be separated into a preparatory phase and a actual running phase [@HSalat responsible for the data organization].

In this version, the model has for now been checked only against the OpenCL implementation of the RAMP-UA project (see #TO_DO list), but potentially it could run in two ways:
1. Python / R implementation, found in [coding/model/microsim](./coding/model/microsim)
2. High performance OpenCL implementation, which can run on both CPU and GPU, which is found in the [coding/model/opencl](./coding/model/opencl) folder. 

Further documentation on the OpenCL model can be found at [microsim/opencl/doc](./microsim/opencl/doc)

Both models should be logically equivalent (with some minor differences). 


## Main differences with RAMP-UA
- The model doesn't use the same options in command line as offered in RAMP-UA, namely the only option is the `--parameters_file` that calls the `yml` where all the options are stored. This means that if you want to change them you have to edit this file only.
The "default" config file is still `default.yml` in `model_parameters` but this needs to be edited according to each user (at first use at least, see next bullet).
- Given some issues with the `get_cwd` command, this has been substituted all over the code with the new parameter `project-dir-absolute-path` that must be defined within `default.yml`. This choice is of course open to discussion and more ideas on this are welcome.
- The folders structure (see below)
- The file `microsim/column_names.py` has been included in the file `coding/constants.py` (see class `ColumnNames`), which is meant to store all constant variables within the project. This can be further implemented.

Conclusive folders structure:

<!-- ![EcoTwins folders structure](https://github.com/Urban-Analytics/RAMP-UA/blob/EcoTwins/img/folders_structure.png){:height="50%" width="50%"} -->
<img src="https://github.com/Urban-Analytics/RAMP-UA/blob/EcoTwins/img/folders_structure.png" width="500">
RampUA model diagram:

![RampUA model](https://github.com/Urban-Analytics/RAMP-UA/blob/EcoTwins/img/model_diagram_rampUA.png)

EcoTwins model diagram:

![EcoTwins model](https://github.com/Urban-Analytics/RAMP-UA/blob/EcoTwins/img/model_diagram_EcoTwins.png)


## Limitations/How it is working now (TO BE UPDATED constantly)
- Currently only the OpenCL has been checked against this version.
- <strike>The current folders structure has the regional files located within each main folder (ie multiple folders within `data/regional_data`, `cache`, `output`), while we probably want to move this structure to a folder for each region and within them a folder for each of `data`, `cache`, `output` (see the document at https://hackmd.io/szqgmlYVTA-E5Pf4OSytZQ?view for some explicative diagrams)</strike>.
- <strike>The re-structuring at the moment only involved the input (ie the parameters and data mainly), so at the moment the cache and the outputs ARE NOT by region yet. This means that if you want to run the model like it is now (11 March 2021) on different regions, the easiest (and lame) way to do it would be to create a copy of the existing project and change the parameters accordingly (see in fact that in the default parameters right now the project is called 'EcoTwins2', where I have stored the WestYorkshire data to try the model on that area).</strike>

## Environment setup
Please follow the instructions for the RAMP-UA project in general.
Though the command line options are not available anymore (as specified above), so now all the parameters are set directly within `model_parameters/default.yml`

### Caching of population initialisation
The population initialisation step runs before either of the models and can be time consuming (up to a couple of hours for Devon data, 5 hours for West Yorkshire).
This step is generated in a first (separate) phase of the model which runs only the 'initialisation' process.


## Documentation
Documentation currently follows the one for RAMP-UA, plus some README files here and there (see for example in the data folder).


## TODO list:
Features that currently are not available, but are to be implemented on this version as well.
- [X] reorganise folders structure depending on the agreed option (discussed separately in the HackMD mentioned above)
- [X] get rid of hard-coded variables depending on the reorganised folders structure
- [X] implement the connection with the 'initialisation' process
- [ ] implement R model compatibility
- [ ] run tests
- [X] understand where/when the `msoa_building_coordinates.json` is created (see `load_msoa_locations.py`)
- [X] correct the hard-coded coordinates and MSOA code within `snapshot.py` add this to the initialisation process (@manluow)
- [ ] fix the `abspath` variable (eliminate if possible)
- [ ] import the snapshot creation for OpenCL version to the initialisation part: IE separate the opencl code (what is in now in `coding/model/opencl/ramp/`) and put the part that generates the opencl snapshot (`cache.npz`) into the initialisation part 
- [ ] what happens when one uses a study area name that already exists? (Raise exception ... overwrite existing files, or use the already existing cache/processed data?)
- [ ] think whether to separate the configuration file (`model_parameters/default.yml`) in two, one for the initialisation and one for the model

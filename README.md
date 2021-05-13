![python-testing](https://github.com/Urban-Analytics/RAMP-UA/workflows/python-testing/badge.svg)
[![codecov](https://codecov.io/gh/Urban-Analytics/RAMP-UA/branch/master/graph/badge.svg)](https://codecov.io/gh/Urban-Analytics/RAMP-UA)
# RAMP-UA and EcoTwins integration

## Notes for first time users (temporarily needed to have the model run)
The model is in a trial phase still, you can see the description and the main differences with RAMP-UA model hereafter. You can have it run though taking into account a few assumptions:
[spoiler: long list but please take the time to pass through all the points]
1. Change in the file `coding/constants.py` some variables names depending on the file you have, namely:
    - `abspath` = "path_to_where_the_project_folder_is_located_in_your_computer"
    - `class MSOAS_RISK_FILE` = name os the csv file that contains the risk for the MSOAS (in RAMP-UA it was called `msoas.csv` in `microsim/opencl/data/`, `msoa_danger_fn.csv` in `init_data/`, `msoas.rda` in `microsim/R/py_int/data/`);
    to have the model run, this file must be located (for now) in `data/processed_data/your_study_area_folder/`, when all ready it will go in the `raw_data` folder, and it will be inputted probably from the MSOAs shp (national file);
    the file contains a table where: the first column is a counter (1,2, ...) with no name, second column is called 'area_code' and has the MSOAs ids, third columns is 'risk' and contains strings values ('low, medium, high').
    - `class INITIAL_CASES` = name of the csv file that contains the seeding/initial cases (in RAMP-UA it was called `devon_initial_cases.csv` in `microsim/opencl/data/`, `devon_cases_fn.csv` in `init_data/`, `gam_cases.rda` in `microsim/R/py_int/data/`);
    similarly to `msoas_risk` file, this is temporarily located in `data/processed_data/your_study_area_folder/` but eventually there will be one singular national file located in `raw_data/reference_data/`;
    the file contains a table where: the first column is a counter (1,2, ...) with no name, second column is called 'um_cases' and is of type integer.
    - `class LIST_MSOAS` = this files contains the list of MSOAs that make your study area, it is made only of one column called `MSOA11CD`, if you have a different column name then change accordingly the constant variable that refers to it (IE the class `MSOAsID` inside `Class ColumnNames` inside the same module);
    the list of msoas file is a csv file and must be located in `model_parameters/`.
2. the TU file is located in `data/raw_data/county_data/` and must be named `tus_hse-[county where MSOAS are located].csv` ... this is temporary, in fact if you don't have the TU file it will be downloaded from Azure on the correct repository, though if you want to try the Test-West_Yorkshire MSOAs you will need to call the corresponding TU file like this, or the model will try to download the data for West Yorskshire that is 10 times bigger (and not needed) - ignore this point if you use Devon.
Note that the name for the county (which the TU file pertain to depending on your chosen MSOAs) is automatically detected by the model from the list of MSOAS using the look-up table, that is in `raw_data/reference_data/`.
3. also the lockdown file (Google mobility `google_mobility_lockdown_daily_14_day_moving_average.csv`) is temporary, use the old Devon one for Devon (ask me for the West Yorkshire one in case you run the test) and must be located in `data/processed_data/your_study_area_folder/`, I have to ask better Hadrien what he would like to do with this file, because he generated a `timeAtHomeIncreaseCTY.csv` file that is a national file (separated by counties) and is automatically downloaded and processed in the initialisation phase (see `coding/initialise/raw_data_handler/`), though I didn't manage to have this file work yet (or the code needs to be adapted to this new version).
4. inside `parameters_file/default.yml` edit:
    - `project-dir-absolute-path` this is really not optimal (I know!) and must be edited together with the same constant variable within 'coding/constants.py
    - `study-area` the name for the folder in `processed_data` and `output` folders that pertains to your chosen study area, the name you assign here will be used throughout the whole process.
5. the model now runs with only one parameter in input (in the command line), check 'Main difference with RAMP-UA' below, so be aware of this (IE if you want to choose whether to run the OpenCL or not, you change the parameter directly in `model_parameters/default.yml`, not by command line);
in any case THIS VERSION ONLY RUNS with OPENCL and anyways NOT WITH THE GUI YET (see the todo list for the dashboard map).
6. please come back to me (@ciupava) for any question



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
- [ ] correct the hard-coded coordinates and MSOA code within `snapshot.py` add this to the initialisation process (@manluow)
- [ ] fix the `project-dir-absolute-path` variable (eliminate if possible)
- [ ] import the snapshot creation for OpenCL version to the initialisation part: IE separate the opencl code (what is in now in `coding/model/opencl/ramp/`) and put the part that generates the opencl snapshot (`cache.npz`) into the initialisation part 
- [ ] what happens when one uses the same study area name? (raise exception ... overwrite existing files, or use the already existing cache/processed data?)

![python-testing](https://github.com/Urban-Analytics/RAMP-UA/workflows/python-testing/badge.svg)
[![codecov](https://codecov.io/gh/Urban-Analytics/RAMP-UA/branch/master/graph/badge.svg)](https://codecov.io/gh/Urban-Analytics/RAMP-UA)
# RAMP-UA and EcoTwins integration (alpha version)

Disclaimer: The model is still under active development.

This branch is the alpha version of the scaling up of the RAMP Urban Analytics project to the national level. It takes the project workflow in its last updated form and aims at making it run on other UK regions (in the current naster branch, only Devon is implemented, see issue https://github.com/Urban-Analytics/RAMP-UA/issues/254).

## Installation steps
1. Clone or Download the repository `Urban-Analytics/RAMP-UA` and switch to the EcoTwins branch 
```bash
$ git clone https://github.com/Urban-Analytics/RAMP-UA.git
$ cd RAMP-UA
$ git checkout EcoTwins
```
2. Install the EcoTwins environment (`$ conda env create -f environment.yml`)
3. Install the RAMP-UA Package (`$ python setup.py install` or `$ python setup.py develop`)
4. TEMPORARY: Manually assign the local path to the `RAMP-UA` folder to the variable `abspath` in `coding/constants.py` 

## Running the model
Make sure that you have selected the correct python interpreter (`python 3.7 (EcoTwins)`) and that you have activated the ecotwins environment (`conda activate ecotwins`)

5. Place inside `model_parameters/` a file containing the list of the MSOAs IDs for your study area (any list of MSOAs in England)
6. Inside `model_parameters/default.yml` edit:
    - `list-of-msoas`: add the name of the file previously placed
    - `study-area`: assign a name for the study area (it will be the name of the folders inside `processed_data` and `output` for that study area, and will be used throughout the whole process)
   
The model is now run in two steps:

7. The "initialisation" step processes the raw data for the chosen area (needs to be run only once for each new area); use:
```bash
$ python coding/main_initialisation.py -p model_parameters/default.yml
```
8. The "model" step runs the actual models and opens a dashboard to visualise the results; use:
```bash
$ python coding/main_model.py -p model_parameters/default.yml
```

## Troubleshooting
**TEMPORARY: SEEDING.**
Until the new COVID-19 cases seeding process is fully implemented, it is necessary to move manually 
- `initial_cases.csv`
- `msoas_risk.csv`

inside the study area folder in `processed_data`. `initial_cases.csv` is the total number of new cases per day for the area (can be modelled using the Devon data as reference, see master branch) and `msoas_risk.csv` can be retrieved from the shapefile dbf at `raw_data/national_data/MSOAS_shp`.

**The initialisation process gets stuck at downloading the QUANT data.**
These data are made of one large archive (about 2.3 Go) and can be difficult to download automatically for some computer set ups. The file can be downloaded directly (`https://ramp0storage.blob.core.windows.net/nationaldata/QUANT_RAMP.tar.gz`) and moved manually to `raw_data/national_data/` before running the process.

## Main differences with the RAMP-UA model
- The model doesn't use the same options in command line as offered in RAMP-UA, namely the only option is the `--parameters_file` that calls the `yml` where all the options are stored. This means that if you want to change them you have to edit this file only.
The "default" config file is still `default.yml` in `model_parameters` but this needs to be edited according to each user (at first use at least, see next bullet).
- Given some issues with the `get_cwd` command, this has been substituted all over the code with the new parameter `project-dir-absolute-path` that must be defined within `default.yml`. This choice is of course open to discussion and more ideas on this are welcome.
- The folders structure (see below) is different
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


### Caching of population initialisation
The population initialisation step runs before either of the models and can be time consuming (up to a couple of hours for Devon data, 5 hours for West Yorkshire).
This step is generated in a first (separate) phase of the model which runs only the 'initialisation' process.

## Documentation
Documentation currently follows the one for RAMP-UA, plus some README files here and there (see for example in the data folder).


## TODO list:
Features that currently are not available, important steps need to be tackled, issues that need to be fixed.
- [X] reorganise folders structure depending on the agreed option (discussed separately in the HackMD mentioned above)
- [X] get rid of hard-coded variables depending on the reorganised folders structure
- [X] implement the connection with the 'initialisation' process
- <strike> [ ] implement R model compatibility </strike>
- [ ] run Python tests
- [X] understand where/when the `msoa_building_coordinates.json` is created (see `load_msoa_locations.py`)
- [X] correct the hard-coded coordinates and MSOA code within `snapshot.py` add this to the initialisation process (@manluow)
- [ ] run tests
- [X] fix the `abspath` variable (eliminate if possible)
- [ ] import the snapshot creation for OpenCL version to the initialisation part: IE separate the opencl code (what is in now in `coding/model/opencl/ramp/`) and put the part that generates the opencl snapshot (`cache.npz`) into the initialisation part 
- [ ] what happens when one uses a study area name that already exists? (Raise exception ... overwrite existing files, or use the already existing cache/processed data?)
- [ ] think whether to separate the configuration file (`model_parameters/default.yml`) in two, one for the initialisation and one for the model

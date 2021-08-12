![python-testing](https://github.com/Urban-Analytics/RAMP-UA/workflows/python-testing/badge.svg)
[![codecov](https://codecov.io/gh/Urban-Analytics/RAMP-UA/branch/master/graph/badge.svg)](https://codecov.io/gh/Urban-Analytics/RAMP-UA)
# RAMP-UA and EcoTwins integration (alpha version)

Disclaimer: The model is still under active development.

This branch is the alpha version of the scaling up of the RAMP Urban Analytics project to the national level.
It takes the project workflow in its last updated form and aims at making it run on other UK regions (in the current master branch, only Devon is implemented, see issue https://github.com/Urban-Analytics/RAMP-UA/issues/254).

## Installation steps
1. Clone or download the repository `Urban-Analytics/RAMP-UA` and switch to the EcoTwins branch 
```bash
$ git clone https://github.com/Urban-Analytics/RAMP-UA.git
$ cd RAMP-UA
$ git checkout EcoTwins
```
2. Create an environment (called by default *ecotwins*) from the provided yml file
   `$ conda env create -f environment.yml`
3. Install the RAMP-UA Package
   `$ python setup.py install` or `$ python setup.py develop`

## Running the model
Make sure that you have activated the *ecotwins* environment (`conda activate ecotwins`) or, if running from an IDE, have selected the correct python interpreter (`python 3.7 (ecotwins)`, for example in MacOS this can be located in a path like `/usr/local/anaconda3/envs/ecotwins/bin/python`)

4. Place inside `model_parameters/` a csv file containing the list of the MSOA IDs for your study area (any list of MSOAs in England that follows the 2011 census boudaries definition) under the column `MSOA11CD`, and the initial number of cases inside each MSOA under the column `cases`.
5. Inside `model_parameters/default.yml` edit:
    - `list-of-msoas`: the name of the file previously placed
    - `study-area`: assign a name for the study area (this will be the name of the folders inside `processed_data` and `output` for that study area, and will be used throughout the whole process)
    - any other parameter that you wish to change from default (the start date of the simulation should be counted as the number of days since the 15th of February 2020)
   
(a custom parameters file could be created ad-hoc; in this case the appropriate file must be set when calling the parameters file in the following step)

The model is then run in two steps:

7. The "initialisation" step processes the raw data for the chosen area (needs to be run only once for each new area) in accordance with the parameters file set above; use:
```bash
$ python coding/main_initialisation.py -p model_parameters/default.yml
```
8. The "model" step runs the actual model and opens a dashboard to visualise the results in accordance with the parameters file set above; use:
```bash
$ python coding/main_model.py -p model_parameters/default.yml
```

## Troubleshooting

**The initialisation process gets stuck at downloading the QUANT data.**
These data are made of one large archive (about 2.3 Go) and can be difficult to download automatically for some computer set ups. The file can be downloaded directly (`https://ramp0storage.blob.core.windows.net/nationaldata/QUANT_RAMP.tar.gz`) and moved manually to `raw_data/national_data/` before running the process.

**Please report any other problem encountered to hsalat@turing.ac.uk or fbenitez@turing.ac.uk**

## Main differences with the RAMP-UA model
- The current version allows for choosing different study areas
- The data is downloaded from a remote repository and is avaialable for all England (UK?), the class that handles this process is `RawDataHandler`
- The model doesn't use the same options in command line as offered in RAMP-UA, the only option is the `--parameters_file`
- <strike> Given some issues with the `get_cwd` command, this has been substituted all over the code with the new parameter `project-dir-absolute-path` that must be defined within `default.yml`. This choice is of course open to discussion and more ideas on this are welcome.</strike>
- The folders structure (see below) is different (see the discussion in https://hackmd.io/szqgmlYVTA-E5Pf4OSytZQ?view)
- The file `microsim/column_names.py` has been included in the file `coding/constants.py` (see class `ColumnNames`)

Folders structure:

<!-- ![EcoTwins folders structure](https://github.com/Urban-Analytics/RAMP-UA/blob/EcoTwins/img/folders_structure.png){:height="50%" width="50%"} -->
<img src="https://github.com/Urban-Analytics/RAMP-UA/blob/EcoTwins/img/folders_structure.png" width="500">
RampUA model diagram:

![RampUA model](https://github.com/Urban-Analytics/RAMP-UA/blob/EcoTwins/img/model_diagram_rampUA.png)

EcoTwins model diagram:
![EcoTwins model](https://github.com/Urban-Analytics/RAMP-UA/blob/EcoTwins/img/model_diagram_EcoTwins.png)


### Caching of population initialisation
The population, within 'initialisation', initialisation step can be time consuming (up to a couple of hours for Devon data, 5 hours for West Yorkshire, IE 2 hours per 100 MSOAs).

Though, this step generates a cache that is stored and can be accessed from the model at any time.

## Documentation
Documentation currently follows the one for RAMP-UA, plus some README files here and there (see for example in the data folder).

## TODO list:
**High priority / Model in working state**
- [ ] run Python tests
- [ ] run multiple test scenarios

**Major updates**
- [ ] implement improved commuting
- [ ] model calibration
- [ ] improve outputs readability

**Possible improvements:**
- [ ] implement detailed timetables (weekends, seasons ... ?)
- [ ] import snapshot creation for OpenCL version to the initialisation part, i.e. separate the opencl code (what is in now in `coding/model/opencl/ramp/`) and put the part that generates the opencl snapshot (`cache.npz`) into the initialisation part 
- [ ] remove print statements in the json creation
- [ ] solve the 'manual' assignment of variables in `param.py`
- [ ] update household transmission simulation
- [ ] implement vaccines simulation
- [ ] give options for what happens when one uses a study area name that already exists (overwrite existing files, or use the already existing cache/processed data, let the user know)
- [ ] think whether to separate the configuration file (`model_parameters/default.yml`) in two, one for the initialisation and one for the model

**Past**
- [X] reorganise folders structure depending on the agreed option (discussed separately in the HackMD mentioned above)
- [X] get rid of hard-coded variables depending on the reorganised folders structure
- [X] implement the connection with the 'initialisation' process
- [X] integrate the data download code into the initialisation, test it
- [ ] <strike> implement R model compatibility </strike>
- [X] create new environment that takes into account also opencl requirements
- [X] solve the error from assertion in popoulation_initialisation (checks the sum up tp 1)
- [X] understand where/when the `msoa_building_coordinates.json` is created (see `load_msoa_locations.py`)
- [X] integrate the shp creation (from step above) in the main code: correct the hard-coded coordinates and MSOA code within `snapshot.py` add this to the initialisation process (@manluow)
- [X] fix the `abspath` variable (eliminate if possible) - ELIMINATED


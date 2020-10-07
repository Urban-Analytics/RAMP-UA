# Microsimulation
Code for running both implementations of the main RAMP-UA microsimulation. The program(s) collect required data, stitch them together, 
and then choose one of the Python/R or OpenCL models to iterate through dynamically.

The purpose of the model is to create a synthetic population of individuals and houses, 
simulate some common movements (e.g. shopping, schooling, leisure, etc.) and look at how different policies might 
be able to suppress the spread of COVID over the period of a few weeks or months.

## Setting up

Most of the data files that the code requires are quite big and are not stored in the repository. They will need to be downloaded separately. For instructions see the [README file in the `data` directory](./data/README.md).

The majority of the code is written in python. Most of the libraries used are fairly standard so should not cause too many problems. 
An [Anaconda environment (yml) file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) has been included ([../environment.yml](environment.yml)) which can be used to create an anaconda environment automatically with the following command:

```
conda env create -f environment.yml
```

Once you have your environment configured correctly, you can run the code by first returning to the main [RAMP-UA](..) directory 
and then running each model as follows:

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

## Process overview

The process that the model runs is as follows:

  1. Read in a synthetic population of individual people and households (created elsewhere).
  1. Read in some additional data, such as health statuses and travel behaviour and link it to the base population.
  1. Read in some flows created by spatial interaction models (created elsewhere) and assign individuals to their most likely venues. These estimate where people are likely to go when they go to school, shopping, etc.
  1. Estimate an initial disease status (one of Susceptible, Exposed, Infected, Recovered).
  1. Step the model so that the disease propogates through the population.

The 'stepping' procedure involves two stages:

  1. People who have the disease visit locations (e.g. shops) and impart some risk onto those places.
  1. Other people visit their most likely places, and receive some of that risk. This will influence the probability of them becoming infected.

Note that until a synthetic individual is actually assigned a disease, the process is deterministic. E.g. if one of the spatial interaction models estimates that an individual will visit two shops with probabilities 0.8 and 0.2, then when they 'visit' those places the risks are assigned proportionally based on those probabilities. Once an individual is assigned a disease status, the model becomes probabilistic.

## Code organisation / overview

The code is under development so this rough sketch needs improving and is likely to change_

The main file is [main.py](main.py). This parses the arguments, runs the population initialisation then chooses which 
timestep model to run (either the Python/R model or the OpenCL model). 

The `PopulationInitialisation` class reads and combines all the data sources and creates a population in a form ready to be iterated by either timestep model. 
The output of this is the `individuals` and `activity_locations` dataframes, described below. 

The `microsim.Microsim` class contains the Python / R model, the `step()` function runs timesteps of the model. 
This mainly works by operating on pandas dataframes them. The most important dataframes are:

### `individuals` dataframe

A table for the individual-level population, containing things like their age, health status, disease status, etc.

There are also columns to record where the individuals do their activities. Because an individual might visit many locations (e.g. shops), we store all possible locations they could visit, and the associated flows/proportions. E.g the column `Retail_Venues` contains a list of the IDs of the venues that the individual might visit (e.g. `[5, 11, 14]`) and `Retail_Flows` contains the probabilities of visitting each venue (e.g. `[0.1, 0.85, 0.05]`). Each possible activity will have a `*_Flows` and `*_Venues` column. The `*_Time` columns store the proportion of time that the individual spends doing the activity.

### `locations` dataframes

The IDs in the `*_Venues` columns (e.g. `[5, 11, 14]` above) point to rows in other dataframes. These other 'locations' dataframes store information about the locations themselves (e.g. individual shops or schools) and, importantly, have a column that represents the 'danger' associated with the location. If lots of people with the infection visit the location then the danger goes up, and increases the probability that someone else who visits the location will contract the disease.

Because we want some flexibility in the types of activities that people do, each activity, and its associated locations, are stored as a dictionary of `ActivityLocation` objects. The dictionary maps a desription of the activity (e.g. 'Retail') to an `ActivtiyLoction` object that, among other things, points to a `locations` dataframe for those particular locations (e.g. shops). That's the dataframe that ultimately stores all the information about the location. The IDs in the `Retail_Venues` column (see above) point to ros in the `ActivityLocation.locations` dataframe.


## Testing

The tests are being implemented in the `microsim.test_microsim_model.py` script. These need some serious development.

#####################################################################################
#####################################################################################
# Multi-parameter calibration of the OpenCL RAMP model using ABC
#####################################################################################
#####################################################################################

#####################################################################################
### Parameter definitions
# The OpenCL RAMP model creates a synthetic population and simulates the movements of individuals between various locations (home, work, school, shops). If an individual is infected then they impart an infection risk on to the locations that they visit. The severity of this infection risk is determined by whether the individual is presymptomatic, asymptomatic or symptomatic (the **individual's hazard rating**). The locations in the model thus come to have an infection risk, based upon how many infected individuals from each infection status have visited. When susceptible individuals without the disease visit these locations their risk of being infected is determined by the location's infection risk, alongside the transmission hazard associated with that kind of location (the location's hazard rating). An additional hazard rating, the current risk beta, is used to control the general transmissability of the disease (?) 

### Parameter calibration
# The current risk beta parameter is calibrated in [this notebook](http://localhost:8888/notebooks/Users/gy17m2a/OneDrive%20-%20University%20of%20Leeds/Project/RAMP-UA/experiments/calibration/abc-2-NewObs.ipynb). This involves determining a prior distribution for both the current risk beta parameter, and for the individual and location hazard parameters. Approximate Bayesian Computation (ABC) is then used to approximate the likelihood of different parameter values by running the model a large number of times. Each time parameter values are drawn randomly from the prior distributions, and the parameter values from the simulations with results closest to the observations are kept. A fitness function is used to assess the similarity of model results to observations. In this case, the Euclidean difference between the observed number of cases per week and the simulated number of cases per week is calculated. In this script, the model is ran using ABC for 105 days, with two populations. 
# In this notebook, the current risk beta parameter value which was associated with the highest fitness in the final population in the calibration process above is set as a constant, and assumed not to change over the life of the disease. However, as the disease evolves, it is expected that the other parameters pertaining to the hazard associated with individuals and locations will change. If a model is calibrated just once using historical data then it will be unable to account for this parameter evolution. However, if it is dynamically calibrated, then it can be optimised in response to changes in parameter values in real time. Dynamic calibration involves re-calibrating the model at each time-step (e.g. after one day, one week etc), and using the best performing parameter value from the previous time-step for the next time-step. 

### Parameter priors
# ABC requires prior distributions to be specified for the parameters.
# 
# <ins>Location hazards</ins> 
#  - `Home` (transmission hazard associated with being with an infected individual at home)
#      - <i>Fixed at 1.0, so all other multipliers are relative to this. This is almost certaintly the most risky activity (i.e. transmission at home is very likely) so no other priors allow values above 1.0. </i>
#  - `Retail` (transmission hazard associated with being with an infected individual in a shop)
#      - <i> Uniform between 0 and 1 </i>
#  - `PrimarySchool` (transmission hazard associated with being with an infected individual at primary school)
#      - <i> Uniform between 0 and 1 </i> 
#  - `SecondarySchool` (transmission hazard associated with being with an infected individual at secondary school)
#      - <i> Uniform between 0 and 1 </i> 
#  - `Work` (transmission hazard associated with being with an infected individual at work)
#      - <i> Uniform between 0 and 1 </i> 
#  
# <ins> Individual hazards</ins> 
#  - `asymptomatic` (transmission hazard associated with asymptomatic individuals)  
#       - <i> This is tricky because we don't know the hazard associated with asymptomatic transmission. James Salter used a prior of: N(0.444, 0.155) which gives the middle 95% as [0.138, 0.75] ([0.138 estimated here](https://www.medrxiv.org/content/10.1101/2020.06.04.20121434v2), [0.58 estimated here](https://jammi.utpjournals.press/doi/abs/10.3138/jammi-2020-0030), [0.75 estimated here (Table 1)](https://www.cdc.gov/coronavirus/2019-ncov/hcp/planning-scenarios.html). </i>
#  - `presymptomatic` (transmission hazard associated with presymptomatic individuals)  
#     - <i> Same as asymptomatic (but abc-2.iypnb fixes them at 1 ? </i>
#  - `symptomatic` (transmission hazard associated with symptomatic individuals)   
#     - <i> Same as asymptomatic (but abc-2.iypnb fixes them at 1 ?)</i>       

##############################################################
# To adapt an ABM to allow it to be optimised in response to data emerging in real time. This would mean performing dynamic calibration (i.e. re calibrating at every model time step) using ABC. -->
# At the moment, however, such predictive ABMs are generally calibrated once, using historical data to adjust their more flexible parameters such that the model predicts present and past conditions well. The models are then allowed to roll forward in time, independent of the world, to make a prediction. As the systems modelled are usually complex, it is likely that over longer time periods such models diverge from realistic estimates. Even over shorter time periods there is little done to check model performance, let alone constrain it. -->
##############################################################

#####################################################################################
#### Import modules required
#####################################################################################
import multiprocessing as mp
import numpy as np
import os
import itertools
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
import datetime
from matplotlib import cm
import time

# PYABC (https://pyabc.readthedocs.io/en/latest/)
import pyabc
from pyabc.transition.multivariatenormal import MultivariateNormalTransition  # For drawing from the posterior
# Quieten down the pyopencl info messages (just print errors)
import logging

logging.getLogger("pyopencl").setLevel(logging.ERROR)

# RAMP model
sys.path.append("../../")
from microsim.initialisation_cache import InitialisationCache

# Import arbitrary distribution class for using posterior estimates as new priors
from arbitrary_distribution import ArbitraryDistribution, GreaterThanZeroParameterTransition

# Bespoke RAMP classes for running the model
from opencl_runner import OpenCLWrapper  # Some additional functions to simplify running the OpenCL model
from opencl_runner import OpenCLRunner

# Set this to False to recalculate all results (good on HPC or whatever).
# If true then it loads pre-calculated results from pickle files (much quicker)
LOAD_PICKLES = True

#####################################################################################
### Create observed cases data
#####################################################################################

#####################################################################################
# Cases_devon_weekly is based on government data recording the number of new cases each week, per MSOA. It has been corrected to account for lags and other shortcomings in the testing process, and summed up to cover the whole of Devon. Full details are here: http://localhost:8888/notebooks/Users/gy17m2a/OneDrive%20-%20University%20of%20Leeds/Project/RAMP-UA/experiments/calibration/observation_data/CreatingObservations-Daily-InterpolateSecond.ipynb
# Cases_devon_daily is created through linear interpolation of the weekly data. This daily data is required for seeding the model. Because the interpolation is carried out for each MSOA seperately, when the data is joined back together for the whole of Devon it is not exactly equivalent to cases_devon_weekly. 
# Currently, we are using the original cases_devon_weekly data in the distance function to evaluate the performance of the model. This makes sense because it is the most accurate weekly data that we have. However, it means when comparing the weekly model results to the observations, the model doesn't seem to be exactly the same as the observations, even during the seeding process. (Within the distance function all particles will be equally far from the observations during the seeding process, so shouldn't negatively affect things in this sense).
#####################################################################################

## Get dataframe with totals for whole of Devon
cases_devon_weekly = pd.read_csv("observation_data/weekly_cases_devon.csv")
# Add column with cumulative sums rather than cases per week
cases_devon_weekly['CumulativeCases'] = cases_devon_weekly['OriginalCases'].cumsum()
# Keep just cumulative cases
cases_devon_weekly = cases_devon_weekly['CumulativeCases'].values

# Read in daily devon case data (interpolated from weekly)
cases_devon_daily = pd.read_csv("observation_data/daily_cases_devon.csv")
# Add column with cumulative sums rather than cases per day
cases_devon_daily['CumulativeCases'] = cases_devon_daily['OriginalCases'].cumsum()
# Keep just cumulative cases
cases_devon_daily = cases_devon_daily['CumulativeCases'].values

# Convert this interpolated data used in seeding back to weekly
# List the index matching the end day of each week (e.g. 7, 14, 21 etc (minus 1 for indexing starting at 0)
n_days = len(cases_devon_daily)
week_end_days = list(range(6,n_days+1,7))

# Keep only the values from the end of each week
cases_devon_daily_summed_weekly = cases_devon_daily[week_end_days]

#####################################################################################
#### Setup Model
#####################################################################################
# Read in parameters
PARAMETERS_FILE = os.path.join("../../","model_parameters", "default.yml")
PARAMS = OpenCLRunner.create_parameters(parameters_file=PARAMETERS_FILE)

# Optionally initialise the population, delete the old OpenCL model snapshot (i.e. an already-initialised model) and
# re-create a new one. Useful if something may have changed (e.g. changed the lockdown file).
OPENCL_DIR = os.path.join("..", "..", "microsim", "opencl")
SNAPSHOT_FILEPATH = os.path.join(OPENCL_DIR, "snapshots", "cache.npz")
assert os.path.isfile(SNAPSHOT_FILEPATH), f"Snapshot doesn't exist: {SNAPSHOT_FILEPATH}"

# Define constants
const_params_dict = {"current_risk_beta": 0.019, "home": 1.0}

#####################################################################################
## Define prior parameter values for running with dynamic calibration 
#####################################################################################
# Current risk beta
current_risk_beta_rv = pyabc.RV("norm", 0.1, 0.155)

# School and retail are all uniform between 0-1
retail_rv, primary_school_rv, secondary_school_rv = ( pyabc.RV("uniform", 0, 1) for _ in range(3)  )
# Work needs some dampening because we know that workplaces are too big in the current implementation
work_rv = pyabc.RV("beta", 0.1, 2)

# Individual multipliers (see justification at the start of this notebook).
# Asymptomatic is normal such that the middle 95% is the range [0.138, 0.75]
presymptomatic_rv, symptomatic_rv, asymptomatic_rv = (pyabc.RV("norm", 0.444, 0.155) for _ in range(3))

# Group all random variables together and give them a string name (this is needed for the distribution later)
all_rv = { "retail": retail_rv, "primary_school": primary_school_rv, "secondary_school": secondary_school_rv, "work": work_rv,
    "presymptomatic": presymptomatic_rv, "symptomatic": symptomatic_rv, "asymptomatic": asymptomatic_rv}

# Can decorate normal distributions later to make sure they are positive
decorated_rvs = {name: pyabc.LowerBoundDecorator(rv, 0.0) for name, rv in all_rv.items()}

# Create a distrubtion from these random variables
original_priors = pyabc.Distribution(**decorated_rvs)

#####################################################################################
# Set-up model for running 
#####################################################################################
# Generate unique timestamp
seconds = int(time.time())
print(seconds)

NUM_SEED_DAYS = 14
USE_GPU = False
ITERATIONS = 42

# Initialise the population
DATA_DIR = os.path.join("..", "..", "devon_data")
cache_dir = os.path.join(DATA_DIR, "caches")
cache = InitialisationCache(cache_dir=cache_dir)
if cache.is_empty():
    raise Exception(f"The cache in {cache_dir} has not been initialised. Probably need to run the code a",
                    "few cells up that initialises the population")

individuals_df, activity_locations = cache.read_from_cache()
print(f"Activity locations: {activity_locations}")

# Dictionary with parameters for running model
admin_params = {"quiet": True, "use_gpu": USE_GPU, "store_detailed_counts": True, "start_day": 0,  "run_length": ITERATIONS,
                "current_particle_pop_df": None,  "parameters_file": PARAMETERS_FILE, "snapshot_file": SNAPSHOT_FILEPATH, 
                "opencl_dir": OPENCL_DIR, "individuals_df": individuals_df, "observations_weekly_array": cases_devon_weekly,
                'num_seed_days' :NUM_SEED_DAYS}

#####################################################################################
# Set-up model to run with dynamic calibration
#####################################################################################
n_pops = 10
n_particles = 150

# Create template for model
template = OpenCLWrapper(const_params_dict, **admin_params)
# Not sure why this is needed. Wthout it we get an error when passing the template object to ABCSMC below
template.__name__ = OpenCLWrapper.__name__

# Set up model
abc = pyabc.ABCSMC(models=template, parameter_priors=original_priors, distance_function=OpenCLWrapper.dummy_distance, 
    sampler=pyabc.sampler.SingleCoreSampler(), transitions=GreaterThanZeroParameterTransition(),
                  population_size = n_particles)

# Prepare to run the model
db_path = ("sqlite:///" + "Outputs/RunModel_OneShotCalibration/ramp_da_{}pops_{}particles_{}days_{}.db".format(n_pops, n_particles,ITERATIONS, seconds))  # Path to database

# abc.new() needs the database location and any observations that we will use (these are passed to the distance_function
# provided to pyabc.ABCSMC above). Currently the observations are provided to the model when it is initialised and 
# these are then used at the end of the model run() function. So they don't need to be provided here.
abc_history = abc.new(db=db_path,observed_sum_stat=None)  # {'observation': observations_array, "individuals": individuals_df}
run_id = abc_history.id

#####################################################################################
# Run model with dynamic calibration
#####################################################################################
print("Running for {} days, {} population, {} particles".format(ITERATIONS, n_pops, n_particles))
abc_history = abc.run(max_nr_populations=n_pops)

fname = "InitialModelCalibration-{}pops-{}particles-{}days_constantCRB.pkl".format(n_pops, n_particles, ITERATIONS)
with open( fname, "wb" ) as f:
        pickle.dump(abc_history, f)

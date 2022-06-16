#####################################################################################
#####################################################################################
# Running RAMP OpenCL model with dynamic calibration

# ### Parameter definitions
#####################################################################################
# The OpenCL RAMP model creates a synthetic population and simulates the movements of individuals between various locations (home, work, school, shops). If an individual is infected then they impart an infection risk on to the locations that they visit. The severity of this infection risk is determined by whether the individual is presymptomatic, asymptomatic or symptomatic (the **individual's hazard rating**). The locations in the model thus come to have an infection risk, based upon how many infected individuals from each infection status have visited. When susceptible individuals without the disease visit these locations their risk of being infected is determined by the location's infection risk, alongside the transmission hazard associated with that kind of location (the location's hazard rating). An additional hazard rating, the current risk beta, is used to control the general transmissability of the disease (?) 

# ### Parameter calibration
# The current risk beta parameter is calibrated in [this notebook](http://localhost:8888/notebooks/Users/gy17m2a/OneDrive%20-%20University%20of%20Leeds/Project/RAMP-UA/experiments/calibration/abc-2-NewObs.ipynb). This involves determining a prior distribution for both the current risk beta parameter, and for the individual and location hazard parameters. Approximate Bayesian Computation (ABC) is then used to approximate the likelihood of different parameter values by running the model a large number of times. Each time parameter values are drawn randomly from the prior distributions, and the parameter values from the simulations with results closest to the observations are kept. A fitness function is used to assess the similarity of model results to observations. In this case, the Euclidean difference between the observed number of cases per week and the simulated number of cases per week is calculated. In this script, the model is ran using ABC for 105 days, with two populations. 
# In this notebook, the current risk beta parameter value which was associated with the highest fitness in the final population in the calibration process above is set as a constant, and assumed not to change over the life of the disease. However, as the disease evolves, it is expected that the other parameters pertaining to the hazard associated with individuals and locations will change. If a model is calibrated just once using historical data then it will be unable to account for this parameter evolution. However, if it is dynamically calibrated, then it can be optimised in response to changes in parameter values in real time. Dynamic calibration involves re-calibrating the model at each time-step (e.g. after one day, one week etc), and using the best performing parameter value from the previous time-step for the next time-step. 

# ### Parameter priors
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

#####################################################################################
#####################################################################################

#####################################################################################
### Import modules
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

# Import arbitrary distribution class for using posterior estimates as new priors
sys.path.append("/nfs/a319/gy17m2a/RAMP-UA/experiments/calibration")
from arbitrary_distribution import ArbitraryDistribution, GreaterThanZeroParameterTransition

# RAMP model
sys.path.append("../../")
from microsim.initialisation_cache import InitialisationCache

# Bespoke RAMP classes for running the model
from opencl_runner import OpenCLWrapper  # Some additional functions to simplify running the OpenCL model
from opencl_runner import OpenCLRunner

# Set this to False to recalculate all results (good on HPC or whatever).
# If true then it loads pre-calculated results from pickle files (much quicker)
LOAD_PICKLES = True

#####################################################################################
### Create observed cases data

# Cases_devon_weekly is based on government data recording the number of new cases each week, per MSOA. It has been corrected to account for lags and other shortcomings in the testing process, and summed up to cover the whole of Devon. Full details are here: http://localhost:8888/notebooks/Users/gy17m2a/OneDrive%20-%20University%20of%20Leeds/Project/RAMP-UA/experiments/calibration/observation_data/CreatingObservations-Daily-InterpolateSecond.ipynb
# Cases_devon_daily is created through linear interpolation of the weekly data. This daily data is required for seeding the model. Because the interpolation is carried out for each MSOA seperately, when the data is joined back together for the whole of Devon it is not exactly equivalent to cases_devon_weekly. 
# Currently, we are using the original cases_devon_weekly data in the distance function to evaluate the performance of the model. This makes sense because it is the most accurate weekly data that we have. However, it means when comparing the weekly model results to the observations, the model doesn't seem to be exactly the same as the observations, even during the seeding process. (Within the distance function all particles will be equally far from the observations during the seeding process, so shouldn't negatively affect things in this sense).
#####################################################################################
# Generate unique timestamp
seconds = int(time.time())
print(seconds)

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

# #### Define currrent risk beta
current_risk_beta_val =0.019

#####################################################################################
## Run model with default parameter values 

# This shows what happens with the 'default' (manually calibrated) model.  
# These parameters are from the 'optimal' particle in the InitialModelCalibration.ipynb script, in which the model is ran with ABC with ten populations for 133 days.
#####################################################################################
#### Initialise model
## Define parameters
ITERATIONS = 105  # Number of days to run for
assert (ITERATIONS /7).is_integer() # check it is divisible by 7 
NUM_SEED_DAYS = 14  # Number of days to seed the population
USE_GPU = False
STORE_DETAILED_COUNTS = False
REPETITIONS = 5
USE_HEALTHIER_POP = True

# Think x by 7 because daily data is used in running model?
assert ITERATIONS < len(cases_devon_weekly)*7,     f"Have more iterations ({ITERATIONS}) than observations ({len(cases_devon_weekly)*7})."

# Initialise the class so that its ready to run the model.
OpenCLRunner.init( iterations = ITERATIONS, repetitions = REPETITIONS, observations =  cases_devon_weekly.T,
    use_healthier_pop = USE_HEALTHIER_POP, use_gpu = USE_GPU, store_detailed_counts = STORE_DETAILED_COUNTS,
    parameters_file = PARAMETERS_FILE, opencl_dir = OPENCL_DIR,snapshot_filepath = SNAPSHOT_FILEPATH,
    num_seed_days = NUM_SEED_DAYS)

#####################################################################################
## Define prior parameter values for running with dynamic calibration 
#####################################################################################
#### Define constants (not touched by ABC. Include parameters that should not be optimised)
const_params_dict = { "current_risk_beta": current_risk_beta_val,"home": 1.0}

#### Define random variables and the prior distributions
# School and retail multipliers are uniform between 0-1
retail_rv, primary_school_rv, secondary_school_rv = (pyabc.RV("uniform", 0, 1) for _ in range(3))
# Work needs some dampening because we know that workplaces are too big in the current implementation
work_rv = pyabc.RV("beta", 0.1, 2)
# Individual multipliers
# Asymptomatic is normal such that the middle 95% is the range [0.138, 0.75]  (see justification in abc)
# No idea about (pre)symptomatic, so use same distribution as asymptomatic
presymptomatic_rv, symptomatic_rv, asymptomatic_rv = (pyabc.RV("norm", 0.444, 0.155) for _ in range(3))

# Group all random variables together and give them a string name (this is needed for the distribution later)
all_rv = {
    "retail": retail_rv, "primary_school": primary_school_rv, "secondary_school": secondary_school_rv, "work": work_rv,
    "presymptomatic": presymptomatic_rv, "symptomatic": symptomatic_rv, "asymptomatic": asymptomatic_rv}

## Create a distrubtion from these random variables
decorated_rvs = {name: pyabc.LowerBoundDecorator(rv, 0.0) for name, rv in all_rv.items()}

# Define the original priors
original_priors = pyabc.Distribution(**decorated_rvs)

#####################################################################################
# Set-up model for running with dynamic calibration
#####################################################################################
# Set the size of a data assimilation window in days:
da_window_size =14

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
admin_params = {"quiet": True, "use_gpu": USE_GPU, "store_detailed_counts": True, "start_day": 0,  "run_length": da_window_size,
                "current_particle_pop_df": None,  "parameters_file": PARAMETERS_FILE, "snapshot_file": SNAPSHOT_FILEPATH, 
                "opencl_dir": OPENCL_DIR, "individuals_df": individuals_df, 
                "observations_weekly_array": cases_devon_weekly,'num_seed_days' :NUM_SEED_DAYS}

# Create dictionaries to store the dfs, weights or history from each window (don't need all of these, but testing for now)
dfs_dict, weights_dict, history_dict = {}, {},{}

# Store starting time to use to calculate how long processing the whole window has taken
starting_windows_time = datetime.datetime.now()

# Define number of and number of populations to run for,
windows = 8
n_pops = 5
next_pop = 7
n_particles = 100

#####################################################################################
# Run model with dynamic calibration
#####################################################################################
print("Running for {} windows, with {} particles per population and {} populations".format(windows, n_particles, n_pops))

# Loop through each window
for window_number in range(1, windows + 1):
  window_start_time = time.time()
  print("Window number: ", window_number)

  # Edit the da_window size in the admin params
  admin_params['run_length'] =  da_window_size * window_number
  print("Running for {} days".format(admin_params['run_length']))

  # Create template for model
  template = OpenCLWrapper(const_params_dict, **admin_params)
  # Not sure why this is needed. Wthout it we get an error when passing the template object to ABCSMC below
  template.__name__ = OpenCLWrapper.__name__

  # Define priors
  # If first window, then use user-specified (original) priors
  if window_number == 1:
      priors = original_priors

  # If a subsequent window, then generate distribution from posterior from previous window
  else:
      priors = ArbitraryDistribution(abc_history)

  # Set up model
  abc = pyabc.ABCSMC(
      models=template,  # Model (could be a list)
      parameter_priors=priors,  # Priors (could be a list)
      # summary_statistics=OpenCLWrapper.summary_stats,  # Summary statistics function (output passed to 'distance')
      distance_function=OpenCLWrapper.dummy_distance,  # Distance function
      sampler=pyabc.sampler.SingleCoreSampler(),
      transitions=GreaterThanZeroParameterTransition(),
      population_size = n_particles)
      # Single core because the model is parallelised anyway (and easier to debug)
      # sampler=pyabc.sampler.MulticoreEvalParallelSampler()  # The default sampler
      #transition=transition,  # Define how to transition from one population to the next

  # Prepare to run the model
  db_path = ("sqlite:///" + "ramp_da_{}.db".format(seconds))  # Path to database

  # abc.new() needs the database location and any observations that we will use (these are passed to the
  # distance_function provided to pyabc.ABCSMC above). Currently the observations are provided to the model
  # when it is initialised and these are then used at the end of the model run() function. So they don't
  # need to be provided here.
  run_id = abc.new(db=db_path,observed_sum_stat=None)  # {'observation': observations_array, "individuals": individuals_df}

  # Run model
  abc_history = abc.run(max_nr_populations=n_pops)

  # Save some info on the posterior parameter distributions.
  for t in range(0, abc.history.max_t + 1):

    # for this t (population) extract the 100 particle parameter values, and their weights
    df, w = abc.history.get_distribution(m=0, t=t)

    # Save these for use in plotting the prior on the plot of parameter values in each population
    dfs_dict["w{},pop{}".format(window_number, t)] = df
    weights_dict["w{}, pop{}".format(window_number, t)] = w
    history_dict["w{}".format(window_number)] = abc_history

    # Add weights to dataframe
    df['weights'] = w

    # Save df
    df.to_csv("Outputs/New/df_t{}_{}windows_window{}, {}pops_{}particles_Crb{}_{}.csv".format(t, windows, window_number, n_pops, n_particles, current_risk_beta_val, seconds))

  # Save abc_history object    
  fname = "Outputs/New/abc_history_{}windows_window{}, {}pops_{}particles_Crb{}_{}.pkl".format(windows, window_number, n_pops, n_particles, current_risk_beta_val, seconds)
  with open( fname, "wb" ) as f:
           pickle.dump(abc_history, f)

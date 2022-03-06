#### Import modules required
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

#os.chdir("C:/Users/gy17m2a/OneDrive - University of Leeds/Project/RAMP-UA/experiments/calibration")

# PYABC (https://pyabc.readthedocs.io/en/latest/)
import pyabc
from pyabc.transition.multivariatenormal import MultivariateNormalTransition  # For drawing from the posterior
# Quieten down the pyopencl info messages (just print errors)
import logging

from experiments import arbitrary_distribution as ad

logging.getLogger("pyopencl").setLevel(logging.ERROR)

# Import arbitrary distribution class for using posterior estimates as new priors
#from arbitrary_distribution import ArbitraryDistribution, GreaterThanZeroParameterTransition

# RAMP model
from microsim.initialisation_cache import InitialisationCache

# Bespoke RAMP classes for running the model
from experiments import opencl_runner as oclr
#from ..opencl_runner import OpenCLWrapper  # Some additional functions to simplify running the OpenCL model
#from ..opencl_runner import OpenCLRunner

# Set this to False to recalculate all results (good on HPC or whatever).
# If true then it loads pre-calculated results from pickle files (much quicker)
LOAD_PICKLES = True





# Observed cases data
cases_msoa_weekly = pd.read_csv("observation_data/weekly_cases_msoas_shifted_mpld_smoothed_IS.csv")
# remove date column
cases_msoa_weekly = cases_msoa_weekly.iloc[:, 1:]
# Transpose
cases_msoa_weekly = cases_msoa_weekly.T

# Observations are cases per msoa.
# Store as an array for us in model (more efficient?)
# (first axis is the msoa number, second is the day)
observations_weekly_array = cases_msoa_weekly.to_numpy()

## Get dataframe with totals for whole of Devon
cases_devon_weekly = pd.read_csv("observation_data/weekly_cases_devon_shifted_mpld_smoothed_IS.csv")
# Create new dataframe with cumulative sums rather than cases per day
cases_devon_weekly['CumulativeCases'] = cases_devon_weekly['OriginalCases'].cumsum()

# Read in daily devon case data (interpoalted from weekly)
cases_devon_daily = pd.read_csv("observation_data/daily_cases_devon_shifted_mpld_smoothed_IS.csv")
# Create new dataframe with cumulative sums rather than cases per day
cases_devon_daily['CumulativeCases'] = cases_devon_daily['OriginalCases'].cumsum()

# Read in daily msoa case data (interpoalted from weekly)
cases_msoa_daily = pd.read_csv("observation_data/daily_cases_msoas_shifted_mpld_smoothed_IS.csv")
#Create new dataframe with cumulative sums rather than cases per day
#cases_msoa_daily['CumulativeCases'] = cases_msoa_daily['OriginalCases'].cumsum()
# remove date column
cases_msoa_daily = cases_msoa_daily.iloc[:, 1:]
cases_msoa_daily = cases_msoa_daily.T
cases_msoa_daily.columns = range(1,(cases_msoa_daily.shape[1])+1)
cases_msoa_daily = cases_msoa_daily.add_prefix('Day')



PARAMETERS_FILE = os.path.join("../../","model_parameters", "default.yml")
PARAMS = oclr.OpenCLRunner.create_parameters(parameters_file=PARAMETERS_FILE)


OPENCL_DIR = os.path.join("..", "..", "microsim", "opencl")
SNAPSHOT_FILEPATH = os.path.join(OPENCL_DIR, "snapshots", "cache.npz")
assert os.path.isfile(SNAPSHOT_FILEPATH), f"Snapshot doesn't exist: {SNAPSHOT_FILEPATH}"





## Define parameters
ITERATIONS = 105  # Number of iterations to run for
assert (ITERATIONS /7).is_integer() # check it is divisible by 7 
NUM_SEED_DAYS = 7  # Number of days to seed the population
USE_GPU = False
STORE_DETAILED_COUNTS = False
REPETITIONS = 2 # XXXX
USE_HEALTHIER_POP = True

# Think x by 7 because daily data is used in running model?
assert ITERATIONS < len(cases_devon_weekly)*7,     f"Have more iterations ({ITERATIONS}) than observations ({len(cases_devon_weekly)*7})."

# Initialise the class so that its ready to run the model.
oclr.OpenCLRunner.init( iterations = ITERATIONS, repetitions = REPETITIONS, observations = cases_devon_weekly,
    use_healthier_pop = USE_HEALTHIER_POP, use_gpu = USE_GPU, store_detailed_counts = STORE_DETAILED_COUNTS,
    parameters_file = PARAMETERS_FILE, opencl_dir = OPENCL_DIR,snapshot_filepath = SNAPSHOT_FILEPATH,
    num_seed_days = NUM_SEED_DAYS)


oclr.OpenCLRunner.update(repetitions=REPETITIONS)  # Temporarily use more repetitions to give a good baseline
oclr.OpenCLRunner.update(store_detailed_counts=True)  # Temporarily output age breakdowns
(fitness_manualcalibration, sim_manualcalibration, obs_manualcalibration, out_params_manualcalibration, summaries_manualcalibration) = oclr.OpenCLRunner.run_model_with_params_abc({}, return_full_details=True, quiet = False)
oclr.OpenCLRunner.update(repetitions=REPETITIONS)
oclr.OpenCLRunner.update(store_detailed_counts=STORE_DETAILED_COUNTS)




best_params = {'retail': 0.5234169902075426, 'primary_school': 0.8189852636000965, 'secondary_school': 1.1516441210574122, 
               'work': 0.07179707294863724, 'presymptomatic': 0.3870375860336805, 'symptomatic': 0.6581503001512558, 
               'asymptomatic': 0.2609686051991401, 'current_risk_beta': 0.01786588871820107}
const_params_dict = {"home": 1.0}
oclr.OpenCLRunner.set_constants(const_params_dict)

## Run model with these best parameters
oclr.OpenCLRunner.update(store_detailed_counts=True)  # Temporarily output age breakdowns
(distance_mc2, sim_mc2, obs_mc2, out_params_mc2, summaries_mc2) = oclr.OpenCLRunner.run_model_with_params_abc(
    best_params, return_full_details=True, quiet = False)
oclr.OpenCLRunner.update(store_detailed_counts=STORE_DETAILED_COUNTS)




# Check the model returns the observations correctly i.e. that theyre the same length
np.array_equal(obs_manualcalibration, cases_devon_daily.loc[:len(sim_manualcalibration)-1,"CumulativeCases"])

# Plot
fig, ax = plt.subplots(1,1)
x = range(len(sim_manualcalibration))
# ax.plot(x, sim_manualcalibration, label="sim", color="orange")
ax.plot(x, sim_mc2, label="sim_BestPerfoming", color="green")
ax.plot(x, obs_manualcalibration, label="obs", color="blue")
ax.legend()




const_params_dict = { "current_risk_beta": 0.025239665550846085,"home": 1.0}
oclr.OpenCLRunner.set_constants(const_params_dict)




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
    "presymptomatic": presymptomatic_rv, "symptomatic": symptomatic_rv, "asymptomatic": asymptomatic_rv
}

# Plot priors
fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
x = np.linspace(-1, 2, 99)  # (specified so that we have some whole numbers)
marker = itertools.cycle((',', '+', '.', 'o', '*'))
for i, (var_name, variable) in enumerate(all_rv.items()):
    # var_name = [ k for k,v in locals().items() if v is variable][0]  # Hack to get the name of the variable
    ax = axes.flatten()[i]
    # ax.plot(x, pyabc.Distribution(param=variable).pdf({"param": x}), label = var_name, marker=next(marker), ms=3)
    ax.plot(x, pyabc.Distribution(param=variable).pdf({"param": x}))
    ax.set_title(var_name)
    ax.axvline(x=0.0, ls='--', color="grey", label="x=0")
    ax.axvline(x=1.0, ls='--', color="grey", label="x=1")
# fig.tight_layout()
fig.suptitle("Priors")
fig.show()

## Create a distrubtion from these random variables
decorated_rvs = {name: pyabc.LowerBoundDecorator(rv, 0.0) for name, rv in all_rv.items()}

# Define the original priors
original_priors = pyabc.Distribution(**decorated_rvs)




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
                "observations_weekly_array": observations_weekly_array,'num_seed_days' :7}
















# Create dictionaries to store the dfs, weights or history from each window (don't need all of these, but testing for now)
dfs_dict, weights_dict, history_dict = {}, {},{}

# Store starting time to use to calculate how long processing the whole window has taken
starting_windows_time = datetime.datetime.now()

# Define number of windows to run for
windows = 2

# Loop through each window
for window_number in range(1, windows + 1):
    print("Window number: ", window_number)

    # Edit the da_window size in the admin params
    admin_params['run_length'] = admin_params['run_length'] * window_number
    print("Running for {} days".format(da_window_size * window_number))

    # Create template for model
    oclr.OpenCLRunner.temp = 1
    template = oclr.OpenCLWrapper(const_params_dict, **admin_params)
    # Not sure why this is needed. Wthout it we get an error when passing the template object to ABCSMC below
    template.__name__ = oclr.OpenCLWrapper.__name__

    # Define priors
    # If first window, then use user-specified (original) priors
    if window_number == 1:
        priors = original_priors
    # If a subsequent window, then generate distribution from posterior from previous window
    else:
        priors = ad.ArbitraryDistribution(abc_history)

    # Set up model
    abc = pyabc.ABCSMC(
        models=template,  # Model (could be a list)
        parameter_priors=priors,  # Priors (could be a list)
        # summary_statistics=OpenCLWrapper.summary_stats,  # Summary statistics function (output passed to 'distance')
        distance_function=oclr.OpenCLWrapper.dummy_distance,  # Distance function
        transitions=ad.GreaterThanZeroParameterTransition(),
        # Single core because the model is parallelised anyway (and easier to debug)
        sampler=pyabc.sampler.SingleCoreSampler(),
        #sampler=pyabc.sampler.MulticoreEvalParallelSampler(),  # The default sampler
        #transition=transition,  # Define how to transition from one population to the next
    )

    # Prepare to run the model
    db_path = ("sqlite:///" + "ramp_da.db")  # Path to database

    # abc.new() needs the database location and any observations that we will use (these are passed to the
    # distance_function provided to pyabc.ABCSMC above). Currently the observations are provided to the model
    # when it is initialised and these are then used at the end of the model run() function. So they don't
    # need to be provided here.
    run_id = abc.new(db=db_path,observed_sum_stat=None)  # {'observation': observations_array, "individuals": individuals_df}

    # Run model
    abc_history = abc.run(max_nr_populations=10)

    # Save some info on the posterior parameter distributions.
    for t in range(0, abc.history.max_t + 1):

        # for this t (population) extract the 100 particle parameter values, and their weights
        df_t1, w_t1 = abc.history.get_distribution(m=0, t=t)

        # Save these for use in plotting the prior on the plot of parameter values in each population
        dfs_dict["w{},pop{}".format(window_number, t)] = df_t1
        weights_dict["w{}, pop{}".format(window_number, t)] = w_t1
        history_dict["w{}".format(window_number)] = abc_history


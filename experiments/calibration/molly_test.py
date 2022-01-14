#### Import modules required
import multiprocessing as mp
import numpy as np
import yaml  # pyyaml library for reading the parameters.yml file
import os
import itertools
import pandas as pd
import unittest
import pickle
import copy
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import datetime
import matplotlib.cm as cm

# For easier plots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# These allow you to plot in a notebook -- may first need to install the jupyter lab plotly extension:
# jupyter labextension install jupyterlab-plotly@4.8.2
from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)

# PYABC (https://pyabc.readthedocs.io/en/latest/)
import pyabc
from pygam import LinearGAM  # For graphing posteriors
from pyabc.transition.multivariatenormal import MultivariateNormalTransition  # For drawing from the posterior
# Quieten down the pyopencl info messages (just print errors)
import logging

logging.getLogger("pyopencl").setLevel(logging.ERROR)

# Import arbitrary distribution class
sys.path.append('..')
from ArbitraryDistribution import ArbitraryDistribution

# RAMP model
from microsim.initialisation_cache import InitialisationCache
# from microsim.opencl.ramp.run import run_headless
# from microsim.opencl.ramp.snapshot_convertor import SnapshotConvertor
# from microsim.opencl.ramp.snapshot import Snapshot
# from microsim.opencl.ramp.params import Params, IndividualHazardMultipliers, LocationHazardMultipliers
# from microsim.opencl.ramp.simulator import Simulator
# from microsim.opencl.ramp.disease_statuses import DiseaseStatus

# Bespoke RAMP classes for running the model
import sys

sys.path.append('..')
from opencl_runner import OpenCLWrapper  # Some additional functions to simplify running the OpenCL model
from opencl_runner import OpenCLRunner

# Set this to False to recalculate all results (good on HPC or whatever).
# If true then it loads pre-calculated results from pickle files (much quicker)
LOAD_PICKLES = True

##########################################################################
##########################################################################
# Read spatial data
##########################################################################
##########################################################################
from microsim.load_msoa_locations import load_osm_shapefile, load_msoa_shapes

# Directory where spatial data is stored
gis_data_dir = ("../../devon_data")
# osm_buildings = load_osm_shapefile(gis_data_dir)
devon_msoa_shapes = load_msoa_shapes(gis_data_dir, visualize=False)
devon_msoa_shapes = devon_msoa_shapes.set_index('Code', drop=True, verify_integrity=True)


devon_msoa_codes = pd.DataFrame({'msoa11cd' :devon_msoa_shapes.index.to_list()})
devon_msoa_codes.to_csv("observation_data/devon_msoa_codes.csv", index=False)

##########################################################################
##########################################################################
# Create observed cases data
# Can maybe move processing into other file and save outputs and just read in
# to avoid confusion
##########################################################################
##########################################################################
# Observed cases data
# These were prepared by Hadrien and made available on the RAMP blob storage (see the observation data README).
cases_msoa = pd.read_csv(os.path.join("observation_data", "england_initial_cases_MSOAs.csv")).set_index('MSOA11CD',
                                                                                                        drop=True,
                                                                                                        verify_integrity=True)

# Merge them to the GIS data for convenience
cases_msoa = cases_msoa.join(other=devon_msoa_shapes, how="inner")  # Joins on the indices (both indices are MSOA code)
assert len(cases_msoa) == len(devon_msoa_shapes)  # Check we don't use any areas in the join

# For some reason we lose the index name when joining
cases_msoa.index.name = "msoa11cd"

# Melt so that cases on each day (D0, D1, ... D404) become a value in a new row
# (Also need to convert the index (area code) to a column)
cases_msoa_melt = pd.melt(cases_msoa.reset_index(), id_vars='MSOA11CD',
                          value_vars=["D" + str(i) for i in range(405)]).rename(columns={'value': 'cases'})
cases_msoa_melt = cases_msoa_melt.set_index('MSOA11CD', drop=True)  # Keep the index as the MSOA
cases_msoa_melt['day'] = cases_msoa_melt['variable'].apply(
    lambda day: int(day[1:]))  # Strip off the initial 'D' to get the day number

# Observations are cases per msoa.
# Store as an array for us in model (more efficient?)
# (first axis is the msoa number, second is the day)
observations_array = cases_msoa.iloc[:, 0:405].to_numpy()

# Reformat into dataframe with one column containing days and one column
# containing cases
observations_msoas_df = cases_msoa.iloc[:, 0:405]
observations_msoas_df.reset_index(level=0, inplace=True)
# Change to MSOA as columns, days as rows
observations_msoas_df = observations_msoas_df.T
# set MSOA codes as column names and remove as a row
observations_msoas_df.rename(columns=observations_msoas_df.iloc[0], inplace=True)
observations_msoas_df.drop(observations_msoas_df.index[0], inplace=True)
# Add column with Day number at front of columns
observations_msoas_df.insert(0, 'Cases', range(0, len(observations_msoas_df)))

# Create new dataframe with cumulative sums rather than cases per day
observations_msoas_cumulative_df = observations_msoas_df.copy()
for colname in observations_msoas_cumulative_df.columns[1:].tolist():
    observations_msoas_cumulative_df[colname] = observations_msoas_cumulative_df[colname].cumsum()

## Create dataframe with totals for whole of Devon
observations_devon_cumulative_df = observations_msoas_cumulative_df.copy()
# Add total across all MSOAs
observations_devon_cumulative_df['Cases'] = observations_devon_cumulative_df.iloc[:, 1:].sum(axis=1)
# Drop MSOA values
observations_devon_cumulative_df.drop(observations_devon_cumulative_df.columns[1:108], axis=1, inplace=True)
# reset index
observations_devon_cumulative_df.reset_index(inplace=True, drop=True)

################################
##########################################################################
# Setup Model
# Optionally initialise the population, delete the old OpenCL model snapshot (i.e. an already-initialised model) and
# re-create a new one. Useful if something may have changed (e.g. changed the lockdown file).
##########################################################################
##########################################################################
OPENCL_DIR = os.path.join("..", "..", "microsim", "opencl")
SNAPSHOT_FILEPATH = os.path.join(OPENCL_DIR, "snapshots", "cache.npz")
assert os.path.isfile(SNAPSHOT_FILEPATH), f"Snapshot doesn't exist: {SNAPSHOT_FILEPATH}"

DATA_DIR = os.path.join("..", "..", "devon_data")
cache_dir = os.path.join(DATA_DIR, "caches")
cache = InitialisationCache(cache_dir=cache_dir)
if cache.is_empty():
    raise Exception(f"The cache in {cache_dir} has not been initialised. Probably need to run the code a",
                    "few cells up that initialises the population")

individuals_df, activity_locations = cache.read_from_cache()

print(f"Activity locations: {activity_locations}")

##########################################################################
##########################################################################
# Define constants
# These are not touched by ABC. Include parameters that should not be optimised.
##########################################################################
##########################################################################
const_params_dict = {
    "current_risk_beta": 0.03,  # Global risk multplier (leave this as it is and allow the other parameters to vary)
    "home": 1.0,
    # Risk associated with being at home. Again leave this constant so the coefficients of other places will vary around it
}

##########################################################################
##########################################################################
# Define random variables and the prior distributions
# Random variables are the global parameters.
##########################################################################
##########################################################################
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

# ax.legend()
# fig.tight_layout()
fig.suptitle("Priors")
fig.show()

## Create a distrubtion from these random variables
decorated_rvs = {name: pyabc.LowerBoundDecorator(rv, 0.0) for name, rv in all_rv.items()}

# Define the original priors
original_priors = pyabc.Distribution(**decorated_rvs)

##########################################################################
##########################################################################
# Setup loop for running model
##########################################################################
##########################################################################
# Path to parameters
parameters_file = os.path.join("../../", "model_parameters/",
                               "default.yml")  # Need to tell it where the default parameters are
# Set the size of a data assimilation window in days:
da_window_size = 14
# Dictionary with parameters for running model
admin_params = {"quiet": True, "use_gpu": True, "store_detailed_counts": True, "start_day": 0,
                "run_length": da_window_size,
                "current_particle_pop_df": None,
                "parameters_file": parameters_file, "snapshot_file": SNAPSHOT_FILEPATH, "opencl_dir": OPENCL_DIR}

# Create dictionaries to store the dfs, weights or history from each window (don't need all of these, but testing for now)
dfs_dict = {}
weights_dict = {}
history_dict = {}

# Store starting time to use to calculate how long processing the whole window has taken
starting_windows_time = datetime.datetime.now()

# Define number of windows to run for
windows = 3

# Loop through each window
for window_number in range(1, windows + 1):
    print("Window number: ", window_number)

    # Edit the da_window size in the admin params
    # print("Running for 14 days")
    admin_params['run_length'] = admin_params['run_length'] * window_number
    print("Running for {} days".format(da_window_size * window_number))

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
        distance_function=OpenCLWrapper.distance,  # Distance function
        sampler=pyabc.sampler.SingleCoreSampler()
        # Single core because the model is parallelised anyway (and easier to debug)
        # sampler=pyabc.sampler.MulticoreEvalParallelSampler()  # The default sampler
        # transition=transition,  # Define how to transition from one population to the next
    )

    # Path to database?
    db_path = ("sqlite:///" + "ramp_da.db")
    run_id = abc.new(db_path, {'observation': observations_array, "individuals": individuals_df})

    # Run model
    abc_history = abc.run(max_nr_populations=2)

    # Save some info on the posterior parameter distributions.
    for t in range(0, abc.history.max_t + 1):
        print(t)
        # for this t (population) extract the 100 particle parameter values, and their weights
        df_t1, w_t1 = abc.history.get_distribution(m=0, t=t)
        # Are these equivalent? yes!
        # df_t1_2, w_t1_2 = abc_history.get_distribution(m=0, t=abc_history.max_t)
        # df_t1.equals(df_t1_2)
        # (w_t1 == w_t1_2).all()

        # Save these for use in plotting the prior on the plot of parameter values in each population
        dfs_dict["w{},pop{}".format(window_number, t)] = df_t1
        weights_dict["w{}, pop{}".format(window_number, t)] = w_t1
        history_dict["w{}".format(window_number)] = abc_history

    # Merge dataframe and weights and sort by weight (highest weight at the top)
    # _df['weight'] = _w
    # posterior_df = _df.sort_values('weight', ascending=False).reset_index()
    # posterior_df.to_csv("Plots/window_number{}_posterior_df.csv".format(window_number), index = False)

#############################################################################################################
#############################################################################################################
# Run the model X (50?) times using paramater values drawn from the posterior
# Plot the results to compare the performance of the model with the observations
#############################################################################################################
#############################################################################################################

# Initialise the class so that its ready to run the model.
## Define parameters
PARAMETERS_FILE = os.path.join("../../", "model_parameters", "default.yml")
PARAMS = OpenCLRunner.create_parameters(parameters_file=PARAMETERS_FILE)

ITERATIONS = 100  # Number of iterations to run for
NUM_SEED_DAYS = 10  # Number of days to seed the population
USE_GPU = True
STORE_DETAILED_COUNTS = False
REPETITIONS = 5
USE_HEALTHIER_POP = True
# assert ITERATIONS < len(OBSERVATIONS), \
# f"Have more iterations ({ITERATIONS}) than observations ({len(OBSERVATIONS)})."

OpenCLRunner.init(iterations=ITERATIONS,
                  repetitions=REPETITIONS,
                  observations=observations_devon_cumulative_df,
                  use_gpu=USE_GPU,
                  use_healthier_pop=USE_HEALTHIER_POP,
                  store_detailed_counts=STORE_DETAILED_COUNTS,
                  parameters_file=PARAMETERS_FILE,
                  opencl_dir=OPENCL_DIR,
                  snapshot_filepath=SNAPSHOT_FILEPATH)

##### define the abc_history object (not necessary as this will be most recent abc_history anyway)
abc_history = history_dict['w3']

# Define the number of samples to take from the posterior distribution of parameters
N_samples = 50
df, w = abc_history.get_distribution(m=0, t=abc_history.max_t)

# Sample from the dataframe of posteriors using KDE
kde = MultivariateNormalTransition(scaling=1)
kde.fit(df, w)
samples = kde.rvs(N_samples)

# Now run N models and store the results of each one
fitness_l = []  # Fitness values for each sample (model)
sim_l = []  # The full simulation results
obs_l = []  # Observations (should be the same for each sample)
out_params_l = []  # The parameters objects used in each sample (all parameters in the model)
out_calibrated_params_l = []  # The values of the specific calibrated parameters for the sample
summaries_l = []  # The summaries objects

negative_count = 0  # Count the number of negatives returned in the KDE posterior
for i, sample in samples.iterrows():
    # Check for negatives. If needed, resample
    while (sample < 0).values.any():
        # while (any(value < 0 for value in sample.values())):
        print("Found negatives. Resampling")
        negative_count += 1
        sample = kde.rvs()
        # Added in this line as the sample was in the wrong format for the while loop
        sample = pd.Series(sample)

    # Create a dictionary with the parameters and their values for this sample
    param_values = {param: sample[str(param)] for param in priors}

    # Run the model
    (_fitness, _sim, _obs, _out_params, _summaries) = \
        OpenCLRunner.run_model_with_params_abc(param_values, return_full_details=True)
    print(f"Fitness: {_fitness}.")
    # print(f"Fitness: {_fitness}. Sample: {sample}")

    fitness_l.append(_fitness)
    sim_l.append(_sim)
    obs_l.append(_obs)
    out_params_l.append(_out_params)
    out_calibrated_params_l.append(param_values)
    summaries_l.append(_summaries)

print(f"Finished sampling. Ignored {negative_count} negative samples.")

# Sanity check - that observations in each case are the same length?
for i in range(len(obs_l) - 1):
    assert np.array_equal(obs_l[0], obs_l[i])


# Save these because it took ages to sample
def pickle_samples(mode, *arrays):
    if mode == "save":
        with open("abc-2-samples.pkl", "wb") as f:
            for x in arrays:
                pickle.dump(x, f)
        return
    elif mode == "load":
        with open("abc-2-samples.pkl", "rb") as f:
            fitness_l = pickle.load(f)
            sim_l = pickle.load(f)
            obs_l = pickle.load(f)
            out_params_l = pickle.load(f)
            out_calibrated_params_l = pickle.load(f)
            summaries_l = pickle.load(f)
        return (fitness_l, sim_l, obs_l, out_params_l, out_calibrated_params_l, summaries_l)
    else:
        raise Exception(f"Unkonwn mode: {mode}")


pickle_samples('save', fitness_l, sim_l, obs_l, out_params_l, out_calibrated_params_l, summaries_l)

# print(f"Original fitness: {round(fitness0)}\nOptimised fitness: {round(fitness)}")

#####################################################################
# Plot the individual results for each sample
#####################################################################
# Normalise fitness to 0-1 to calculate transparency
_fitness = np.array(fitness_l)  # Easier to do maths on np.array
fitness_norm = (_fitness - min(_fitness)) / (max(_fitness) - min(_fitness))

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
x = range(len(sim_l[0]))
for i in range(len(summaries_l)):
    ax.plot(x, OpenCLRunner.get_cumulative_new_infections(summaries_l[i]),
            # label=f"Particle {df.index[sample_idx[i]]}",
            color="black", alpha=1 - fitness_norm[i]  # (1-x because high fitness is bad)
            )

    # ax.text(x=len(sim_l[i]), y=sim_l[i][-1], s=f"Fitness {round(fitness_l[i])}", fontsize=8)
    # ax.text(x=len(sim_l[i]), y=sim_l[i][-1], s=f"P{df.index[sample_idx[i]]}, F{round(fitness_l[i])}", fontsize=8)
# Plot observations
ax.plot(x, obs_l[0], label="Observations", color="blue")
# Plot result from manually calibrated model
# ax.plot(x, OpenCLRunner.get_cumulative_new_infections(summaries0), label="Initial sim", color="orange")
ax.legend()
# plot_summaries(summaries=summaries_l[0], plot_type="error_bars", observations=OBSERVATIONS)
plt.xlabel("Days")
plt.ylabel("Cases")
plt.show()

del _fitness, fitness_norm

# ##########################################################################
# ##########################################################################
# ### Save dicts
# ##########################################################################
# with open('8windows_14days_each_finalpop_dfs_dict.pkl', 'wb') as f:
#     pickle.dump(dfs_dict, f)
# with open('8windows_14days_each_finalpop_ws_dict.pkl', 'wb') as f:
#     pickle.dump(weights_dict, f)
# with open('8windows_14days_each_finalpop_history_dict.pkl', 'wb') as f:
#     pickle.dump(history_dict, f)
#
# with open('8windows_14days_each_finalpop_dfs_dict.pkl', 'rb') as f:
#     dfs_dict = pickle.load(f)
# with open('8windows_14days_each_finalpop_ws_dict.pkl', 'rb') as f:
#     weights_dict = pickle.load(f)
# with open('8windows_14days_each_finalpop_history_dict.pkl', 'rb') as f:
#     history_dict = pickle.load(f)
#
# ##########################################################################
# ##########################################################################
# ### Plot the final population for each window
# ##########################################################################
# ##########################################################################
# evenly_spaced_interval = np.linspace(0, 1, 8)
# colors = [cm.autumn_r(x) for x in evenly_spaced_interval]
#
# fig, axes = plt.subplots(3,int(len(original_priors)/2), figsize=(12,10))
# for i, param in enumerate(original_priors.keys()):
#     ax = axes.flat[i]
#     color_i =0
#     for history_name, history in history_dict.items():
#         color = colors[color_i]
#         df, w = history.get_distribution(m=0, t=history.max_t)
#         pyabc.visualization.plot_kde_1d(df, w, x=param, ax=ax,
#                 label=history_name,
#                 #alpha=1.0 if t==0 else float(t)/abc_history.max_t, # Make earlier populations transparent
#                 color= color)
#         if param!="work":
#                 ax.set_xlim(0,1)
#         if param=="secondary_school" or param=='presymptomatic' or param =='symptomatic':
#              ax.set_ylim(0,2.5)
#         elif param == 'retail' or param == 'primary_school':
#              ax.set_ylim(0,1.4)
#         elif param == 'work' :
#              ax.set_ylim(0,20)
#              ax.set_xlim(0,0.4)
#         elif param == 'asymptomatic' :
#              ax.set_ylim(0,7)
#         ax.legend(fontsize="small")
#         #ax.axvline(x=posterior_df.loc[1,param], color="grey", linestyle="dashed")
#         #ax.set_title(f"{param}: {posterior_df.loc[0,param]}")
#         ax.set_title(f"{param}")
#         handles, labels = ax.get_legend_handles_labels()
#         ax.get_legend().remove()
#         color_i = color_i +1
# fig.legend(handles, labels, loc='center right', fontsize = 17,
#            bbox_to_anchor=(1.01, 0.17))
#           # ncol = 8, bbox_to_anchor=(0.5, -0.07))
# axes[2,2].set_axis_off()
# axes[2,1].set_axis_off()
# fig.tight_layout()
# fig.show()
# fig.savefig("Plots/8windows_14days_each_finalpop.jpg")
#
#
# # ##########################################################################
# # ##########################################################################
# # ### Plot all populations for each window
# # ##########################################################################
# # ##########################################################################
# colors = [cm.autumn_r(x) for x in evenly_spaced_interval]
# alphas= [1, 1]
# linestyles = ['dotted', 'solid']
# fig, axes = plt.subplots(3,int(len(original_priors)/2), figsize=(12,10))
# for i, param in enumerate(original_priors.keys()):
#     ax = axes.flat[i]
#     col_i = 0
#     for history_name, history in history_dict.items():
#         for t in range(history.max_t + 1):
#             print(t)
#             df, w = history.get_distribution(m=0, t=t)
#             pyabc.visualization.plot_kde_1d(df, w, x=param, ax=ax,
#                 label="{}, pop {}".format(history_name, t),
#                 alpha= alphas[t],
#                 color = colors[col_i],
#                 linestyle = linestyles[t])
#             if param!="work":
#                 ax.set_xlim(0,1)
#             if param=="secondary_school" or param=='presymptomatic' or param =='symptomatic':
#                  ax.set_ylim(0,2.5)
#             elif param == 'retail' or param == 'primary_school':
#                  ax.set_ylim(0,1.4)
#             elif param == 'work' :
#                  ax.set_ylim(0,20)
#             elif param == 'asymptomatic' :
#                  ax.set_ylim(0,7)
#             ax.legend(fontsize="small")
#             #ax.axvline(x=posterior_df.loc[1,param], color="grey", linestyle="dashed")
#             #ax.set_title(f"{param}: {posterior_df.loc[0,param]}")
#             ax.set_title(f"{param}")
#             handles, labels = ax.get_legend_handles_labels()
#             ax.get_legend().remove()
#         col_i = col_i+1
# axes[2,2].set_axis_off()
# axes[2,1].set_axis_off()
# fig.legend(handles, labels, loc='center right', fontsize = 17,ncol =2,
#            bbox_to_anchor=(1.01, 0.17))
# fig.tight_layout()
# fig.show()
# fig.savefig("Plots/8windows_14days_each_allpops.jpg")


#     # # ##########################################################################
#     # # ##########################################################################
#     # # ### Algorithm diagnostics
#     # # ##########################################################################
#     # # ##########################################################################
#     # _, arr_ax = plt.subplots(2, 2)
#     #
#     # pyabc.visualization.plot_sample_numbers(abc_history, ax=arr_ax[0][0])
#     # pyabc.visualization.plot_epsilons(abc_history, ax=arr_ax[0][1])
#     # #pyabc.visualization.plot_credible_intervals(
#     # #    history, levels=[0.95, 0.9, 0.5], ts=[0, 1, 2, 3, 4],
#     # #    show_mean=True, show_kde_max_1d=True,
#     # #    refval={'mean': 2.5},
#     # #    arr_ax=arr_ax[1][0])
#     # pyabc.visualization.plot_effective_sample_sizes(abc_history, ax=arr_ax[1][1])
#     #
#     # plt.gcf().set_size_inches((12, 8))
#     # plt.gcf().tight_layout()
#     # plt.savefig("Plots/window_number{}_algorithm_diagnostics.jpg".format(window_number))
#     # plt.show()
#     #
#     # # ########## Plot the marginal posteriors
#     # fig, axes = plt.subplots(3,int(len(original_priors)/2), figsize=(12,10))
#     #
#     # #cmap = { 0:'k',1:'b',2:'y',3:'g',4:'r' }  # Do this automatically for len(params)
#     #
#     # for i, param in enumerate(original_priors.keys()):
#     #     ax = axes.flat[i]
#     #     for t in range(abc_history.max_t + 1):
#     #         df, w = abc_history.get_distribution(m=0, t=t)
#     #         pyabc.visualization.plot_kde_1d(df, w, x=param, ax=ax,
#     #             label=f"{param} PDF t={t}",
#     #             alpha=1.0 if t==0 else float(t)/abc_history.max_t, # Make earlier populations transparent
#     #             color= "black" if t==abc_history.max_t else None # Make the last one black
#     #         )
#     #         if param!="work":
#     #             ax.set_xlim(0,1)
#     #         ax.legend(fontsize="small")
#     #         #ax.axvline(x=posterior_df.loc[1,param], color="grey", linestyle="dashed")
#     #         #ax.set_title(f"{param}: {posterior_df.loc[0,param]}")
#     #         ax.set_title(f"{param}")
#     #
#     # fig.tight_layout()
#     # fig.show()
#     # fig.savefig("Plots/window_number{}_marginal_posteriors.jpg".format(window_number))
#     #
#     # print("Finished window {} in {}".format(window_number, datetime.datetime.now()- starting_windows_time))
#     # ##os.remove("ramp_da2.db")
#
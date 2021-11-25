#### Import modules required
import multiprocessing as mp
import numpy as np
import yaml # pyyaml library for reading the parameters.yml file
import os
import itertools
import pandas as pd
import unittest
import pickle
import copy
import random
import matplotlib.pyplot as plt
import scipy.stats as stats

# For easier plots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# These allow you to plot in a notebook -- may first need to install the jupyter lab plotly extension:
# jupyter labextension install jupyterlab-plotly@4.8.2
from plotly.offline import init_notebook_mode
init_notebook_mode(connected = True)

# PYABC (https://pyabc.readthedocs.io/en/latest/)
import pyabc
from pygam import LinearGAM  # For graphing posteriors
from pyabc.transition.multivariatenormal import MultivariateNormalTransition  # For drawing from the posterior
# Quieten down the pyopencl info messages (just print errors)
import logging
logging.getLogger("pyopencl").setLevel(logging.ERROR)

# RAMP model
from microsim.initialisation_cache import InitialisationCache
#from microsim.opencl.ramp.run import run_headless
#from microsim.opencl.ramp.snapshot_convertor import SnapshotConvertor
#from microsim.opencl.ramp.snapshot import Snapshot
#from microsim.opencl.ramp.params import Params, IndividualHazardMultipliers, LocationHazardMultipliers
#from microsim.opencl.ramp.simulator import Simulator
#from microsim.opencl.ramp.disease_statuses import DiseaseStatus

# Bespoke RAMP classes for running the model
import sys
sys.path.append('..')
from opencl_runner import OpenCLWrapper # Some additional functions to simplify running the OpenCL model

# Set this to False to recalculate all results (good on HPC or whatever).
# If true then it loads pre-calculated results from pickle files (much quicker)
LOAD_PICKLES = True


##########################################################################
##########################################################################
# Read case data and spatial data
##########################################################################
##########################################################################
from microsim.load_msoa_locations import load_osm_shapefile, load_msoa_shapes

# Directory where spatial data is stored
gis_data_dir = ("../../devon_data")
#osm_buildings = load_osm_shapefile(gis_data_dir)
devon_msoa_shapes = load_msoa_shapes(gis_data_dir, visualize=False)
devon_msoa_shapes = devon_msoa_shapes.set_index('Code', drop=True, verify_integrity=True)

# Observed cases data
# These were prepared by Hadrien and made available on the RAMP blob storage (see the observation data README).
cases_msoa = pd.read_csv(os.path.join("observation_data", "england_initial_cases_MSOAs.csv")).set_index('MSOA11CD', drop=True, verify_integrity=True)

# Merge them to the GIS data for convenience
cases_msoa = cases_msoa.join(other = devon_msoa_shapes, how="inner")  # Joins on the indices (both indices are MSOA code)
assert len(cases_msoa) == len(devon_msoa_shapes)  # Check we don't use any areas in the join

# For some reason we lose the index name when joining
cases_msoa.index.name = "MSOA11CD"

# Melt so that cases on each day (D0, D1, ... D404) become a value in a new row
# (Also need to convert the index (area code) to a column)
cases_msoa_melt = pd.melt(cases_msoa.reset_index(), id_vars='MSOA11CD',
                          value_vars=[ "D"+str(i) for i in range(405) ]).rename(columns={'value':'cases'})
cases_msoa_melt = cases_msoa_melt.set_index('MSOA11CD', drop=True) # Keep the index as the MSOA
cases_msoa_melt['day'] = cases_msoa_melt['variable'].apply(lambda day: int(day[1:])) # Strip off the initial 'D' to get the day number

##########################################################################
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
    "current_risk_beta": 0.03, # Global risk multplier (leave this as it is and allow the other parameters to vary)
    "home": 1.0,  # Risk associated with being at home. Again leave this constant so the coefficients of other places will vary around it
}

##########################################################################
##########################################################################
# Define random variables and the prior distribution
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

## Create a distrubtion from these random variables
decorated_rvs = { name: pyabc.LowerBoundDecorator(rv, 0.0) for name, rv in all_rv.items() }

# The distribution is a collection of the priors and their name (e.g. 'asymptomatic'=asymptomatic_rv, 'work'=work_rv, .. )
# so can just unpack the decorated_rvs dictionary and pass it straight in
priors = pyabc.Distribution(**decorated_rvs)

##########################################################################
##########################################################################
# Create model 'template'
# Constants and random variables need to be handed differently. To set constants,
# create an instance of OpenCLWrapper; this is a 'template' instance of the model.
# Then ABC calls that instance directly via the __call__ function, which allows random
# variables to be passed. I followed the suggestions from here: https://github.com/ICB-DCM/pyABC/issues/446
##########################################################################
##########################################################################
parameters_file = os.path.join("../../", "model_parameters/", "default.yml")  # Need to tell it where the default parameters are

da_window_size = 14 # Set the size of a data assimilation window in days:

admin_params = { "quiet":True, "use_gpu": True, "store_detailed_counts": True, "start_day": 0, "run_length": da_window_size,
                "current_particle_pop_df": None,
                "parameters_file": parameters_file, "snapshot_file": SNAPSHOT_FILEPATH, "opencl_dir": OPENCL_DIR
               }

template = OpenCLWrapper(const_params_dict, **admin_params)

# Not sure why this is needed. Wthout it we get an error when passing the template object to ABCSMC below
template.__name__ = OpenCLWrapper.__name__

print(f"Created a template model with: \n\tconstant params: {const_params_dict}\n\tadmin params:{admin_params}")


##########################################################################
##########################################################################
# First window
##########################################################################
##########################################################################
abc = pyabc.ABCSMC(
    models=template, # Model (could be a list)
    parameter_priors=priors, # Priors (could be a list)
    summary_statistics=OpenCLWrapper.summary_stats,  # Summary statistics function (output passed to 'distance')
    distance_function=OpenCLWrapper.distance,  # Distance function
    sampler = pyabc.sampler.SingleCoreSampler()  # Single core because the model is parallelised anyway (and easier to debug)
    #sampler=pyabc.sampler.MulticoreEvalParallelSampler()  # The default sampler
    #transition=transition,  # Define how to transition from one population to the next
    )

# Observations are cases per msoa, but make it into a numpy array because this may be more efficient
# (first axis is the msoa number, second is the day)
observations = cases_msoa.iloc[:,0:405].to_numpy()

db_path = ("sqlite:///" + "ramp_da2.db")
run_id = abc.new(db_path, {'observation': observations, "individuals":individuals_df})

# Run :
print('here')
history = abc.run(max_nr_populations=2)

print(f"Running new ABC with id {run_id}.... ", flush=True)
history

##########################################################################
##########################################################################
### Algorithm diagnostics
##########################################################################
##########################################################################

_, arr_ax = plt.subplots(2, 2)

pyabc.visualization.plot_sample_numbers(history, ax=arr_ax[0][0])
pyabc.visualization.plot_epsilons(history, ax=arr_ax[0][1])
#pyabc.visualization.plot_credible_intervals(
#    history, levels=[0.95, 0.9, 0.5], ts=[0, 1, 2, 3, 4],
#    show_mean=True, show_kde_max_1d=True,
#    refval={'mean': 2.5},
#    arr_ax=arr_ax[1][0])
pyabc.visualization.plot_effective_sample_sizes(history, ax=arr_ax[1][1])

plt.gcf().set_size_inches((12, 8))
plt.gcf().tight_layout()

#### Algorithm diagnostics
_, arr_ax = plt.subplots(2, 2)

pyabc.visualization.plot_sample_numbers(history, ax=arr_ax[0][0])
pyabc.visualization.plot_epsilons(history, ax=arr_ax[0][1])
#pyabc.visualization.plot_credible_intervals(
#    history, levels=[0.95, 0.9, 0.5], ts=[0, 1, 2, 3, 4],
#    show_mean=True, show_kde_max_1d=True,
#    refval={'mean': 2.5},
#    arr_ax=arr_ax[1][0])
pyabc.visualization.plot_effective_sample_sizes(history, ax=arr_ax[1][1])

plt.gcf().set_size_inches((12, 8))
plt.gcf().tight_layout()

########## Plot the marginal posteriors
fig, axes = plt.subplots(3,int(len(priors)/2), figsize=(12,10))

#cmap = { 0:'k',1:'b',2:'y',3:'g',4:'r' }  # Do this automatically for len(params)

for i, param in enumerate(priors.keys()):
    ax = axes.flat[i]
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(df, w, x=param, ax=ax,
            label=f"{param} PDF t={t}",
            alpha=1.0 if t==0 else float(t)/history.max_t, # Make earlier populations transparent
            color= "black" if t==history.max_t else None # Make the last one black
        )
        if param!="work":
            ax.set_xlim(0,1)
        ax.legend(fontsize="small")
        #ax.axvline(x=posterior_df.loc[1,param], color="grey", linestyle="dashed")
        #ax.set_title(f"{param}: {posterior_df.loc[0,param]}")
        ax.set_title(f"{param}")
fig.tight_layout()

### Analyse the posterior
# Posterior distribution for the final population.
# This is made up of the posterior estimates for each particle in the population and the associated weight.
_df, _w = history.get_distribution(m=0, t=history.max_t)
# Merge dataframe and weights and sort by weight (highest weight at the top)
_df['weight'] = _w
posterior_df = _df.sort_values('weight', ascending=False).reset_index()
posterior_df

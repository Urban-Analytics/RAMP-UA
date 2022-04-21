#!/usr/bin/env python
# coding: utf-8

# # Running RAMP OpenCL model with dynamic calibration
# 
# ### Parameter definitions
# The OpenCL RAMP model creates a synthetic population and simulates the movements of individuals between various locations (home, work, school, shops). If an individual is infected then they impart an infection risk on to the locations that they visit. The severity of this infection risk is determined by whether the individual is presymptomatic, asymptomatic or symptomatic (the **individual's hazard rating**). The locations in the model thus come to have an infection risk, based upon how many infected individuals from each infection status have visited. When susceptible individuals without the disease visit these locations their risk of being infected is determined by the location's infection risk, alongside the transmission hazard associated with that kind of location (the location's hazard rating). An additional hazard rating, the current risk beta, is used to control the general transmissability of the disease (?) 
# 
# ### Parameter calibration
# The current risk beta parameter is calibrated in [this notebook](http://localhost:8888/notebooks/Users/gy17m2a/OneDrive%20-%20University%20of%20Leeds/Project/RAMP-UA/experiments/calibration/abc-2-NewObs.ipynb). This involves determining a prior distribution for both the current risk beta parameter, and for the individual and location hazard parameters. Approximate Bayesian Computation (ABC) is then used to approximate the likelihood of different parameter values by running the model a large number of times. Each time parameter values are drawn randomly from the prior distributions, and the parameter values from the simulations with results closest to the observations are kept. A fitness function is used to assess the similarity of model results to observations. In this case, the Euclidean difference between the observed number of cases per week and the simulated number of cases per week is calculated. In this script, the model is ran using ABC for 105 days, with two populations. 
# 
# In this notebook, the current risk beta parameter value which was associated with the highest fitness in the final population in the calibration process above is set as a constant, and assumed not to change over the life of the disease. However, as the disease evolves, it is expected that the other parameters pertaining to the hazard associated with individuals and locations will change. If a model is calibrated just once using historical data then it will be unable to account for this parameter evolution. However, if it is dynamically calibrated, then it can be optimised in response to changes in parameter values in real time. Dynamic calibration involves re-calibrating the model at each time-step (e.g. after one day, one week etc), and using the best performing parameter value from the previous time-step for the next time-step. 
# 
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

# ### Import modules

# In[1]:


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

# os.chdir("C:/Users/gy17m2a/OneDrive - University of Leeds/Project/RAMP-UA/experiments/calibration")

# PYABC (https://pyabc.readthedocs.io/en/latest/)
import pyabc
from pyabc.transition.multivariatenormal import MultivariateNormalTransition  # For drawing from the posterior
# Quieten down the pyopencl info messages (just print errors)
import logging

logging.getLogger("pyopencl").setLevel(logging.ERROR)

# Import arbitrary distribution class for using posterior estimates as new priors
from arbitrary_distribution import ArbitraryDistribution, GreaterThanZeroParameterTransition

# RAMP model
from microsim.initialisation_cache import InitialisationCache

# Bespoke RAMP classes for running the model
from opencl_runner import OpenCLWrapper  # Some additional functions to simplify running the OpenCL model
from opencl_runner import OpenCLRunner

# Set this to False to recalculate all results (good on HPC or whatever).
# If true then it loads pre-calculated results from pickle files (much quicker)
LOAD_PICKLES = True


# #### Create observed cases data
# Cases_devon_weekly is based on government data recording the number of new cases each week, per MSOA. It has been corrected to account for lags and other shortcomings in the testing process, and summed up to cover the whole of Devon. Full details are here: http://localhost:8888/notebooks/Users/gy17m2a/OneDrive%20-%20University%20of%20Leeds/Project/RAMP-UA/experiments/calibration/observation_data/CreatingObservations-Daily-InterpolateSecond.ipynb
# 
# Cases_devon_daily is created through linear interpolation of the weekly data. This daily data is required for seeding the model. Because the interpolation is carried out for each MSOA seperately, when the data is joined back together for the whole of Devon it is not exactly equivalent to cases_devon_weekly. 
# 
# Currently, we are using the original cases_devon_weekly data in the distance function to evaluate the performance of the model. This makes sense because it is the most accurate weekly data that we have. However, it means when comparing the weekly model results to the observations, the model doesn't seem to be exactly the same as the observations, even during the seeding process. (Within the distance function all particles will be equally far from the observations during the seeding process, so shouldn't negatively affect things in this sense).

# In[2]:


## Get dataframe with totals for whole of Devon
cases_devon_weekly = pd.read_csv("observation_data/weekly_cases_devon.csv")
# Add column with cumulative sums rather than cases per week
cases_devon_weekly['CumulativeCases'] = cases_devon_weekly['OriginalCases'].cumsum()
# Keep just cumulative cases
cases_devon_weekly = cases_devon_weekly['CumulativeCases'].values


# In[3]:


# Read in daily devon case data (interpolated from weekly)
cases_devon_daily = pd.read_csv("observation_data/daily_cases_devon.csv")
# Add column with cumulative sums rather than cases per day
cases_devon_daily['CumulativeCases'] = cases_devon_daily['OriginalCases'].cumsum()
# Keep just cumulative cases
cases_devon_daily = cases_devon_daily['CumulativeCases'].values


# In[4]:


# Convert this interpolated data used in seeding back to weekly
# List the index matching the end day of each week (e.g. 7, 14, 21 etc (minus 1 for indexing starting at 0)
n_days = len(cases_devon_daily)
week_end_days = list(range(6,n_days+1,7))

# Keep only the values from the end of each week
cases_devon_daily_summed_weekly = cases_devon_daily[week_end_days]


# #### Setup Model

#  Read in parameters

# In[5]:


PARAMETERS_FILE = os.path.join("../../","model_parameters", "default.yml")
PARAMS = OpenCLRunner.create_parameters(parameters_file=PARAMETERS_FILE)


# Optionally initialise the population, delete the old OpenCL model snapshot (i.e. an already-initialised model) and
# re-create a new one. Useful if something may have changed (e.g. changed the lockdown file).

# In[6]:


OPENCL_DIR = os.path.join("..", "..", "microsim", "opencl")
SNAPSHOT_FILEPATH = os.path.join(OPENCL_DIR, "snapshots", "cache.npz")
assert os.path.isfile(SNAPSHOT_FILEPATH), f"Snapshot doesn't exist: {SNAPSHOT_FILEPATH}"


# #### Define currrent risk beta

# In[7]:


current_risk_beta_val =0.019


# ## Run model with default parameter values 
# 
# This shows what happens with the 'default' (manually calibrated) model.  
# These parameters are from the 'optimal' particle in the InitialModelCalibration.ipynb script, in which the model is ran with ABC with ten populations for 133 days.

# #### Initialise model

# In[8]:


## Define parameters
ITERATIONS = 105  # Number of days to run for
assert (ITERATIONS /7).is_integer() # check it is divisible by 7 
NUM_SEED_DAYS = 14  # Number of days to seed the population
USE_GPU = True
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


# #### Run model

# In[9]:


# These are from InitialModelCalibration.ipynb
best_params_manual_calibration = {"asymptomatic":0.411005, "current_risk_beta": 0.019494, "presymptomatic":0.371609, 
                                  "primary_school": 0.609788, "retail": 0.640238, "secondary_school": 0.536117, 
                                  "symptomatic": 0.737517, "work":0.008547}


# In[10]:


OpenCLRunner.update(repetitions=5)  # Temporarily use more repetitions to give a good baseline
OpenCLRunner.update(store_detailed_counts=True)  # Temporarily output age breakdowns
(fitness_manualcalibration, sim_manualcalibration, obs_manualcalibration, out_params_manualcalibration, summaries_manualcalibration) = OpenCLRunner.run_model_with_params_abc(
    best_params_manual_calibration, return_full_details=True, quiet = False)
OpenCLRunner.update(repetitions=REPETITIONS)
OpenCLRunner.update(store_detailed_counts=STORE_DETAILED_COUNTS)


# #### Plot results

# In[11]:


# Check the model returns the observations correctly i.e. that theyre the same length
np.array_equal(obs_manualcalibration, cases_devon_daily[:len(sim_manualcalibration)-1])

# Plot
fig, ax = plt.subplots(1,1)
x = range(len(sim_manualcalibration))
ax.plot(x, sim_manualcalibration, label="sim", color="orange")
ax.plot(x, obs_manualcalibration, label="obs", color="blue")
ax.legend()


# ## Run model with dynamic calibration
# ### Define parameter values
# #### Define constants
# These are not touched by ABC. Include parameters that should not be optimised.

# In[12]:


const_params_dict = { "current_risk_beta": current_risk_beta_val,"home": 1.0}


# #### Define random variables and the prior distributions
# Random variables are the global parameters.

# In[13]:


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


# #### Define parameters for running model

# In[14]:


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
admin_params = {"quiet": True, "use_gpu": True, "store_detailed_counts": True, "start_day": 0,  "run_length": da_window_size,
                "current_particle_pop_df": None,  "parameters_file": PARAMETERS_FILE, "snapshot_file": SNAPSHOT_FILEPATH, 
                "opencl_dir": OPENCL_DIR, "individuals_df": individuals_df, 
                "observations_weekly_array": cases_devon_weekly,'num_seed_days' :NUM_SEED_DAYS}


# #### Define dynamic calibration loop

# In[ ]:


# Create dictionaries to store the dfs, weights or history from each window (don't need all of these, but testing for now)
dfs_dict, weights_dict, history_dict = {}, {},{}

# Store starting time to use to calculate how long processing the whole window has taken
starting_windows_time = datetime.datetime.now()

# Define number of and number of populations to run for,
windows = 7
n_pops = 3

# Loop through each window
for window_number in range(1, windows + 1):
    print("Window number: ", window_number)

    # Edit the da_window size in the admin params
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
        distance_function=OpenCLWrapper.dummy_distance,  # Distance function
        sampler=pyabc.sampler.SingleCoreSampler(),
        transitions=GreaterThanZeroParameterTransition())
        # Single core because the model is parallelised anyway (and easier to debug)
        # sampler=pyabc.sampler.MulticoreEvalParallelSampler()  # The default sampler
        #transition=transition,  # Define how to transition from one population to the next

    # Prepare to run the model
    db_path = ("sqlite:///" + "ramp_da.db")  # Path to database

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
        df_t1, w_t1 = abc.history.get_distribution(m=0, t=t)

        # Save these for use in plotting the prior on the plot of parameter values in each population
        dfs_dict["w{},pop{}".format(window_number, t)] = df_t1
        weights_dict["w{}, pop{}".format(window_number, t)] = w_t1
        history_dict["w{}".format(window_number)] = abc_history
        
    fname = "{}windows_window{}, {}pops_Crb{}.pkl".format(windows, window_number, n_pops, current_risk_beta)
    with open( fname, "wb" ) as f:
            pickle.dump( history_dict, f)


# #### Save abc_history object

# In[ ]:


# fname = "{}windows_{}pops_Crb{}.pkl".format(n_pops, windows, current_risk_beta)
# with open( fname, "wb" ) as f:
#         pickle.dump( history_dict, f)
# # # with open(fname, "rb") as f:
# # #             history_dict = pickle.load(f)


# ## Assess how parameter values change over time with dynamic calibration

# ### Parameter values of final population for each window

# # In[17]:


# # define colour map
# evenly_spaced_interval = np.linspace(0.35, 1, 3)
# colors = [cm.Greens(x) for x in evenly_spaced_interval]
# colors =  ['orange', 'darkred']

# fig, axes = plt.subplots(3,int(len(original_priors)/2), figsize=(12,10))
# for i, param in enumerate(original_priors.keys()):
#     color_i =0
#     ax = axes.flat[i]
#     # Add parameter priors
#     priors_x = np.linspace(-1, 2, 99)  # (specified so that we have some whole numbers)
#     ax.plot(priors_x, pyabc.Distribution(param=all_rv[param]).pdf({"param": priors_x}), 
#             color = 'black', label = 'Prior', linewidth  = 1, linestyle ='dashed')
    
#     for history_name, history in history_dict.items():
#         color = colors[color_i]
#         df, w = history.get_distribution(m=0, t=history.max_t)
#         pyabc.visualization.plot_kde_1d(df, w, x=param, ax=ax,
#                 label=history_name, linewidth = 4,
#                 color= color)
#         ax.legend(fontsize="small")
#         ax.set_title(f"{param}")
#         handles, labels = ax.get_legend_handles_labels()
#         ax.get_legend().remove()
#         color_i = color_i +1
# fig.legend(handles, labels, loc='center right', fontsize = 17,
#             bbox_to_anchor=(1.01, 0.17))
#           # ncol = 8, bbox_to_anchor=(0.5, -0.07))
# axes[2,2].set_axis_off()
# axes[2,1].set_axis_off()
# fig.tight_layout()
# fig.show()
# # fig.savefig("Plots/8windows_14days_each_finalpop.jpg")


# # ### Parameter values for all populations for each window

# # In[18]:


# # define colour map and line style to use in intermediate populations
# evenly_spaced_interval = np.linspace(0.35, 1, 5)
# colors = [cm.Greens(x) for x in evenly_spaced_interval]
# linestyles = ['dashed','dashed','dashed','dashed', 'solid'] # check this is same length as n populations
# linewidths = [2,2,2,2,4]
# colors =  ['orange', 'darkred']

# # Set up plot
# fig, axes = plt.subplots(3,int(len(original_priors)/2), figsize=(12,10))
# for i, param in enumerate(original_priors.keys()):
#     color_i =0
#     ax = axes.flat[i]
#     # Add parameter priors
#     priors_x = np.linspace(-1, 2, 99)  # (specified so that we have some whole numbers)
#     ax.plot(priors_x, pyabc.Distribution(param=all_rv[param]).pdf({"param": priors_x}), 
#             color = 'black', label = 'Prior', linewidth  = 3, linestyle ='solid')
#     for history_name, history in history_dict.items():
#         color = colors[color_i]
#         for t in range(history.max_t + 1):
#             df, w = history.get_distribution(m=0, t=t)
#             pyabc.visualization.plot_kde_1d(df, w, x=param, ax=ax,
#                 label="{}, pop {}".format(history_name, t),
#                 color = color, linestyle = linestyles[t],linewidth = linewidths[t])
#             ax.legend(fontsize="small")
#             ax.set_title(f"{param}")
#             handles, labels = ax.get_legend_handles_labels()
#             ax.get_legend().remove()
#         color_i = color_i +1
#     if param!="work":
#             ax.set_xlim(-0.5,1.5)        
#     if param=="work":
#         ax.set_xlim(-0.05,0.1)
#     if param =='asymptomatic':
#          ax.set_xlim(-0.05,0.1)    
# fig.legend(handles, labels, loc='center right', fontsize = 12,
#             bbox_to_anchor=(1.01, 0.17))
# axes[2,2].set_axis_off()
# axes[2,1].set_axis_off()
# fig.tight_layout()
# fig.show()


# # ## Compare model predictions from dynamically calibrated model against observations

# # ### Model predictions from within dynamic calibration window
# # #### Get model predictions, and particle distances for each particle in all of the populations 

# # In[19]:


# # Create dictionary to store results for each window
# abc_sum_stats = {}

# # Loop through each calibration window, defining the number of days it covered
# for t in range(0,history.n_populations):
#     print(t)
#     for window, n_days in { "w1": 14, "w2":28}.items():

#         # Create lists to store values for each particle
#         distance_l, daily_preds_l, params_l = [],[],[]

#         # get the history for this window    
#         history_wx  = history_dict[window]   

#         # Get parameter values
#         parameter_vals_df, w = history_wx.get_distribution(m=0, t=t)

#         # Get the summary stats for the final population for this window ([1] means keep just the 
#         # dataframe and not the array of weights)
#         weighted_sum_stats_t0 = history_wx.get_weighted_sum_stats_for_model(t=t)[1]
     
#         # Loop through each particle and save their distance and predictions into the lists
#         for particle_no in range(0,100):
#             # Get data for just this particle
#             particle_x_dict = weighted_sum_stats_t0[particle_no]

#             # Get daily predictions
#             cumulative_model_diseased_devon = particle_x_dict["model_daily_cumulative_infections"]     
#             cumulative_model_diseased_devon = cumulative_model_diseased_devon[0:n_days]

#             # Add daily predictions for this particle to list
#             daily_preds_l.append(cumulative_model_diseased_devon)

#             # Add distance to list
#             distance_l.append(particle_x_dict['distance'])

#             # Add parameter values to list
#             params_l.append(parameter_vals_df.iloc[particle_no])

#         # Add to dictionary for this window
#         abc_sum_stats["{}_{}".format(window,t)] = {'distance_l':distance_l, 'daily_preds_l' :daily_preds_l, 'params_l':params_l}


# # #### For each window, plot the predictions made by each particle in the final population 

# # In[24]:


# # Create figure
# fig, axes = plt.subplots(1, 2, figsize=(12,4))
# axes_number = 0
# for window, n_days in { "w1": 14, "w2":28}.items():
#     t = history.max_t
#     # Find the best particle
#     best_particle_idx = abc_sum_stats["{}_{}".format(window,t)]['distance_l'].index(min(abc_sum_stats["{}_{}".format(window,t)]['distance_l']))

#     # Get data for this window
#     daily_preds_l  = abc_sum_stats["{}_{}".format(window,t)]['daily_preds_l']   
#     distance_l = abc_sum_stats["{}_{}".format(window,t)]['distance_l']   

#     # Normalise distance to 0-1 to calculate transparency
#     _distance = np.array(distance_l)  # Easier to do maths on np.array
#     distance_norm = (_distance - min(_distance)) / (max(_distance) - min(_distance))

#     # define number of days these results relate to
#     x=range(1,n_days+1)    

#     # For each particle, plot the predictions, coloured by distance
#     for i in range(0,len(daily_preds_l)):
#         axes[axes_number].plot(x, daily_preds_l[i], color="black", alpha=1-distance_norm[i])  # (1-x because high distance is bad)

#     # Add the best particle
#     axes[axes_number].plot(x, daily_preds_l[best_particle_idx], color="green", linewidth = 3,
#                           label = 'Best particle')  # (1-x because high distance is bad)

#     # Add observations
#     axes[axes_number].plot(x, cases_devon_daily[0:len(daily_preds_l[0])], label="Observations",
#                            linewidth = 3, color="darkred")

#     # Apply labels
#     axes[axes_number].set_xlabel("Day")
#     axes[axes_number].set_ylabel("Number infections")
#     axes[axes_number].set_title("{}".format(window))

#     # Apply legend
#     axes[axes_number].legend(fontsize="large")

#     axes_number =axes_number +1

# # Set full plot title
# fig.suptitle('POPULATION {} -- Number of infections predicted by each particle within each window'.format(t))


# # #### For the final window, plot the predictions made by each particle in each population

# # In[28]:


# # Create figure
# fig, axes = plt.subplots(1, 4, figsize=(15,4), sharey=True)
# axes_number = 0
# for t in range(0, history.max_t):
#     window, n_days = 'w2', 28
#     # Find the best particle
#     best_particle_idx = abc_sum_stats["{}_{}".format(window,t)]['distance_l'].index(min(abc_sum_stats["{}_{}".format(window,t)]['distance_l']))

#     # Get data for this window
#     daily_preds_l  = abc_sum_stats["{}_{}".format(window,t)]['daily_preds_l']   
#     distance_l = abc_sum_stats["{}_{}".format(window,t)]['distance_l']   

#     # Normalise distance to 0-1 to calculate transparency
#     _distance = np.array(distance_l)  # Easier to do maths on np.array
#     distance_norm = (_distance - min(_distance)) / (max(_distance) - min(_distance))

#     # define number of days these results relate to
#     x=range(1,n_days+1)    

#     # For each particle, plot the predictions, coloured by distance
#     for i in range(0,len(daily_preds_l)):
#         axes[axes_number].plot(x, daily_preds_l[i], color="black", alpha=1-distance_norm[i])  # (1-x because high distance is bad)

#     # Add the best particle
#     axes[axes_number].plot(x, daily_preds_l[best_particle_idx], color="green", linewidth = 3,
#                           label = 'Best particle')  # (1-x because high distance is bad)

#     # Add observations
#     axes[axes_number].plot(x, cases_devon_daily[0:len(daily_preds_l[0])], label="Observations",
#                            linewidth = 3, color="darkred")

#     # Apply labels
#     axes[axes_number].set_xlabel("Day")
#     axes[axes_number].set_ylabel("Number infections")
#     axes[axes_number].set_title("Pop {}".format(t))

#     # Apply legend
#     axes[axes_number].legend(fontsize="large")

#     axes_number =axes_number +1

# # Set full plot title
# fig.suptitle('Number of infections predicted by each particle within each window'.format(t))


# # ### Model predictions for future

# # #### Set up OpenCL runner

# # In[32]:


# # Initialise the class so that its ready to run the model.
# OpenCLRunner.init(iterations = 105,  repetitions = 5,observations = cases_devon_weekly, use_healthier_pop = True, 
#     use_gpu = True, store_detailed_counts = False, parameters_file = PARAMETERS_FILE,opencl_dir = OPENCL_DIR,
#     snapshot_filepath = SNAPSHOT_FILEPATH,num_seed_days = 7) 
# # Set constants 
# const_params_dict = { "current_risk_beta": current_risk_beta_val,"home": 1.0}
# OpenCLRunner.set_constants(const_params_dict)


# # ### Use parameters from dynamic calibration process to run model for 105 days

# # In[35]:


# # Define the number of samples to take from the posterior distribution of parameters
# N_samples = 30

# # # Create dictionary to store results for each window
# windows_dict ={}

# for window in ['w1', 'w2']:

#     # Define abc_history object from final window
#     abc_history = history_dict[window]

#     # Get dataframe of posterior parameter values
#     df, w = abc_history.get_distribution(m=0, t=abc_history.max_t)

#     # Sample from the dataframe of posteriors using KDE
#     kde = MultivariateNormalTransition(scaling=1)
#     kde.fit(df, w)
#     samples = kde.rvs(N_samples)

#     # Now run N models and store the results of each one
#     distance_l, sim_l, obs_l, out_params_l,out_calibrated_params_l, summaries_l = [],[],[],[],[],[] 

#     negative_count = 0  # Count the number of negatives returned in the KDE posterior
#     for i, sample in samples.iterrows():
#         # Check for negatives. If needed, resample
#         while (sample < 0).values.any():
#             #print("Found negatives. Resampling")
#             negative_count += 1
#             sample = kde.rvs()
#             # Added in this line as the sample was in the wrong format for the while loop
#             sample = pd.Series(sample)

#         # Create a dictionary with the parameters and their values for this sample
#         param_values = sample.to_dict()
#         #print(param_values)

#         # _summaries = 
#         (_distance, _sim, _obs, _out_params, _summaries) =             OpenCLRunner.run_model_with_params_abc(param_values, return_full_details=True)
#         #print(f"distance: {_distance}.")

#         distance_l.append(_distance)
#         sim_l.append(_sim)
#         obs_l.append(_obs)
#         out_params_l.append(_out_params)
#         out_calibrated_params_l.append(param_values)
#         summaries_l.append(_summaries)

#     print(f"Finished sampling. Ignored {negative_count} negative samples.")

#     # add to dictionary
#     windows_dict[window] = {'distance': distance_l, 'sim': sim_l, 'obs':obs_l, 'summaries': summaries_l, 
#                            'out_calibrated_params': out_calibrated_params_l, 'out_params': out_params_l}


# # #### Plot the individual results for each sample and compare to observations

# # In[33]:


# windows= ['w1','w2']

# # create 3x1 subplots
# fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True, figsize = (12,8))
# fig.suptitle('Predicted cases for 105 days')

# # clear subplots
# for ax in axs:
#     ax.remove()

# # add subfigure per subplot
# gridspec = axs[0].get_subplotspec().get_gridspec()
# subfigs = [fig.add_subfigure(gs) for gs in gridspec]

# windows_i = 0
# for row, subfig in enumerate(subfigs):
#     subfig.suptitle('Window {}'.format(row+1), fontsize = 15)
#     this_windows_dict = windows_dict[windows[windows_i]]
#     # create 1x3 subplots per subfig
#     axs = subfig.subplots(nrows=1, ncols=2, sharey= True)
#     for col, ax in enumerate(axs):
        
#         # Plot model values
#         _distance = np.array(this_windows_dict['distance'])  # Easier to do maths on np.array
#         distance_norm = (_distance - min(_distance)) / (max(_distance) - min(_distance))
        
#         # Plot observations
#         if col == 0:
#             x = range(len(this_windows_dict['sim'][0]))
#             for i in range(len(this_windows_dict['summaries'])):
#                 ax.plot(x, this_windows_dict['sim'][i],
#                 color="black", alpha=1 - distance_norm[i]) # (1-x because high distance is bad)
#             ax.plot(x, cases_devon_daily_summed_weekly[0:int((105/7))], label="Observations", linewidth=5, color="blue")
#             ax.set_title('Weekly')    
#         elif col ==1:
#             x = range(len(OpenCLRunner.get_cumulative_daily_infections(this_windows_dict['summaries'][1])))
#             for i in range(len(this_windows_dict['summaries'])):
#                 ax.plot(x, OpenCLRunner.get_cumulative_daily_infections(this_windows_dict['summaries'][i]),
#                 color="black", alpha=1 - distance_norm[i]) # (1-x because high distance is bad)
#             ax.plot(x, cases_devon_daily[0:105], label="Observations", linewidth = 5, color="blue")
#             ax.set_title('Daily')
    
#     # move on to next window
#     windows_i = windows_i+1


# # ### Run model with single 'best' estimate of parameter values 

# # #### Find the sample drawn from the posterior distribution which has the smallest distance (best performing!)

# # In[51]:


# # Find the index of the lowest distance value
# best_params_dict = {}
# best_sim_dict = {}

# for window in ['w1', 'w2']:
#     # Find the index of the lowest distance value
#     best_model_idx = np.argmin(windows_dict[window]['distance'])
#     # Find the corresponding parameter values
#     best_params = windows_dict[window]['out_calibrated_params'][best_model_idx]

#     ## Run model with these best parameters
#     OpenCLRunner.update(store_detailed_counts=True)  # Temporarily output age breakdowns
#     (distance_bestparams, sim_bestparams, obs_bestparams, out_params_bestparams, summaries_bestparams) = OpenCLRunner.run_model_with_params_abc(
#         best_params, return_full_details=True, quiet = True)
#     OpenCLRunner.update(store_detailed_counts=STORE_DETAILED_COUNTS)

#     # Add to dictionary
#     best_params_dict['{}'.format(window)]= best_params
#     best_sim_dict['{}'.format(window)] = sim_bestparams


# # #### See how the parameters from the particle with the lowest distance relate to the marginal posteriors

# # In[52]:


# # define colour map
# evenly_spaced_interval = np.linspace(0.35, 1, 3)
# colors = [cm.Greens(x) for x in evenly_spaced_interval]

# fig, axes = plt.subplots(3,int(len(original_priors)/2), figsize=(12,10))
# for i, param in enumerate(original_priors.keys()):
#     color_i =0
#     ax = axes.flat[i]  
#     for history_name, history in history_dict.items():
#         color = colors[color_i]
#         df, w = history.get_distribution(m=0, t=history.max_t)
#         pyabc.visualization.plot_kde_1d(df, w, x=param, ax=ax,
#                 label=history_name, linewidth = 4,
#                 color= color)
#         ax.legend(fontsize="small")
#         ax.set_title(f"{param}")
#         handles, labels = ax.get_legend_handles_labels()
#         ax.get_legend().remove()
#         color_i = color_i +1
#         ax.axvline(x=best_params[param], color="grey", linestyle="solid")
#     ax.text(x=best_params[param], y=0.9*ax.get_ylim()[1], s=str(round(best_params[param],3)), fontsize=8)
#     ax.set_title(f"{param}")    
# fig.legend(handles, labels, loc='center right', fontsize = 17,
#             bbox_to_anchor=(1.01, 0.17))
#           # ncol = 8, bbox_to_anchor=(0.5, -0.07))
# axes[2,2].set_axis_off()
# axes[2,1].set_axis_off()
# fig.tight_layout()
# fig.show()
# # fig.savefig("Plots/8windows_14days_each_finalpop.jpg")


# # #### Run model with 'best' parameter values

# # In[57]:


# fig, ax = plt.subplots(1,1)
# # ax.plot(x, sim_manualcalibration, label="manual_calibration", color="green")
# ax.plot(range(len(cases_devon_daily_summed_weekly[0:(int(ITERATIONS/7))])), 
#        cases_devon_daily_summed_weekly[0:(int(ITERATIONS/7))], label="obs", color="blue")

# for window in ['w1', 'w2']:
#     best_sim = best_sim_dict["{}".format(window)]
#     col = 'red' if window == 'w1' else 'green'
#     # Add to plot
#     ax.plot(range(len(best_sim)), best_sim, label="BestParams - {}".format(window), color=col,
#            linestyle='solid')
#     ax.legend()


# # ### Compare spatial distribution
# # 
# # Can't do this currently as OpenCLRunner doesn't return a breakdown by MSOA

# # ### Save/pickle

# # In[41]:


# fname = "2windows_7seeds_OldDistance_abc.pkl"
# with open( fname, "wb" ) as f:
#         pickle.dump( history, f)


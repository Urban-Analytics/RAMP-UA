# Very simple pyabc model, used for experimenting with the pyabc library
# This extends the pyabc_example-basic.py script by implementing Bayesian updating,
# i.e. taking the posterior from one run and using it as the prior for the next run.

import pyabc
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union

from pyabc import History
from pyabc.transition.multivariatenormal import MultivariateNormalTransition  # For drawing from the posterior
from pyabc.random_variables import Distribution
from pyabc.parameters import Parameter

pyabc.settings.set_figure_params('pyabc')  # for beautified plots


class ArbitraryDistribution(Distribution):
    """
    Define an arbitrary distribution. Used so that the posterior from an ABC run can be re-used
    as the prior in a new run. Does this by taking the abc_history object (output from an
    ABC run) and generating a MultivariateNormalTransition (KDE) that can be sampled
    """

    def __init__(self, abc_hist: History):
        # Get the dataframe of particles (parameter point estimates) and associated weights
        dist_df, dist_w = abc_hist.get_distribution(m=0, t=abc_history.max_t)
        # Create a KDE using the particles
        self.kde = MultivariateNormalTransition(scaling=1)
        self.kde.fit(dist_df, dist_w)

    def rvs(self) -> Parameter:
        """Sample from the joint distribution, returning a Parameter object.
           Just calls rvs() on the underlying kde"""
        return self.kde.rvs()

    def pdf(self, x: Union[Parameter, pd.Series, pd.DataFrame]) -> Union[float, np.ndarray]:
        """Get probability density at point `x` (product of marginals).
        Just calls pdf(x) on the underlying kde"""
        return self.kde.pdf(x)

    def __repr__(self):
        return f"<ArbitraryDistribution\n    " + \
               ",\n    ".join(str(p) for p in self.kde.X.columns) + ">"

    def copy(self) -> Distribution:
        """Copy the distribution.

        Returns
        -------
        copied_distribution: Distribution
            A copy of the distribution.
        """
        return copy.deepcopy(self)


# Define our 'model'.
def my_model(input_params_dict: dict) -> dict:
    """my_model takes a dictionary that contains the values of the parameters.
      I expect two parametrs, param_a and param_b.
      The 'model' just plugs these into a quadratic equation and returns the result.
      Note that the result is returned as an entry in a dictionary."""
    # Uncomment to see what parameters the model is receiving (useful for debugging)
    # print(f"Model received parameters: {input_params_dict}")
    if "param_a" not in input_params_dict:
        raise Exception(f"Error: the model is expecting an input called 'param_a, but got: {input_params_dict}")
    if "param_b" not in input_params_dict:
        raise Exception(f"Error: the model is expecting an input called 'param_b, but got: {input_params_dict}")

    a = input_params_dict["param_a"]
    b = input_params_dict["param_b"]

    x = 5  # Arbitrary value for the quadratic
    result = a * x + b * x ** 2

    # Return the result as a dictionary. We can put anything that we want in here.
    # I'm returning the result of the equation associated with a dictionary key 'model_result'
    # We could have more than one thing in the dictionary in case your distance function
    # calculates the distance based on lots of different pieces of information
    return {"model_result": result}


# Summary statistics convert raw model output into a statistical summary
def summary_stats(model_result_dict: dict) -> dict:
    """Summary statistics can be used to convert raw model output (created by the my_model() function)
    to summary information. This is then passed to the distance() function (see below).
    Summary stats are optional as the distance() function can always convert the raw output to a summary,
    but anything included in the dictionary returned by this function *is added to the results database*.
    This can be really useful e.g. for storing the state of a particular model. The data can be retrieved
    from the database with `abc_history.get_population_extended()`, which returns a DataFrame.
    (Here all the function does is return the model raw model results, but includes an additional
    dictionary item 'test'."""
    # 'model_result' was the key in the dict defined in my_model
    model_result_summary = model_result_dict['model_result']
    return {"model_summary": model_result_summary, "test": "SUMMARY_TEST"}


# Define the distance function (how close is the model result to some observations).
# This takes the output from the summary_stats() function as well as observations
def distance(model_result_summary_dict: dict, observations_dict: dict) -> float:
    """Simple distance function just returns the absolute difference between the model
    result and the observations.
    Note that these inputs are dictionaries."""
    # 'model_summary' was the key in the dict that is returned by summary_stats
    model_summary = model_result_summary_dict['model_summary']
    observation = observations_dict[
        'observation']  # Later we will create a dictionary of results and use the key 'observation' (they key could be anything)
    return abs(model_summary - observation)


# ** Lets check the model and distance function work **

# First define the 'true' values of our parameters. In reality we won't know these
PARAM_A = 1.4
PARAM_B = 6.7

# Generate the observation by running the model once with the perfect parameters.
# Normally this observation would be something we have measured in the real world.
# Note that { .. } is short-hand for creating a dictionary. Remember that the model expects to get it's parameters
# in a dictionary.
truth_model = my_model({"param_a": PARAM_A, "param_b": PARAM_B})
obs = truth_model['model_result']  # (Rememmber the model retuns a dictionary. We want the actual result

print(f"Assuming 'truth' parameters are {PARAM_A} and {PARAM_B}. They give a model result of {obs}.")

# Check that the distance function gives a larger distance for models that are worse
good_result = my_model({"param_a": PARAM_A + 0.3, "param_b": PARAM_B - 0.2})
bad_result = my_model({"param_a": PARAM_A + 7.5, "param_b": PARAM_B - 3.8})

# (Remember that the distance function wants the simulation result and observations in dictionaries,
# and the model results need to be passed through the summary_stats() function first before
# being given to distance()).
good_result_distance = distance(summary_stats(good_result), {'observation': obs})
bad_result_distance = distance(summary_stats(bad_result), {'observation': obs})

print(f"Good model distance is {good_result_distance}. Bad distance is {bad_result_distance}.")
assert good_result_distance < bad_result_distance

# Now lets see if pyabc can work out what those original parameter values were, but we only give it the observation

# Define priors
# Assume both paramerers are normally distributed around 5 and 6 respectivly (mostly arbitrary)
param_a_rv = pyabc.RV("norm", 5, 1)
param_b_rv = pyabc.RV("norm", 6, 1)

# Plot them to check they look OK
X = np.linspace(-0, 10, 1000)
plt.plot(X, pyabc.Distribution(param=param_a_rv).pdf({"param": X}), '--',
         label="Paremter A", lw=3)
plt.plot(X, pyabc.Distribution(param=param_b_rv).pdf({"param": X}), ':',
         label="Parameter B", lw=3)
plt.autoscale(tight=True)
plt.legend(title=r"Model parameters");
plt.show()

# Decorate the RVs so that they wont go below 0 (optional) and create the prior distribution
# The names of the variables in the distribution (e.g. 'param_a' must match those that the
# 'my_model' function is expecting
priors = pyabc.Distribution(
    param_a=pyabc.LowerBoundDecorator(param_a_rv, 0.0),
    param_b=pyabc.LowerBoundDecorator(param_b_rv, 0.0)
)

# **Run ABC**

# Prepare the ABC model
abc = pyabc.ABCSMC(
    models=my_model,  # Model (could be a list of models)
    parameter_priors=priors,  # Priors (again could be a list if we have different priors for different models)
    distance_function=distance,  # Distance function defined earlier
    summary_statistics=summary_stats,  # Function takes raw model output and calculates a summary
    #sampler=pyabc.sampler.SingleCoreSampler()  # Single core for testing (optional)
    sampler=pyabc.sampler.MulticoreEvalParallelSampler()  # The default sampler
)

# The results are stored in a database. We use a simple file database (sqlite) that creates a database
# file in the current directory
db_path = ("sqlite:///" + "pyabc_example.db")

# Each time you run it you get a new 'ID' (useful if you want to look up runs in the database)
# Note that this is where we give it the observations as well. Our distance function is expecting
# a dictionary with an 'observation' key in it, so we create a dictionary on the fly with { .. }
run_id = abc.new(db_path, {'observation': obs})
print(f"Running new ABC with id {run_id}.... ", flush=True)

# Run the algorithm!
abc_history = abc.run(max_nr_populations=3, minimum_epsilon=0.05)

# Algorithm diagnostics. I copied most of this from the docs
_, arr_ax = plt.subplots(2, 2)
pyabc.visualization.plot_sample_numbers(abc_history, ax=arr_ax[0][0])
pyabc.visualization.plot_epsilons(abc_history, ax=arr_ax[0][1])
pyabc.visualization.plot_effective_sample_sizes(abc_history, ax=arr_ax[1][1])

plt.gcf().set_size_inches((12, 8))
plt.gcf().tight_layout()
plt.show()

# Marginal posteriors. These are the *marginal* estimates of the individual parameters.
fig, axes = plt.subplots(2, int(len(priors) / 2) + 1, figsize=(12, 8))

for i, param in enumerate(priors.keys()):
    ax = axes.flat[i]
    for t in range(abc_history.max_t + 1):
        df_raw_, w = abc_history.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(df_raw_, w, x=param, ax=ax,
                                        label=f"{param} PDF t={t}",
                                        alpha=1.0 if t == 0 else float(t) / abc_history.max_t,
                                        # Make earlier populations transparent
                                        color="black" if t == abc_history.max_t else None  # Make the last one black
                                        )
        ax.legend()
        ax.set_title(f"{param}")
fig.tight_layout()
fig.show()

# %% 2D correlations
pyabc.visualization.plot_histogram_matrix(abc_history, size=(12, 10))
plt.show()

# *********************************************************
# Now try running again, using the posterior as a new prior
# *********************************************************

new_priors = ArbitraryDistribution(abc_history)

# Prepare the ABC model
abc = pyabc.ABCSMC(
    models=my_model,  # Model (could be a list of models)
    parameter_priors=new_priors,  # Priors (again could be a list if we have different priors for different models)
    distance_function=distance,  # Distance function defined earlier
    summary_statistics=summary_stats,  # Function takes raw model output and calculates a summary
    #sampler=pyabc.sampler.SingleCoreSampler()  # Single core for testing (optional)
    sampler=pyabc.sampler.MulticoreEvalParallelSampler()  # The default sampler
)

# The results are stored in a database. We use a simple file database (sqlite) that creates a database
# file in the current directory
db_path = ("sqlite:///" + "pyabc_example.db")

# Each time you run it you get a new 'ID' (useful if you want to look up runs in the database)
# Note that this is where we give it the observations as well. Our distance function is expecting
# a dictionary with an 'observation' key in it, so we create a dictionary on the fly with { .. }
run_id = abc.new(db_path, {'observation': obs})
print(f"Running new ABC with id {run_id}.... ", flush=True)

# Run the algorithm! (you can look up the max_nr_populations and minimum_epsilon arguments).
abc_history = abc.run(max_nr_populations=20, minimum_epsilon=0.05)

# Re-do the Algorithm diagnostics. I copied most of this from the docs
_, arr_ax = plt.subplots(2, 2)
pyabc.visualization.plot_sample_numbers(abc_history, ax=arr_ax[0][0])
pyabc.visualization.plot_epsilons(abc_history, ax=arr_ax[0][1])
pyabc.visualization.plot_effective_sample_sizes(abc_history, ax=arr_ax[1][1])

plt.gcf().set_size_inches((12, 8))
plt.gcf().tight_layout()
plt.show()

# Marginal posteriors. These are the *marginal* estimates of the individual parameters.
fig, axes = plt.subplots(2, int(len(priors) / 2) + 1, figsize=(12, 8))

for i, param in enumerate(priors.keys()):
    ax = axes.flat[i]
    for t in range(abc_history.max_t + 1):
        df_raw_, w = abc_history.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(df_raw_, w, x=param, ax=ax,
                                        label=f"{param} PDF t={t}",
                                        alpha=1.0 if t == 0 else float(t) / abc_history.max_t,
                                        # Make earlier populations transparent
                                        color="black" if t == abc_history.max_t else None  # Make the last one black
                                        )
        ax.legend()
        ax.set_title(f"{param}")
fig.tight_layout()
fig.show()

# %% 2D correlations
pyabc.visualization.plot_histogram_matrix(abc_history, size=(12, 10))
plt.show()

# Generic functions that are used in the experiments notebooks
# Useful to put them in here so that they can be shared across notebooks
# and can be tested (see tests/experiements/experiments_functions_tests.py)
import os
import numpy as np
import multiprocessing
import itertools
import yaml
import time
import tqdm

from microsim.opencl.ramp.snapshot import Snapshot
from microsim.opencl.ramp.simulator import Simulator
from microsim.opencl.ramp.run import run_headless
from microsim.opencl.ramp.params import Params, LocationHazardMultipliers, IndividualHazardMultipliers


class Functions():
    """Includes useful functions for the notebooks"""

    @staticmethod
    def fit_l2(obs: np.ndarray, sim: np.ndarray):
        """Calculate the fitness of a model.
        
         Parameters
        ----------
        obs : array_like
              The observations data..
        sim : array_like
              The simulated data."""

        if len(obs) != len(sim):
            raise Exception(f"Lengths should be the same, not {len(obs)}) and {len(sim)}")
        if np.array(obs).shape != np.array(sim).shape:
            raise Exception("fShapes should be the same")

        return np.linalg.norm(np.array(obs) - np.array(sim))

    @staticmethod
    def get_mean_total_counts(summaries, disease_status: int):
        """
        Get the mean total counts for a given disease status at every iteration over a number of model repetitions

        :param summaries: A list of Summary objects created by running the OpenCL model
        :param disease_status: The disease status number, e.g.  `DiseaseStatus.Exposed.value`
        """
        reps = len(summaries)  # Number of repetitions
        iters = len(summaries[0].total_counts[disease_status])  # Number of iterations for each repetition
        matrix = np.zeros(shape=(reps, iters))
        for rep in range(reps):
            matrix[rep] = summaries[rep].total_counts[disease_status]
        mean = np.mean(matrix, axis=0)
        return mean

    @staticmethod
    def create_parameters(parameters_file: str = None,
                          current_risk_beta: float = None,
                          proportion_asymptomatic: float = None):
        """Create a params object with the given arguments."""

        # If no parameters are provided then read the default parameters from a yml file
        if parameters_file is None:
            parameters_file = os.path.join(".", "model_parameters", "default.yml")
        elif not os.path.isfile(parameters_file):
            raise Exception(f"The given parameters file '{parameters_file} is not a file.")

        with open(parameters_file) as f:
            parameters = yaml.load(f, Loader=yaml.SafeLoader)

        sim_params = parameters["microsim"]  # Parameters for the dynamic microsim (python)
        calibration_params = parameters["microsim_calibration"]
        disease_params = parameters["disease"]  # Parameters for the disease model (r)

        if current_risk_beta is None:
            current_risk_beta = disease_params['current_risk_beta']

        # The OpenCL model incorporates the current risk beta by pre-multiplying the hazard multipliers with it
        location_hazard_multipliers = LocationHazardMultipliers(
            retail=calibration_params["hazard_location_multipliers"]["Retail"] * current_risk_beta,
            primary_school=calibration_params["hazard_location_multipliers"]["PrimarySchool"] * current_risk_beta,
            secondary_school=calibration_params["hazard_location_multipliers"]["SecondarySchool"] * current_risk_beta,
            home=calibration_params["hazard_location_multipliers"]["Home"] * current_risk_beta,
            work=calibration_params["hazard_location_multipliers"]["Work"] * current_risk_beta,
        )

        # Individual hazard multipliers can be passed straight through
        individual_hazard_multipliers = IndividualHazardMultipliers(
            presymptomatic=calibration_params["hazard_individual_multipliers"]["presymptomatic"],
            asymptomatic=calibration_params["hazard_individual_multipliers"]["asymptomatic"],
            symptomatic=calibration_params["hazard_individual_multipliers"]["symptomatic"]
        )

        if proportion_asymptomatic is None:
            proportion_asymptomatic = disease_params["asymp_rate"]

        return Params(
            location_hazard_multipliers=location_hazard_multipliers,
            individual_hazard_multipliers=individual_hazard_multipliers,
            proportion_asymptomatic=proportion_asymptomatic
        )

    @staticmethod
    def run_opencl_model(i: int, iterations: int, snapshot_filepath: str, params,
                         opencl_dir: str, num_seed_days: int, use_gpu: bool,
                         store_detailed_counts: bool = True, quiet=False) -> (np.ndarray, np.ndarray):
        """
        Run the OpenCL model.

        :param i: Simulation number (i.e. if run as part of an ensemble)
        :param iterations: Number of iterations to ru the model for
        :param snapshot_filepath: Location of the snapshot (the model must have already been initialised)
        :param params: a Params object containing the parameters used to define how the model behaves
        :param opencl_dir: Location of the OpenCL code
        :param num_seed_days: number of days to seed the model
        :param use_gpu: Whether to use the GPU to process it or not
        :param store_detailed_counts: Whether to store the age distributions for diseases (default True, if
          false then the model runs much more quickly).
        :param quiet: Whether to print a message when the model starts
        :return: A summary python array that contains the results for each iteration and a final state

        """

        # load snapshot
        snapshot = Snapshot.load_full_snapshot(path=snapshot_filepath)

        # set params
        snapshot.update_params(params)

        # set the random seed of the model for each repetition, otherwise it is completely deterministic
        snapshot.seed_prngs(i)

        # seed initial infections using GAM initial cases
        data_dir = os.path.join(opencl_dir, "data")
        snapshot.seed_initial_infections(num_seed_days=num_seed_days, data_dir=data_dir)

        # Create a simulator and upload the snapshot data to the OpenCL device
        kernel_dir = os.path.join(opencl_dir, "ramp", "kernels")
        simulator = Simulator(snapshot, kernel_dir=kernel_dir, gpu=use_gpu)
        simulator.upload_all(snapshot.buffers)

        if not quiet:
            print(f"Running simulation {i + 1}.")
        summary, final_state = run_headless(simulator, snapshot, iterations, quiet=True,
                                            store_detailed_counts=store_detailed_counts)
        return summary, final_state

    #
    # Functions to run the model in multiprocess mode.
    # Don't wory currently on OS X, something to do with calling multiprocessing from a notebook
    # This is a workaround to allow multiprocessing.Pool to work in the pf_experiments_plots notebook.
    # The function called by pool.map ('count_wiggles') needs to be defined in this separate file and imported.
    # https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror/42383397
    #
    @staticmethod
    def run_opencl_model_multi(
            repetitions: int, iterations: int, params: Params,
            num_seed_days: int = 10,
            use_gpu: bool = False, store_detailed_counts: bool = False,
            opencl_dir=os.path.join(".", "microsim", "opencl"),
            snapshot_filepath=os.path.join(".", "microsim", "opencl", "snapshots", "cache.npz"),
            multiprocess=False):
        """Run a number of models and return a list of summaries.

        :param multiprocess: Whether to run in mutliprocess mode (default False)
        """
        # Prepare the function arguments. We need one set of arguments per repetition
        l_i = [i for i in range(repetitions)]
        l_iterations = [iterations] * repetitions
        l_snapshot_filepath = [snapshot_filepath] * repetitions
        l_params = [params] * repetitions
        l_opencl_dir = [opencl_dir] * repetitions
        l_num_seed_days = [num_seed_days] * repetitions
        l_use_gpu = [use_gpu] * repetitions
        l_store_detailed_counts = [store_detailed_counts] * repetitions
        l_quiet = [True] * repetitions  # Don't print info

        args = zip(l_i, l_iterations, l_snapshot_filepath, l_params, l_opencl_dir, l_num_seed_days, l_use_gpu,
                   l_store_detailed_counts, l_quiet)
        to_return = None
        start_time = time.time()
        if multiprocess:
            try:
                print("Running multiple models in multiprocess mode ... ", end="", flush=True)
                with multiprocessing.Pool(processes=int(os.cpu_count())) as pool:
                    to_return = pool.starmap(Functions.run_opencl_model, args)
            finally:  # Make sure they get closed (shouldn't be necessary)
                pool.close()
        else:
            # Return as a list to force the models to execute (otherwise this is delayed because starmap returns
            # a generator. Also means we can use tqdm to get a progress bar, which is nice.
            results = itertools.starmap(Functions.run_opencl_model, args)
            to_return = [x for x in tqdm.tqdm(results, desc="Running models", total=repetitions)]

        print(f".. finished, took {round(float(time.time() - start_time), 2)}s)", flush=True)
        return to_return


# FOR TESTING


def __run_model_current_risk_beta(x, return_full_details=False, multiprocess=False):
    """Run a model REPETITIONS times using the provided value for the current_risk_beta.

    :param return_details: If True then rather than just returning the fitness,
        return a tuple of (fitness, summaries_list, final_results_list).
    :return: The mean fitness across all model runs

    """
    # Sometimes x might be passed as a number in a list
    if isinstance(x, np.ndarray) or isinstance(x, list):
        if len(x) > 1:
            raise Exception(
                f"The curent risk beta value (x) should either be a 1-element array or a single number, not {x}")
        x = x[0]
    params = Functions.create_parameters(parameters_file=os.path.join("../..", "model_parameters", "default.yml"),
                                         current_risk_beta=x)

    results = Functions.run_opencl_model_multi(
        repetitions=REPETITIONS, iterations=ITERATIONS, params=params,
        opencl_dir=os.path.join("../..", "microsim", "opencl"),
        snapshot_filepath=os.path.join("../..", "microsim", "opencl", "snapshots", "cache.npz"),
        multiprocess=multiprocess
    )

    summaries = [x[0] for x in results]
    final_results = [x[1] for x in results]  # These aren't used, just left here for reference

    # Get mean cases per day from the summary object
    sim = Functions.get_mean_total_counts(summaries, DiseaseStatus.Exposed.value)
    # Compare these to the observations
    obs = observations.loc[:ITERATIONS - 1, "Cases"].values
    assert len(sim) == len(obs)
    fitness = fit(sim, obs)
    if return_full_details:
        return (fitness, sim, obs)
    else:
        return fitness


if __name__ == "__main__":
    import multiprocessing as mp
    import numpy as np
    import yaml  # pyyaml library for reading the parameters.yml file
    import os
    import pandas as pd
    import unittest
    import pickle
    import copy

    import matplotlib.pyplot as plt

    from microsim.opencl.ramp.run import run_headless
    from microsim.opencl.ramp.snapshot_convertor import SnapshotConvertor
    from microsim.opencl.ramp.snapshot import Snapshot
    from microsim.opencl.ramp.params import Params, IndividualHazardMultipliers, LocationHazardMultipliers
    from microsim.opencl.ramp.simulator import Simulator
    from microsim.opencl.ramp.disease_statuses import DiseaseStatus

    PARAMETERS_FILENAME = "default.yml"  # Creates default parameters using the default.yml file
    PARAMS = Functions.create_parameters(
        parameters_file=os.path.join("../../", "model_parameters", PARAMETERS_FILENAME))

    OPENCL_DIR = "../../microsim/opencl"
    SNAPSHOT_FILEPATH = os.path.join(OPENCL_DIR, "snapshots", "cache.npz")
    assert os.path.isfile(SNAPSHOT_FILEPATH), f"Snapshot doesn't exist: {SNAPSHOT_FILEPATH}"

    observations = pd.read_csv("observation_data/gam_cases.csv", header=0, names=["Day", "Cases"], )
    # The fitness function is defined in experiments_functions.py (so that it can be tested)
    fit = Functions.fit_l2

    ITERATIONS = 100  # Number of iterations to run for
    NUM_SEED_DAYS = 10  # Number of days to seed the population
    USE_GPU = False
    STORE_DETAILED_COUNTS = False
    REPETITIONS = 5

    assert ITERATIONS < len(observations), \
        f"Have more iterations ({ITERATIONS}) than observations ({len(observations)})."

    (fitness, sim, obs) = __run_model_current_risk_beta(0.001, return_full_details=True, multiprocess=False)
    # fitness = run_model_current_risk_beta(0.001)

    print(f"fitness: {fitness}")
    # list(zip(obs,sim))

    from scipy.optimize import minimize

    # x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    x0 = np.array([0.005])  # initial guess for each variable

    res = minimize(__run_model_current_risk_beta, x0, method='nelder-mead',
                   options={'xatol': 1e-8, 'disp': True})

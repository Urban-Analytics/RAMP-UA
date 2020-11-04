# Generic functions that are used in the experiments notebooks
# Useful to put them in here so that they can be shared across notebooks
# and can be tested (see tests/experiements/opencl_runner_tests.py)
import os
import numpy as np
import multiprocessing
import itertools
import yaml
import time
import tqdm
import pandas as pd

from typing import List
from microsim.opencl.ramp.snapshot import Snapshot
from microsim.opencl.ramp.simulator import Simulator
from microsim.opencl.ramp.run import run_headless
from microsim.opencl.ramp.params import Params, IndividualHazardMultipliers, LocationHazardMultipliers
from microsim.opencl.ramp.disease_statuses import DiseaseStatus


class OpenCLRunner:
    """Includes useful functions for running the OpenCL model in notebooks"""

    @classmethod
    def init(cls, iterations: int, repetitions: int, observations: pd.DataFrame, use_gpu: bool,
             store_detailed_counts: bool, parameters_file: str, opencl_dir: str, snapshot_filepath: str):
        """
        The class variables determine how the model should run. They need to be class variables
        because the 'run_model_with_params' function, which is called by calibration libraries, can only take
        one parameter argument -- the parameters to calibrate -- not any others. The init() function sets these values.

        :param iterations: Number of iterations to run for
        :param repetitions: Number of repetitions
        :param observations: A dataframe with the observation used to calculate fitness
        :param use_gpu: Whether to use the GPU
        :param store_detailed_counts: Whether to store age-related exposure information
        :param parameters_file:
        :param opencl_dir:
        :param snapshot_filepath:
        :return:
        """
        cls.ITERATIONS = iterations
        cls.REPETITIONS = repetitions
        cls.OBSERVATIONS = observations
        cls.USE_GPU = use_gpu
        cls.STORE_DETAILED_COUNTS = store_detailed_counts
        cls.PARAMETERS_FILE = parameters_file
        cls.OPENCL_DIR = opencl_dir
        cls.SNAPSHOT_FILEPATH = snapshot_filepath
        cls.initialised = True

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
                          proportion_asymptomatic: float = None,
                          infection_log_scale: float = None,
                          infection_mode: float = None
                          ):
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

        # current_risk_beta needs to be set first  as the OpenCL model pre-multiplies the hazard multipliers by it
        if current_risk_beta is None:
            current_risk_beta = disease_params['current_risk_beta']

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

        # Some parameters are set in the default.yml file and can be overridden
        if proportion_asymptomatic is None:
            proportion_asymptomatic = disease_params["asymp_rate"]

        p = Params(
            location_hazard_multipliers=location_hazard_multipliers,
            individual_hazard_multipliers=individual_hazard_multipliers,
            proportion_asymptomatic=proportion_asymptomatic
        )

        # Remaining parameters are defined within the Params class and have to be manually overridden
        if infection_log_scale is not None:
            p.infection_log_scale = infection_log_scale
        if infection_mode is not None:
            p.infection_mode = infection_mode

        return p

    @staticmethod
    def run_opencl_model(i: int, iterations: int, snapshot_filepath: str, params,
                         opencl_dir: str, use_gpu: bool,
                         store_detailed_counts: bool = True, quiet=False) -> (np.ndarray, np.ndarray):
        """
        Run the OpenCL model.

        :param i: Simulation number (i.e. if run as part of an ensemble)
        :param iterations: Number of iterations to ru the model for
        :param snapshot_filepath: Location of the snapshot (the model must have already been initialised)
        :param params: a Params object containing the parameters used to define how the model behaves
        :param opencl_dir: Location of the OpenCL code
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

        # Create a simulator and upload the snapshot data to the OpenCL device
        simulator = Simulator(snapshot, opencl_dir=opencl_dir, gpu=use_gpu)
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
        l_use_gpu = [use_gpu] * repetitions
        l_store_detailed_counts = [store_detailed_counts] * repetitions
        l_quiet = [True] * repetitions  # Don't print info

        args = zip(l_i, l_iterations, l_snapshot_filepath, l_params, l_opencl_dir, l_use_gpu,
                   l_store_detailed_counts, l_quiet)
        to_return = None
        start_time = time.time()
        if multiprocess:
            try:
                print("Running multiple models in multiprocess mode ... ", end="", flush=True)
                with multiprocessing.Pool(processes=int(os.cpu_count())) as pool:
                    to_return = pool.starmap(OpenCLRunner.run_opencl_model, args)
            finally:  # Make sure they get closed (shouldn't be necessary)
                pool.close()
        else:
            # Return as a list to force the models to execute (otherwise this is delayed because starmap returns
            # a generator. Also means we can use tqdm to get a progress bar, which is nice.
            results = itertools.starmap(OpenCLRunner.run_opencl_model, args)
            to_return = [x for x in tqdm.tqdm(results, desc="Running models", total=repetitions)]

        print(f".. finished, took {round(float(time.time() - start_time), 2)}s)", flush=True)
        return to_return

    @classmethod
    def run_model_with_params(cls, input_params: List, return_full_details=False):
        """Run a model REPETITIONS times using the provided parameter values.

        :param input_params: The parameter values to pass, as a list. These need to correspond to specific parameters. Currently they are:
           input_params[0] -> current_risk_beta
        :param return_details: If True then rather than just returning the fitness,
            return a tuple of (fitness, summaries_list, final_results_list).
        :return: The mean fitness across all model runs

        """
        if not cls.initialised:
            raise Exception("The OpenCLRunner class needs to be initialised first. "
                            "Call the OpenCLRunner.init() function")

        current_risk_beta = input_params[0]
        proportion_asymptomatic = input_params[1]
        infection_log_scale = input_params[2]
        infection_mode = input_params[3]

        params = OpenCLRunner.create_parameters(
            parameters_file=cls.PARAMETERS_FILE,
            current_risk_beta=current_risk_beta,
            proportion_asymptomatic=proportion_asymptomatic,
            infection_log_scale=infection_log_scale,
            infection_mode=infection_mode)

        results = OpenCLRunner.run_opencl_model_multi(
            repetitions=cls.REPETITIONS, iterations=cls.ITERATIONS, params=params,
            opencl_dir=cls.OPENCL_DIR,
            snapshot_filepath=cls.SNAPSHOT_FILEPATH,
            multiprocess=False
        )

        summaries = [x[0] for x in results]
        final_results = [x[1] for x in results]

        # Get mean cases per day from the summary object
        sim = OpenCLRunner.get_mean_total_counts(summaries, DiseaseStatus.Exposed.value)
        # Compare these to the observations
        obs = cls.OBSERVATIONS.loc[:cls.ITERATIONS - 1, "Cases"].values
        assert len(sim) == len(obs)
        fitness = OpenCLRunner.fit_l2(sim, obs)
        if return_full_details:
            return fitness, sim, obs, params, summaries
        else:
            return fitness

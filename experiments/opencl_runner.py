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
import random

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
            use_healthier_pop: bool, store_detailed_counts: bool, parameters_file: str, opencl_dir: str, snapshot_filepath: str):
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
        cls.USE_HEALTHIER_POP = use_healthier_pop
        cls.STORE_DETAILED_COUNTS = store_detailed_counts
        cls.PARAMETERS_FILE = parameters_file
        cls.OPENCL_DIR = opencl_dir
        cls.SNAPSHOT_FILEPATH = snapshot_filepath
        cls.initialised = True

    @classmethod
    def update(cls, iterations: int = None, repetitions: int = None, observations: pd.DataFrame = None,
               use_gpu: bool = None, use_healthier_pop = None, store_detailed_counts: bool = None, parameters_file: str = None,
               opencl_dir: str = None, snapshot_filepath: str = None):
        """
        Update any of the variables that have already been initialised
        """
        if not cls.initialised:
            raise Exception("The OpenCLRunner class needs to be initialised first; call OpenCLRunner.init()")
        if iterations is not None:
            cls.ITERATIONS = iterations
        if repetitions is not None:
            cls.REPETITIONS = repetitions
        if observations is not None:
            cls.OBSERVATIONS = observations
        if use_gpu is not None:
            cls.USE_GPU = use_gpu
        if use_healthier_pop is not None:
            cls.USE_HEALTHIER_POP = use_healthier_pop
        if store_detailed_counts is not None:
            cls.STORE_DETAILED_COUNTS = store_detailed_counts
        if parameters_file is not None:
            cls.PARAMETERS_FILE = parameters_file
        if opencl_dir is not None:
            cls.OPENCL_DIR = opencl_dir
        if snapshot_filepath is not None:
            cls.SNAPSHOT_FILEPATH = snapshot_filepath

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
    def get_mean_total_counts(summaries, disease_status: int, get_sd=False):
        """
        Get the mean total counts for a given disease status at every iteration over a number of model repetitions

        :param summaries: A list of Summary objects created by running the OpenCL model
        :param disease_status: The disease status number, e.g.  `DiseaseStatus.Exposed.value`
        :param get_sd: Optionally get the standard deviation as well

        :return: The mean total counts of the disease status per iteration, or (if get_sd=True)
            or a tuple of (mean,sd)

        """
        reps = len(summaries)  # Number of repetitions
        iters = len(summaries[0].total_counts[disease_status])  # Number of iterations for each repetition
        matrix = np.zeros(shape=(reps, iters))
        for rep in range(reps):
            matrix[rep] = summaries[rep].total_counts[disease_status]
        mean = np.mean(matrix, axis=0)
        sd = np.std(matrix, axis=0)
        if get_sd:
            return mean, sd
        else:
            return mean

    @staticmethod
    def get_cumulative_new_infections(summaries):
        """
        Get cumulative infections per day by summing all the non-susceptible people

        :param summaries: A list of Summary objects created by running the OpenCL model
        """
        iters = len(summaries[0].total_counts[DiseaseStatus.Exposed.value])  # Number of iterations for each repetition
        total_not_susceptible = np.zeros(iters)  # Total people not susceptible per iteration
        for d, disease_status in enumerate(DiseaseStatus):
            if disease_status != DiseaseStatus.Susceptible:
                mean = OpenCLRunner.get_mean_total_counts(summaries, d)  # Mean number of people with that disease
                total_not_susceptible = total_not_susceptible + mean
        return total_not_susceptible


    @staticmethod
    def create_parameters(parameters_file: str = None,
                          current_risk_beta: float = None,
                          infection_log_scale: float = None,
                          infection_mode: float = None,
                          presymptomatic: float = None,
                          asymptomatic: float = None,
                          symptomatic: float = None
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
            presymptomatic=calibration_params["hazard_individual_multipliers"]["presymptomatic"] \
                if presymptomatic is None else presymptomatic,
            asymptomatic=calibration_params["hazard_individual_multipliers"]["asymptomatic"] \
                if asymptomatic is None else asymptomatic,
            symptomatic=calibration_params["hazard_individual_multipliers"]["symptomatic"] \
                if symptomatic is None else symptomatic
        )

        # Some parameters are set in the default.yml file and can be overridden
        pass  # None here yet
        
        obesity_multipliers = np.array([disease_params["overweight"], disease_params["obesity_30"],disease_params["obesity_35"], disease_params["obesity_40"]])
        
        cvd = disease_params["cvd"]
        diabetes = disease_params["diabetes"]
        bloodpressure = disease_params["bloodpressure"]
        overweight_sympt_mplier = disease_params["overweight_sympt_mplier"]
        
        p = Params(
            location_hazard_multipliers=location_hazard_multipliers,
            individual_hazard_multipliers=individual_hazard_multipliers,
        )

        # Remaining parameters are defined within the Params class and have to be manually overridden
        if infection_log_scale is not None:
            p.infection_log_scale = infection_log_scale
        if infection_mode is not None:
            p.infection_mode = infection_mode
        
        p.obesity_multipliers = obesity_multipliers
        p.cvd_multiplier = cvd
        p.diabetes_multiplier = diabetes
        p.bloodpressure_multiplier = bloodpressure
        p.overweight_sympt_mplier = overweight_sympt_mplier
        return p

    @staticmethod
    def run_opencl_model(i: int, iterations: int, snapshot_filepath: str, params,
                         opencl_dir: str, use_gpu: bool,
                         use_healthier_pop: bool,
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
            # load snapshot
        snapshot = Snapshot.load_full_snapshot(path=snapshot_filepath)
        prev_obesity = np.copy(snapshot.buffers.people_obesity)
        if use_healthier_pop:
            snapshot.switch_to_healthier_population()
        print("testing obesity arrays not equal")
        print(np.mean(prev_obesity))
        print(np.mean(snapshot.buffers.people_obesity))
       # assert not np.array_equal(prev_obesity, snapshot.buffers.people_obesity)
       # print("arrays not equal")
        # set params
        snapshot.update_params(params)
        
       #print(params.use_healthier_pop)
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
            use_gpu: bool = False, use_healthier_pop: bool = True, store_detailed_counts: bool = False,
            opencl_dir=os.path.join(".", "microsim", "opencl"),
            snapshot_filepath=os.path.join(".", "microsim", "opencl", "snapshots", "cache.npz"),
            multiprocess=False,
            random_ids=False):
        """Run a number of models and return a list of summaries.

        :param multiprocess: Whether to run in mutliprocess mode (default False)
        """
        # Prepare the function arguments. We need one set of arguments per repetition
        l_i = [i for i in range(repetitions)] if not random_ids else \
            [random.randint(1, 100000) for _ in range(repetitions)]
        l_iterations = [iterations] * repetitions
        l_snapshot_filepath = [snapshot_filepath] * repetitions
        l_params = [params] * repetitions
        l_opencl_dir = [opencl_dir] * repetitions
        l_use_gpu = [use_gpu] * repetitions
        l_use_healthier_pop = [use_healthier_pop] * repetitions
        l_store_detailed_counts = [store_detailed_counts] * repetitions
        l_quiet = [False] * repetitions  # Don't print info
        
        print(f"Using healthier population - {use_healthier_pop}")
        #print(params[0])
        args = zip(l_i, l_iterations, l_snapshot_filepath, l_params, l_opencl_dir, l_use_gpu, l_use_healthier_pop, l_store_detailed_counts, l_quiet)
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
            results = itertools.starmap(OpenCLRunner.run_opencl_model, args)
            # Return as a list to force the models to execute (otherwise this is delayed because starmap returns
            # a generator. Also means we can use tqdm to get a progress bar, which is nice.
            to_return = [x for x in tqdm.tqdm(results, desc="Running models", total=repetitions)]
      
        print(f".. finished, took {round(float(time.time() - start_time), 2)}s)", flush=True)
        #print(params)
        return to_return

    @classmethod
    def run_model_with_params(cls, input_params: List, return_full_details=False):
        """Run a model REPETITIONS times using the provided parameter values.

        :param input_params: The parameter values to pass, as a list. These need to correspond to specific parameters. Currently they are:
           input_params[0] -> current_risk_beta
        :param return_full_details: If True then rather than just returning the fitness,
            return a tuple of (fitness, summaries_list, final_results_list).
        :return: The mean fitness across all model runs

        """
        if not cls.initialised:
            raise Exception("The OpenCLRunner class needs to be initialised first. "
                            "Call the OpenCLRunner.init() function")

        current_risk_beta = input_params[0]
        infection_log_scale = input_params[1]
        infection_mode = input_params[2]
        presymptomatic = input_params[3]
        asymptomatic = input_params[4]
        symptomatic = input_params[5]

        params = OpenCLRunner.create_parameters(
            parameters_file=cls.PARAMETERS_FILE,
            current_risk_beta=current_risk_beta,
            infection_log_scale=infection_log_scale,
            infection_mode=infection_mode,
            presymptomatic=presymptomatic,
            asymptomatic=asymptomatic,
            symptomatic=symptomatic)

        results = OpenCLRunner.run_opencl_model_multi(
            repetitions=cls.REPETITIONS, iterations=cls.ITERATIONS, params=params,
            opencl_dir=cls.OPENCL_DIR, snapshot_filepath=cls.SNAPSHOT_FILEPATH, use_gpu=cls.USE_GPU,
            store_detailed_counts=cls.STORE_DETAILED_COUNTS, multiprocess=False
        )

        summaries = [x[0] for x in results]
        final_results = [x[1] for x in results]

        # Get the cumulative number of new infections per day
        sim = OpenCLRunner.get_cumulative_new_infections(summaries)
        # Compare these to the observations
        obs = cls.OBSERVATIONS.loc[:cls.ITERATIONS - 1, "Cases"].values
        assert len(sim) == len(obs)
        fitness = OpenCLRunner.fit_l2(sim, obs)
        if return_full_details:
            return fitness, sim, obs, params, summaries
        else:
            return fitness

    @classmethod
    def run_model_with_params_abc(cls, input_params_dict: dict, return_full_details=False):
        """
        TEMP to work with ABC. Parameters are passed in as a dictionary.

        :param return_full_details: If True then rather than just returning the normal results,
            it returns a tuple of the following:
             (fitness value, simulated results, observations, the Params object, summaries list)
        :return: The number of cumulative new infections per day (as a list value in a
            dictionary as required by the pyabc package) unless return_full_details is True.
        """

        if not cls.initialised:
            raise Exception("The OpenCLRunner class needs to be initialised first. "
                            "Call the OpenCLRunner.init() function")

        # Check that all input parametrers are not negative
        for k, v in input_params_dict.items():
            if v < 0:
                raise Exception(f"The parameter {k}={v} < 0. "
                                f"All parameters: {input_params_dict}")

        # Splat the input_params_dict to automatically set any parameters that have been inlcluded
        params = OpenCLRunner.create_parameters(
            parameters_file=cls.PARAMETERS_FILE,
            **input_params_dict
        )

        results = OpenCLRunner.run_opencl_model_multi(
            repetitions=cls.REPETITIONS, iterations=cls.ITERATIONS, params=params,
            opencl_dir=cls.OPENCL_DIR, snapshot_filepath=cls.SNAPSHOT_FILEPATH, use_gpu=cls.USE_GPU,
            store_detailed_counts=cls.STORE_DETAILED_COUNTS, multiprocess=False, random_ids=True
        )

        summaries = [x[0] for x in results]
        # Get the cumulative number of new infections per day (i.e. simulated results)
        sim = OpenCLRunner.get_cumulative_new_infections(summaries)
        print(f"Ran Model. {str(input_params_dict)} ("
              f"Sum result: {sum(sim)})")

        if return_full_details:
            # Can compare these to the observations to get a fitness
            obs = cls.OBSERVATIONS.loc[:cls.ITERATIONS - 1, "Cases"].values
            assert len(sim) == len(obs)
            fitness = OpenCLRunner.fit_l2(sim, obs)
            return fitness, sim, obs, params, summaries
        else:  # Return the expected counts in a dictionary
            return {"data": sim}

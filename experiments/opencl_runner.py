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
from microsim.opencl.ramp.summary import Summary


class OpenCLRunner:
    """Includes useful functions for running the OpenCL model in notebooks"""

    # Need a list to optionally store additional constant parameter values that cannot
    # be passed through one of the run model functions.
    constants = {}

    @classmethod
    def init(cls, iterations: int, repetitions: int, observations: pd.DataFrame, use_gpu: bool,
             use_healthier_pop: bool, store_detailed_counts: bool, parameters_file: str, opencl_dir: str,
             snapshot_filepath: str):
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
               use_gpu: bool = None, use_healthier_pop=None, store_detailed_counts: bool = None,
               parameters_file: str = None,
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

    @classmethod
    def set_constants(cls, constants):
        """Set any constant variables (parameters) that override the defaults.
        :param constants: This should be a dist of parameter_nam -> value
        """
        cls.constants = constants

    @classmethod
    def clear_constants(cls):
        cls.constants = {}

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
                          symptomatic: float = None,
                          retail: float = None,
                          primary_school: float = None,
                          secondary_school: float = None,
                          home: float = None,
                          work: float = None,
                          ):
        """Create a params object with the given arguments. This replicates the functionality in
        microsim.main.create_params() but rather than just reading parameters from the parameters
        json file, it allows some of the parameters to be set manually.

        Also note that some (constant) parameters can be set by calling the `set_constants` method.
        This is useful for cases where parameters should override the defaults specified in the
        parameters file but cannot be called directly by the function that is running the model"""

        # Read the default parameters from a yml file, then override with any provided
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
        current_risk_beta = OpenCLRunner._check_if_none("current_risk_beta", current_risk_beta,
                                                        disease_params['current_risk_beta'])

        # Location hazard multipliers can be passed straight through to the LocationHazardMultipliers object.
        # If no argument was passed then the default in the parameters file is used. Note that they need to
        # be multiplied by the current_risk_beta
        location_hazard_multipliers = LocationHazardMultipliers(
            retail=current_risk_beta * OpenCLRunner._check_if_none("retail",
                                                                   retail,
                                                                   calibration_params["hazard_location_multipliers"][
                                                                       "Retail"]),
            primary_school=current_risk_beta * OpenCLRunner._check_if_none("primary_school",
                                                                           primary_school, calibration_params[
                                                                               "hazard_location_multipliers"][
                                                                               "PrimarySchool"]),
            secondary_school=current_risk_beta * OpenCLRunner._check_if_none("secondary_school",
                                                                             secondary_school, calibration_params[
                                                                                 "hazard_location_multipliers"][
                                                                                 "SecondarySchool"]),
            home=current_risk_beta * OpenCLRunner._check_if_none("home",
                                                                 home,
                                                                 calibration_params["hazard_location_multipliers"][
                                                                     "Home"]),
            work=current_risk_beta * OpenCLRunner._check_if_none("work",
                                                                 work,
                                                                 calibration_params["hazard_location_multipliers"][
                                                                     "Work"]),
        )

        # Individual hazard multipliers can be passed straight through
        individual_hazard_multipliers = IndividualHazardMultipliers(
            presymptomatic=OpenCLRunner._check_if_none("presymptomatic",
                                                       presymptomatic,
                                                       calibration_params["hazard_individual_multipliers"][
                                                           "presymptomatic"]),
            asymptomatic=OpenCLRunner._check_if_none("asymptomatic",
                                                     asymptomatic, calibration_params["hazard_individual_multipliers"][
                                                         "asymptomatic"]),
            symptomatic=OpenCLRunner._check_if_none("symptomatic",
                                                    symptomatic,
                                                    calibration_params["hazard_individual_multipliers"]["symptomatic"])
        )

        # Some parameters are set in the default.yml file and can be overridden
        pass  # None here yet

        obesity_multipliers = np.array(
            [disease_params["overweight"], disease_params["obesity_30"], disease_params["obesity_35"],
             disease_params["obesity_40"]])

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

    @classmethod
    def _check_if_none(cls, param_name, param_value, default_value):
        """Checks whether the given param is None. If so, it will return a constant value, if it has
         one, or failing that the provided default if it is"""
        if param_value is not None:
            # The value has been provided. Return it, but check a constant hasn't been set as well
            # (it's unlikely that someone would set a constant and then also provide a value for the same parameter)
            if param_name in cls.constants.keys():
                raise Exception(f"A parameter {param_name} has been provided, but it has also been set as a constant")
            return param_value
        else:  # No value provided, return a constant, if there is one, or the default otherwise
            if param_name in cls.constants.keys():
                return cls.constants[param_name]
            else:
                return default_value

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
            use_gpu: bool = False, use_healthier_pop: bool = False, store_detailed_counts: bool = False,
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

        args = zip(l_i, l_iterations, l_snapshot_filepath, l_params, l_opencl_dir, l_use_gpu, l_use_healthier_pop,
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
            results = itertools.starmap(OpenCLRunner.run_opencl_model, args)
            # Return as a list to force the models to execute (otherwise this is delayed because starmap returns
            # a generator. Also means we can use tqdm to get a progress bar, which is nice.
            to_return = [x for x in tqdm.tqdm(results, desc="Running models", total=repetitions)]

        print(f".. finished, took {round(float(time.time() - start_time), 2)}s)", flush=True)
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
        Run the model, compatible with pyABC. Random variables (parameters) are passed in as a dictionary.
        For constant parameters that override the defaults (in the default.yml file) set them first
        with the `set_constants` method.

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

        # Check if there are any constants that should

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
        print(f"Ran Model with {str(input_params_dict)}")

        if return_full_details:
            # Can compare these to the observations to get a fitness
            obs = cls.OBSERVATIONS.loc[:cls.ITERATIONS - 1, "Cases"].values
            assert len(sim) == len(obs)
            fitness = OpenCLRunner.fit_l2(sim, obs)
            return fitness, sim, obs, params, summaries
        else:  # Return the expected counts in a dictionary
            return {"data": sim}


class OpenCLWrapper(object):
    """A simplified version of the OpenCLRunner designed specifically to support using ABC as a day to do
    data assimilation. Uses some features of OpenCLRunner but isn't intended to run a model from start to
    finish."""

    def __init__(self, const_params_dict,
                 quiet, use_gpu, store_detailed_counts, start_day, run_length,
                 current_particle_pop_df, parameters_file, snapshot_file, opencl_dir,
                 _random_params_dict=None):
        """The constructor accepts the following:

         - `const_params_dict`: a dictionary of constant model parameters (e.g. hazard coefficients) that are
         not known about by pyabc
         - `current_particle_pop_df`: a dictionary containing the current pyABC population of particles. These
         can be used to re-start previous particles from previous data assimilation windows. If 'None' then
         the model is just starting so there is no previous population.
         - A load of administrative parameters (not used in the model logic).
         - The location of a `parameters_file`, that has default values for parameters (if None then then
         a default location is assumed, see `OpenCLRunner.create_parameters`).
         - The location of the 'snapshot_file' which has the location of a pre-created model snapshot

        Random variables (in `_random_params_dict`) should not be passed directly to the constructor, they should be
        passed via the __call__  function by instantiating an instance of OpenCLWrapper first and then calling that
        object directly. E.g. like this:
        ``m = OpenCLWrapper(const_params_dict={"const_param1":1.0, "const_param2":0.1})
        m(random_variables_dict=={"random_param1":2.0, "random_param2":0.5})``
        That method was suggested here: https://github.com/ICB-DCM/pyABC/issues/446
        """
        # Set administrative parameters
        self.quiet = quiet
        self.use_gpu = use_gpu
        self.store_detailed_counts = store_detailed_counts
        self.start_day = start_day
        self.run_length = run_length
        self.parameters_file = parameters_file
        self.snapshot_file = snapshot_file
        self.opencl_dir = opencl_dir
        self.current_particle_pop_df = current_particle_pop_df

        # Now deal with the model parameters
        self.const_params_dict = const_params_dict
        if _random_params_dict is None:  # Only have constant params
            final_params = const_params_dict
        else:  # Have constants *and* random variables
            # Check the parameters are distinct
            for key in const_params_dict:
                if key in _random_params_dict:
                    raise Exception(
                        f"Parameter {key} in the constants dict is also in the random variables dict {_random_params_dict}")
            final_params = {**const_params_dict, **_random_params_dict}

        # Have a single params dict now ('final_params'), can create the Parameters object
        if self.parameters_file is None:
            self.params = OpenCLRunner.create_parameters(**final_params)
        else:
            self.params = OpenCLRunner.create_parameters(parameters_file=self.parameters_file, **final_params)

    def __call__(self, random_params_dict) -> dict:
        """This function is used by pyABC to run the model and pass in random variables.

        :param random_params_dict: Dictionary with random variable values that should be used
        to run the model.

        :return: a dictionary containing model results and other useful pieces of information,
        like the current model state."""

        # Create a new model with previously created constant parameters (set through the constructor)
        # and random variables passed here
        # (Is creation of m actually necessary? Probably not. Advantageous though for when we want
        # to call methods like m.run()
        m = OpenCLWrapper(const_params_dict=self.const_params_dict,
                          quiet=self.quiet, use_gpu=self.use_gpu, store_detailed_counts=self.store_detailed_counts,
                          start_day=self.start_day, run_length=self.run_length,
                          current_particle_pop_df=self.current_particle_pop_df,
                          parameters_file=self.parameters_file, snapshot_file=self.snapshot_file,
                          opencl_dir=self.opencl_dir,
                          _random_params_dict=random_params_dict)
        return m.run()

    def run(self):

        # If this is the first data assimilation window, we can just run the model as normal
        if self.start_day == 0:
            assert self.current_particle_pop_df is None  # Shouldn't have any preivously-created particles
            # load snapshot
            snapshot = Snapshot.load_full_snapshot(path=self.snapshot_file)
            # set params
            snapshot.update_params(self.params)
            # Can set the random seed to make it deterministic (None means np will choose one randomly)
            snapshot.seed_prngs(seed=None)

            # Create a simulator and upload the snapshot data to the OpenCL device
            simulator = Simulator(snapshot, opencl_dir=self.opencl_dir, gpu=self.use_gpu)
            simulator.upload_all(snapshot.buffers)

            if not self.quiet:
                # print(f"Running simulation {sim_number + 1}.")
                print(f"Running simulation")

            params = Params.fromarray(snapshot.buffers.params)  # XX Why extract Params? Can't just use PARAMS?

            summary = Summary(snapshot,
                              store_detailed_counts=self.store_detailed_counts,
                              max_time=self.run_length  # Total length of the simulation
                              )

            # only show progress bar in quiet mode
            timestep_iterator = range(self.run_length) if self.quiet \
                else tqdm(range(self.quiet), desc="Running simulation")

            iter_count = 0  # Count the total number of iterations
            # Run for iterations days
            for _ in timestep_iterator:
                # Update parameters based on lockdown
                params.set_lockdown_multiplier(snapshot.lockdown_multipliers, iter_count)
                simulator.upload("params", params.asarray())

                # Step the simulator
                simulator.step()
                iter_count += 1

            # Update the statuses
            simulator.download("people_statuses", snapshot.buffers.people_statuses)
            summary.update(iter_count, snapshot.buffers.people_statuses)

            if not self.quiet:
                for i in range(self.run_length):
                    print(f"\nDay {i}")
                    summary.print_counts(i)

            if not self.quiet:
                print("\nFinished")

            # Download the snapshot from OpenCL to host memory
            # XX This is 'None'.
            final_state = simulator.download_all(snapshot.buffers)

            pass
        else:  # Otherwise we need to restart previous models stored in the current_particle_pop_df
            # XXXX CAN GET OLD MODEL STATES, WITH ALL DISEASE STATUSES, FROM THE DF. TWO ISSUES
            # 1. But need to work out how to draw these appropriately; can't assume they are each as good as
            # each other. THIS SHOULD BE OK, surely there's a way to go from the final particles and weights
            # to the DF of state vectors. Particle ID? Just try it out.
            # 2. Also: what to do about stochasticity. For a given (global) parameter combination, we will
            # get quite different results depending on the mode state. - I DON'T THINK THIS IS A PROBLEM.
            # ABC Commonly used with stochastic models. E.g. https://eprints.lancs.ac.uk/id/eprint/80439/1/mainR1.pdf
            #
            raise Exception("Not implemented yet")

        # Return the current state of the model in a dictionary describing what it is
        #return {"simulator": simulator}
        return {"simulator": snapshot}

    @staticmethod
    def summary_stats(raw_model_results: dict) -> dict:
        """Takes raw model results, as output from `__call__` and passed them on to the
        `distance` function. This doesn't actually calculate summary statistics, nor do
        anything else useful in itself, but is useful because anything returned in the
        dictionary is added to the results database, so we can recover a model state, not
        just it's results.

        :param raw_model_results: dictionary of model results as output from __call__.
        :return: processed model results.
        """
        # Check that we receive everything that we expect to
        if "simulator" not in raw_model_results.keys():
            raise Exception(f"No 'simulator' item found in the model results that are passed "
                            f"to summary_stats: {raw_model_results}")

        # Just pass the model results on. The 'distance' function can work out how good the results
        # are. The important thing is that the model summary will now be stored by ABC in the database
        return raw_model_results

    @staticmethod
    def distance(sim, obs):
        """Calculate the difference (error) between simulated and observed data.

        :param sim: a dictionary containing the simulated data
        :param obs: a dictionary containing the observed (real) data
        :return: a single distance measure (float). Lower is better."""
        # Check that we receive everything that we expect to
        if "simulator" not in sim.keys():
            raise Exception(f"No 'simulator' item found in the model results that are passed "
                            f"to summary_stats: {sim}")

        # TODO HERE. Calculate the distance.
        return 1

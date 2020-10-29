# Generic functions that are used in the experiments notebooks
# Useful to put them in here so that they can be shared across notebooks
# and can be tested (see tests/experiements/experiments_functions_tests.py)
import os
import numpy as np
import multiprocessing
import itertools
import yaml

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
    def create_parameters(parameters_file: str=None, current_risk_beta=None):
        """Create a params object with the given arguments"""

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

        if current_risk_beta == None:
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

        proportion_asymptomatic = disease_params["asymp_rate"]

        return Params(
            location_hazard_multipliers=location_hazard_multipliers,
            individual_hazard_multipliers=individual_hazard_multipliers,
            proportion_asymptomatic=proportion_asymptomatic
        )

    @staticmethod
    def run_opencl_model(i: int, iterations: int, snapshot_filepath: str, params,
                         opencl_dir: str, num_seed_days: int, use_gpu: bool,
                         store_detailed_counts: bool = True) -> (np.ndarray, np.ndarray):
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
        l_iterations = [iterations] *repetitions
        l_snapshot_filepath = [snapshot_filepath] * repetitions
        l_params = [params] * repetitions
        l_opencl_dir = [opencl_dir] * repetitions
        l_num_seed_days = [num_seed_days] * repetitions
        l_use_gpu = [use_gpu] * repetitions
        l_store_detailed_counts = [store_detailed_counts] * repetitions

        if multiprocess:
            try:
                with multiprocessing.Pool(processes=int(os.cpu_count())) as pool:
                    return pool.starmap(Functions.run_opencl_model, zip(
                        l_i, l_iterations, l_snapshot_filepath, l_params, l_opencl_dir, l_num_seed_days, l_use_gpu,
                        l_store_detailed_counts
                    ))
            finally:  # Make sure they get closed (shouldn't be necessary)
                pool.close()
        else:
            # Note: wrapping as a list forces the execution of the model, otherwise starmap returns a generator
            # so execution is delayed
            return list(itertools.starmap(Functions.run_opencl_model, zip(
                        l_i, l_iterations, l_snapshot_filepath, l_params, l_opencl_dir, l_num_seed_days, l_use_gpu,
                        l_store_detailed_counts
                    )))




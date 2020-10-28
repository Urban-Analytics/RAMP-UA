# Generic functions that are used in the experiments notebooks
# Useful to put them in here so that they can be shared across notebooks
# and can be tested (see tests/experiements/experiments_functions_tests.py)
import numpy as np

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
import os
import multiprocessing

from microsim.opencl.ramp.snapshot import Snapshot
from microsim.opencl.ramp.simulator import Simulator
from microsim.opencl.ramp.run import run_headless


def run_opencl_model_multiprocess(*args):
    #*al_i, l_snapshot_filepath, l_params, l_opencl_dir, l_num_seed_days, l_use_gpu):
    try:
        with multiprocessing.Pool(processes=int(os.cpu_count())) as pool:
            results = pool.starmap(_run_opencl_model, zip(*args))
            return results

    finally:  # Make sure they get closed (shouldn't be necessary)
        pool.close()



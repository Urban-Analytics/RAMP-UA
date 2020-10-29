# This is a workaround to allow multiprocessing.Pool to work in the pf_experiments_plots notebook.
# The function called by pool.map ('count_wiggles') needs to be defined in this separate file and imported.
# https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror/42383397
import os
import multiprocessing
import itertools # TEMP

from microsim.opencl.ramp.snapshot import Snapshot
from microsim.opencl.ramp.simulator import Simulator
from microsim.opencl.ramp.run import run_headless


def run_opencl_model_multiprocess(*args):
    #*al_i, l_snapshot_filepath, l_params, l_opencl_dir, l_num_seed_days, l_use_gpu):
    try:
        with multiprocessing.Pool(processes=int(os.cpu_count())) as pool:
            results = pool.starmap(_run_opencl_model, zip(*args))
            #results = itertools.starmap(_run_opencl_model, zip(*args))
            return results

    finally:  # Make sure they get closed (shouldn't be necessary)
        pool.close()


def _run_opencl_model(i, iterations, snapshot_filepath, params, opencl_dir, num_seed_days, use_gpu,
                      store_detailed_counts=True):

    # load snapshot
    snapshot = Snapshot.load_full_snapshot(path=snapshot_filepath)
    
    # set params
    snapshot.update_params(params)
    
    # set the random seed of the model for each repetition, otherwise it is completely deterministic
    snapshot.seed_prngs(i)

    # Create a simulator and upload the snapshot data to the OpenCL device
    simulator = Simulator(snapshot, opencl_dir=opencl_dir, gpu=use_gpu, num_seed_days=num_seed_days)
    simulator.upload_all(snapshot.buffers)
    
    print(f"Running simulation {i+1}.")
    summary, final_state = run_headless(simulator, snapshot, iterations, quiet=True,
                                        store_detailed_counts=store_detailed_counts)
    return summary, final_state

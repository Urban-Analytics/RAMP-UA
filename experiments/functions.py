# Some notebook functionality (like multiprocessing) only works with functions defined in separate
# files. Those functions can go here.
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


def _run_opencl_model(i, iterations, snapshot_filepath, params, opencl_dir, num_seed_days, use_gpu):

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
    
    print(f"Running simulation {i+1}.")
    summary, final_state = run_headless(simulator, snapshot, iterations, quiet=True)
    return (summary, final_state)

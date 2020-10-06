#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core RAMP-UA model.

Created on Wed Apr 29 19:59:25 2020

@author: nick
"""
import sys

sys.path.append("microsim")  # This is only needed when testing. I'm so confused about the imports
import multiprocessing
import pandas as pd

pd.set_option('display.expand_frame_repr', False)  # Don't wrap lines when displaying DataFrames
# pd.set_option('display.width', 0)  # Automatically find the best width
import os
import click  # command-line interface
import pickle  # to save data
from yaml import load, dump, SafeLoader  # pyyaml library for reading the parameters.yml file
from shutil import copyfile

from microsim.quant_api import QuantRampAPI
from microsim.population_initialisation import PopulationInitialisation
from microsim.microsim_model import Microsim
from microsim.opencl.ramp.run import run_opencl
from microsim.opencl.ramp.snapshot_convertor import SnapshotConvertor
from microsim.opencl.ramp.snapshot import Snapshot
from microsim.initialisation_cache import InitialisationCache


# ********
# PROGRAM ENTRY POINT
# Uses 'click' library so that it can be run from the command line
# ********
@click.command()
@click.option('-p', '--parameters_file', default="./model_parameters/default.yml", type=click.Path(exists=True),
              help="Parameters file to use to configure the model. Default: ./model_parameters/default.yml")
@click.option('-npf', '--no-parameters-file', is_flag=True,
              help="Don't read a parameters file, use command line arguments instead")
@click.option('-i', '--iterations', default=10, help='Number of model iterations. 0 means just run the initialisation')
@click.option('-s', '--scenario', default="default", help="Name this scenario; output results will be put into a "
                                                          "directory with this name.")
@click.option('--data-dir', default="devon_data", help='Root directory to load data from')
@click.option('--output/--no-output', default=True,
              help='Whether to generate output data (default yes).')
@click.option('--output-every-iteration/--no-output-every-iteration', default=False,
              help='Whether to generate output data at every iteration rather than just at the end (default no).')
@click.option('--debug/--no-debug', default=False, help="Whether to run some more expensive checks (default no debug)")
@click.option('-r', '--repetitions', default=1, help="How many times to run the model (default 1)")
@click.option('-l', '--lockdown-file', default="google_mobility_lockdown_daily.csv",
              help="Optionally read lockdown mobility data from a file (default use google mobility). To have no "
                   "lockdown pass an empty string, i.e. --lockdown-file='' ")
@click.option('--quant-dir', default=None, help='Directory to QUANT data, set to None to use Devon data')
@click.option('-s', '--use-cache/--dont-use-cache', default=False,
              help="Whether to cache the population data initialisation")
@click.option('-s', '--opencl/--no-opencl', default=False, help="Run OpenCL model")
@click.option('-s', '--opencl-gui/--no-opencl-gui', default=False,
              help="Use GUI visualisation for OpenCL model (if false then run in headless mode")
@click.option('-s', '--opencl-gpu/--no-opencl-gpu', default=False,
              help="Run OpenCL model on the GPU (if false then run using CPU")
def main(parameters_file, no_parameters_file, iterations, scenario, data_dir, output, output_every_iteration,
               debug, repetitions, lockdown_file, quant_dir, use_cache, opencl, opencl_gui, opencl_gpu):
    # First see if we're reading a parameters file or using command-line arguments.
    if no_parameters_file:
        print("Not reading a parameters file")
    else:
        print(f"Reading parameters file: {parameters_file}. Any other command-line arguments are being ignored")
        with open(parameters_file, 'r') as f:
            parameters = load(f, Loader=SafeLoader)
            sim_params = parameters["microsim"]  # Parameters for the dynamic microsim (python)
            calibration_params = parameters["microsim_calibration"]
            disease_params = parameters["disease"]  # Parameters for the disease model (r)
            # TODO Implement a more elegant way to set the parameters and pass them to the model. E.g.:
            #         self.params, self.params_changed = Model._init_kwargs(params, kwargs)
            #         [setattr(self, key, value) for key, value in self.params.items()]
            # Utility parameters
            scenario = sim_params["scenario"]
            iterations = sim_params["iterations"]
            data_dir = sim_params["data-dir"]
            output = sim_params["output"]
            output_every_iteration = sim_params["output-every-iteration"]
            debug = sim_params["debug"]
            repetitions = sim_params["repetitions"]
            lockdown_file = sim_params["lockdown-file"]
            quant_dir = sim_params["quant-dir"]

    # Check the parameters are sensible
    if iterations < 0:
        raise ValueError("Iterations must be > 0")
    if repetitions < 1:
        raise ValueError("Repetitions must be greater than 0")
    if (not output) and output_every_iteration:
        raise ValueError("Can't choose to not output any data (output=False) but also write the data at every "
                         "iteration (output_every_iteration=True)")

    print(f"Running model with the following parameters:\n"
          f"\tParameters file: {parameters_file}\n"
          f"\tScenario directory: {scenario}\n"
          f"\tNumber of iterations: {iterations}\n"
          f"\tData dir: {data_dir}\n"
          f"\tOutputting results?: {output}\n"
          f"\tOutputting results at every iteration?: {output_every_iteration}\n"
          f"\tDebug mode?: {debug}\n"
          f"\tNumber of repetitions: {repetitions}\n"
          f"\tLockdown file: {lockdown_file}\n"
          f"\tCalibration parameters: {'N/A (not reading parameters file)' if no_parameters_file else str(calibration_params)}\n")

    if iterations == 0:
        print("Iterations = 0. Not stepping model, just assigning the initial risks.")

    # To fix file path issues, use absolute/full path at all times
    # Pick either: get working directory (if user starts this script in place, or set working directory
    # Option A: copy current working directory:
    base_dir = os.getcwd()  # get current directory
    data_dir = os.path.join(base_dir, data_dir)
    r_script_dir = os.path.join(base_dir, "R", "py_int")

    # Temporarily only want to use Devon MSOAs
    # devon_msoas = pd.read_csv(os.path.join(data_dir, "devon_msoas.csv"), header=None,
    #                           names=["x", "y", "Num", "Code", "Desc"])

    # check whether to use QUANT or Devon data
    if quant_dir is None:
        quant_object = None
        print("Using Devon data")
    else:
        print("Using QUANT data")
        # we only need 1 QuantRampAPI object even if we do multiple iterations
        # the quant_object object will be called by each microsim object
        if os.path.isdir(os.path.join(data_dir, quant_dir)):
            print(os.path.join(data_dir, quant_dir))
        else:
            raise Exception("QUANT directory does not exist, please check input")
        quant_object = QuantRampAPI(os.path.join(data_dir, quant_dir))

    # args for population initialisation
    population_args = {"data_dir": data_dir, "debug": debug, "lockdown_file": lockdown_file, "use_cache": True,
                       "quant_object": quant_object}

    # Use same arguments whether running 1 repetition or many
    msim_args = {"data_dir": data_dir, "r_script_dir": r_script_dir, "output": output,
                 "output_every_iteration": output_every_iteration}

    if not no_parameters_file:  # When using a parameters file, include the calibration parameters
        msim_args.update(**calibration_params)  # python calibration parameters are unpacked now
        # Also read the R calibration parameters (this is a separate section in the .yml file)
        if disease_params is not None:
            # (If the 'disease_params' section is included but has no calibration variables then we want to ignore it -
            # it will be turned into an empty dictionary by the Microsim constructor)
            msim_args["disease_params"] = disease_params  # R parameters kept as a dictionary and unpacked later

    # Temporily use dummy data for testing
    # data_dir = os.path.join(base_dir, "dummy_data")
    # m = Microsim(data_dir=data_dir, testing=True, output=output)

    cache = InitialisationCache(cache_dir=base_dir + "/microsim/temp_cache/")

    # generate new population dataframes if we aren't using the cache
    if not use_cache:
        population = PopulationInitialisation(**population_args)
        individuals, activity_locations, time_activity_multiplier = population.generate()

        # store in cache so we can load later
        cache.store_in_cache(individuals, activity_locations, time_activity_multiplier)
    else:
        individuals, activity_locations, time_activity_multiplier = cache.read_from_cache()

    # Select which model implementation to run
    if opencl:
        run_opencl_model(individuals, activity_locations, time_activity_multiplier, iterations, data_dir, base_dir,
                         opencl_gui, opencl_gpu, use_cache)
    else:
        run_python_model(individuals, activity_locations, time_activity_multiplier, msim_args, iterations,
                         repetitions, parameters_file)


def run_opencl_model(individuals_df, activity_locations_df, time_activity_multiplier, iterations, data_dir, base_dir,
                     use_gui, use_gpu, use_cache):
    print("\nRunning OpenCL model")

    snapshot_cache_filepath = base_dir + "/microsim/opencl/snapshots/cache.npz"

    if not use_cache or not os.path.exists(snapshot_cache_filepath):
        snapshot_converter = SnapshotConvertor(individuals_df, activity_locations_df, time_activity_multiplier, data_dir)
        snapshot = snapshot_converter.generate_snapshot()
        snapshot.save(snapshot_cache_filepath) # store snapshot in cache so we can load later
    else:  # load cached snapshot
        snapshot = Snapshot.load_full_snapshot(path=snapshot_cache_filepath)

    # set the random seed of the model
    snapshot.seed_prngs(42)

    run_opencl(snapshot, iterations, data_dir, use_gui, use_gpu, quiet=False)


def run_python_model(individuals_df, activity_locations_df, time_activity_multiplier, msim_args, iterations,
                     repetitions, parameters_file):
    print("\nRunning Python / R model")

    # Create a microsim object
    m = Microsim(individuals_df, activity_locations_df, time_activity_multiplier, **msim_args)
    copyfile(parameters_file, os.path.join(m.SCEN_DIR, "parameters.yml"))

    # Run the Python / R model
    if repetitions == 1:
        m.run(iterations, 0)
    elif repetitions >= 1:  # Run it multiple times on lots of cores
        try:
            with multiprocessing.Pool(processes=int(os.cpu_count())) as pool:
                # Copy the model instance so we don't have to re-read the data each time
                # (Use a generator so we don't need to store all the models in memory at once).
                models = (Microsim._make_a_copy(m) for _ in range(repetitions))
                pickle_out = open(os.path.join("Models_m.pickle"), "wb")
                pickle.dump(m, pickle_out)
                pickle_out.close()
                # models = ( Microsim(msim_args) for _ in range(repetitions))
                # Also need a list giving the number of iterations for each model (same for each model)
                iters = (iterations for _ in range(repetitions))
                repnr = (r for r in range(repetitions))
                # Run the models by passing each model and the number of iterations
                pool.starmap(_run_multicore, zip(models, iters, repnr))
        finally:  # Make sure they get closed (shouldn't be necessary)
            pool.close()


def _run_multicore(m, iter, rep):
    return m.run(iter, rep)


if __name__ == "__main__":
    main()
    print("End of program")

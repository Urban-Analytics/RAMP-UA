#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core RAMP-UA model.

Created on Wed Apr 29 19:59:25 2020

@author: nick
@edits: Anna (Feb 2021)
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
from yaml import load, SafeLoader  # pyyaml library for reading the parameters.yml file
from shutil import copyfile

from microsim.quant_api import QuantRampAPI
from microsim.population_initialisation import PopulationInitialisation
from microsim.microsim_model import MicrosimModel
from microsim.opencl.ramp.run import run_opencl
from microsim.opencl.ramp.snapshot_convertor import SnapshotConvertor
from microsim.opencl.ramp.snapshot import Snapshot
from microsim.opencl.ramp.params import Params, IndividualHazardMultipliers, LocationHazardMultipliers
from microsim.initialisation_cache import InitialisationCache
from microsim.constants import Constants


# ********
# PROGRAM ENTRY POINT
# Uses 'click' library so that it can be run from the command line
# ********


## actual script


@click.command()
@click.option('-p',
              '--parameters-file',
              type=click.Path(exists=True),
              help="Parameters file to use to configure the model. This must be located in the working directory.")
# @click.option('-ocl', '--opencl/--no-opencl', default=True, help="Run OpenCL model (runs in headless mode by default")
def main(parameters_file):
    """
    Main function which runs the population initialisation, then chooses which model to run, either the Python/R
    model or the OpenCL model
    """
    
    # If parameters file is missing, raise exception:
    #raise Exception("Missing parameters file, please check input")
    # If parameters file path doesn't exist, raise exception:
    # this is automatically checked already by the option click.path(exists=True) above
    # If parameters file is wrong type (not yml) raise exception:

    # First check that the parameters file has been assigned in input.
    print(f"--\nReading parameters file: {parameters_file}\n--")

    try:
        with open(parameters_file, 'r') as f:
            #print(f"Reading parameters file: {parameters_file}. ")
            parameters = load(f,
                              Loader=SafeLoader)
            sim_params = parameters["microsim"]  # Parameters for the dynamic microsim (python)
            calibration_params = parameters["microsim_calibration"]
            disease_params = parameters["disease"]  # Parameters for the disease model (r)
            # TODO Implement a more elegant way to set the parameters and pass them to the model. E.g.:
            #         self.params, self.params_changed = Model._init_kwargs(params, kwargs)
            #         [setattr(self, key, value) for key, value in self.params.items()]
            # Utility parameters
            scenario = sim_params["scenario"]
            initialise = sim_params["initialise"]
            iterations = sim_params["iterations"]
            Constants.Paths.PROJECT_FOLDER_ABSOLUTE_PATH = sim_params["project-dir-absolute-path"]
            selected_region_folder_name = sim_params["selected-region-folder-name"]
            output = sim_params["output"]
            output_every_iteration = sim_params["output-every-iteration"]
            debug = sim_params["debug"]
            repetitions = sim_params["repetitions"]
            lockdown_file = sim_params["lockdown-file"]
            #quant_dir = sim_params["quant-dir"]
            use_cache = sim_params["use-cache"]
            open_cl_model = sim_params["opencl-model"]
            opencl_gui = sim_params["opencl-gui"]
            opencl_gpu = sim_params["opencl-gpu"]
    except Exception as error:
        print('Error in parameters file format')
        raise error

    # Check the parameters are sensible
    if iterations < 1:
        raise ValueError("Iterations must be > 1. If you want to just initialise the model and then exit,"
                        "set initalise : true")
    if repetitions < 1:
        raise ValueError("Repetitions must be greater than 0")
    if (not output) and output_every_iteration:
        raise ValueError("Can't choose to not output any data (output=False) but also write the data at every "
                        "iteration (output_every_iteration=True)")


    # To fix file path issues, use absolute/full path at all times
    # Pick either: get working directory (if user starts this script in place, or set working directory)
    # Option A: copy current working directory:
    ###### current_working_dir = os.getcwd()  # get current directory
    # TODO: change this working dir because it's not correct and had to add the ".." in the 2 paths under here

    # Check that working directory is as expected
    # path = os.path.join(current_working_dir, "..", Constants.Paths.DATA_FOLDER, Constants.Paths.REGIONAL_DATA_FOLDER)
    # if not os.path.exists(os.path.join(current_working_dir, "..", Constants.Paths.DATA_FOLDER, Constants.Paths.REGIONAL_DATA_FOLDER)):
    if not os.path.exists(os.path.join(Constants.Paths.PROJECT_FOLDER_ABSOLUTE_PATH,
                                       Constants.Paths.DATA_FOLDER,
                                       Constants.Paths.REGIONAL_DATA_FOLDER)):
        raise Exception("Data folder structure not valid. Make sure you are running within correct working directory.")


    # regional_data_dir_full_path = os.path.join(current_working_dir,
    #                                            "..",
    #                                            Constants.Paths.DATA_FOLDER,
    #                                            Constants.Paths.REGIONAL_DATA_FOLDER,
    #                                            regional_data_dir)
    selected_region_folder_full_path = os.path.join(Constants.Paths.PROJECT_FOLDER_ABSOLUTE_PATH,
                                               Constants.Paths.DATA_FOLDER,
                                               Constants.Paths.REGIONAL_DATA_FOLDER,
                                               selected_region_folder_name)
    common_data_dir_full_path = os.path.join(Constants.Paths.PROJECT_FOLDER_ABSOLUTE_PATH,
                                             Constants.Paths.DATA_FOLDER,
                                             Constants.Paths.COMMON_DATA_FOLDER)
    if not os.path.exists(selected_region_folder_full_path):
        raise Exception("Regional data folder doesn't exist")
    
    # population_args = {"project_dir": project_dir, "regional_data_dir": regional_data_dir_full_path, "debug": debug}
    population_args = {"common_data_dir": common_data_dir_full_path,
                       "regional_data_dir": selected_region_folder_full_path,
                       "debug": debug}


    # r_script_dir = os.path.join(current_working_dir, "R", "py_int")
    r_script_dir = os.path.join(Constants.Paths.PROJECT_FOLDER_ABSOLUTE_PATH,
                                "R",
                                "py_int")


#     # args for Python/R Microsim. Use same arguments whether running 1 repetition or many
    msim_args = {"data_dir": selected_region_folder_full_path,
                 "r_script_dir": r_script_dir,
                 "scen_dir": scenario,
                 "output": output,
                 "output_every_iteration": output_every_iteration}

    msim_args.update(**calibration_params)  # python calibration parameters are unpacked now
    # Also read the R calibration parameters (this is a separate section in the .yml file)
    if disease_params is not None:
        # (If the 'disease_params' section is included but has no calibration variables then we want to ignore it -
        # it will be turned into an empty dictionary by the Microsim constructor)
        msim_args["disease_params"] = disease_params  # R parameters kept as a dictionary and unpacked later

#     # Temporarily use dummy data for testing
#     # data_dir = os.path.join(base_dir, "dummy_data")
#     # m = Microsim(data_dir=data_dir, testing=True, output=output)

#     # cache to hold previously calculate population data
    cache = InitialisationCache(cache_dir=os.path.join(Constants.Paths.PROJECT_FOLDER_ABSOLUTE_PATH,
                                                       Constants.Paths.CACHE_FOLDER))

#     # generate new population dataframes if we aren't using the cache, or if the cache is empty
#     if not use_cache or cache.is_empty():
    if not use_cache or cache.is_empty():
        print(f'Reading population data because {"caching is disabled" if not use_cache else "the cache is empty"}')
        # args for population initialisation
        population = PopulationInitialisation(**population_args)
        individuals = population.individuals
        activity_locations = population.activity_locations

        # store in cache so we can load later
        cache.store_in_cache(individuals, activity_locations)
    else:  # load from cache
        print("Loading data from previous cache")
        individuals, activity_locations = cache.read_from_cache()

#     # Calculate the time-activity multiplier (this is for implementing lockdown)
    time_activity_multiplier = None
    if lockdown_file != "":
        print(f"Implementing a lockdown with time activities from {lockdown_file}")
        time_activity_multiplier: pd.DataFrame = \
            PopulationInitialisation.read_time_activity_multiplier(os.path.join(selected_region_folder_full_path,
                                                                                lockdown_file))

#     # Select which model implementation to run
    if open_cl_model:
        run_opencl_model(individuals,
                         activity_locations,
                         time_activity_multiplier,
                         iterations,
                         selected_region_folder_full_path,
                         Constants.Paths.PROJECT_FOLDER_ABSOLUTE_PATH,
                         opencl_gui,
                         opencl_gpu,
                         use_cache,
                         initialise,
                         calibration_params,
                         disease_params)
    else:
        # If -init flag set then don't run the model. Note for the opencl model this check needs to happen
        # after the snapshots have been created in run_opencl_model
        if initialise:
            print("Have finished initialising model. -init flag is set so not running it. Exiting")
            return
        run_python_model(individuals,
                         activity_locations,
                         time_activity_multiplier,
                         msim_args,
                         iterations,
                         repetitions,
                         parameters_file)


def run_opencl_model(individuals_df,
                     activity_locations,
                     time_activity_multiplier,
                     iterations,
                     regional_data_dir_full_path,
                     project_dir,
                     use_gui,
                     use_gpu,
                     use_cache,
                     initialise,
                     calibration_params,
                     disease_params):
    snapshot_cache_filepath = os.path.join(Constants.Paths.PROJECT_FOLDER_ABSOLUTE_PATH,
                                           Constants.Paths.CACHE_FOLDER,
                                           Constants.Paths.OPENCL.OPENCL_SNAPSHOTS_FOLDER,
                                           Constants.Paths.OPENCL.OPENCL_CACHE_FILE)
    # project_dir + "microsim/opencl/snapshots/cache.npz"  #TODO: THIS IS HARD-CODED!!

    # Choose whether to load snapshot file from cache, or create a snapshot from population data
    if not use_cache or not os.path.exists(snapshot_cache_filepath):
        print("\nGenerating Snapshot for OpenCL model")
        snapshot_converter = SnapshotConvertor(individuals_df,
                                               activity_locations,
                                               time_activity_multiplier,
                                               regional_data_dir_full_path)
        snapshot = snapshot_converter.generate_snapshot()
        snapshot.save(snapshot_cache_filepath)  # store snapshot in cache so we can load later
    else:  # load cached snapshot
        snapshot = Snapshot.load_full_snapshot(path=snapshot_cache_filepath)

    # set the random seed of the model
    snapshot.seed_prngs(42)

    # set params
    if calibration_params is not None and disease_params is not None:
        snapshot.update_params(create_params(calibration_params, disease_params))

        if disease_params["improve_health"]:
            print("Switching to healthier population")
            snapshot.switch_to_healthier_population()
    if initialise:
        print("Have finished initialising model. -init flag is set so not running it. Exiting")
        return

    run_mode = "GUI" if use_gui else "headless"
    print(f"\nRunning OpenCL model in {run_mode} mode")
    run_opencl(snapshot,
               regional_data_dir_full_path,
               iterations,
               use_gui,
               use_gpu,
               num_seed_days=disease_params["seed_days"],
               quiet=False)


def run_python_model(individuals_df, activity_locations_df, time_activity_multiplier, msim_args, iterations,
                     repetitions, parameters_file):
    print("\nRunning Python / R model")

    # Create a microsim object
    m = MicrosimModel(individuals_df, activity_locations_df, time_activity_multiplier, **msim_args)
    copyfile(parameters_file, os.path.join(m.SCEN_DIR, "parameters.yml"))# use: copyfile(microsim,destination)

    # Run the Python / R model
    if repetitions == 1:
        m.run(iterations, 0)
    elif repetitions >= 1:  # Run it multiple times on lots of cores
        try:
            with multiprocessing.Pool(processes=int(os.cpu_count())) as pool:
                # Copy the model instance so we don't have to re-read the data each time
                # (Use a generator so we don't need to store all the models in memory at once).
                models = (MicrosimModel._make_a_copy(m) for _ in range(repetitions))
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


def create_params(calibration_params, disease_params):
    current_risk_beta = disease_params["current_risk_beta"]

    # NB: OpenCL model incorporates the current risk beta by pre-multiplying the hazard multipliers with it
    location_hazard_multipliers = LocationHazardMultipliers(
        retail=calibration_params["hazard_location_multipliers"]["Retail"] * current_risk_beta,
        primary_school=calibration_params["hazard_location_multipliers"]["PrimarySchool"] * current_risk_beta,
        secondary_school=calibration_params["hazard_location_multipliers"]["SecondarySchool"] * current_risk_beta,
        home=calibration_params["hazard_location_multipliers"]["Home"] * current_risk_beta,
        work=calibration_params["hazard_location_multipliers"]["Work"] * current_risk_beta,
    )

    individual_hazard_multipliers = IndividualHazardMultipliers(
        presymptomatic=calibration_params["hazard_individual_multipliers"]["presymptomatic"],
        asymptomatic=calibration_params["hazard_individual_multipliers"]["asymptomatic"],
        symptomatic=calibration_params["hazard_individual_multipliers"]["symptomatic"]
    )

    obesity_multipliers = [disease_params["overweight"], disease_params["obesity_30"], disease_params["obesity_35"],
                           disease_params["obesity_40"]]

    return Params(
        location_hazard_multipliers=location_hazard_multipliers,
        individual_hazard_multipliers=individual_hazard_multipliers,
        obesity_multipliers=obesity_multipliers,
        cvd_multiplier=disease_params["cvd"],
        diabetes_multiplier=disease_params["diabetes"],
        bloodpressure_multiplier=disease_params["bloodpressure"],
    )


if __name__ == "__main__":
    main()
    print("End of program")

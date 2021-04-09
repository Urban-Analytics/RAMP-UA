#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core RAMP-UA model.

Created on Tue Apr 6

@author: AnnaZ
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

# from microsim.quant_api import QuantRampAPI
# from microsim.population_initialisation import PopulationInitialisation
from microsim.microsim_model import MicrosimModel  `# are we keeping the R/Python model or not?
from microsim.opencl.ramp.run import run_opencl
from microsim.opencl.ramp.snapshot_convertor import SnapshotConvertor
from microsim.opencl.ramp.snapshot import Snapshot
from microsim.opencl.ramp.params import Params, IndividualHazardMultipliers, LocationHazardMultipliers
# from microsim.initialisation_cache import InitialisationCache
from microsim.constants import Constants
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

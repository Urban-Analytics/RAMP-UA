#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core RAMP-UA model initialisation.
Created on Tue Apr 6 2021
@author: Anna on Nick's original version
"""
import sys

# sys.path.append("code")
import multiprocessing
import pandas as pd

pd.set_option('display.expand_frame_repr', False)  # Don't wrap lines when displaying DataFrames
import os
import click  # command-line interface
import pickle  # to save data
from yaml import load, SafeLoader  # pyyaml library for reading the parameters.yml file
from shutil import copyfile
# Packages from Hadrien's code
import requests
import os
import tarfile
import zipfile
import geopandas as gpd
import numpy as np
# Packages as in the original model code
from coding.constants import Constants
# from initialise.quant_api import QuantRampAPI
from coding.initialise.population_initialisation import PopulationInitialisation
from coding.initialise.initialisation_cache import InitialisationCache
from coding.initialise.raw_data_handler import RawDataHandler

# ********
# PROGRAM ENTRY POINT
# Uses 'click' library so that it can be run from the command line
# ********


"""
Actual initialisation process
"""


@click.command()
@click.option('-p',
              '--parameters-file',
              type=click.Path(exists=True),
              help="Parameters file to use to configure the model. This must be located in the working directory.")
# @click.option('-ocl', '--opencl/--no-opencl', default=True, help="Run OpenCL model (runs in headless mode by default")
def main(parameters_file):
    """
    Main function which runs the population initialisation
    """

    # If parameters file is missing, raise exception:
    # raise Exception("Missing parameters file, please check input")
    # If parameters file path doesn't exist, raise exception:
    # this is automatically checked already by the option click.path(exists=True) above
    # If parameters file is wrong type (not yml) raise exception:

    # First check that the parameters file has been assigned in input.
    print(f"--\nReading parameters file: {parameters_file}\n--")

    try:
        with open(parameters_file, 'r') as f:
            # print(f"Reading parameters file: {parameters_file}. ")
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
            study_area = sim_params["study-area"]
            # selected_region_folder_name = sim_params["selected-region-folder-name"]
            output = sim_params["output"]
            output_every_iteration = sim_params["output-every-iteration"]
            debug = sim_params["debug"]
            repetitions = sim_params["repetitions"]
            lockdown_file = sim_params["lockdown-file"]
            # quant_dir = sim_params["quant-dir"]
            use_cache = sim_params["use-cache"]
            # open_cl_model = sim_params["opencl-model"]
            # opencl_gui = sim_params["opencl-gui"]
            # opencl_gpu = sim_params["opencl-gpu"]
    except Exception as error:
        print('Error in parameters file format')
        raise error

    # Check the parameters are sensible
    if iterations < 1:
        raise ValueError("Iterations must be > 1. If you want to just initialise the model and then exit,"
                         "set initialise : true")
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

    # Check that working directory is as expected, ie 'project/data/raw_data'
    if not os.path.exists(Constants.Paths.RAW_DATA.FULL_PATH_FOLDER):
        raise Exception("Data folder structure not valid. Make sure you are running within correct working directory.")

    """
    Download and Unpack raw data files from the www
    """
    raw_data_handler = RawDataHandler()

    # selected_region_folder_full_path = # put here the TU file county name
    # common_data_dir_full_path = os.path.join(Constants.Paths.PROJECT_FOLDER_ABSOLUTE_PATH,
    #                                          Constants.Paths.DATA_FOLDER,
    #                                          Constants.Paths.COMMON_DATA_FOLDER)
    # if not os.path.exists(selected_region_folder_full_path):
    #     raise Exception("Regional data folder doesn't exist")

    # population_args = {"project_dir": project_dir, "regional_data_dir": regional_data_dir_full_path, "debug": debug}
    population_args = {"debug": debug, "raw_data_handler_param": raw_data_handler}

    # The disease parameters for the model were defined here, this part was removed as now it's done inside OpenCL

    #     # Temporarily use dummy data for testing
    #     # data_dir = os.path.join(base_dir, "dummy_data")
    #     # m = Microsim(data_dir=data_dir, testing=True, output=output)

    #     # cache to hold previously calculate population data
    study_area_folder_in_processed_data = os.path.join(Constants.Paths.PROCESSED_DATA.FULL_PATH_FOLDER,
                                                       study_area)  # this generates the folder name
    print(f"study area folder {study_area_folder_in_processed_data}")
    if not os.path.exists(study_area_folder_in_processed_data):
        os.makedirs(study_area_folder_in_processed_data)
    cache = InitialisationCache(cache_dir=study_area_folder_in_processed_data)

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

    ### SEPARATE HERE!!! ###

    else:  # load from cache
        # print("Loading data from previous cache")
        # individuals, activity_locations = cache.read_from_cache()
        print("***\n"
              "A cache of the processed data already exists for the area you selected, you can run the model module.\n"
              "***")


if __name__ == "__main__":
    main()
    print("End of program")

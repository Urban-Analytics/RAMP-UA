#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core RAMP-UA model.

Created on Wed Apr 29 19:59:25 2020

@author: nick
"""

import sys

sys.path.append("microsim")  # This is only needed when testing. I'm so confused about the imports
from activity_location import ActivityLocation
from r_interface import RInterface
from column_names import ColumnNames
from utilities import Optimise
import multiprocessing
import copy

import pandas as pd

pd.set_option('display.expand_frame_repr', False)  # Don't wrap lines when displaying DataFrames
# pd.set_option('display.width', 0)  # Automatically find the best width
import numpy as np
import glob
import os
import random
import time
import re  # For analysing file names
import warnings
from collections.abc import Iterable  # drop `.abc` with Python 2.7 or lower
from typing import List, Dict
from tqdm import tqdm  # For a progress bar
import click  # command-line interface
import pickle  # to save data
import swifter  # For speeding up apply functions (e.g. df.swifter.apply)
import rpy2.robjects as ro  # For calling R scripts
from yaml import load, dump, SafeLoader  # pyyaml library for reading the parameters.yml file


# import pandas.rpy.common as com



class Microsim:
    """
    A class used to represent the microsimulation model.

    TODO: Document class. Please use reStructuredText format:
    https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
    ...

    Attributes
    ----------
    XXXX

    Methods
    -------
    XXXX
    """

    def __init__(self,
                 data_dir: str = "./data/", r_script_dir: str = "./R/py_int/",
                 hazard_multiplier_presymptomatic: float = 1.0,
                 hazard_multiplier_asymptomatic: float = 1.0,
                 hazard_multiplier_symptomatic: float = 1.0,
                 risk_multiplier: float = 1.0,
                 lockdown_from_file: bool = True,
                 random_seed: float = None, read_data: bool = True,
                 testing: bool = False,
                 output: bool = True,
                 debug=False,
                 disable_disease_status=False,
                 disease_params=dict()
                 ):
        """
        Microsim constructor. This reads all of the necessary data to run the microsimulation.
        ----------
        :param data_dir: A data directory from which to read the source data
        :param r_script_dir: A directory with the required R scripts in (these are used to estimate disease status)
        :param random_seed: A optional random seed to use when creating the class instance. If None then
            the current time is used.
        :param read_data: Optionally don't read in the data when instantiating this Microsim (useful
            in debugging).
        :param testing: Optionally turn off some exceptions and replace them with warnings (only good when testing!)
        :param output: Whether to create files to store the results (default True)
        :param debug: Whether to do some more intense error checks (e.g. for data inconsistencies)
        :param disable_disease_status: Optionally turn off the R interface. This will mean we cannot calculate new
            disease status. Only good for testing.
        :param disease_params: Optional parameters that are passed to the R code that estimates disease status
            (a dictionary, assumed to be empty)
        """

        # Administrative variables that need to be defined
        Microsim.DATA_DIR = data_dir  # TODO (minor) pass the data_dir to class functions directly so no need to have it defined at class level
        self.DATA_DIR = data_dir
        self.iteration = 0
        self.hazard_multiplier_presymptomatic = hazard_multiplier_presymptomatic
        self.hazard_multiplier_asymptomatic = hazard_multiplier_asymptomatic
        self.hazard_multiplier_symptomatic = hazard_multiplier_symptomatic
        self.risk_multiplier = risk_multiplier
        self.lockdown_from_file = lockdown_from_file
        self.random = random.Random(random_seed)

        self.output = output
        self.r_script_dir = r_script_dir
        Microsim.debug = debug
        self.disable_disease_status = disable_disease_status
        self.disease_params = disease_params
        Microsim.testing = testing
        if self.testing:
            warnings.warn("Running in testing mode. Some exceptions will be disabled.")

        if not read_data:  # Optionally can not do this, usually for debugging
            return

        # We need an interface to R code to calculate disease status, but don't initialise it until the run()
        # method is called so that the R process is initiatied in the same process as the Microsim object
        self.r_int = None

        # If we do output information, then it will go to this directory. This is determined in run(), rather than
        # here, because otherwise all copies of this object will have the same output directory.
        self.output_dir = None

        # Now the main chunk of initialisation is to read the input data.

        # Begin by reading the individuals. This includes core information about the population as well as the
        # durations that people spend doing activities.
        # This also creates flows and venues columns for the journeys of individuals to households, and makes a new
        # households dataset to replace the one we read in above.
        home_name = "Home"  # How to describe flows to people's houses
        self.individuals, self.households = Microsim.read_individual_time_use_and_health_data(home_name)

        # Extract a list of all MSOAs in the study area. Will need this for the new SIMs
        self.all_msoas = Microsim.extract_msoas_from_indiviuals(self.individuals)


        #
        # ********** How to assign activities for the population **********
        #
        # For each 'activity' (e.g shopping), we need to store the following things:
        #
        # 1. A data frame of the places where the activities take place (e.g. a list of shops). Refered to as
        # the 'locations' dataframe. Importantly this will have a 'Danger' column which records whether infected
        # people have visited the location.
        #
        # 2. Columns in the individuals data frame that says which locations each individual is likely to do that
        # activity (a list of 'venues'), how likely they are to do to the activity (a list of 'flows'), and the
        # duration spent doing the activity.
        #
        # For most activities, there are a number of different possible locations that the individual
        # could visit. The 'venues' column stores a list of indexes to the locations dataframe. E.g. one individual
        # might have venues=[2,54,19]. Those numbers refer to the *row numbers* of locations in the locations
        # dataframe. So venue '2' is the third venue in the list of all the locaitons associated with that activity.
        # Flows are also a list, e.g. for that individual the flows might be flows=[0.8,0.1,0.1] which means they
        # are most likely to go to venue with index 2, and less likely to go to the other two.
        #
        # For some activities, e.g. 'being at home', each individual has only a single location and one flow, so their
        # venues and flows columns will only be single-element lists.
        #
        # For multi-venue activities, the process is as follows (easiest to see the retail or shopping example):
        # 1. Create a dataframe of individual locations and use a spatial interation model to estimate flows to those
        # locations, creating a flow matrix. E.g. the `read_retail_flows_data()` function
        # 2. Run through the flow matrix, assigning all individuals in each MSOA the appropriate flows. E.g. the
        # `add_individual_flows()` function.
        # 3. Create an `ActivityLocation` object to store information about these locations in a standard way.
        # When they are created, these `ActivityLocation` objects will also add another column
        # to the individuals dataframe that records the amount of time they spend doing the activity
        # (e.g. 'RETAIL_DURATION'). These raw numbers were attached earlier in `attach_time_use_and_health_data`.
        # (Again see the retail example below).
        # These ActivityLocation objects are stored in a dictionary (see `activity_locations` created above). This makes
        # it possible to run through all activities and calculate risks and dangers using the same code.
        #

        # For each type of activity (store, retail, etc), create ActivityLocation objects to keep all the
        # required information together.
        self.activity_locations: Dict[str, ActivityLocation] = {}

        # Create 'activity locations' for the activity of being at home. (This is done for other activities,
        # like retail etc, when those data are read in later.
        self.activity_locations[home_name] = ActivityLocation(name=home_name, locations=self.households,
                                                              flows=None, individuals=self.individuals,
                                                              duration_col="phome")

        # Generate travel time columns and assign travel modes to some kind of risky activity (not doing this yet)
        # self.individuals = Microsim.generate_travel_time_colums(self.individuals)
        # One thing we do need to do (this would be done in the function) is replace NaNs in the time use data with 0
        # for col in ["pwork", "_pschool", "pshop", "pleisure", "ptransport", "pother"]:
        for col in ["punknown", "phome", "pworkhome", "pwork", "_pschool", "pshop", "pservices", "pleisure",
                   "pescort", "ptransport", "pnothome", "phometot", "pmwalk", "pmcycle", "pmprivate",
                   "pmpublic", "pmunknown"]:
            self.individuals[col].fillna(0, inplace=True)


        # Read Retail flows data
        retail_name = "Retail"  # How to refer to this in data frame columns etc.
        stores, stores_flows = Microsim.read_retail_flows_data(self.all_msoas)  # (list of shops and a flow matrix)
        Microsim.check_sim_flows(stores, stores_flows)
        # Assign Retail flows data to the individuals
        self.individuals = Microsim.add_individual_flows(retail_name, self.individuals, stores_flows)
        self.activity_locations[retail_name] = \
            ActivityLocation(retail_name, stores, stores_flows, self.individuals, "pshop")

        # Read Schools (primary and secondary)
        primary_name = "PrimarySchool"
        secondary_name = "SecondarySchool"
        schools, primary_flows, secondary_flows = \
            Microsim.read_school_flows_data(self.all_msoas)  # (list of schools and a flow matrix)
        Microsim.check_sim_flows(schools, primary_flows)
        Microsim.check_sim_flows(schools, secondary_flows)
        # Assign Schools
        # TODO: need to separate primary and secondary school duration. At the moment everyone is given the same
        # duration, 'pschool', which means that children will be assigned a PrimarySchool duration *and* a
        # seconary school duration, regardless of their age. I think the only way round this is to
        # make two new columns - 'pschool_primary' and 'pschool_seconary', and set these to either 'pschool'
        # or 0 depending on the age of the child.
        self.individuals = Microsim.add_individual_flows(primary_name, self.individuals, primary_flows)
        self.activity_locations[primary_name] = \
            ActivityLocation(primary_name, schools.copy(), primary_flows, self.individuals, "pschool-primary")
        self.individuals = Microsim.add_individual_flows(secondary_name, self.individuals, secondary_flows)
        self.activity_locations[secondary_name] = \
            ActivityLocation(secondary_name, schools.copy(), secondary_flows, self.individuals, "pschool-secondary")
        del schools  # No longer needed as we gave copies to the ActivityLocation

        # Assign work. Each individual will go to a virtual office depending on their occupation (all accountants go
        # to the virtual accountant office etc). This means we don't have to calculate a flows matrix (similar to homes)
        # Occupation is taken from column soc2010 in individuals df
        self.individuals['soc2010'] = self.individuals['soc2010'].astype(str)  # These are integers but we need string
        possible_jobs = sorted(self.individuals.soc2010.unique())  # list of possible jobs in alphabetical order
        workplaces = pd.DataFrame({'ID': range(0, 0 + len(possible_jobs))})  # df with all possible 'virtual offices'
        Microsim._add_location_columns(workplaces, location_names=possible_jobs)
        work_name = "Work"
        self.individuals = Microsim.add_work_flows(work_name, self.individuals, workplaces)
        self.activity_locations[work_name] = ActivityLocation(name=work_name, locations=workplaces, flows=None,
                                                              individuals=self.individuals, duration_col="pwork")

        ## Some flows will be very complicated numbers. Reduce the numbers of decimal places across the board.
        ## This makes it easier to write out the files and to make sure that the proportions add up properly
        ## Use multiprocessing because swifter doesn't work properly for some reason (wont paralelise)
        #with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
        #   for name in tqdm(self.activity_locations.keys(), desc="Rounding all flows"):
        #       rounded_flows = pool.map( Microsim._round_flows, list(self.individuals[f"{name}{ColumnNames.ACTIVITY_FLOWS}"]))
        #       self.individuals[f"{name}{ColumnNames.ACTIVITY_FLOWS}"] = rounded_flows
        #   # Use swifter, but for some reason it wont paralelise the problem. Not sure why.
        #   #self.individuals[f"{name}{ColumnNames.ACTIVITY_FLOWS}"] = \
        #   #        self.individuals.loc[:,f"{name}{ColumnNames.ACTIVITY_FLOWS}"].\
        #   #            swifter.allow_dask_on_strings(enable=True).progress_bar(True, desc=name).\
        #   #            apply(lambda flows: [round(flow, 5) for flow in flows])

        # Round the durations
        for name in tqdm(self.activity_locations.keys(), desc="Rounding all durations"):
            self.individuals[f"{name}{ColumnNames.ACTIVITY_DURATION}"] = \
                self.individuals[f"{name}{ColumnNames.ACTIVITY_DURATION}"].apply(lambda x: round(x, 5))

        # Some people's activity durations will not add up to 1.0 because we don't model all their activities.
        # Extend the amount of time at home to make up for this
        self.individuals = Microsim.pad_durations(self.individuals, self.activity_locations)

        # Now that we have everone's initial activities, remember the proportions of times that they spend doing things
        # so that if these change (e.g. under lockdown) they can return to 'normality' later
        for activity_name in self.activity_locations.keys():
            self.individuals[f"{activity_name}{ColumnNames.ACTIVITY_DURATION_INITIAL}"] = \
                self.individuals[f"{activity_name}{ColumnNames.ACTIVITY_DURATION}"]

        # Add some necessary columns for the disease
        self.individuals = Microsim.add_disease_columns(self.individuals)

        # Read a file that tells us how much more time people should spend at home than normal (this is much greater
        # after lockdown
        self.time_activity_multiplier: pd.DataFrame = Microsim.read_time_activity_multiplier()

        print(" ... finished initialisation.")

        return  # finish __init__

    @staticmethod
    def _find_new_directory(dir):
        """
        Find a new directory and make one to store results in starting from 'dir'.
        :param dir: Start looking from this directory
        :return: The new directory (full path)
        """
        # Find a new directory for this initialisation (may have old ones)
        i = 0
        while os.path.exists(os.path.join(dir, str(i))):
            i += 1
        # Create a directory for these results
        results_subdir = os.path.join(dir, str(i))
        try:
            os.mkdir(results_subdir)
        except FileExistsError as e:
            print("Directory ", results_subdir, " already exists")
            raise e
        return results_subdir

    @staticmethod
    def _round_flows(flows):
        return [round(flow, 5) for flow in flows]

    @classmethod
    def _check_no_homeless(cls, individuals, households, warn=True):
        """
        Check that each individual has a household. NOTE: this only works for the raw mirosimulation data.
        Once the health data has been attached this wont work becuase the unique identifiers change.
        If this function is still needed then it will need to take the specific IDs as arguments, but this is
        a little complicated because some combination of [area, HID, (PID)] is needed for unique identification.

        :param individuals:
        :param households:
        :param warn: Whether to warn (default, True) or raise an exception (False)
        :return: True if there are no homeless, False otherwise (unless `warn==False` in which case an
        exception is raised).
        :raise: An exception if `warn==False` and there are individuals without a household
        """
        print("Checking no homeless (all individuals assigned to a household) ...", )
        # This will fail if used on anything other than the raw msm data because once I read in the
        # health data the PID and HID columns are renamed to prevent them being accidentally used.
        assert "PID" in individuals.columns and "HID" in households.columns
        # Households in the msm are uniquely identified by [area,HID] combination.
        # Individuals are identified by [House_OA,HID,PID]
        hids = households.set_index(["area", "HID"])  # Make a new dataset with a unique index for households
        # Find individuals who do not have a related entry in the households dataset
        homeless = [(area, hid, pid) for area, hid, pid in individuals.loc[:, ["House_OA", "HID", "PID"]].values if
                    (area, hid) not in hids.index]
        # (version using apply isn't quicker)
        # h2 = individuals.reset_index().loc[:, ["House_OA", "HID", "PID"]].swifter.apply(
        #    lambda x: x[2] if (x[0], x[1]) in hids.index else None, axis=1)
        # (Vectorised version doesn't quite work sadly)
        # h2 = np.where(individuals.loc[:, ["House_OA", "HID", "PID"]].isin(hids.index), True, False)
        if len(homeless) > 0:
            msg = f"There are {len(homeless)} individuals without an associated household (HID)."
            if warn:
                warnings.warn(msg)
                return False
            else:
                raise Exception(msg)
        print("... finished checking homeless")
        return True

    @classmethod
    def extract_msoas_from_indiviuals(cls, individuals: pd.DataFrame) -> List[str]:
        """
        Analyse a DataFrame of individuals and extract the unique MSOA codes, returning them as a list in ascending
        order
        :param individuals:
        :return:
        """
        areas = list(individuals.area.unique())
        areas.sort()
        return areas

    @classmethod
    def read_individual_time_use_and_health_data(cls, home_name: str) -> pd.DataFrame:
        """
        Read a population of individuals. Includes time-use & health info.

        :param home_name: A string to describe flows to people's homes (probably 'Home')
        :return A tuple with new dataframes of individuals and households
        """
        print("Reading time use and health data ... ", )
        # filename = os.path.join(cls.DATA_DIR, "devon-tu_health", "Devon_simulated_TU_health.txt")
        # filename = os.path.join(cls.DATA_DIR, "devon-tu_health", "Devon_keyworker.txt")
        # filename = os.path.join(cls.DATA_DIR, "devon-tu_health", "Devon_Complete.txt")
        filename = os.path.join(cls.DATA_DIR, "devon-tu_health", "Devon_simulated_TU_keyworker_health.csv")

        tuh = pd.read_csv(filename)
        tuh = Optimise.optimize(tuh)  # Reduce memory of tuh where possible.

        # Drop people that weren't matched to a household originally
        nohh = len(tuh.loc[tuh.hid == -1])
        if nohh > 0:
            warnings.warn(f"{nohh} / {len(tuh)} individuals in the TUH data had not originally been matched "
                          f"to a household. They're being removed")
        tuh = tuh.loc[tuh.hid != -1]

        # Indicate that HIDs and PIDs shouldn't be used as indices as they don't uniquely
        # identify indivuals / households in this health data
        tuh = tuh.rename(columns={'hid': '_hid', 'pid': '_pid'})

        # Make a new, unique id for each individual (PIDs have been replicated so no longer uniquely idenfity individuals}
        assert len(tuh.index.unique()) == len(tuh)  # Index should have been set to row number when tuh was read in
        tuh.insert(0, "ID", tuh.index, allow_duplicates=False)  # Insert into first position

        #
        # ********** Create households dataframe *************
        #

        # Go through each individual. House members can be identified because they have the same [Area, HID]
        # combination.
        # Maintain a dictionary of (Area, HID) -> House_ID that records a new ID for each house
        # Each time a new [Area, HID] combination is found, create a new entry in the households dictionary for that
        # household, generate a House_ID, and record that in the dictionary.
        # When existing (Area, HID) combinations are found, look up the ID in the dataframe and record it for that
        # individual
        # Also, maintain a list of house_ids in the same order as individuals in the tuh data which can be used later
        # when we link from the individuls in the TUH data to their house id

        # This is the main dictionary. It maps (Area, HID) to house id numbers, along with some more information:
        house_ids_dict = {}  # (Area, HID) -> [HouseIDNumber, NumPeople, area, hid]

        house_ids_list = []  # ID of each house for each individual
        house_id_counter = 0  # Counter to generate new HouseIDNumbers
        unique_individuals = []  # Also store all [Area, HID, PID] combinations to check they're are unique later

        # Maybe quicker to loop over 3 lists simultaneously than through a DataFrame
        _areas = list(tuh["area"])
        _hids = list(tuh["_hid"])
        _pids = list(tuh["_pid"])

        for i, (area, hid, pid) in enumerate(zip(_areas, _hids, _pids)):
            # print(i, area, hid, pid)
            unique_individuals.append((area, hid, pid))
            house_key = (area, hid)  # Uniqely identifies a household
            house_id_number = -1
            try:  # If this lookup works then we've seen this house before. Get it's ID number and increase num people in it
                house_info = house_ids_dict[house_key]
                # Check the area and hid are the same as the one previously stored in the dictionary
                assert area == house_info[2] and hid == house_info[3]
                # Also check that the house key (Area, HID) matches the area and HID
                assert house_key[0] == house_info[2] and house_key[1] == house_info[3]
                # We need the ID number to tell the individual which their house is
                house_id_number = house_info[0]
                # Increse the number of people in the house and create a new list of info for this house
                people_per_house = house_info[1] + 1
                house_ids_dict[house_key] = [house_id_number, people_per_house, area, hid]
            except KeyError:  # If the lookup failed then this is the first time we've seen this house. Make a new ID.
                house_id_number = house_id_counter
                house_ids_dict[house_key] = [house_id_number, 1, area,
                                             hid]  # (1 is beacuse 1 person so far in the hosue)
                house_id_counter += 1
            assert house_id_number > -1
            house_ids_list.append(house_id_number)  # Remember the house for this individual

        assert len(unique_individuals) == len(tuh)
        assert len(house_ids_list) == len(tuh)
        assert len(house_ids_dict) == house_id_counter

        # While we're here, may as well also check that [Area, HID, PID] is a unique identifier of individuals
        if len(tuh) != len(set(unique_individuals)):
            # TODO FIND OUT FROM KARYN WHY THERE ARE ~20,000 NON-UNIQUE PEOPLE
            warnings.warn(f"There are {len(tuh)-len(set(unique_individuals))} / {len(tuh)} non-unique individuals.")

        # Done! Now can create the households dataframe
        households_df = pd.DataFrame(house_ids_dict.values(), columns=['House_ID', 'Num_People', 'area', '_hid'])
        households_df = Optimise.optimize(households_df)

        # And tell the individuals which house they live in
        tuh["House_ID"] = house_ids_list  # Assign each individuals to their household

        # Check all house IDs are unique and have same number as in TUH data
        assert len(frozenset(households_df.House_ID.unique())) == len(households_df)
        assert len(tuh.area.unique()) == len(tuh.area.unique())
        # Check that the area that the individual lives in is the same as the area their house is in
        temp_merge = tuh.merge(households_df, how="left", on=["House_ID"], validate="many_to_one")
        assert len(temp_merge) == len(tuh)
        assert (temp_merge['area_x'] == temp_merge['area_y']).all()  # (all says 'all are true')

        # Check that NumPople in the house dataframe is the same as number of people in the indivdiuals dataframe
        # with this house id
        if Microsim.debug:
            for house_id, num_people in tqdm(zip(households_df.House_ID, households_df.Num_People),
                                             desc="Checking household sizes match"):  # I know you shouldn't loop, but I can't work out the apply way (and this only happens once)
                num_people2 = len(tuh.loc[tuh.House_ID == house_id])  # Number of individuals who link to this house
                assert num_people == num_people2, f"House {house_id} doesn't match: {num_people} / {num_people2}"

        # Add some required columns
        Microsim._add_location_columns(households_df, location_names=list(households_df.House_ID),
                                       location_ids=households_df.House_ID)
        # The new ID column should be the same as the House_ID
        assert (households_df.House_ID == households_df[ColumnNames.LOCATION_ID]).all()

        # Later we need time spent in primary and secondary school. But currently we just have 'pschool'. Make
        # two new columns separating out primary and secondary based on age
        tuh["pschool"] = tuh["pschool"].fillna(0)
        tuh["pschool-primary"] = 0.0
        tuh["pschool-secondary"] = 0.0
        children_idx = tuh.index[tuh["age"] < 11]
        teen_idx = tuh.index[(tuh["age"] >= 11) & (tuh["age"] < 19)]

        assert len(children_idx) > 0
        assert len(teen_idx) > 0

        tuh.loc[children_idx, "pschool-primary"] = tuh.loc[children_idx, "pschool"]
        tuh.loc[teen_idx, "pschool-secondary"] = tuh.loc[teen_idx, "pschool"]

        # Check that people have been allocated correctly
        adults_in_school = tuh.loc[~(tuh["pschool-primary"] + tuh["pschool-secondary"] == tuh["pschool"]),
                                   ["age", "pschool", "pschool-primary", "pschool-secondary"]]
        if len(adults_in_school) > 0:
            warnings.warn(f"{len(adults_in_school)} people > 18y/o go to school, but they are not being assigned to a "
                          f"primary or secondary school (so their schooling is ignored at the moment).")

        tuh = tuh.rename(columns={"pschool": "_pschool"})  # Indicate that the pschool column shouldn't be used now

        # For some reason, we get some *very* large households. Can demonstrate this with:
        # households_df.Num_People.hist(bins=10000)
        # This needs to be resolved, but in the meantime just remove all households that have more than 10 people
        large_house_idx = frozenset(households_df.index[households_df.Num_People > 10])  # Indexes of large houses
        # For each person, get a house_id, or -1 if the house is very large
        large_people_idx = tuh["House_ID"].apply(lambda x: -1 if x in large_house_idx else x)
        if len(large_house_idx) > 0:
            warnings.warn(f"There are {len(large_house_idx)} households with more than 10 people in them. This covers "
                      f"{len(large_people_idx[large_people_idx == -1])} people. These households are being removed.")
        tuh["TEMP_HOUSE_ID"] = large_people_idx  # Use this colum to remove people (all people with HOUSE_ID == -1)
        # Check the numbers add up (normal house len + large house len = original len)
        assert (len(tuh.loc[tuh.TEMP_HOUSE_ID != -1]) + len(large_people_idx[large_people_idx == -1])) == len(tuh)
        assert (len(households_df.loc[~households_df.House_ID.isin(large_house_idx)]) + len(large_house_idx)) == len(
            households_df)
        # Remove people, but leave the households (no one will live there so they wont affect anything)
        tuh = tuh[tuh.TEMP_HOUSE_ID != -1]
        # TODO Work out why removing households kills the model later. - it's probably because houses are removed but the indexes and IDs don't change, so indexes will end up larger than size of the households list. Probably would need to recalculate the index and House_ID so that they are ascending again (pain, can't be bothered).
        # households_df = households_df.loc[~households_df.House_ID.isin(large_house_idx)]
        # households_df = households_df.loc[households_df.Num_People <= 10]
        # Check that the large house ids no longer exist in the individuals df (use House_ID rather than index to be sure, but they're the same anyway)
        id_set = frozenset(households_df.loc[households_df.Num_People > 10, "House_ID"].values)
        assert True not in list(tuh["House_ID"].apply(lambda x: x in id_set))
        del tuh["TEMP_HOUSE_ID"]

        # Add flows for each individual (this is easy, it's just converting their House_ID and flow (1.0) into a
        # one-value lists).
        venues_col = f"{home_name}{ColumnNames.ACTIVITY_VENUES}"  # Names for the new columns
        flows_col = f"{home_name}{ColumnNames.ACTIVITY_FLOWS}"
        tuh[venues_col] = tuh["House_ID"].apply(lambda x: [x])
        tuh[flows_col] = [[1.0]] * len(tuh)

        # Later we also record the individual risks for each activity per individual. It's nice if the columns for
        # each activity are grouped together, so create that column now.
        tuh[f"{home_name}{ColumnNames.ACTIVITY_RISK}"] = [-1] * len(tuh)

        print(f"... finished reading TU&H data. There are {len(tuh)} individuals in {len(households_df)} houses "
              f"over {len(tuh.area.unique())} MSOAs")

        return tuh, households_df

    @classmethod
    def generate_travel_time_colums(cls, individuals: pd.DataFrame) -> pd.DataFrame:
        """
        TODO Read the raw travel time columns and create standard ones to show how long individuals
        spend travelling on different modes. Ultimately these will be turned into activities
        :param individuals:
        :return:
        """

        # Some sanity checks for the time use data
        # Variables pnothome, phome add up to 100% of the day and
        # pwork +pschool +pshop+ pleisure +pescort+ ptransport +pother = phome

        # TODO go through some of these with Karyn, they don't all pass
        # Time at home and not home should sum to 1.0
        if False in list((individuals.phome + individuals.pnothome) == 1.0):
            raise Exception("Time at home (phome) + time not at home (pnothome) does not always equal 1.0")
        # These columns should equal time not at home
        # if False in list(tuh.loc[:, ["pwork", "pschool", "pshop", "pleisure",  "ptransport", "pother"]]. \
        #                         sum(axis=1, skipna=True) == tuh.pnothome):
        #    raise Exception("Times doing activities don't add up correctly")

        # Temporarily (?) remove NAs from activity columns (I couldn't work out how to do this in 1 line like:
        for col in ["pwork", "pschool", "pshop", "pleisure", "ptransport", "pother"]:
            individuals[col].fillna(0, inplace=True)

        # TODO assign activities properly. Need to map from columns in the dataframe to standard names
        # Assign time use for Travel (just do this arbitrarily for now, the correct columns aren't in the data).
        # travel_cols = [ x + ColumnNames.ACTIVITY_DURATION for x in
        #                 [ ColumnNames.TRAVEL_CAR, ColumnNames.TRAVEL_BUS, ColumnNames.TRAVEL_TRAIN, ColumnNames.TRAVEL_WALK ] ]
        # for col in travel_cols:
        #    tuh[col] = 0.0
        # OLD WAY OF HARD-CODING TIME USE CATEGORIES FOR EACH INDIVIDUAL
        # For now just hard code broad categories. Ultimately will have different values for different activities.
        # activities = ["Home", "Retail", "PrimarySchool", "SecondarySchool", "Work", "Leisure"]
        # col_names = []
        # for act in activities:
        #    col_name = act + ColumnNames.ACTIVITY_DURATION
        #    col_names.append(col_name)
        #    if act=="Home":
        #        # Assume XX hours per day at home (this is whatever not spent doing other activities)
        #        individuals[col_name] = 14/24
        #    elif act == "Retail":
        #        individuals[col_name] = 1.0/24
        #    elif act == "PrimarySchool":
        #        # Assume 8 hours per day for all under 12
        #        individuals[col_name] = 0.0 # Default 0
        #        individuals.loc[individuals[ColumnNames.INDIVIDUAL_AGE] < 12, col_name] = 8.0/24
        #    elif act == "SecondarySchool":
        #        # Assume 8 hours per day for 12 <= x < 19
        #        individuals[col_name] = 0.0  # Default 0
        #        individuals.loc[individuals[ColumnNames.INDIVIDUAL_AGE] < 19, col_name] = 8.0 / 24
        #        individuals.loc[individuals[ColumnNames.INDIVIDUAL_AGE] < 12, col_name] = 0.0
        #    elif act == "Work":
        #        # Opposite of school
        ##        individuals[col_name] = 0.0 # Default 0
        #        individuals.loc[individuals[ColumnNames.INDIVIDUAL_AGE] >= 19, col_name] = 8.0/24
        #    elif act == "Leisure":
        #        individuals[col_name] = 1.0/24
        #    else:
        #        raise Exception(f"Unrecognised activity: {act}")

        # Check that proportions add up to 1.0
        # For some reason this fails, but as far as I can see the proportions correctly sum to 1 !!
        # assert False not in (individuals.loc[:, col_names].sum(axis=1).round(decimals=4) == 1.0)

        ## Add travel data columns (no values yet)
        # travel_cols = [ x + ColumnNames.ACTIVITY_DURATION for x in ["Car", "Bus", "Walk", "Train"] ]
        # for col in travel_cols:
        #    individuals[col] = 0.0
        return individuals

    @classmethod
    def read_school_flows_data(cls, study_msoas: List[str]) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Read the flows between each MSOA and the most likely schools attended by pupils in this area.
        All schools are initially read together, but flows are separated into primary and secondary

        :param study_msoas: A list of MSOAs in the study area (flows outside of this will be ignored)
        :return: A tuple of three dataframes. All schools, then the flows to primary and secondary
        (Schools, PrimaryFlows, SeconaryFlows). Although all the schools are one dataframe, no primary flows will flow
        to secondary schools and vice versa).
        """
        # TODO Need to read full school flows, not just those of Devon
        print("Reading school flow data for Devon...", )
        dir = os.path.join(cls.DATA_DIR, "devon-schools")

        # Read the schools (all of them)
        schools = pd.read_csv(os.path.join(dir, "exeter schools.csv"))
        # Add some standard columns that all locations need
        schools_ids = list(schools.index + 1)  # Mark counts from 1, not zero, so indices need to start from 1 not 0
        schools_names = schools.EstablishmentName  # Standard name for the location
        Microsim._add_location_columns(schools, location_names=schools_names, location_ids=schools_ids)

        # Read the flows
        primary_rows = []  # Build up all the rows in the matrix gradually then add all at once
        secondary_rows = []
        with open(os.path.join(dir, "DJS002.TXT")) as f:
            # Mark's file comes in batches of 3 lines, each giving different data. However, some lines overrun and are
            # read as several lines rather than 1 (hence use of dests_tmp and flows_tmp)
            count = 1
            oa = None
            oa_name = ""
            num_dests = None
            dests = None
            flows = None
            dests_tmp = None
            flows_tmp = None

            for lineno, raw_line in enumerate(f):
                # print(f"{lineno}: '{raw_line}'")
                line_list = raw_line.strip().split()
                if count == 1:  # primary/secondary school, OA and number of schools
                    sch_type = int(line_list[0])
                    assert sch_type == 1 or sch_type == 2  # Primary schools are 1, secondary 2
                    oa = int(line_list[1])
                    oa_name = study_msoas[oa - 1]  # The OA names are stored in a separate file temporarily
                    num_dests = int(line_list[2])
                elif count == 2:  # school ids
                    dests_tmp = [int(x) for x in line_list[0:]]  # Make the destinations numbers
                    # check if dests exists from previous iteration and add dests_tmp
                    if dests == None:
                        dests = dests_tmp
                    else:
                        dests.extend(dests_tmp)
                    if len(dests) < num_dests:  # need to read next line
                        count = 1  # counteracts count being increased by 1 later
                    else:
                        assert len(dests) == num_dests
                elif count == 3:  # Flows per 1,000 pupils
                    flows_tmp = [float(x) for x in line_list[0:]]  # Make the destinations numbers
                    # check if dests exists from previous iteration and add dests_tmp
                    if flows == None:
                        flows = flows_tmp
                    else:
                        flows.extend(flows_tmp)
                    if len(flows) < num_dests:  # need to read next line
                        count = 2  # counteracts count being increased by 1 later
                    else:
                        assert len(flows) == num_dests

                        # Have read all information for this area. Store the info in the flows matrix

                        # We should have one line in the matrix for each OA, and OA codes are incremental
                        # assert len(flow_matrix) == oa - 1
                        row = [0.0 for _ in range(len(schools))]  # Initially assume all flows are 0
                        for i in range(num_dests):  # Now add the flows
                            dest = dests[i]
                            flow = flows[i]
                            row[dest - 1] = flow  # (-1 because destinations are numbered from 1, not 0)
                        assert len([x for x in row if x > 0]) == num_dests  # There should only be N >0 flows
                        row = [oa, oa_name] + row  # Insert the OA number and code (don't know this yet now)

                        # Luckily Mark's file does all primary schools first, then all secondary schools, so we
                        # know that all schools in this area are one or the other
                        if sch_type == 1:
                            primary_rows.append(row)
                        else:
                            secondary_rows.append(row)
                        # rows.append(row)

                        # Add the row to the matrix. As the OA numbers are incremental they should equal the number
                        # of rows
                        # flow_matrix.loc[oa-1] = row
                        # assert len(flow_matrix) == oa
                        count = 0
                        # reset dests and flows
                        dests = None
                        flows = None

                count += 1

        # Have finished reading the file, now create the matrices. MSOAs as rows, school locations as columns
        columns = ["Area_ID", "Area_Code"]  # A number (ID) and full code for each MSOA
        columns += [f"Loc_{i}" for i in schools.index]  # Columns for each school

        primary_flow_matrix = pd.DataFrame(data=primary_rows, columns=columns)
        secondary_flow_matrix = pd.DataFrame(data=secondary_rows, columns=columns)
        # schools_flows = schools_flows.iloc[0:len(self.study_msoas), :]
        print(f"... finished reading school flows.")

        return schools, primary_flow_matrix, secondary_flow_matrix

    @classmethod
    def add_work_flows(cls, flow_type: str, individuals: pd.DataFrame, workplaces: pd.DataFrame) \
            -> (pd.DataFrame):
        """
        Create a dataframe of (virtual) work locations that individuals
        travel to. Unlike retail etc, each individual will have only one work location with 100% of flows there.
        :param flow_type: The name for these flows (probably something like 'Work')
        :param individuals: The dataframe of synthetic individuals
        :param workplaces:  The dataframe of workplaces (i.e. occupations)
        :return: The new 'individuals' dataframe (with new columns)
        """
        # Later on, add a check to see if occupations in individuals df are same as those in workplaces df??
        # Tell the individuals about which virtual workplace they go to
        venues_col = f"{flow_type}{ColumnNames.ACTIVITY_VENUES}"
        flows_col = f"{flow_type}{ColumnNames.ACTIVITY_FLOWS}"

        # Lists showing where individuals go, and what proption (here only 1 flow as only 1 workplace)
        # Need to do the flows in venues in 2 stages: first just add the venue, then turn that venu into a single-element
        # list (pandas complains about 'TypeError: unhashable type: 'list'' if you try to make the single-item lists
        # directly in the apply
        with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
            print("Assigning work venues ... ",)
            venues = pool.starmap(Microsim._assign_work_flow,
                                  zip(list(individuals["soc2010"]), (workplaces for _ in range(len(individuals))) ) )
        #venues = list(individuals["soc2010"].swifter.progress_bar(True, desc="Assigning work venues").apply(
        #    lambda job: workplaces.index[workplaces[ColumnNames.LOCATION_NAME] == job].values[0]))
        venues = [[x] for x in venues]
        individuals[venues_col] = venues
        individuals[flows_col] = [[1.0] for _ in range(len(individuals))]  # Flows are easy, [1.0] to the single venue
        # Later we also record the individual risks for each activity per individual. It's nice if the columns for
        # each activity are grouped together, so create that column now.
        individuals[f"{flow_type}{ColumnNames.ACTIVITY_RISK}"] = [-1] * len(individuals)
        return individuals

    @staticmethod
    def _assign_work_flow(job, workplaces):
        return workplaces.index[workplaces[ColumnNames.LOCATION_NAME] == job].values[0]

    @classmethod
    def read_retail_flows_data(cls, study_msoas: List[str]) -> (pd.DataFrame, pd.DataFrame):
        """
        Read the flows between each MSOA and the most commonly visited shops

        :param study_msoas: A list of MSOAs in the study area (flows outside of this will be ignored)
        :return: A tuple of two dataframes. One containing all of the flows and another
        containing information about the stores themselves.
        """
        # TODO Need to read full retail flows, not just those of Devon (temporarily created by Mark).
        # Will also need to subset the flows into areas of interst, but at the moment assume that we area already
        # working with Devon subset of flows
        print("Reading retail flow data for Devon...", )
        dir = os.path.join(cls.DATA_DIR, "devon-retail")

        # Read the stores
        stores = pd.read_csv(os.path.join(dir, "devon smkt.csv"))
        # Add some standard columns that all locations need
        stores_ids = list(stores.index + 1)  # Mark counts from 1, not zero, so indices need to start from 1 not 0
        store_names = stores.store_name  # Standard name for the location
        Microsim._add_location_columns(stores, location_names=store_names, location_ids=stores_ids)

        # Read the flows
        rows = []  # Build up all the rows in the matrix gradually then add all at once
        total_flows = 0  # For info & checking
        with open(os.path.join(dir, "DJR002.TXT")) as f:
            count = 1  # Mark's file comes in batches of 3 lines, each giving different data

            # See the README for info about these variables. This is only tempoarary so I can't be bothered
            # to explain properly
            oa = None
            oa_name = ""
            num_dests = None
            dests = None
            flows = None

            for lineno, raw_line in enumerate(f):
                # print(f"{lineno}: '{raw_line}'")
                line_list = raw_line.strip().split()
                if count == 1:  # OA and number of destinations
                    oa = int(line_list[0])
                    if oa > len(study_msoas):
                        msg = f"Attempting to read more output areas ({oa}) than are present in the study area {study_msoas}."
                        if cls.testing:
                            warnings.warn(msg)
                        else:
                            raise Exception(msg)
                    oa_name = study_msoas[oa - 1]  # The OA names are stored in a separate file temporarily
                    num_dests = int(line_list[1])
                elif count == 2:  # Top N (=10) destinations in the OA
                    # First number is the OA (don't need this), rest are the destinations
                    assert int(line_list[0]) == oa
                    dests = [int(x) for x in line_list[1:]]  # Make the destinations numbers
                    assert len(dests) == num_dests
                elif count == 3:  # Distance to each store (not currently used)
                    pass
                elif count == 4:  # Flows per 1,000 trips
                    # First number is the OA (don't need this), rest are the destinations
                    assert int(line_list[0]) == oa
                    flows = [float(x) for x in line_list[1:]]  # Make the destinations numbers
                    assert len(flows) == num_dests
                    total_flows += sum(flows)

                    # Have read all information for this area. Store the info in the flows matrix

                    # We should have one line in the matrix for each OA, and OA codes are incremental
                    # assert len(flow_matrix) == oa - 1
                    row = [0.0 for _ in range(len(stores))]  # Initially assume all flows are 0
                    for i in range(num_dests):  # Now add the flows
                        dest = dests[i]
                        flow = flows[i]
                        row[dest - 1] = flow  # (-1 because destinations are numbered from 1, not 0)
                    assert len([x for x in row if x > 0]) == num_dests  # There should only be positive flows (no 0s)
                    row = [oa, oa_name] + row  # Insert the OA number and code (don't know this yet now)

                    rows.append(row)

                    # Add the row to the matrix. As the OA numbers are incremental they should equal the number
                    # of rows
                    # flow_matrix.loc[oa-1] = row
                    # assert len(flow_matrix) == oa
                    count = 0

                count += 1

        # Have finished reading the file, now create the matrix. MSOAs as rows, retail locations as columns
        columns = ["Area_ID", "Area_Code"]  # A number (ID) and full code for each MSOA
        columns += [f"Loc_{i}" for i in stores.index]  # Columns for each store
        flow_matrix = pd.DataFrame(data=rows, columns=columns)

        # Check that we haven't lost any flows (need two sums, once to get the flows for each row, then
        # to add up all rows
        total_flows2 = flow_matrix.iloc[:, 2:].apply(lambda row: sum(row)).sum()
        assert total_flows == total_flows2

        print(f"... read {total_flows} flows from {len(flow_matrix)} areas.")

        return stores, flow_matrix

    @classmethod
    def check_sim_flows(cls, locations: pd.DataFrame, flows: pd.DataFrame):
        """
        Check that the flow matrix looks OK, raising an error if not
        :param locations: A DataFrame with information about each location (destination)
        :param flows: The flow matrix itself, showing flows from origin MSOAs to destinations
        :return:
        """
        # TODO All MSOA codes are unique
        # TODO Locations have 'Danger' and 'ID' columns
        # TODO Number of destination columns ('Loc_*') matches number of locaitons
        # TODO Number of origins (rows) in the flow matrix matches number of OAs in the locations
        return

    @classmethod
    def _add_location_columns(cls, locations: pd.DataFrame, location_names: List[str], location_ids: List[int] = None):
        """
        Add some standard columns to DataFrame (in place) that contains information about locations.
        :param locations: The dataframe of locations that the columns will be added to
        :param location_names: Names of the locations (e.g shop names)
        :param location_ids: Can optionally include a list of IDs. An 'ID' column is always created, but if no specific
        IDs are provided then the ID will be the same as the index (i.e. the row number). If ids are provided then
        the ID column will be set to the given IDs, but the index will still be the row number.
        :return: None; the columns are added to the input dataframe inplace.
        """
        # Make sure the index will always be the row number
        locations.reset_index(inplace=True, drop=True)
        if location_ids is None:
            # No specific index provided, just use the index
            locations[ColumnNames.LOCATION_ID] = locations.index
        else:
            # User has provided a specific list of indices to use
            if len(location_ids) != len(locations):
                raise Exception(f"When adding the standard columns to a locations dataframe, a list of specific",
                                f"IDs has ben passed, but this list (length {len(location_ids)}) is not the same"
                                f"length as the locations dataframe (length {len(locations)}. The list of ids passed"
                                f"is: {location_ids}.")
            locations[ColumnNames.LOCATION_ID] = location_ids
        if len(location_names) != len(locations):
            raise Exception(f"The list of location names is not the same as the number of locations in the dataframe",
                            f"({len(location_names)} != {len(locations)}.")
        locations[ColumnNames.LOCATION_NAME] = location_names  # Standard name for the location
        locations[ColumnNames.LOCATION_DANGER] = 0  # All locations have a disease danger of 0 initially
        # locations.set_index(ColumnNames.LOCATION_ID, inplace=True, drop=False)
        return None  # Columns added in place so nothing to return

    @classmethod
    def add_individual_flows(cls, flow_type: str, individuals: pd.DataFrame, flow_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Take a flow matrix from MSOAs to (e.g. retail) locations and assign flows to individuals.

        It a assigns the id of the destination of the flow according to its column in the matrix. So the first column
        that has flows for a destination is given index 0, the second is index 1, etc. This is probably not the same as
        the ID of the venue that they point to (e.g. the first store probably has ID 1, but will be given the index 0)
        so it is important that when the activity_locations are created, they are created in the same order as the
        columns that appear in the matix. The first column in the matrix must also be the first row in the locations
        data.
        :param flow_type: What type of flows are these. This will be appended to the column names. E.g. "Retail".
        :param individuals: The DataFrame contining information about all individuals
        :param flow_matrix: The flow matrix, created by (e.g.) read_retail_flows_data()
        :return: The DataFrame of individuals with new locations and probabilities added
        """

        # Check that there aren't any individuals who wont be given any flows
        if len(individuals.loc[-individuals.area.isin(flow_matrix.Area_Code)]) > 0:
            raise Exception(f"Some individuals will not be assigned any flows to: '{flow_type}' because their"
                            f"MSOA is not in the flow matrix: "
                            f"{individuals.loc[-individuals.area.isin(flow_matrix.Area_Code)]}.")

        # Check that there aren't any duplicate flows
        if len(flow_matrix) != len(flow_matrix.Area_Code.unique()):
            raise Exception("There are duplicate area codes in the flow matrix: ", flow_matrix.Area_Code)

        # Names for the new columns
        venues_col = f"{flow_type}{ColumnNames.ACTIVITY_VENUES}"
        flows_col = f"{flow_type}{ColumnNames.ACTIVITY_FLOWS}"

        # Create empty lists to hold the vanues and flows for each individuals
        individuals[venues_col] = [[] for _ in range(len(individuals))]
        individuals[flows_col] = [[] for _ in range(len(individuals))]

        # Later we also record the individual risks for each activity per individual. It's nice if the columns for
        # each activity are grouped together, so create that column now.
        individuals[f"{flow_type}{ColumnNames.ACTIVITY_RISK}"] = [-1] * len(individuals)

        # Use a hierarchical index on the Area to speed up finding all individuals in an area
        # (not sure this makes much difference).
        individuals.set_index(["area", "ID"], inplace=True, drop=False)

        for area in tqdm(flow_matrix.values,
                         desc=f"Assigning individual flows for {flow_type}"):  # Easier to operate over a 2D matrix rather than a dataframe
            oa_num: int = area[0]
            oa_code: str = area[1]
            # Get rid of the area codes, so are now just left with flows to locations
            area = list(area[2:])
            # Destinations with positive flows and the flows themselves
            dests = []
            flows = []
            for i, flow in enumerate(area):
                if flow > 0.0:
                    dests.append(i)
                    flows.append(flow)

            # Normalise the flows
            flows = Microsim._normalise(flows)

            # Now assign individuals in those areas to those flows
            # This ridiculous 'apply' line is the only way I could get pandas to update the particular
            # rows required. Something like 'individuals.loc[ ...] = dests' (see below) didn't work becuase
            # instead of inserting the 'dests' list itself, pandas tried to unpack the list and insert
            # the individual values instead.
            # individuals.loc[individuals.area == oa_code, f"{flow_type}_Venues"] = dests
            # individuals.loc[individuals.area == oa_code, f"{flow_type}_Probabilities"] = flow
            #
            # A quicker way to do this is probably to create N subsets of individuals (one table for
            # each area) and then concatenate them at the end.
            individuals.loc[oa_code, venues_col] = \
                individuals.loc[oa_code, venues_col].apply(lambda _: dests).values
            individuals.loc[oa_code, flows_col] = \
                individuals.loc[oa_code, flows_col].apply(lambda _: flows).values
            # individuals.loc[individuals.area=="E02004189", f"{flow_type}_Venues"] = \
            #    individuals.loc[individuals.area=="E02004189", f"{flow_type}_Venues"].apply(lambda _: dests)
            # individuals.loc[individuals.area=="E02004189", f"{flow_type}_Probabilities"] = \
            #    individuals.loc[individuals.area=="E02004189", f"{flow_type}_Probabilities"].apply(lambda _: flows)

        # Reset the index so that it's not the PID
        individuals.reset_index(inplace=True, drop=True)

        # Check everyone has some flows (all list lengths are >0)
        assert False not in (individuals.loc[:, venues_col].apply(lambda cell: len(cell)) > 0).values
        assert False not in (individuals.loc[:, flows_col].apply(lambda cell: len(cell)) > 0).values

        return individuals

    @classmethod
    def pad_durations(cls, individuals, activity_locations) -> pd.DataFrame:
        """
        Some indvidiuals' activity durations don't add up to 1. In these cases pad them out with extra time at home.
        :param individuals:
        :param activity_locations:
        :return: The new individuals dataframe
        """
        total_duration = [0.0] * len(individuals)  # Add up all the different activity durations
        for activity in activity_locations.keys():
            total_duration = total_duration + individuals.loc[:, f"{activity}{ColumnNames.ACTIVITY_DURATION}"]
        total_duration = total_duration.apply(lambda x: round(x, 5))
        assert (total_duration <= 1.0).all()  # None should be more than 1.0 (after rounding)

        missing_duration = 1.0 - total_duration  # Amount of activity time that needs to be added on to home
        #missing_duration = missing_duration.apply(lambda x: round(x,5))
        individuals[f"Home{ColumnNames.ACTIVITY_DURATION}"] = \
            (individuals[f"Home{ColumnNames.ACTIVITY_DURATION}"] + missing_duration).apply(lambda x: round(x, 5))

        Microsim.check_durations_sum_to_1(individuals, activity_locations.keys())

        return individuals

    @classmethod
    def check_durations_sum_to_1(cls, individuals, activities):
        total_duration = [0.0] * len(individuals)  # Add up all the different activity durations
        for activity in activities:
            total_duration = total_duration + individuals.loc[:, f"{activity}{ColumnNames.ACTIVITY_DURATION}"]
        if not (total_duration.apply(lambda x: round(x, 5)) == 1.0).all():
            print("Some activity durations don't sum to 1", flush=True)
            print(total_duration[total_duration!=1.0], flush=True)
            raise Exception("Some activity durations don't sum to 1")


    @classmethod
    def read_time_activity_multiplier(cls) -> pd.DataFrame:
        """
        Some times people should spend more time at home than normal. E.g. after lockdown. This function
        reads a file that tells us how much more time should be spent at home on each day.
        :return: A dataframe with 'day' and 'timeout_multiplier' columns
        """
        print("Reading time activity multiplier data...", )
        time_activity = pd.read_csv(os.path.join(cls.DATA_DIR, "google_mobility_lockdown_daily.csv"))
        # Cap at 1.0 (it's a curve so some times peaks above 1.0)=
        time_activity["timeout_multiplier"] = time_activity.loc[:,"timeout_multiplier"].\
            apply(lambda x: 1.0 if x > 1.0 else x)

        return time_activity

    @classmethod
    def _normalise(cls, l: List[float], decimals=3) -> List[float]:
        """
        Normalise a list so that it sums to almost 1.0. Rounding might cause it not to add exactly to 1

        :param decimals: Optionally round to the number of decimal places. Default 3. If 'None' the do no rounding.
        """
        if not isinstance(l, Iterable):
            raise Exception("Can only work with iterables")
        if len(l) == 1:  # Special case for 1-item iterables
            return [1.0]

        l = np.array(l)  # Easier to work with numpy vectorised operators
        total = l.sum()
        l = l / total
        if decimals is None:
            return list(l)
        return [round(x, decimals) for x in l]

    def _init_output(self):
        """
        Might need to write out some data if saving output for analysis later. If so, creates a new directory for the
        results as a subdirectory of the data directory.
        Also store some information for use in the visualisations and analysis.
        """
        if self.output:
            print("Saving initial models for analysis ... ", )
            # Find a directory to use, within the 'outut' directory
            self.output_dir = Microsim._find_new_directory(os.path.join(self.DATA_DIR, "output"))

            # save initial model
            pickle_out = open(os.path.join(self.output_dir, "m0.pickle"), "wb")
            pickle.dump(self, pickle_out)
            pickle_out.close()

            # collect disease status in new df (for analysis/visualisation)
            self.individuals_to_pickle = self.individuals.copy()
            self.individuals_to_pickle[ColumnNames.DISEASE_STATUS + "000"] = self.individuals_to_pickle[
                ColumnNames.DISEASE_STATUS]

            # collect location dangers at time 0 in new df(for analysis/visualisation)
            self.activities_to_pickle = {}
            for name in self.activity_locations:
                # Get the details of the location activity
                activity = self.activity_locations[name]  # Pointer to the ActivityLocation object
                loc_name = activity.get_name()  # retail, school etc
                loc_ids = activity.get_ids()  # List of the IDs of the locations
                loc_dangers = activity.get_dangers()  # List of the current dangers
                # XXXX HERE CREATE NEW DATAFRAME FROM PREVIOUS ONE AND ADD Danger0 Column
                self.activities_to_pickle[loc_name] = activity.get_dataframe_copy()
                self.activities_to_pickle[loc_name]['Danger0'] = loc_dangers
                assert False not in list(loc_ids == self.activities_to_pickle[loc_name]["ID"].values)
                # Just use ID and Danger columns
                # self.activities_to_pickle[loc_name] = pd.DataFrame(
                #    list(zip(loc_ids, loc_dangers)), columns=['ID', 'Danger0'])

    @classmethod
    def add_disease_columns(cls, individuals: pd.DataFrame) -> pd.DataFrame:
        """Adds columns required to estimate disease prevalence"""
        individuals[ColumnNames.DISEASE_STATUS] = 0
        individuals[ColumnNames.DISEASE_STATUS_CHANGED] = False
        # individuals[ColumnNames.DAYS_WITH_STATUS] = 0  # Also keep the number of days that have elapsed with this status
        individuals[ColumnNames.CURRENT_RISK] = 0  # This is the risk that people get when visiting locations.

        # No longer update disease counts per MSOA etc. Not needed
        #individuals[ColumnNames.MSOA_CASES] = 0  # Useful to count cases per MSOA
        #individuals[ColumnNames.HID_CASES] = 0  # Ditto for the household

        individuals[ColumnNames.DISEASE_PRESYMP] = -1
        individuals[ColumnNames.DISEASE_SYMP_DAYS] = -1
        individuals[ColumnNames.DISEASE_EXPOSED_DAYS] = -1
        return individuals

    def update_behaviour_during_lockdown(self):
        """
        Unilaterally alter the proportions of time spent on different activities before and after 'lockddown'
        Otherwise this doesn't do anything update_behaviour_during_lockdown.

        Note: ignores people who are currently showing symptoms (`ColumnNames.DiseaseStatus.SYMPTOMATIC`)
        """
        # Are we doing any lockdown at all? in this iteration?
        if self.lockdown_from_file:
            # Only change the behaviour of people who aren't showing symptoms. If you are showing symptoms then you
            # will be mostly at home anyway, so don't want your behaviour overridden by lockdown.
            uninfected = self.individuals.index[
                self.individuals[ColumnNames.DISEASE_STATUS] != ColumnNames.DiseaseStatuses.SYMPTOMATIC]
            if len(uninfected) < len(self.individuals):
                print(f"\t{len(self.individuals) - len(uninfected)} people are symptomatic so not affected by lockdown")

            # Reduce all activities, replacing the lost time with time spent at home
            non_home_activities = set(self.activity_locations.keys())
            non_home_activities.remove("Home")
            # Need to remember the total duration of time lost for non-home activities
            total_duration = pd.Series(data=[0.0] * len(self.individuals.loc[uninfected]), name="TotalDuration")

            if self.lockdown_from_file:  # Reduce the initial activity proportion of time by a particular amount per day
                timeout_multiplier = self.time_activity_multiplier.loc[
                    self.time_activity_multiplier.day == self.iteration, "timeout_multiplier"].values[0]
                print(f"\tApplying regular (google mobility) multiplier {timeout_multiplier}")
                for activity in non_home_activities:
                    # Need to be careful with new_duration because we don't want to keep the index used in
                    # self.individuals as this will be missing out people who aren't infected so will have gaps
                    new_duration = pd.Series(list(self.individuals.loc[uninfected, activity + ColumnNames.ACTIVITY_DURATION_INITIAL] * timeout_multiplier), name="NewDuration")
                    total_duration += new_duration
                    self.individuals.loc[uninfected, activity + ColumnNames.ACTIVITY_DURATION] = list(new_duration)

            else:  # Should not be able to get here
                assert False

            assert (total_duration <= 1.0).all() and (new_duration <= 1.0).all()
            # Now set home duration to fill in the time lost from doing other activities.
            self.individuals.loc[uninfected, 'Home' + ColumnNames.ACTIVITY_DURATION] = list(1 - total_duration)

            # Check they still sum correctly (if not then they probably need rounding)
            # (If you want to print the durations)
            # self.individuals.loc[:, [ x+ColumnNames.ACTIVITY_DURATION for x in self.activity_locations.keys() ]+
            #     [ x+ColumnNames.ACTIVITY_DURATION_INITIAL for x in self.activity_locations.keys()   ]   ]
            Microsim.check_durations_sum_to_1(self.individuals, self.activity_locations.keys())

    def update_venue_danger_and_risks(self, decimals=8):
        """
        Update the danger score for each location, based on where the individuals who have the infection visit.
        Then look through the individuals again, assigning some of that danger back to them as 'current risk'.

        :param risk_multiplier: Risk is calcuated as duration * flow * risk_multiplier.
        :param decimals: Number of decimals to round the indivdiual risks and dangers to (defult 10). If 'None'
                        then do no rounding
        """
        print("\tUpdating danger associated with visiting each venue")

        # Make a new list to keep the new risk for each individual (better than repeatedly accessing the dataframe)
        # Make this 0 initialy as the risk is not cumulative; it gets reset each day
        current_risk = [0] * len(self.individuals)

        # for name in tqdm(self.activity_locations, desc=f"Updating dangers and risks for activity locations"):
        for activty_name in self.activity_locations:

            #
            # ***** 1 - update dangers of each venue (infected people visitting places)
            #

            print(f"\t\t{activty_name} activity")
            # Get the details of the location activity
            activity_location = self.activity_locations[activty_name]  # Pointer to the ActivityLocation object
            # Create a list to store the dangers associated with each location for this activity.
            # Assume 0 initially, it should be reset each day
            loc_dangers = [0] * len(activity_location.get_dangers())
            # loc_dangers = activity_location.get_dangers()  # List of the current dangers associated with each place

            # Now look up those venues in the table of individuals
            venues_col = f"{activty_name}{ColumnNames.ACTIVITY_VENUES}"  # The names of the venues and
            flows_col = f"{activty_name}{ColumnNames.ACTIVITY_FLOWS}"  # flows in the individuals DataFrame
            durations_col = f"{activty_name}{ColumnNames.ACTIVITY_DURATION}"  # flows in the individuals DataFrame

            # 2D lists, for each individual: the venues they visit, the flows to the venue (i.e. how much they visit it)
            # and the durations (how long they spend doing it)
            statuses = self.individuals[ColumnNames.DISEASE_STATUS]
            venues = self.individuals.loc[:, venues_col]
            flows = self.individuals.loc[:, flows_col]
            durations = self.individuals.loc[:, durations_col]
            assert len(venues) == len(flows) and len(venues) == len(statuses)
            for i, (v, f, s, duration) in enumerate(zip(venues, flows, statuses, durations)):  # For each individual
                # Only people with the disease who are infectious will add danger to a place
                if s == ColumnNames.DiseaseStatuses.PRESYMPTOMATIC \
                        or s == ColumnNames.DiseaseStatuses.SYMPTOMATIC \
                        or s == ColumnNames.DiseaseStatuses.ASYMPTOMATIC:
                    # The hazard multiplier depends on the type of disease status that this person has
                    hazard_multiplier = None
                    if s == ColumnNames.DiseaseStatuses.PRESYMPTOMATIC:
                        hazard_multiplier = self.hazard_multiplier_presymptomatic
                    elif s == ColumnNames.DiseaseStatuses.SYMPTOMATIC:
                        hazard_multiplier = self.hazard_multiplier_symptomatic
                    elif s == ColumnNames.DiseaseStatuses.ASYMPTOMATIC:
                        hazard_multiplier = self.hazard_multiplier_asymptomatic
                    assert hazard_multiplier is not None
                    # v and f are lists of flows and venues for the individual. Go through each one
                    for venue_idx, flow in zip(v, f):
                        # print(i, venue_idx, flow, duration)
                        # Increase the danger by the flow multiplied by some disease risk
                        danger_increase = (flow * duration * hazard_multiplier)
                        warnings.warn("Temporarily reduce danger for work while we have virtual work locations")
                        if activty_name == "Work":
                            work_danger = float(danger_increase / 20)
                            loc_dangers[venue_idx] += work_danger
                        else:
                            loc_dangers[venue_idx] += danger_increase

            #
            # ***** 2 - risks for individuals who visit dangerous venues
            #

            # It's useful to report the specific risks associated with *this* activity for each individual
            activity_specific_risk = [0] * len(self.individuals)

            for i, (v, f, s, duration) in enumerate(zip(venues, flows, statuses, durations)):  # For each individual
                # v and f are lists of flows and venues for the individual. Go through each one
                for venue_idx, flow in zip(v, f):
                    #  Danger associated with the location (we just created these updated dangers in the previous loop)
                    danger = loc_dangers[venue_idx]
                    risk_increase = flow * danger * duration * self.risk_multiplier
                    current_risk[i] += risk_increase
                    activity_specific_risk[i] += risk_increase

            # Remember the (rounded) risk for this activity
            if decimals is not None:
                activity_specific_risk = [round(x, decimals) for x in activity_specific_risk]
            self.individuals[f"{activty_name}{ColumnNames.ACTIVITY_RISK}"] = activity_specific_risk

            # Now we have the dangers associated with each location, apply these back to the main dataframe
            if decimals is not None:  # Round the dangers?
                loc_dangers = [round(x, decimals) for x in loc_dangers]
            activity_location.update_dangers(loc_dangers)

        # Round the current risk
        if decimals is not None:
            current_risk = [round(x, decimals) for x in current_risk]

        # Sanity check
        assert len(current_risk) == len(self.individuals)
        assert min(current_risk) >= 0  # Should not be risk less than 0
        # Santity check - do the risks of each activity add up to the total?
        # (I can't get this to work, there are some really minor differences, but on the whole it looks fine)
        # (I can't get this to work, there are some really minor differences, but on the whole it looks fine)
        # if Microsim.debug:  # replace with .debug
        #    total_risk = [0.0] * len(self.individuals)
        #    for activty_name in self.activity_locations:
        #        total_risk = [i + j for (i, j) in zip(total_risk, list(self.individuals[f"{activty_name}{ColumnNames.ACTIVITY_RISK}"]))]
        #    # Round both
        #    total_risk = [round(x, 5) for x in total_risk]
        #    current_risk_temp = [round(x, 5) for x in current_risk]
        #    assert current_risk_temp == total_risk

        self.individuals[ColumnNames.CURRENT_RISK] = current_risk

        return

    # No longer update disease counts per MSOA etc. Not needed
    #def update_disease_counts(self):
    #    """Update some disease counters -- counts of diseases in MSOAs & households -- which are useful
    #    in estimating the probability of contracting the disease"""
    #    # Update the diseases per MSOA and household
    #    # TODO replace Nan's with 0 (not a problem with MSOAs because they're a cateogry so the value_counts()
    #    # returns all, including those with 0 counts, but with HID those with 0 count don't get returned
    #    # Get rows with cases
    #    cases = self.individuals.loc[(self.individuals[ColumnNames.DISEASE_STATUS] == 1) |
    #                                 (self.individuals[ColumnNames.DISEASE_STATUS] == 2), :]
    #    # Count cases per area (convert to a dataframe)
    #    case_counts = cases["area"].value_counts()
    #    case_counts = pd.DataFrame(data={"area": case_counts.index, "Count": case_counts}).reset_index(drop=True)
    #    # Link this back to the orignal data
    #    self.individuals[ColumnNames.MSOA_CASES] = self.individuals.merge(case_counts, on="area", how="left")["Count"]
    #    self.individuals[ColumnNames.MSOA_CASES].fillna(0, inplace=True)
    #
    #     # Update HID cases
    #    case_counts = cases["House_ID"].value_counts()
    #    case_counts = pd.DataFrame(data={"House_ID": case_counts.index, "Count": case_counts}).reset_index(drop=True)
    #    self.individuals[ColumnNames.HID_CASES] = self.individuals.merge(case_counts, on="House_ID", how="left")[
    #        "Count"]
    #    self.individuals[ColumnNames.HID_CASES].fillna(0, inplace=True)

    def calculate_new_disease_status(self) -> None:
        """
        Call an R function to calculate the new disease status for all individuals.
        Update the indivdiuals dataframe in place
        :return: . Update the dataframe inplace
        """
        # Remember the old status so that we can calculate whether it has changed
        old_status: pd.Series = self.individuals[ColumnNames.DISEASE_STATUS].copy()

        # Calculate the new status (will return a new dataframe)
        self.individuals = self.r_int.calculate_disease_status(self.individuals, self.iteration, self.disease_params)

        # Remember whose status has changed
        new_status: pd.Series = self.individuals[ColumnNames.DISEASE_STATUS].copy()
        self.individuals[ColumnNames.DISEASE_STATUS_CHANGED] = list(new_status != old_status)

        # For info, find out how the statuses have changed.
        # Make a dict with all possible changes, then loop through and count them.
        change = dict()
        for old in ColumnNames.DiseaseStatuses.ALL:
            for new in ColumnNames.DiseaseStatuses.ALL:
                change[(old, new)] = 0
        for (old, new) in zip(old_status, new_status):
            if new != old:
                change[(old,  new)] += 1

        assert sum(change.values()) == len(new_status[new_status != old_status])

        print(f"\t{len(new_status[new_status != old_status])} individuals have a different status. Status changes:")
        for old in ColumnNames.DiseaseStatuses.ALL:
            print(f"\t\t{old} -> ", end="")
            for new in ColumnNames.DiseaseStatuses.ALL:
                print(f" {new}:{change[(old,new)]} \t", end="")
            print()

    def change_behaviour_with_disease(self) -> None:
        """
        When people have the disease the proportions that they spend doing activities changes. This function applies
        those changes inline to the individuals dataframe
        :return: None. Update the dataframe inplace
        """
        #print("Changing behaviour of infected individuals ... ",)
        # Find out which people have changed status
        change_idx = self.individuals.index[self.individuals[ColumnNames.DISEASE_STATUS_CHANGED] == True]

        # Now set their new behaviour
        self.individuals.loc[change_idx] = \
            self.individuals.loc[change_idx].apply(
                func=Microsim._set_new_behaviour, args=( list(self.activity_locations.keys()), ), axis=1)
        # self.individuals.loc[change_idx].swifter.progress_bar(True, desc="Changing behaviour of infected"). \

        print(f"\tCurrent statuses:"
              f"\n\t\tSusceptible ({ColumnNames.DiseaseStatuses.SUSCEPTIBLE}): {len(self.individuals.loc[self.individuals[ColumnNames.DISEASE_STATUS] == ColumnNames.DiseaseStatuses.SUSCEPTIBLE])}"
              f"\n\t\tExposed ({ColumnNames.DiseaseStatuses.EXPOSED}): {len(self.individuals.loc[self.individuals[ColumnNames.DISEASE_STATUS] == ColumnNames.DiseaseStatuses.EXPOSED])}"
              f"\n\t\tPresymptomatic ({ColumnNames.DiseaseStatuses.PRESYMPTOMATIC}): {len(self.individuals.loc[self.individuals[ColumnNames.DISEASE_STATUS] == ColumnNames.DiseaseStatuses.PRESYMPTOMATIC])}"
              f"\n\t\tSymptomatic ({ColumnNames.DiseaseStatuses.SYMPTOMATIC}): {len(self.individuals.loc[self.individuals[ColumnNames.DISEASE_STATUS] == ColumnNames.DiseaseStatuses.SYMPTOMATIC])}"
              f"\n\t\tAsymptomatic ({ColumnNames.DiseaseStatuses.ASYMPTOMATIC}): {len(self.individuals.loc[self.individuals[ColumnNames.DISEASE_STATUS] == ColumnNames.DiseaseStatuses.ASYMPTOMATIC])}"
              f"\n\t\tRecovered ({ColumnNames.DiseaseStatuses.RECOVERED}): {len(self.individuals.loc[self.individuals[ColumnNames.DISEASE_STATUS] == ColumnNames.DiseaseStatuses.RECOVERED])}"
              f"\n\t\tRemoved/dead ({ColumnNames.DiseaseStatuses.DEAD}): {len(self.individuals.loc[self.individuals[ColumnNames.DISEASE_STATUS] == ColumnNames.DiseaseStatuses.DEAD])}")

        #self.individuals.loc[change_idx].apply(func=self._set_new_behaviour, axis=1)
        #print("... finished")

    @staticmethod
    def _set_new_behaviour(row: pd.Series, activities: List[str]):
        """
        Define how someone with the disease should behave. This is called for every individual whose disease status
        has changed between the current iteration and the previous one
        :param row: A row of the individuals dataframe
        :return: An updated row with new ACTIVITY_DURATION columns to reflect the changes in proportion of time that
        the individual spends doing the different activities.
        """
        # Maybe need to put non-symptomatic people back to normal behaviour (or do nothing if they e.g. transfer from
        # Susceptible to Pre-symptomatic, which means they continue doing normal behaviour)
        # Minor bug: this will erode any changes caused by lockdown behaviour for the rest of this iteration, but this
        # only affects people whose status has just changed so only a minor problem
        if row[ColumnNames.DISEASE_STATUS] in [ColumnNames.DiseaseStatuses.SUSCEPTIBLE,
                                               ColumnNames.DiseaseStatuses.EXPOSED,
                                               ColumnNames.DiseaseStatuses.PRESYMPTOMATIC,
                                               ColumnNames.DiseaseStatuses.ASYMPTOMATIC,
                                               ColumnNames.DiseaseStatuses.RECOVERED,
                                               ColumnNames.DiseaseStatuses.DEAD]:
            for activity in activities:
                row[f"{activity}{ColumnNames.ACTIVITY_DURATION}"] = \
                    row[f"{activity}{ColumnNames.ACTIVITY_DURATION_INITIAL}"]

        # Put newly symptomatic people at home
        elif row[ColumnNames.DISEASE_STATUS] == ColumnNames.DiseaseStatuses.SYMPTOMATIC:
            # Reduce all activities, replacing the lost time with time spent at home
            non_home_activities = set(activities)
            non_home_activities.remove("Home")
            total_duration = 0.0  # Need to remember the total duration of time lost for non-home activities
            for activity in non_home_activities:
                #new_duration = row[f"{activity}{ColumnNames.ACTIVITY_DURATION}"] * 0.10
                new_duration = row[f"{activity}{ColumnNames.ACTIVITY_DURATION}"] * 0.50
                total_duration += new_duration
                row[f"{activity}{ColumnNames.ACTIVITY_DURATION}"] = new_duration
            # Now set home duration to fill in the time lost from doing other activities.
            row[f"Home{ColumnNames.ACTIVITY_DURATION}"] = (1 - total_duration)
        else:
            raise Exception(f"Unrecognised disease state for individual {row['ID']}: {row[ColumnNames.DISEASE_STATUS] }")
        return row

    @staticmethod
    def _make_a_copy(m):
        """
        When copying a microsim object, reset the seed

        :param m: A Microsim object
        :return: A deep copy of the microsim object
        """
        m.random.seed()
        return copy.deepcopy(m)

    def step(self) -> None:
        """
        Step (iterate) the model for 1 iteration

        :return:
        """
        self.iteration += 1

        print(f"\nIteration: {self.iteration}\n")

        # Unilaterally adjust the proportions of time that people spend doing different activities after lockdown
        self.update_behaviour_during_lockdown()

        # Update the danger associated with each venue (i.e. the as people with the disease visit them they
        # become more dangerous) then update the risk to each individual of going to those venues.
        self.update_venue_danger_and_risks()

        # Update disease counters. E.g. count diseases in MSOAs & households
        # No longer update disease counts per MSOA etc. Not needed
        #self.update_disease_counts()

        # Calculate new disease status and update the people's behaviour
        if not self.disable_disease_status:
            self.calculate_new_disease_status()
            self.change_behaviour_with_disease()

    def run(self, iterations: int) -> None:
        """
        Run the model (call the step() function) for the given number of iterations
        :param iterations:
        """
        # Create directories for the results
        self._init_output()

        # Initialise the R interface. Do this here, rather than in init, because when in multiprocessing mode
        # at this point the Microsim object will be in its own process
        if not self.disable_disease_status:
            self.r_int = RInterface(self.r_script_dir)

        # Step the model
        for i in range(iterations):
            iter_start_time = time.time()  # time how long each iteration takes (for info)
            self.step()

            # Add to items to pickle for visualisations
            if self.output:
                print("\tGenerating output ... ",)
                # (Force column names to have leading zeros)
                self.individuals_to_pickle[f"{ColumnNames.DISEASE_STATUS}{(i + 1):03d}"] = self.individuals[
                    ColumnNames.DISEASE_STATUS]
                fname = os.path.join(self.output_dir, "Individuals")
                with open(fname + ".pickle", "wb") as pickle_out:
                    pickle.dump(self.individuals_to_pickle, pickle_out)
                # Also make a (compressed) csv file for others
                self.individuals_to_pickle.to_csv(fname + ".csv.gz", compression='gzip')

                for name in self.activity_locations:
                    # Get the details of the location activity
                    activity = self.activity_locations[name]  # Pointer to the ActivityLocation object
                    loc_name = activity.get_name()  # retail, school etc
                    # loc_ids = activity.get_ids()  # List of the IDs of the locations
                    loc_dangers = activity.get_dangers()  # List of the current dangers
                    # Add a new danger column to the previous dataframe
                    self.activities_to_pickle[loc_name][f"{ColumnNames.LOCATION_DANGER}{(i + 1):03d}"] = loc_dangers
                    # Save this activity location
                    fname = os.path.join(self.output_dir, loc_name)
                    with open(fname + ".pickle", "wb") as pickle_out:
                        pickle.dump(self.activities_to_pickle[loc_name], pickle_out)
                    # Also make a (compressed) csv file for others
                    self.activities_to_pickle[loc_name].to_csv(fname + ".csv.gz", compression='gzip')
                    # self.activities_to_pickle[loc_name].to_csv(fname+".csv")  # They not so big so don't compress
                print(" ... finished ", )

            print(f"\tIteration {i} took {round(float(time.time() - iter_start_time), 2)}s")

        print(f"Model finished running (iterations: {i+1})")

        # TEMP WRITE OUTPUT AT END
        #fname = os.path.join(self.output_dir, "Individuals")
        #with open(fname + ".pickle", "wb") as pickle_out:
        #    pickle.dump(self.individuals_to_pickle, pickle_out)
        #self.individuals_to_pickle.to_csv(fname + ".csv.gz", compression='gzip')
        #for name in self.activity_locations:
        #    loc_name = self.activity_locations[name].get_name()
        #    fname = os.path.join(self.output_dir, loc_name)
        #    with open(fname + ".pickle", "wb") as pickle_out:
        #        pickle.dump(self.activities_to_pickle[loc_name], pickle_out)
        #    # Also make a (compressed) csv file for others
        #    self.activities_to_pickle[loc_name].to_csv(fname + ".csv.gz", compression='gzip')


# ********
# PROGRAM ENTRY POINT
# Uses 'click' library so that it can be run from the command line
# ********
@click.command()
@click.option('-p', '--parameters_file', default="./model_parameters/default.yml", type=click.Path(exists=True),
              help="Parameters file to use to configure the model. Default: ./model_parameters.yml")
@click.option('-npf', '--no-parameters-file', is_flag=True,
              help="Don't read a parameters file, use command line arguments instead")
@click.option('-i', '--iterations', default=10, help='Number of model iterations. 0 means just run the initialisation')
@click.option('--data-dir', default="devon_data", help='Root directory to load data from')
@click.option('--output/--no-output', default=True,
              help='Whether to generate output data (default yes).')
@click.option('--debug/--no-debug', default=False, help="Whether to run some more expensive checks (default no debug)")
@click.option('-r', '--repetitions', default=1, help="How many times to run the model (default 1)")
@click.option('-l', '--lockdown-from-file/--no-lockdown-from-file', default=True,
              help="Optionally read lockdown mobility data from a file (default True)")
def run_script(parameters_file, no_parameters_file, iterations, data_dir, output, debug, repetitions, lockdown_from_file):

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
            iterations = sim_params["iterations"]
            data_dir = sim_params["data-dir"]
            output = sim_params["output"]
            debug = sim_params["debug"]
            repetitions = sim_params["repetitions"]
            lockdown_from_file = sim_params["lockdown-from-file"]
            ## Calibration parameters
            #hazard_multiplier_presymptomatic = calibration_params["hazard_multiplier_presymptomatic"]
            #hazard_multiplier_asymptomatic = calibration_params["hazard_multiplier_asymptomatic"]
            #hazard_multiplier_symptomatic = calibration_params["hazard_multiplier_symptomatic"]
            #risk_multiplier = calibration_params["risk_multiplier"]

    # Check the parameters are sensible
    if iterations < 0:
        raise ValueError("Iterations must be > 0")
    if repetitions < 1:
        raise ValueError("Repetitions must be greater than 0")

    print(f"Running model with the following parameters:\n"
          f"\tParameters file: {parameters_file}\n"
          f"\tNumber of iterations: {iterations}\n"
          f"\tData dir: {data_dir}\n"
          f"\tOutputting results?: {output}\n"
          f"\tDebug mode?: {debug}\n"
          f"\tNumber of repetitions: {repetitions}\n"
          f"\tLockdown from file? : {lockdown_from_file}\n"
          f"\tCalibration parameters: {'N/A (not reading parameters file)' if no_parameters_file else str(calibration_params)}\n")
          #f"\thazard_multiplier_presymptomatic: {hazard_multiplier_presymptomatic}\n"
          #f"\thazard_multiplier_asymptomatic: {hazard_multiplier_asymptomatic}\n"
          #f"\thazard_multiplier_symptomatic: {hazard_multiplier_symptomatic}\n"
          #f"\trisk_multiplier: {risk_multiplier}\n")

    if iterations == 0:
        print("Iterations = 0. Not stepping model, just assigning the initial risks.")

    # To fix file path issues, use absolute/full path at all times
    # Pick either: get working directory (if user starts this script in place, or set working directory
    # Option A: copy current working directory:
    base_dir = os.getcwd()  # get current directory
    data_dir = os.path.join(base_dir, data_dir)
    r_script_dir = os.path.join(base_dir, "R", "py_int")

    # Temporarily only want to use Devon MSOAs
    devon_msoas = pd.read_csv(os.path.join(data_dir, "devon_msoas.csv"), header=None,
                              names=["x", "y", "Num", "Code", "Desc"])

    # Use same arguments whether running 1 repetition or many
    msim_args = {"data_dir": data_dir, "r_script_dir": r_script_dir,
                 "output": output, "debug": debug,
                 "lockdown_from_file": lockdown_from_file,
                 }

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

    # Run it!
    if repetitions == 1:
        # Create a microsim object
        m = Microsim(**msim_args)
        m.run(iterations)
    else:  # Run it multiple times in lots of cores
        try:
            with multiprocessing.Pool(processes=int(os.cpu_count())) as pool:
                # Copy the model instance so we don't have to re-read the data each time
                # (Use a generator so we don't need to store all the models in memory at once).
                m = Microsim(**msim_args)
                models = (Microsim._make_a_copy(m) for _ in range(repetitions))
                # models = ( Microsim(msim_args) for _ in range(repetitions))
                # Also need a list giving the number of iterations for each model (same for each model)
                iters = (iterations for _ in range(repetitions))
                # Run the models by passing each model and the number of iterations
                pool.starmap(_run_multicore, zip(models, iters))
        finally:  # Make sure they get closed (shouldn't be necessary)
            pool.close()

    print("End of program")


def _run_multicore(m, iter):
    return m.run(iter)


if __name__ == "__main__":
    run_script()
    print("End of program")

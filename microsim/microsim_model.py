#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core RAMP-UA model.

Created on Wed Apr 29 19:59:25 2020

@author: nick
"""

import pandas as pd
import glob
import os
import random
import time
import re  # For analysing file names
import typing
from enum import Enum  # For disease status
from typing import List, Dict
from tqdm import tqdm  # For a progress bar
import click  # command-line interface

from microsim.microsim_analysis import MicrosimAnalysis


class DiseaseStatus(Enum):
    """Represent the different statuses of the disease"""
    SUSCEPTIBLE = 0
    EXPOSED = 1
    INFECTED = 2
    RECOVERED = 3

class ActivityLocation():
    """Class to represent information about activity locations, e.g. retail destinations, workpaces, etc."""
    def __init__(self, name: str, locations: pd.DataFrame, flows: pd.DataFrame):
        """
        Initialise an ActivityLocation
        :param name: A name to use to refer to this activity. Column names in the big DataFrame of individuals
        will be named according to this
        :param locations: A dataframe containing information about each loction
        :param flows: A dataframe containing flows
        """
        self._name = name
        # Check that the locations DataFrame has an ID column and a Danger column
        if "ID" not in locations.columns or "Danger" not in locations.columns:
            raise Exception(f"Activity '{name}' dataframe needs columns called 'ID' and 'Danger."
                            f"It only has: {locations.columns}")
        # Check that the DataFrame's ID column is also an index, this is to ensure that the IDs always
        # refer to the same thing
        if locations.index.name != "ID" or False in (locations.index == locations.ID):
            raise Exception(f"Activity '{name}' dataframe needs to have an index column called 'ID'"
                            f"that is equal to the 'ID' columns.")
        self._locations = locations
        self._flows = flows

    def __repr__(self):
        return f"<{self._name} ActivityLocation>"

    def get_dangers(self) -> List[float]:
        """Get the danger associated with each location as a list. These will be in the same order as the
        location IDs returned by `get_ids()`"""
        return list(self._locations.Danger)

    def get_ids(self) -> List[int]:
        """Retrn the IDs of each retail destination"""
        return list(self._locations.ID)

    def update_dangers(self, dangers: List[float]):
        """
        Update the danger associated with each location
        :param dangers: A list of dangers for each location. Must be in the same order as the locations as
        returned by `get_ids`.
        """
        if len(dangers) != len(self._locations):
            raise Exception(f"The number of danger scores ({len(dangers)}) is not the same as the number of"
                            f"activity locations ({len(self._locations)}).")
        self._locations["Danger"] = dangers



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

    DATA_DIR = "./data/"

    def __init__(self, study_msoas: List[str] = [], random_seed: float = None, read_data: bool = True):
        """
        Microsim constructor.
        ----------
        :param study_msoas: An optional list of MSOA codes to restrict the model to
        :param random_seed: A optional random seed to use when creating the class instance. If None then
            the current time is used.
        :param read_data: Optionally don't read in the data when instantiating this Microsim (useful
            in debugging).
        """

        # Administrative variables that need to be defined
        self.iteration = 0
        self.random = random.Random(time.time() if random_seed is None else random_seed)


        # Now the main chunk of initialisation is to read the input data.
        if not read_data:  # Optionally can not do this, usually for debugging
            return

        # This is the main population of individuals and their households
        self.individuals, self.households = Microsim.read_msm_data()

        # The individuals are at MSOA level, so use that file to construct a list of areas
        self.all_msoas = Microsim.extract_msoas_from_indiviuals(self.individuals)

        # See if we need to restrict by a study area (optional parameter passed by the user).
        # If so, then remove individuals and households not in the study area
        self.study_msoas, self.individuals, self.households = \
            Microsim.check_study_area(self.all_msoas, study_msoas, self.individuals, self.households)

        # Attach a load of health attributes to each individual
        self.individuals = Microsim.attach_health_data(self.individuals)

        # Attach the labour force data
        self.individuals = Microsim.attach_labour_force_data(self.individuals)

        # Attach a load of transport attributes to each individual
        self.individuals = Microsim.attach_time_use_data(self.individuals)

        # Now we have the 'core population'. Keep a copy of this but continue to use the 'individuals' data frame
        self.core_population = self.individuals.copy()

        # Read the locations of schools, workplaces, etc.
        # For each type of activity (store, retail, etc), create ActivityLocation objects to keep all the
        # required information together.
        self.activity_locatons: Dict[str, ActivityLocation] = {}

        # Read Retail
        retail_name = "Retail" # How to refer to this in data frame columns etc.
        stores, stores_flows = Microsim.read_retail_flows_data(self.study_msoas)  # (list of shops and a flow matrix)
        Microsim.check_sim_flows(stores, stores_flows)

        # Read Schools
        school_name = "School"
        schools, schools_flows = Microsim.read_school_flows_data(self.study_msoas)  # (list of schools and a flow matrix)
        print("TEMPORARILY TRIMMING SCHOOLS FLOWS")
        schools_flows = schools_flows.iloc[0:107, :]



        # Workplaces etc.
        # self.workplaces = Microsim.read_workplace_data()
        # self.schools = Microsim.read_school_data()

        # Assign probabilities that each individual will go to each location (most of these will be 0!)
        # Do this by adding two new columns, both storing lists. One lists ids of the locations that
        # the individual may visit, the other has the probability of them visitting those places

        # Assign Retail
        self.individuals = Microsim.add_individual_flows(retail_name, self.individuals, stores_flows)
        self.activity_locatons[retail_name] = ActivityLocation(retail_name, stores, stores_flows)

        # Assign Schools
        self.individuals = Microsim.add_individual_flows(school_name, self.individuals, schools_flows)
        self.activity_locatons[school_name] = ActivityLocation(school_name, schools, schools_flows)

        # Assign initial SEIR status
        self.individuals = Microsim.assign_initial_disease_status(self.individuals)

        print(".. End of __init__() .. ")
        return

    @classmethod
    def read_msm_data(cls) -> (pd.DataFrame, pd.DataFrame):
        """Read the csv files that have the indivduls and households

        :return a tuple with two pandas dataframes representing individuls (0) and households (1)
        """

        msm_dir = os.path.join(cls.DATA_DIR, "msm_data")

        # Can't just read in all the files because the microsimulation restarts the house and index numbering with
        # each file, but we need the IDs to be consistent across the whole area. So read the files in one-by-one
        # and make sure houses and individual IDs are unique
        household_files = glob.glob(msm_dir + '/ass_hh_*_OA11_2020.csv')
        if len(household_files) == 0:
            raise Exception(f"No household csv files found in {msm_dir}.",
                            f"Have you downloaded and extracted the necessary data? (see {cls.DATA_DIR} README)")
        individual_files = glob.glob(msm_dir + '/ass_*_MSOA11_2020.csv')
        if len(individual_files) == 0:
            raise Exception(f"No individual csv files found in {msm_dir}.",
                            f"Have you downloaded and extracted the necessary data? (see {cls.DATA_DIR} README)")
        assert (len(household_files) == len(individual_files))
        household_files.sort()
        individual_files.sort()

        # Create a DataFrame from each file, then concatenate them later
        house_dfs = []
        indiv_dfs = []

        # Keep track of the house and person indices
        PID_counter = 0
        HID_counter = 0
        for i in tqdm(range(len(household_files)), desc="Reading raw microsim data"):
            house_file = household_files[i]
            indiv_file = individual_files[i]
            area = re.search(r".*?ass_hh_(E\d.*?)_OA.*", house_file).group(1)  # Area is in the file name
            # (and check that both files refer to the same area)
            assert area == re.search(r".*?ass_(E\d.*?)_MSOA.*", indiv_file).group(1)

            house_df = pd.read_csv(house_file)
            indiv_df = pd.read_csv(indiv_file)

            # Increment the counters
            house_df["HID"] = house_df["HID"].apply(lambda x: x + HID_counter)
            house_df["HRPID"] = house_df["HRPID"].apply(lambda x: x + PID_counter)  # Also increase the HRP

            indiv_df["PID"] = indiv_df["PID"].apply(lambda x: x + PID_counter)
            indiv_df["HID"] = indiv_df["HID"].apply(lambda x: x + HID_counter)  # Also increase the link to HID

            HID_counter = max(house_df["HID"]) + 1  # Want next counter to start at one larger than current
            PID_counter = max(indiv_df["PID"]) + 1

            # Save the dataframes for concatination later
            house_dfs.append(house_df)
            indiv_dfs.append(indiv_df)

        households = pd.concat(house_dfs)
        households.set_index("HID", inplace=True, drop=False)
        individuals = pd.concat(indiv_dfs)
        individuals.set_index("PID", inplace=True, drop=False)

        # Make sure HIDs and PIDs are unique
        assert len(households["HID"].unique()) == len(households)
        assert len(individuals["PID"].unique()) == len(individuals)

        # THE FOLLOWING SHOULD BE DONE AS PART OF A TEST SUITE
        # TODO: check that correct numbers of rows have been read.
        # TODO: check that each individual has a household
        # TODO: check that individuals are correctly linked to households (I had to re-do the PID and HID indexing)
        # TODO: graph number of people per household just to sense check

        print("Have read files:",
              f"\n\tHouseholds:  {len(house_dfs)} files with {len(households)}",
              f"households in {len(households.Area.unique())} areas",
              f"\n\tIndividuals: {len(indiv_dfs)} files with {len(individuals)}",
              f"individuals in {len(individuals.Area.unique())} areas")

        # TODO Join individuls to households (create lookup columns in each)

        return (individuals, households)

    @classmethod
    def extract_msoas_from_indiviuals(cls, individuals: pd.DataFrame) -> List[str]:
        """
        Analyse a DataFrame of individuals and extract the unique MSOA codes, returning them as a list in ascending
        order
        :param individuals:
        :return:
        """
        areas = list(individuals.Area.unique())
        areas.sort()
        return areas

    @classmethod
    def check_study_area(cls, all_msoas: List[str], study_msoas: List[str], individuals: pd.DataFrame,
                         households: pd.DataFrame) \
            -> (List[str], pd.DataFrame, pd.DataFrame):
        """
        It is possible to optionally subset all MSOAs used in the analysis (i.e. create a study area). If so, then
        remove all individuals and households who are outside of the study area, returning new DataFrames.
        :param all_msoas: All areas that could be used (e.g. all MSOAs in the UK)
        :param study_msoas: A subset of those areas that could be used
        :param individuals: The DataFrame of individuals.
        :param households:  The DataFrame of households
        :return: A tuple containing:
          - [0] a list of the MSOAs being used (either all, or just those in the study area)
          - [1] the new list of individuals (which might be shorter than the original if using a smaller study area)
          - [2] the new list of households
        """
        # No study area subset provided, use the whole area.
        if study_msoas is None or len(study_msoas) == 0:
            return all_msoas, individuals, households
        # Check that all areas in both arrays are unique
        for d, l in [("all msoas", all_msoas), ("study area msoas", study_msoas)]:
            if len(l) != len(set(l)):
                raise Exception(f"There are some duplicate areas in the {d} list: {l}.")
        for area in study_msoas:
            if area not in all_msoas:
                raise Exception(f"Area '{area}' in the list of case study areas is not in the national dataset")

        # Which individuals and houeholds to keep
        individuals_to_keep = individuals.loc[individuals.Area.isin(study_msoas), :]
        assert (len(individuals_to_keep.Area.unique()) == len(study_msoas))
        households_to_keep = households.loc[households.HID.isin(individuals_to_keep.HID), :]
        print(f"\tUsing a subset study area consisting of {len(study_msoas)} MSOAs.\n"
              f"\tBefore subsetting: {len(individuals)} individuals, {len(households)} househods.",
              f"\tAfter subsetting: {len(individuals_to_keep)} individuals, {len(households_to_keep)} househods.")

        return (study_msoas, individuals_to_keep, households_to_keep)


    @classmethod
    def attach_health_data(cls, individuals: pd.DataFrame) -> pd.DataFrame:
        """

        :param individuals:
        :return:
        """
        print("Attaching health data ... ", )
        pass
        print("... finished.")
        return individuals

    @classmethod
    def attach_time_use_data(cls, individuals: pd.DataFrame) -> pd.DataFrame:
        print("Attaching time use data ... ", )
        pass
        print("... finished.")
        return individuals

    @classmethod
    def attach_labour_force_data(cls, individuals: pd.DataFrame) -> pd.DataFrame:
        print("Attaching labour force ... ", )
        pass
        print("... finished.")
        return individuals


    @classmethod
    def read_school_flows_data(cls, study_msoas: List[str]) -> (pd.DataFrame, pd.DataFrame):
        """
        Read the flows between each MSOA and the most likely schools attended by pupils in this area
        ??? We will not allocate a single school to a given pupil due to uncertainty in catchment area
        Instead, like in the retail flow, each pupil will visit schools with a given likelihood ???

        :type study_msoas: A list of MSOAs in the study area (flows outside of this will be ignored)
        :return: A tuple of two dataframes. One containing all of the flows and another
        containing information about the schools themselves.
        """
        # TODO Need to read full school flows, not just those of Devon
        print("WARNING: not currently subsetting school flows")
        print("Reading retail flow data...", )
        dir = os.path.join(cls.DATA_DIR, "temp-schools")

        # Read the schools
        schools = pd.read_csv(os.path.join(dir, "exeter schools.csv"))
        schools['ID'] = list(schools.index + 1)  # Mark counts from 1, not zero, so indices need to start from 1
        schools['Danger'] = 0 # All schools have a disease danger associated with them, initialise it to 0
        schools.set_index("ID", inplace=True, drop=False)

        # Read the flows
        rows = []  # Build up all the rows in the matrix gradually then add all at once
        with open(os.path.join(dir, "DJS002.TXT")) as f:
            count = 1  # Mark's file comes in batches of 3 lines, each giving different data. However, some lines overrun and are read as several lines rather than 1 (hence use of dests_tmp and flows_tmp)
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
                    oa = int(line_list[1])
                    oa_name = study_msoas[oa-1] # The OA names are stored in a separate file temporarily
                    num_dests = int(line_list[2])
                elif count == 2:  # school ids
                    dests_tmp = [int(x) for x in line_list[0:]]  # Make the destinations numbers
                    # check if dests exists from previous iteration and add dests_tmp
                    if dests == None:
                        dests = dests_tmp
                    else:
                        dests.extend(dests_tmp)
                    if len(dests) < num_dests: # need to read next line
                        count = 1 # counteracts count being increased by 1 later
                    else:
                        assert len(dests) == num_dests
                elif count == 3:  # Flows per 1,000 pupils
                    flows_tmp = [float(x) for x in line_list[0:]]  # Make the destinations numbers
                    # check if dests exists from previous iteration and add dests_tmp
                    if flows == None:
                        flows = flows_tmp
                    else:
                        flows.extend(flows_tmp)
                    if len(flows) < num_dests: # need to read next line
                        count = 2 # counteracts count being increased by 1 later
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

                        rows.append(row)

                        # Add the row to the matrix. As the OA numbers are incremental they should equal the number
                        # of rows
                        # flow_matrix.loc[oa-1] = row
                        # assert len(flow_matrix) == oa
                        count = 0
                        # reset dests and flows
                        dests = None
                        flows = None

                count += 1

        # Have finished reading the file, now create the matrix. MSOAs as rows, school locations as columns
        columns = ["Area_ID", "Area_Code"]  # A number (ID) and full code for each MSOA
        columns += [f"Loc_{i}" for i in schools.index]  # Columns for each school
        flow_matrix = pd.DataFrame(data=rows, columns=columns)

        return schools, flow_matrix



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
        print("WARNING: not currently subsetting retail flows")
        print("Reading retail flow data...", )
        dir = os.path.join(cls.DATA_DIR, "temp-retail")

        # Read the stores
        stores = pd.read_csv(os.path.join(dir, "devon smkt.csv"))
        stores['ID'] = list(stores.index + 1)  # Mark counts from 1, not zero, so indices need to start from 1
        stores['Danger'] = 0 # All stores have a disease danger associated with them, initialise it to 0
        stores.set_index("ID", inplace=True, drop=False)

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
                    assert len([x for x in row if x > 0]) == num_dests  # There should only be N >0 flows
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

    @classmethod
    def add_individual_flows(cls, flow_type: str, individuals: pd.DataFrame, flow_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Take a flow matrix from MSOAs to (e.g. retail) locations and assign flows to individuals.
        :param flow_type: What type of flows are these. This will be appended to the column names. E.g. "Retail".
        :param individuals: The DataFrame contining information about all individuals
        :param flow_matrix: The flow matrix, created by (e.g.) read_retail_flows_data()
        :return: The DataFrame of individuals with new locations and probabilities added
        """

        # Check that there aren't any individuals who wont be given any flows
        if len(individuals.loc[-individuals.Area.isin(flow_matrix.Area_Code)]) > 0:
            raise Exception(f"Some individuals will not be assigned any flows to: '{flow_type}' because their"
                            f"MSOA is not in the flow matrix: "
                            f"{individuals.loc[-individuals.Area.isin(flow_matrix.Area_Code)]}.")

        # Check that there aren't any duplicate flows
        if len(flow_matrix) != len(flow_matrix.Area_Code.unique()):
            raise Exception("There are duplicate area codes in the flow matrix: ", flow_matrix.Area_Code)

        # Names for the new columns
        venues_col = f"{flow_type}_Venues"
        flows_col = f"{flow_type}_Flows"

        # Create empty lists to hold the vanues and flows for each individuals
        individuals[venues_col] = [[] for _ in range(len(individuals))]
        individuals[flows_col] = [[] for _ in range(len(individuals))]

        # Use a hierarchical index on the Area to speed up finding all individuals in an area
        # (not sure this makes much difference).
        individuals.set_index(["Area", "PID"], inplace=True, drop=False)

        for area in tqdm(flow_matrix.values, desc=f"Assigning individual flows for {flow_type}"):  # Easier to operate over a 2D matrix rather than a dataframe
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

            # Now assign individuals in those areas to those flows
            # This ridiculous 'apply' line is the only way I could get pandas to update the particular
            # rows required. Something like 'individuals.loc[ ...] = dests' (see below) didn't work becuase
            # instead of inserting the 'dests' list itself, pandas tried to unpack the list and insert
            # the individual values instead.
            # individuals.loc[individuals.Area == oa_code, f"{flow_type}_Venues"] = dests
            # individuals.loc[individuals.Area == oa_code, f"{flow_type}_Probabilities"] = flow
            #
            # A quicker way to do this is probably to create N subsets of individuals (one table for
            # each area) and then concatenate them at the end.

            individuals.loc[oa_code, venues_col] = \
                individuals.loc[oa_code, venues_col].apply(lambda _: dests).values
            individuals.loc[oa_code, flows_col] = \
                individuals.loc[oa_code, flows_col].apply(lambda _: flows).values
            # individuals.loc[individuals.Area=="E02004189", f"{flow_type}_Venues"] = \
            #    individuals.loc[individuals.Area=="E02004189", f"{flow_type}_Venues"].apply(lambda _: dests)
            # individuals.loc[individuals.Area=="E02004189", f"{flow_type}_Probabilities"] = \
            #    individuals.loc[individuals.Area=="E02004189", f"{flow_type}_Probabilities"].apply(lambda _: flows)

        # Reset the index so that it's just PID
        individuals.set_index("PID", inplace=True, drop=False)

        # Check everyone has some flows (all list lengths are >0)
        assert False not in (individuals.loc[:, venues_col].apply(lambda cell: len(cell)) > 0).values
        assert False not in (individuals.loc[:, flows_col].apply(lambda cell: len(cell)) > 0).values

        return individuals

    @classmethod
    def assign_initial_disease_status(cls, individuals: pd.DataFrame) -> pd.DataFrame:
        """
        Create a new column to represent the initial disease status of the individuals and assign them
        an initial status.
        :param individuals: The dataframe containin synthetic individuals
        :return: A new DataFrame for the individuals with the additional column
        """
        print("Assigning initial disease status ...",)
        individuals["Disease_Status"] = [random.choice(list(DiseaseStatus)) for _ in range(len(individuals))]
        print(f"... finished assigning initial status for {len(individuals)} individuals.")
        return individuals


    def update_venue_danger(self):
        print("\tUpdating danger associated with visiting each venue")
        print("XXXX HERE")
        for name in tqdm(self.activity_locatons, desc=f"Updating dangers for activity locations"):
            print(f"\tAnalysing {name} activity")
            # Get the details of the location activity
            activity = self.activity_locatons[name]
            loc_ids = activity.get_ids() # Locations where the activity can take place
            loc_dangers = activity.get_dangers() # Current dangers associated with each place

            # Now look up those venues in the table of individuals

            venues_col = f"{name}_Venues" # The names of the venues and
            flows_col = f"{name}_Flows"   # flows in the individuals DataFrame

            # 2D lists, for each individual the venues they visit and the flows to the venue (i.e. how much they visit it)
            statuses = self.individuals.Disease_Status
            venues = self.individuals.loc[:, venues_col]
            flows = self.individuals.loc[:, flows_col]
            assert len(venues) == len(flows) and len(venues) == len(statuses)
            for i, (v, f, s) in enumerate(zip(venues, flows, statuses)): # For each individual
                if s == DiseaseStatus.INFECTED:
                    # v and f are lists of flows and venues for the individual. Go through each one
                    for venue, flow in zip(v, f):
                        # Increase the danger by the flow multiplied by some disease risk
                        loc_dangers[venue] += ( flow * 0.1)

            # Now we have the dangers associated with each location, apply these back to the main dataframe
            activity.update_dangers(loc_dangers)

        return


    def update_venue_risks(self):
        print("\tAssigning risks of visiting different venues to the individuals")
        print("XXXX HERE")
        x=1
        return

    def step(self) -> None:
        """Step (iterate) the model"""
        self.iteration += 1
        print(f"Iteration: {self.iteration}")

        # Update the danger associated with each venue (i.e. the as people with the disease visit them they
        # become more dangerous
        self.update_venue_danger()

        # Update the risks to individuals who visit those venues
        self.update_venue_risks()

        # Do some analysis
        fig = MicrosimAnalysis.population_distribution(self.individuals, ["DC1117EW_C_AGE"])
        fig.show()
        #MicrosimAnalysis.location_danger_distribution(self.activity_locatons['Retail'], ["Danger"])



# PROGRAM ENTRY POINT
# Uses 'click' library so that it can be run from the command line
@click.command()
@click.option('--iterations', default=10, help='Number of model iterations')
def run(iterations):
    num_iter = iterations

    # Temporarily only want to use Devon MSOAs
    devon_msoas = pd.read_csv("./data/devon_msoas.csv", header=None, names=["x", "y", "Num", "Code", "Desc"])

    m = Microsim(study_msoas=list(devon_msoas.Code))
    for i in range(num_iter):
        m.step()
    print("End of program")


if __name__ == "__main__":
    run()
    print("End of program")

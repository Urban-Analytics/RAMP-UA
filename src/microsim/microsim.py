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
import re # For analysing file names
import typing
from tqdm import tqdm # For a progress bar

import click  # command-line interface


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

    DATA_DIR = "../../data/"

    def __init__(self, random_seed: float = None, read_data: bool = True):
        """
        Microsim constructor.
        ----------
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

        # Attach a load of health attributes to each individual
        self.individuals = Microsim.attach_health_data(self.individuals)

        # Attach the labour force data
        self.individuals = Microsim.attach_labour_force_data(self.individuals)

        # Attach a load of transport attributes to each individual
        self.individuals = Microsim.attach_time_use_data(self.individuals)

        # Now we have the 'core population'. Keep a copy of this but continue to use the 'individuals' data frame
        self.core_population = self.individuals.copy()

        # Read the locations of schools, workplaces, etc.
        # self.schools = Microsim.read_school_data()
        self.stores, self.stores_flows = Microsim.read_retail_flows_data()  # (list of shops and a flow matrix)
        # self.workplaces = Microsim.read_workplace_data()

        # Assign probabilities that each individual will go to each location (most of these will be 0!)
        # Do this by adding two new columns, both storing lists. One lists ids of the locations that
        # the individual may visit, the other has the probability of them visitting those places
        self.individuals = Microsim.add_individual_flows("Retail", self.individuals, self.stores_flows)

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
            area = re.search(r".*?ass_hh_(E\d.*?)_OA.*", house_file).group(1) # Area is in the file name
            # (and check that both files refer to the same area)
            assert area == re.search(r".*?ass_(E\d.*?)_MSOA.*", indiv_file).group(1)

            house_df = pd.read_csv(house_file)
            indiv_df = pd.read_csv(indiv_file)

            # Increment the counters
            house_df["HID"] = house_df["HID"].apply(lambda x: x+HID_counter)
            house_df["HRPID"] = house_df["HRPID"].apply(lambda x: x+PID_counter) # Also increase the HRP

            indiv_df["PID"] = indiv_df["PID"].apply(lambda x: x+PID_counter)
            indiv_df["HID"] = indiv_df["HID"].apply(lambda x: x+HID_counter) # Also increase the link to HID

            HID_counter = max(house_df["HID"]) + 1 # Want next counter to start at one larger than current
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
    def read_retail_flows_data(cls) -> (pd.DataFrame, pd.DataFrame):
        """
        Read the flows between each MSOA and the most commonly visited shops

        :return: A tuple of two dataframes. One containing all of the flows and another
        containing information about the stores themselves.
        """
        dir = os.path.join(cls.DATA_DIR, "temp-retail")

        # Read the stores
        stores = pd.read_csv(os.path.join(dir, "devon smkt.csv"))
        stores['ID'] = list(stores.index + 1)  # Mark counts from 1, not zero, so indices need to start from 1
        stores = stores.set_index("ID")

        # Read the flows
        rows = []  # Build up all the rows in the matrix gradually then add all at once
        with open(os.path.join(dir, "DJR002.TXT")) as f:
            count = 1  # Mark's file comes in batches of 3 lines, each giving different data

            # See the README for info about these variables. This is only tempoarary so I can't be bothered
            # to explain properly
            oa = None
            num_dests = None
            dests = None
            flows = None

            for lineno, raw_line in enumerate(f):
                # print(f"{lineno}: '{raw_line}'")
                line_list = raw_line.strip().split()
                if count == 1:  # OA and number of destinations
                    oa = int(line_list[0])
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

                    # Have read all information for this area. Store the info in the flows matrix

                    # We should have one line in the matrix for each OA, and OA codes are incremental
                    # assert len(flow_matrix) == oa - 1
                    row = [0.0 for _ in range(len(stores))]  # Initially assume all flows are 0
                    for i in range(num_dests):  # Now add the flows
                        dest = dests[i]
                        flow = flows[i]
                        row[dest - 1] = flow  # (-1 because destinations are numbered from 1, not 0)
                    assert len([x for x in row if x > 0]) == num_dests  # There should only be N >0 flows
                    row = [oa, "NA"] + row  # Insert the OA number and code (don't know this yet now)

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

        return stores, flow_matrix

    @classmethod
    def add_individual_flows(cls, flow_type: str, individuals: pd.DataFrame, flow_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Take a flow matrix from MSOAs to (e.g. retail) locations and assign flows to individuals
        :param flow_type: What type of flows are these. This will be appended to the column names. E.g. "Retail".
        :param individuals: The DataFrame contining information about all individuals
        :param flow_matrix: The flow matrix, created by (e.g.) read_retail_flows_data()
        :return: The DataFrame of individuals with new locations and probabilities added
        """
        print("TEMP adding individual flows")
        # TODO First need to go back to read_retail_flows_data and include the MSOA code and an AREA_ID

        for area in tqdm(flow_matrix.values): # Easier to operate over a 2D matrix rather than a dataframe
            oa_num:int = area[0]
            oa_code:str = area[1]
            # Get rid of the area codes, so are now just left with flows to locations
            area = list(area[2:])
            # Destinations with positive flows and the flows themselves
            dests = []
            flows = []
            for i, flow in enumerate(area):
                if flow > 0.0:
                    dests.append(i)
                    flows.append(flow)

            # Create empty lists to hold the vanues and flows for each individuals
            individuals[f"{flow_type}_Venues"] = [ [] for _ in range(len(individuals)) ]
            individuals[f"{flow_type}_Probabilities"] = [ [] for _ in range(len(individuals)) ]

            # Now assign individuals in those areas to those flows
            # This ridiculous 'apply' line is the only way I could get pandas to update the particular
            # rows required. Something like 'individuals.loc[ ...] = dests' (see below) didn't work becuase
            # instead of inserting the 'dests' list itself, pandas tried to unpack the list and insert
            # the individual values instead.
            #individuals.loc[individuals.Area == oa_code, f"{flow_type}_Venues"] = dests
            #individuals.loc[individuals.Area == oa_code, f"{flow_type}_Probabilities"] = flow

            # TODO make this quicker by creating a AreaNumber for easier lookup OR make the Area part of the index (?)

            # Use a hierarchical index on the Area to speed up finding all individuals in an area (?)
            individuals.set_index(["Area", "PID"], inplace=True, drop=False)

            individuals.loc["E02004189", f"{flow_type}_Venues"] = \
               individuals.loc["E02004189", f"{flow_type}_Venues"].apply(lambda _: dests).values
            individuals.loc["E02004189", f"{flow_type}_Probabilities"] = \
                individuals.loc["E02004189", f"{flow_type}_Probabilities"].apply(lambda _: flows).values
            #individuals.loc[individuals.Area=="E02004189", f"{flow_type}_Venues"] = \
            #    individuals.loc[individuals.Area=="E02004189", f"{flow_type}_Venues"].apply(lambda _: dests)
            #individuals.loc[individuals.Area=="E02004189", f"{flow_type}_Probabilities"] = \
            #    individuals.loc[individuals.Area=="E02004189", f"{flow_type}_Probabilities"].apply(lambda _: flows)

        print("HERE")


    def step(self) -> None:
        """Step (iterate) the model"""
        self.iteration += 1
        print(f"Iteration: {self.iteration}")
        # XXXX HERE


# PROGRAM ENTRY POINT
# Uses 'click' library so that it can be run from the command line
@click.command()
@click.option('--iterations', default=10, help='Number of model iterations')
def run(iterations):
    num_iter = iterations
    m = Microsim()
    for i in range(num_iter):
        m.step()
    print("End of program")


if __name__ == "__main__":
    run()
    print("End of program")

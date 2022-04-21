#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Used to create data that can be used to parameterise the disease estimates. These data
need to be created so that the coefficients can be calculated.

Created on Wed 39 May 14:30

@author: nick
"""
import sys
sys.path.append("microsim")
import click  # command-line interface
import os
import pandas as pd
import random
import copy
import multiprocessing

from microsim.microsim_model import Microsim
from microsim.column_names import ColumnNames


class MicrosimInit(Microsim):
    """
    Create some preliminary data that are needed to estimate coefficients for the disease estimates.

    Process is as follows:
      1. Read a list of the most high risk MSOAs per day for N (=6?) days
      2. Read the total number of incidences per day.
      3. Then iterate for N days:
         - At the start of each day, randomly distribute cases to randomly chosen individuals to
         the high-risk MSOAs
         - Run step the model to calculate a danger score
         - Save the danger score for each individual each day

    Also repeat the above process M times.
    """

    def __init__(self, msoa_danger: pd.DataFrame, cases: pd.DataFrame, results_dir: str,
                 *args, **kwargs):
        """
        Initialise the MicrosimInit. Some specific parameters are required for initialisation,
        everything else is passed to the parent constructor

        :param msoa_danger: A list of MSOAs with an associated danger rating 'high', 'medium', 'low'
        :param cases: Number of cases per iteration.
        :param results_dir: The root results directory (actual results from this model will be written
        into a unique subdirectory

        """
        super(MicrosimInit, self).__init__(*args, **kwargs)
        self.msoa_danger = msoa_danger
        self.results_dir = results_dir
        self.cases = cases
        # Work out which areas and individuals are high risk (might be assigned to be a case)
        self.high_risk_msoas = self.msoa_danger.loc[self.msoa_danger.risk == "High", "area"].values
        # Now get the indices of the high risk individuals (all those who live in high risk areaa)
        self.high_risk_individuals = self.individuals.index[
                                     self.individuals["Area"].isin(self.high_risk_msoas)]
        assert len(self.individuals.loc[self.high_risk_individuals, :]["Area"].unique()) == len(self.high_risk_msoas)

    @staticmethod
    def run(m: Microsim, results_subdirectory: str):
        """
        Run the initialisation for a model.

        :param m: A microsim object run
        :param results_subdirectory: Where to write the results for this model
        :return:
        """
        # Create a directory for these results
        try:
            os.makedirs(results_subdirectory)
        except FileExistsError as e:
            print("Directory ", results_subdirectory, " already exists")
            raise e

        # Maintain a new dataframe that has the risk per individual for each iteration
        individual_risks = pd.DataFrame(m.individuals.loc[:, ["Area", ColumnNames.CURRENT_RISK] ])
        # Do the same for each activity
        activity_dangers = {}
        for name, activity_location in m.activity_locations.items():
            activity_dangers[name] = activity_location._locations.copy()

        if len(m.cases) > 100:
            raise Exception("Internal error. If there are more than 100 days of cases then the format"
                            "line below which is used to create a new column for each day as a two-digit "
                            "number, will need to be adapted to make 3-digit numbers.")

        # Loop for every iteration (get this from cases)
        for i, row in m.cases.iterrows():
            print(i, row['date'], row['new_cases'])
            # Reset everyone's disease status
            m.individuals.loc[:,ColumnNames.DISEASE_STATUS] = 0

            # Manually change people's activity durations after lockdown
            if i > 39:  # After day 39 - March 23RD in new cases
                total_duration = 0.0
                for colum_name in ['Retail', 'PrimarySchool', 'SecondarySchool', 'Work']:
                    new_duration = m.individuals.loc[:, colum_name+ ColumnNames.ACTIVITY_DURATION] * 0.33
                    # Round the new duration to prevent tiny numbers
                    new_duration = round(new_duration, 10)
                    total_duration += new_duration
                    m.individuals.loc[:, colum_name + ColumnNames.ACTIVITY_DURATION] = new_duration

                # Now set home
                m.individuals.loc[:, 'Home'+ ColumnNames.ACTIVITY_DURATION] = (1 - total_duration)

                # If you want to loop over all activities this is how you do it:
                #for name, activity_location in m.activity_locations.items():
                    # Reduce the duration of all activites by 0.9:
                    #m.individuals.loc[ : , name+ColumnNames.ACTIVITY_DURATION  ]  =
                        #m.individuals.loc[ : , name+ColumnNames.ACTIVITY_DURATION  ] * 0.9

            #  Randomly assign cases to individuals
            num_cases = row['new_cases']
            random.seed()  # Sometimes different Processes can be given the same generator and seed
            infected_individuals = random.sample(list(m.high_risk_individuals), num_cases)
            assert len(infected_individuals) == num_cases
            m.individuals.loc[infected_individuals, ColumnNames.DISEASE_STATUS] = 1
            assert len(m.individuals.loc[m.individuals[ColumnNames.DISEASE_STATUS] == 1]) == num_cases
            # Aggregate and count the cases per area and check they are the correct numbers of cases
            assert m.individuals.loc[m.individuals.Disease_Status > 0, :]["Area"].value_counts().sum() == num_cases

            # Step the model
            m.step()

            # Write out some of the info about individuals as a csv (verbose format string is to make the iteration two-digit)
            #self.individuals.to_csv(os.path.join(results_subdirectory,"individuals-{0:0=2d}.csv".format(i)))

            # Remember the information about the activity locations
            for name, activity_location in m.activity_locations.items():
                activity_dangers[name][ColumnNames.LOCATION_DANGER+"{0:0=2d}".format(i)] = \
                    activity_location._locations[ColumnNames.LOCATION_DANGER]

            # Remember the current risk per individual per day
            individual_risks[ColumnNames.CURRENT_RISK+"{0:0=2d}".format(i)] = m.individuals.loc[:, ColumnNames.CURRENT_RISK]

        # Write out the risks per individual each day
        individual_risks.to_csv(os.path.join(results_subdirectory, "individual_risks.csv"))
        # And the locations dangers per day
        for name, activity_df in activity_dangers.items():
            activity_df.to_csv(os.path.join(results_subdirectory, f"{name}.csv"))

        return individual_risks

    @staticmethod
    def make_a_copy(m: Microsim):
        """When copying a microsim object, reset the seed"""
        m.random.seed()
        return copy.deepcopy(m)


# PROGRAM ENTRY POINT
# Uses 'click' library so that it can be run from the command line
@click.command()
@click.option('--repetitions', default=5, help='Number of times to repeat the initialisation process')
@click.option('--data_dir', default="devon_data", help='Root directory to load main model data from')
@click.option('--init_dir', default="init_data", help="Directory that stores initialisation data, and where outputs are written")
@click.option('--multiprocess', default=False, help="Whether to run multiprocess or not")
@click.option('--debug', default=False, help="Whether to run some more expensive checks (default True)")
def run_script(repetitions, data_dir, init_dir, multiprocess, debug):
    # To fix file path issues, use absolute/full path at all times
    base_dir = os.getcwd()  # get current directory
    data_dir = os.path.join(base_dir, data_dir)
    init_dir = os.path.join(base_dir, init_dir)
    results_dir = os.path.join(init_dir, "results/")

    print(f"Running initialisation process {repetitions} times.\n\t"
          f"Using model data in {data_dir}, initialisation data in {init_dir}\n\t"
          f"Storing results in {results_dir}.")

    # Read the initialisation data
    cases = pd.read_csv(os.path.join(init_dir, "devon_cases_fn.csv"))
    # Cut off cases after 10
    #cases = cases.loc[0:10, :]
    msoa_danger = pd.read_csv(os.path.join(init_dir, "msoa_danger_fn.csv"))

    # Sense check: all MSOAs in Devon should have a danger score
    devon_msoas = pd.read_csv(os.path.join(data_dir, "devon_msoas.csv"), header=None,
                              names=["x", "y", "Num", "Code", "Desc"])
    assert len(devon_msoas) == len(msoa_danger)
    assert set(devon_msoas.Code) == set(msoa_danger.area)

    # Initialise a model (MirosimInit init is a child of Microsim)
    m = MicrosimInit(msoa_danger=msoa_danger, cases=cases, results_dir=results_dir,
                     # These are for the parent Microsim object
                     study_msoas=list(devon_msoas.Code), data_dir=data_dir, output=False, debug=debug,
                     # TODO Since implementing this we have the disease status stuff working. Check that deactivating it works.
                     disable_disease_status=True # Turn off disease status calculation as we want to run it without this
                     )

    # Find a new directory for this initialisation (may have old ones)
    i = 0
    while os.path.exists(os.path.join(results_dir, str(i))):
        i += 1
    # Create a directory for these results
    results_subdir = os.path.join(results_dir, str(i) )
    try:
        os.mkdir(results_subdir)
    except FileExistsError as e:
        print("Directory ", results_subdir, " already exists")
        raise e

    # Write out the individuals as they were before the model(s) started iterating
    m.individuals.to_csv(os.path.join(results_subdir, "individuals.csv"))

    # For each repetition, remember risks of each individual and dangers of each location
    individual_risks = []
    if repetitions <= 1 or not multiprocess: # Run in a single process
        for j in range(0, repetitions):
            # Now find subdirectories for each individual run
            subdir = os.path.join(results_subdir, str(j))
            # Run it, writing out data, and returing the risks per day.
            # Copy the model initialisation instance each time (although maybe this not strictly necessary
            individual_risks.append(
                MicrosimInit.run(MicrosimInit.make_a_copy(m), subdir))
    else:  # Run as multiprocess
        subdirs = [ os.path.join(results_subdir, str(j)) for j in range(repetitions) ]
        models = (MicrosimInit.make_a_copy(m) for _ in range(repetitions))  # Generator so dont need to do all copies at once
        with multiprocessing.Pool(processes=int(os.cpu_count()) ) as pool:
            try:
                #individual_risks = pool.map(MicrosimInit.run, zip(models, subdirs))
                individual_risks = pool.starmap(MicrosimInit.run, zip(models, subdirs))
            finally:
                pool.close()

    print("End of initialisation")

if __name__ == "__main__":
    run_script()
    print("End of initialisation")


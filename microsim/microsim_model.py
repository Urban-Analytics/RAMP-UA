import sys

sys.path.append("microsim")  # This is only needed when testing. I'm so confused about the imports
from microsim.r_interface import RInterface
from microsim.column_names import ColumnNames
from microsim.utilities import check_durations_sum_to_1
import pandas as pd
pd.set_option('display.expand_frame_repr', False)  # Don't wrap lines when displaying DataFrames
# pd.set_option('display.width', 0)  # Automatically find the best width

import os
import time
from typing import List, Dict
import pickle
import copy
import random


class Microsim:
    """
    Class containing code for running timesteps of the Python/ R microsim model.
    This operates on two main dataframes: individuals and activity_locations.
    """
    def __init__(self,
                 individuals,
                 activity_locations,
                 time_activity_multiplier=None,
                 random_seed: float = None,
                 disable_disease_status=False,
                 r_script_dir: str = "./R/py_int/",
                 data_dir: str = "./data/",
                 scen_dir: str = "default",
                 output: bool = True,
                 output_every_iteration=False,
                 hazard_individual_multipliers: Dict[str, float] = {},
                 hazard_location_multipliers: Dict[str, float] = {},
                 risk_multiplier: float = 1.0,
                 disease_params: Dict = {}
                 ):
        """
        PopulationInitialisation constructor. This reads all of the necessary data to run the microsimulation.
        ----------
        :param individuals: dataframe of population data
        :param activity_locations: dataframe of location data
        :param time_activity_multiplier: activity multipliers based on lockdown data
        :param random_seed: A optional random seed to use when creating the class instance. This is passed
          directly to `random.Random()` (including if None).
        :param disable_disease_status: Optionally turn off the R interface. This will mean we cannot calculate new
            disease status. Only good for testing.
        :param r_script_dir: A directory with the required R scripts in (these are used to estimate disease status)
        :param scen_dir: A data directory to write the output to (i.e. a name for this model run)
        :param data_dir: A data directory from which to read the source data
        :param output: Whether to create files to store the results (default True)
        :param output_every_iteration: Whether to create files to store the results at every iteration rather than
            just at the end (default False)
        :param hazard_individual_multipliers: A dictionary containing multipliers that can make particular disease
            statuses more hazardous to the locations that the people visit. See 'hazard_individual_multipliers' section
            of the parameters file (e.g. model_parameters/default.yml). Default is {} which means the multiplier will
            be 1.0
        :hazard_location_multipliers: A dictionary containing multipliers that can make certain locations
            more hazardous (e.g. easier to get the disease if you visit). See 'hazard_location_multipliers' section
            of the parameters file (e.g. model_parameters/default.yml). Default is {} which means the multiplier will
            be 1.0
        :param disease_params: Optional parameters that are passed to the R code that estimates disease status
            (a dictionary, assumed to be empty)
        """
        self.individuals = individuals
        self.activity_locations = activity_locations
        self.random = random.Random(random_seed)
        self.disable_disease_status = disable_disease_status
        self.r_script_dir = r_script_dir
        self.output = output
        self.output_every_iteration = output_every_iteration

        Microsim.DATA_DIR = data_dir  # TODO (minor) pass the data_dir to class functions directly so no need to have it defined at class level
        self.DATA_DIR = data_dir
        self.SCEN_DIR = scen_dir

        # create full path for scenario dir, also check if scenario dir exists and if so, add nr
        self.SCEN_DIR = self._find_new_directory(os.path.join(self.DATA_DIR, "output"), self.SCEN_DIR)

        # We need an interface to R code to calculate disease status, but don't initialise it until the run()
        # method is called so that the R process is initiatied in the same process as the Microsim object
        self.r_int = None

        # If we do output information, then it will go to this directory. This is determined in run(), rather than
        # here, because otherwise all copies of this object will have the same output directory.
        self.output_dir = None

        self.iteration = 0
        self.time_activity_multiplier = time_activity_multiplier

        self.risk_multiplier = risk_multiplier

        self.hazard_individual_multipliers = hazard_individual_multipliers
        Microsim.__check_hazard_location_multipliers(hazard_location_multipliers)
        self.hazard_location_multipliers = hazard_location_multipliers

        self.disease_params = disease_params

        self.repnr = -1  # This is a unique ID for the model used if this model is run as part of an ensemble

    def run(self, iterations: int, repnr: int) -> None:
        """
        Run the model (call the step() function) for the given number of iterations.
        :param iterations: The number of iterations to run
        :param repnr: The repetition number of this model. Like an ID. Used to create new unique directory for this model instance.
        """
        # Now that this model is being run we know it's ID (repetition number)
        assert self.repnr == -1  # The ID should not have been set yet
        self.repnr = repnr

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
                print("\tPreparing output ... " +
                      "(but not writing anything until the end)" if not self.output_every_iteration else "", )
                # Add the new column names for this iteration's disease statuses
                # (Force column names to have leading zeros)
                self.individuals_to_pickle[f"{ColumnNames.DISEASE_STATUS}{(i + 1):03d}"] = self.individuals[
                    ColumnNames.DISEASE_STATUS]
                # Write the output at the end, or at every iteration
                if i == (iterations - 1) or self.output_every_iteration:
                    print("\t\tWriting individuals file... ")
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
                    if i == (iterations - 1) or self.output_every_iteration:
                        print(f"\t\tWriting activity file for {name}... ")
                        fname = os.path.join(self.output_dir, loc_name)
                        with open(fname + ".pickle", "wb") as pickle_out:
                            pickle.dump(self.activities_to_pickle[loc_name], pickle_out)
                        # Also make a (compressed) csv file for others
                        self.activities_to_pickle[loc_name].to_csv(fname + ".csv.gz", compression='gzip')
                        # self.activities_to_pickle[loc_name].to_csv(fname+".csv")  # They not so big so don't compress

            print(f"\tIteration {i} took {round(float(time.time() - iter_start_time), 2)}s")

        print(f"Model finished running (iterations: {iterations})")

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
        # self.update_disease_counts()

        # Calculate new disease status and update the people's behaviour
        if not self.disable_disease_status:
            self.calculate_new_disease_status()
            self.change_behaviour_with_disease()

        # TEMP WRITE OUTPUT AT END
        # fname = os.path.join(self.output_dir, "Individuals")
        # with open(fname + ".pickle", "wb") as pickle_out:
        #    pickle.dump(self.individuals_to_pickle, pickle_out)
        # self.individuals_to_pickle.to_csv(fname + ".csv.gz", compression='gzip')
        # for name in self.activity_locations:
        #    loc_name = self.activity_locations[name].get_name()
        #    fname = os.path.join(self.output_dir, loc_name)
        #    with open(fname + ".pickle", "wb") as pickle_out:
        #        pickle.dump(self.activities_to_pickle[loc_name], pickle_out)
        #    # Also make a (compressed) csv file for others
        #    self.activities_to_pickle[loc_name].to_csv(fname + ".csv.gz", compression='gzip')

    def update_behaviour_during_lockdown(self):
        """
        Unilaterally alter the proportions of time spent on different activities before and after 'lockddown'
        Otherwise this doesn't do anything update_behaviour_during_lockdown.

        Note: ignores people who are currently showing symptoms (`ColumnNames.DiseaseStatus.SYMPTOMATIC`)
        """
        # Are we doing any lockdown at all? in this iteration?
        if self.time_activity_multiplier is not None:
            # Only change the behaviour of people who aren't showing symptoms. If you are showing symptoms then you
            # will be mostly at home anyway, so don't want your behaviour overridden by lockdown.
            uninfected = self.individuals.index[
                self.individuals[ColumnNames.DISEASE_STATUS] != ColumnNames.DiseaseStatuses.SYMPTOMATIC]
            if len(uninfected) < len(self.individuals):
                print(f"\t{len(self.individuals) - len(uninfected)} people are symptomatic so not affected by lockdown")

            # Reduce all activities, replacing the lost time with time spent at home
            non_home_activities = set(self.activity_locations.keys())
            non_home_activities.remove(ColumnNames.Activities.HOME)
            # Need to remember the total duration of time lost for non-home activities
            total_duration = pd.Series(data=[0.0] * len(self.individuals.loc[uninfected]), name="TotalDuration")

            # Reduce the initial activity proportion of time by a particular amount per day
            timeout_multiplier = self.time_activity_multiplier.loc[
                self.time_activity_multiplier.day == self.iteration, "timeout_multiplier"].values[0]
            print(f"\tApplying regular (google mobility) lockdown multiplier {timeout_multiplier}")
            for activity in non_home_activities:
                # Need to be careful with new_duration because we don't want to keep the index used in
                # self.individuals as this will be missing out people who aren't infected so will have gaps
                new_duration = pd.Series(list(self.individuals.loc[
                                                  uninfected, activity + ColumnNames.ACTIVITY_DURATION_INITIAL] * timeout_multiplier),
                                         name="NewDuration")
                total_duration += new_duration
                self.individuals.loc[uninfected, activity + ColumnNames.ACTIVITY_DURATION] = list(new_duration)

            assert (total_duration <= 1.0).all() and (new_duration <= 1.0).all()
            # Now set home duration to fill in the time lost from doing other activities.
            self.individuals.loc[uninfected, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION}"] = list(
                1 - total_duration)

            # Check they still sum correctly (if not then they probably need rounding)
            # (If you want to print the durations)
            # self.individuals.loc[:, [ x+ColumnNames.ACTIVITY_DURATION for x in self.activity_locations.keys() ]+
            #     [ x+ColumnNames.ACTIVITY_DURATION_INITIAL for x in self.activity_locations.keys()   ]   ]

            check_durations_sum_to_1(self.individuals, self.activity_locations.keys())
        else:
            print("\tNot applying a lockdown multiplier")

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
                    # There may be different multipliers passed in a dictionary as calibration parameters, if not
                    # then assume the multiplier is 1.0
                    individual_hazard_multiplier: float = None
                    if not self.hazard_individual_multipliers:  # The dictionary is empty
                        individual_hazard_multiplier = 1.0
                    else:  # A dict was passed, so find out what the values of the multiplier are by disease status
                        if s == ColumnNames.DiseaseStatuses.PRESYMPTOMATIC:
                            individual_hazard_multiplier = self.hazard_individual_multipliers['presymptomatic']
                        elif s == ColumnNames.DiseaseStatuses.SYMPTOMATIC:
                            individual_hazard_multiplier = self.hazard_individual_multipliers['symptomatic']
                        elif s == ColumnNames.DiseaseStatuses.ASYMPTOMATIC:
                            individual_hazard_multiplier = self.hazard_individual_multipliers['asymptomatic']
                    assert individual_hazard_multiplier is not None
                    # There may also be a hazard multiplier for locations (i.e. some locations become more hazardous
                    # than others
                    location_hazard_multiplier = None
                    if not self.hazard_location_multipliers:  # The dictionary is empty
                        location_hazard_multiplier = 1.0
                    else:
                        location_hazard_multiplier = self.hazard_location_multipliers[activty_name]
                    assert location_hazard_multiplier is not None

                    # v and f are lists of flows and venues for the individual. Go through each one
                    for venue_idx, flow in zip(v, f):
                        # print(i, venue_idx, flow, duration)
                        # Increase the danger by the flow multiplied by some disease risk
                        danger_increase = (flow * duration * individual_hazard_multiplier * location_hazard_multiplier)
                        # if activty_name == ColumnNames.Activities.WORK:
                        #    warnings.warn("Temporarily reduce danger for work while we have virtual work locations")
                        #    work_danger = float(danger_increase / 20)
                        #    loc_dangers[venue_idx] += work_danger
                        # else:
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
        # if PopulationInitialisation.debug:  # replace with .debug
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
    # def update_disease_counts(self):
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
        self.individuals = self.r_int.calculate_disease_status(
            self.individuals, self.iteration, self.repnr, self.disease_params)

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
                change[(old, new)] += 1

        assert sum(change.values()) == len(new_status[new_status != old_status])

        print(f"\t{len(new_status[new_status != old_status])} individuals have a different status. Status changes:")
        for old in ColumnNames.DiseaseStatuses.ALL:
            print(f"\t\t{old} -> ", end="")
            for new in ColumnNames.DiseaseStatuses.ALL:
                print(f" {new}:{change[(old, new)]} \t", end="")
            print()

    def change_behaviour_with_disease(self) -> None:
        """
        When people have the disease the proportions that they spend doing activities changes. This function applies
        those changes inline to the individuals dataframe
        :return: None. Update the dataframe inplace
        """
        # print("Changing behaviour of infected individuals ... ",)
        # Find out which people have changed status
        change_idx = self.individuals.index[self.individuals[ColumnNames.DISEASE_STATUS_CHANGED] == True]

        # Now set their new behaviour
        self.individuals.loc[change_idx] = \
            self.individuals.loc[change_idx].apply(
                func=Microsim._set_new_behaviour, args=(list(self.activity_locations.keys()),), axis=1)
        # self.individuals.loc[change_idx].swifter.progress_bar(True, desc="Changing behaviour of infected"). \

        print(f"\tCurrent statuses:"
              f"\n\t\tSusceptible ({ColumnNames.DiseaseStatuses.SUSCEPTIBLE}): {len(self.individuals.loc[self.individuals[ColumnNames.DISEASE_STATUS] == ColumnNames.DiseaseStatuses.SUSCEPTIBLE])}"
              f"\n\t\tExposed ({ColumnNames.DiseaseStatuses.EXPOSED}): {len(self.individuals.loc[self.individuals[ColumnNames.DISEASE_STATUS] == ColumnNames.DiseaseStatuses.EXPOSED])}"
              f"\n\t\tPresymptomatic ({ColumnNames.DiseaseStatuses.PRESYMPTOMATIC}): {len(self.individuals.loc[self.individuals[ColumnNames.DISEASE_STATUS] == ColumnNames.DiseaseStatuses.PRESYMPTOMATIC])}"
              f"\n\t\tSymptomatic ({ColumnNames.DiseaseStatuses.SYMPTOMATIC}): {len(self.individuals.loc[self.individuals[ColumnNames.DISEASE_STATUS] == ColumnNames.DiseaseStatuses.SYMPTOMATIC])}"
              f"\n\t\tAsymptomatic ({ColumnNames.DiseaseStatuses.ASYMPTOMATIC}): {len(self.individuals.loc[self.individuals[ColumnNames.DISEASE_STATUS] == ColumnNames.DiseaseStatuses.ASYMPTOMATIC])}"
              f"\n\t\tRecovered ({ColumnNames.DiseaseStatuses.RECOVERED}): {len(self.individuals.loc[self.individuals[ColumnNames.DISEASE_STATUS] == ColumnNames.DiseaseStatuses.RECOVERED])}"
              f"\n\t\tRemoved/dead ({ColumnNames.DiseaseStatuses.DEAD}): {len(self.individuals.loc[self.individuals[ColumnNames.DISEASE_STATUS] == ColumnNames.DiseaseStatuses.DEAD])}")

        # self.individuals.loc[change_idx].apply(func=self._set_new_behaviour, axis=1)
        # print("... finished")

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
            non_home_activities.remove(ColumnNames.Activities.HOME)
            total_duration = 0.0  # Need to remember the total duration of time lost for non-home activities
            for activity in non_home_activities:
                # new_duration = row[f"{activity}{ColumnNames.ACTIVITY_DURATION}"] * 0.10
                new_duration = row[f"{activity}{ColumnNames.ACTIVITY_DURATION}"] * 0.10
                total_duration += new_duration
                row[f"{activity}{ColumnNames.ACTIVITY_DURATION}"] = new_duration
            # Now set home duration to fill in the time lost from doing other activities.
            row[f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION}"] = (1 - total_duration)
        else:
            raise Exception(f"Unrecognised disease state for individual {row['ID']}: {row[ColumnNames.DISEASE_STATUS]}")
        return row

    def _init_output(self):
        """
        Might need to write out some data if saving output for analysis later. If so, creates a new directory for the
        results as a subdirectory of the data directory.
        Also store some information for use in the visualisations and analysis.
        """
        assert self.repnr >= 0  # If -1 then the repetition number has not been initialised correctly
        if self.output:
            print("Saving initial models for analysis ... ", )
            # Find a directory to use, within the 'output' directory
            #            if repnr == 0:
            #                self.SCEN_DIR = PopulationInitialisation._find_new_directory(os.path.join(self.DATA_DIR, "output"), self.SCEN_DIR)
            self.output_dir = os.path.join(self.SCEN_DIR, str(self.repnr))
            os.mkdir(self.output_dir)

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
                self.activities_to_pickle[loc_name] = activity.get_dataframe_copy()
                self.activities_to_pickle[loc_name]['Danger0'] = loc_dangers
                assert False not in list(loc_ids == self.activities_to_pickle[loc_name]["ID"].values)

    @staticmethod
    def __check_hazard_location_multipliers(hazard_location_multipliers):
        """Check that the hazard multipliation multipliers correspond to the actual locations being used
        in the model (we don't want a hazard for a location that doesn't exist, and need all locations
        to have hazards). If there are no hazard location multipliers (an empty dict) then we're not
        using them so just return

        :return None if all is OK. Throw an exception if not
        :raise Exception if the multipliers don't align."""
        if not hazard_location_multipliers:
            return
        hazard_activities = set(hazard_location_multipliers.keys())
        all_activities = set(ColumnNames.Activities.ALL)
        if hazard_activities != all_activities:
            raise Exception(f"The hzard location multipliers: '{hazard_activities} don't match the "
                            f"activities in the model: {all_activities}")

    @staticmethod
    def _find_new_directory(dir, scendir):
        """
        Find a new directory and make one to store results in starting from 'dir'.
        :param dir: Start looking from this directory
        :return: The new directory (full path)
        """
        results_subdir = os.path.join(dir, scendir)
        # if it exists already, try adding numbers
        i = 0
        while os.path.isdir(results_subdir):
            i += 1
            newdir = scendir + "_" + str(i)
            results_subdir = os.path.join(dir, newdir)
        # Create a directory for these results
        # results_subdir = os.path.join(dir, str(i))
        try:
            os.mkdir(results_subdir)
        except FileExistsError as e:
            print("Directory ", results_subdir, " already exists")
            raise e
        return results_subdir

    @staticmethod
    def _make_a_copy(m):
        """
        When copying a microsim object, reset the seed

        :param m: A Microsim object
        :return: A deep copy of the microsim object
        """
        m.random.seed()
        return copy.deepcopy(m)

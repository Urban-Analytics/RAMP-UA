import pandas as pd
import numpy as np
import os


class InitialCases:
    def __init__(self, area_codes, not_home_probs, data_dir="microsim/opencl/data/"):
        """
        This class loads the initial cases data for seeding infections in the model.
        Once the data is loaded it selects the people from higher risk area codes who
        spend more time outside of their home.

        :param area_codes: Area codes, from the dataframe of individuals
        :param not_home_probs: `pnothome` column from the dataframe of individuals.
            (Basically the amount of time per day people are expected not to be at home).
        """

        # load initial case data
        self.initial_cases = pd.read_csv(os.path.join(data_dir, "devon_initial_cases.csv"))

        msoa_risks_df = pd.read_csv(os.path.join(data_dir, "msoas.csv"), usecols=[1, 2])

        # combine into a single dataframe to allow easy filtering based on high risk area codes and
        # not home probabilities
        people_df = pd.DataFrame({"area_code": area_codes,
                                  "not_home_prob": not_home_probs})
        people_df = people_df.merge(msoa_risks_df, on="area_code")

        # get people_ids for people in high risk MSOAs and high not home probability
        self.high_risk_ids = np.where((people_df["risk"] == "High") & (people_df["not_home_prob"] > 0.3))[0]

    def get_seed_people_ids_for_day(self, day):
        """Randomly choose a given number of people ids from the high risk people"""
        
        num_cases = self.initial_cases.loc[day, "num_cases"]
        if num_cases > self.high_risk_ids.shape[0]:  # if there aren't enough high risk individuals then return all of them
            return self.high_risk_ids

        selected_ids = np.random.choice(self.high_risk_ids, num_cases, replace=False)

        # remove people from high_risk_ids so they are not chosen again
        self.high_risk_ids = np.setdiff1d(self.high_risk_ids, selected_ids)

        return selected_ids

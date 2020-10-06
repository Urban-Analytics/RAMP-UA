import pandas as pd
import os
import pickle


class InitialisationCache:
    """
    Class to handle caching of initialisation data, eg. individuals and activity locations dataframes
    """
    def __init__(self, cache_dir="./temp_cache"):
        self.cache_dir = cache_dir
        self.individuals_filepath = self.cache_dir + "individuals.pkl"
        self.activity_locations_filepath = self.cache_dir + "activity_locations.pkl"
        self.time_activity_multiplier_filepath = self.cache_dir + "time_activity_multiplier.pkl"
        self.all_cache_filepaths = [self.individuals_filepath, self.activity_locations_filepath,
                                    self.time_activity_multiplier_filepath]

    def store_in_cache(self, individuals, activity_locations, time_activity_multiplier):
        individuals.to_pickle(self.individuals_filepath)
        with open(self.activity_locations_filepath, 'wb') as handle:
            pickle.dump(activity_locations, handle)
        time_activity_multiplier.to_pickle(self.time_activity_multiplier_filepath)

    def read_from_cache(self):
        if self.cache_files_exist():
            individuals = pd.read_pickle(self.individuals_filepath)
            with open(self.time_activity_multiplier_filepath, 'rb') as handle:
                activity_locations = pickle.load(handle)
            time_activity_multiplier = pd.read_pickle(self.time_activity_multiplier_filepath)
            return individuals, activity_locations, time_activity_multiplier
        else:
            print("\nWARNING: attempting to load files from cache but they do not exist!")

    def cache_files_exist(self):
        files_exist = [os.path.exists(cache_file) for cache_file in self.all_cache_filepaths]
        return all(files_exist)

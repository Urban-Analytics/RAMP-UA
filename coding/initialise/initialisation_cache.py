import pandas as pd
import os
import pickle


class InitialisationCache:
    """
    Class to handle caching of initialisation data, eg. individuals and activity locations dataframes
    """
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.individuals_filepath = os.path.join(self.cache_dir, "individuals.pkl")
        self.activity_locations_filepath = os.path.join(self.cache_dir, "activity_locations.pkl")
        self.all_cache_filepaths = [self.individuals_filepath, self.activity_locations_filepath]

    def store_in_cache(self, individuals, activity_locations):
        individuals.to_pickle(self.individuals_filepath)
        with open(self.activity_locations_filepath, 'wb') as handle:
            pickle.dump(activity_locations, handle)

    def read_from_cache(self):
        individuals = pd.read_pickle(self.individuals_filepath)
        with open(self.activity_locations_filepath, 'rb') as handle:
            activity_locations = pickle.load(handle)
        return individuals, activity_locations

    def cache_files_exist(self):
        files_exist = [os.path.exists(cache_file) for cache_file in self.all_cache_filepaths]
        return all(files_exist)

    def is_empty(self):
        return not self.cache_files_exist()

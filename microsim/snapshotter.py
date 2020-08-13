import numpy as np
import pandas as pd
import os
import pickle


class Snapshotter:
    """
    Take snapshot of internal model state before running the python model, so this "snapshot" of model
    state can be saved and transferred to the GPU version of the model
    """

    def __init__(self, individuals, activity_locations, snapshot_dir, cache_inputs=True):
        self.snapshot_dir = snapshot_dir

        if individuals is None:
            self.individuals = self.load_from_cache("individuals_cache.pkl", is_dataframe=True)
        else:
            self.individuals = individuals
            if cache_inputs:
                self.write_to_cache("individuals_cache.pkl", self.individuals, is_dataframe=True)

        if activity_locations is None:
            self.activity_names = self.load_from_cache("activity_names.pkl")
        else:
            self.activity_names = list(activity_locations.keys())
            if cache_inputs:
                self.write_to_cache("activity_names.pkl", self.activity_names, is_dataframe=False)

        self.locations = dict()
        for activity_name in self.activity_names:
            cache_filename = "activity_locations_" + activity_name + "_cache.pkl"
            if activity_locations is None:
                self.locations[activity_name] = self.load_from_cache(cache_filename, is_dataframe=True)
            else:
                self.locations[activity_name] = activity_locations[activity_name]._locations
                if cache_inputs:
                    self.write_to_cache(cache_filename, self.locations[activity_name], is_dataframe=True)

    def store_snapshots(self):
        self.store_people_snapshots()
        self.store_place_snapshots()

    def store_people_snapshots(self):
        num_people = self.individuals['ID'].count()
        ages = self.individuals['age']
        age_array = ages.to_numpy(dtype=np.uint16)
        filepath = os.path.join(self.snapshot_dir, 'people_ages.npz')
        print(f"Saving ages for {num_people} people to {filepath}")
        np.savez(filepath, age_array=age_array)

    def store_place_snapshots(self):
        pass

    def load_from_cache(self, cache_filename, is_dataframe=False):
        cache_filepath = os.path.join(self.snapshot_dir, cache_filename)
        if os.path.isfile(cache_filepath):
            print(f"Reading cached data from {cache_filepath}")
            if is_dataframe:
                return pd.read_pickle(cache_filepath)
            else:
                return pickle.load(open(cache_filepath, "rb"))
        else:
            print(f"WARNING: Could not load {cache_filepath} from cache, file does not exist")

    def write_to_cache(self, cache_filename, data, is_dataframe=False):
        cache_filepath = os.path.join(self.snapshot_dir, cache_filename)
        print(f"Writing cache data to {cache_filepath}")
        if is_dataframe:
            data.to_pickle(cache_filepath)
        else:
            pickle.dump(data, open(cache_filepath, "wb"))


def main():
    base_dir = os.getcwd()
    snapshot_dir = os.path.join(base_dir, "snapshots")
    snapshotter = Snapshotter(individuals=None, activity_locations=None, snapshot_dir=snapshot_dir)
    snapshotter.store_snapshots()


if __name__ == '__main__':
    main()

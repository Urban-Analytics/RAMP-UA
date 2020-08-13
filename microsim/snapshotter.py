import numpy as np
import pandas as pd
import os


class Snapshotter:
    """
    Take snapshot of internal model state before running the python model, so this "snapshot" of model
    state can be saved and transferred to the GPU version of the model
    """

    def __init__(self, individuals, activity_locations, snapshot_dir, use_cache=True):
        self.snapshot_dir = snapshot_dir

        if use_cache:
            self.individuals = self.cache_or_load(individuals, "individuals_cache.pkl")
            self.activity_locations = self.cache_or_load(activity_locations, "activity_locations_cache.pkl")
        else:
            self.individuals = individuals
            self.activity_locations = activity_locations

    def store_snapshots(self):
        self.store_people_snapshots()
        self.store_place_snapshots()

    def store_people_snapshots(self):
        num_people = self.individuals['ID'].count()
        ages = self.individuals['age']
        age_array = ages.to_numpy(dtype=np.uint16)
        filepath = os.path.join(self.snapshot_dir, 'people_ages.npz')
        print(f"Saving ages for {num_people} people to {filepath}")
        np.savez(filepath, age_array)

    def store_place_snapshots(self):
        pass

    def cache_or_load(self, data, cache_filename):
        cache_filepath = os.path.join(self.snapshot_dir, cache_filename)
        if data is None:
            print(f"Reading cached data from {cache_filepath}")
            return pd.read_pickle(cache_filepath)
        else:
            print(f"Writing cache data to {cache_filepath}")
            data.to_pickle(cache_filepath)
            return data


def main():
    base_dir = os.getcwd()
    snapshot_dir = os.path.join(base_dir, "snapshots")
    snapshotter = Snapshotter(individuals=None, activity_locations=None, snapshot_dir=snapshot_dir, use_cache=True)
    snapshotter.store_snapshots()


if __name__ == '__main__':
    main()

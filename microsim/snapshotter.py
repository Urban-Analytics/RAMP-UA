import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm


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

        self.num_people = self.individuals['ID'].count()
        self.global_place_id_lookup = self.create_global_place_ids()

    def store_snapshots(self):
        ages = self.get_people_ages()
        people_place_ids, people_flows = self.get_people_place_data()

        filepath = os.path.join(self.snapshot_dir, 'people_snapshot.npz')
        print(f"Saving data for {self.num_people} people to {filepath}")
        np.savez(filepath, ages=ages, people_place_ids=people_place_ids, people_baseline_flows=people_flows)

    def create_global_place_ids(self):
        max_id = 0
        global_place_id_lookup = dict()

        for activity_name in self.activity_names:
            activity_ids = self.locations[activity_name]['ID'].to_numpy(dtype=np.uint32)
            num_activity_ids = activity_ids.shape[0]
            starting_id = activity_ids[0]

            # check that activity IDs increase by one
            assert activity_ids[-1] == num_activity_ids - 1 + starting_id

            # subtract starting ID in case the first local activity ID is not zero
            global_ids = activity_ids + max_id - starting_id
            global_place_id_lookup[activity_name] = {
                "ids": global_ids,
                "id_offset": starting_id
            }
            max_id += num_activity_ids

        return global_place_id_lookup

    def get_global_place_id(self, activity_name, local_place_id):
        ids_for_activity = self.global_place_id_lookup[activity_name]
        global_id_location = local_place_id - ids_for_activity["id_offset"]
        return ids_for_activity["ids"][global_id_location]

    def get_people_ages(self):
        return self.individuals['age'].to_numpy(dtype=np.uint16)

    def get_people_place_data(self):
        max_places_per_person = 16
        people_place_ids = np.full((self.num_people, max_places_per_person), np.nan, dtype=np.uint32)
        people_flows = np.full((self.num_people, max_places_per_person), np.nan, dtype=np.float32)

        for people_id in tqdm(range(self.num_people), desc="Calculating place flows for all people"):
            person_global_place_ids = list()
            person_place_flows = list()

            person_row = self.individuals.loc[people_id]

            for activity_name in self.activity_names:
                activity_venue_col = activity_name + "_Venues"
                local_place_ids = np.array(person_row[activity_venue_col])

                activity_duration_col = activity_name + "_Duration"
                activity_duration = np.float32(person_row[activity_duration_col])

                activity_flow_col = activity_name + "_Flows"
                activity_flows = np.array(person_row[activity_flow_col])

                activity_flows = activity_flows * activity_duration

                # check dimensions match
                assert local_place_ids.shape[0] == activity_flows.shape[0]

                person_place_flows += list(activity_flows)

                for (i, local_place_id) in enumerate(local_place_ids):
                    person_global_place_ids.append(self.get_global_place_id(activity_name, local_place_id))

            person_place_data = np.array(list(zip(person_global_place_ids, person_place_flows)))

            # sort by flow value descending so we can take the n places with highest flows
            person_place_data = person_place_data[person_place_data[:, 1].argsort()[::-1]]
            person_place_data = person_place_data[:max_places_per_person]

            num_places = person_place_data.shape[0]
            people_place_ids[people_id][0:num_places] = person_place_data[:, 0]
            people_flows[people_id][0:num_places] = person_place_data[:, 1]

        return people_place_ids, people_flows

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

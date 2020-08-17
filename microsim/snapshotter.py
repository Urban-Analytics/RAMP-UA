import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from convertbng.util import convert_lonlat


class Snapshotter:
    """
    Take snapshot of internal model state before running the python model, so this "snapshot" of model
    state can be saved and transferred to the GPU version of the model
    """

    def __init__(self, individuals, activity_locations, snapshot_dir, cache_inputs=True):
        self.snapshot_dir = snapshot_dir

        # load individuals dataframe from cache
        if individuals is None:
            self.individuals = self.load_from_cache("individuals_cache.pkl", is_dataframe=True)
        else:
            self.individuals = individuals
            if cache_inputs:
                self.write_to_cache("individuals_cache.pkl", self.individuals, is_dataframe=True)

        # load names of activities from cache
        if activity_locations is None:
            self.activity_names = self.load_from_cache("activity_names.pkl")
        else:
            self.activity_names = list(activity_locations.keys())
            if cache_inputs:
                self.write_to_cache("activity_names.pkl", self.activity_names, is_dataframe=False)

        # load locations dataframe from cache
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
        self.global_place_id_lookup, self.num_places = self.create_global_place_ids()

    def store_snapshots(self):
        ages = self.get_people_ages()
        people_place_ids, people_flows = self.get_people_place_data()

        filepath = os.path.join(self.snapshot_dir, 'people_snapshot.npz')
        print(f"Saving data for {self.num_people} people to {filepath}")
        np.savez(filepath, ages=ages, people_place_ids=people_place_ids, people_baseline_flows=people_flows)

        place_type_enum, place_types, place_coordinates = self.get_place_data()
        filepath = os.path.join(self.snapshot_dir, 'place_snapshot.npz')
        print(f"Saving data for {self.num_places} people to {filepath}")
        np.savez(filepath,
                 place_type_enum=place_type_enum,
                 place_types=place_types,
                 place_coordinates=place_coordinates)

    def create_global_place_ids(self):
        max_id = 0
        global_place_id_lookup = dict()

        for activity_name in self.activity_names:
            locations_ids = self.locations[activity_name]['ID'].to_numpy(dtype=np.uint32)
            num_activity_ids = locations_ids.shape[0]
            starting_id = locations_ids[0]

            # check that activity IDs increase by one
            assert locations_ids[-1] == num_activity_ids - 1 + starting_id

            # subtract starting ID in case the first local activity ID is not zero
            global_ids = locations_ids + max_id - starting_id
            global_place_id_lookup[activity_name] = {
                "ids": global_ids,
                "id_offset": starting_id
            }
            max_id += num_activity_ids

        num_places = max_id
        return global_place_id_lookup, num_places

    def get_global_place_id(self, activity_name, local_place_id):
        ids_for_activity = self.global_place_id_lookup[activity_name]
        global_id_location = local_place_id - ids_for_activity["id_offset"]
        return ids_for_activity["ids"][global_id_location]

    def get_people_ages(self):
        return self.individuals['age'].to_numpy(dtype=np.uint16)

    def get_people_place_data(self):
        max_places_per_person = 10  # assume upper limit so we can use a fixed size array
        people_place_flows = np.full((self.num_people, max_places_per_person, 2), np.nan, dtype=np.float32)

        for people_id, person_row in tqdm(self.individuals.iterrows(), total=self.num_people,
                                          desc="Calculating place flows for all people"):
            num_places_added = 0

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

                num_places_to_add = local_place_ids.shape[0]

                start_idx = num_places_added
                end_idx = start_idx + num_places_to_add
                people_place_flows[people_id, start_idx:end_idx, 0] = np.array(
                    [self.get_global_place_id(activity_name, local_place_id)
                     for local_place_id in local_place_ids])

                people_place_flows[people_id, start_idx:end_idx, 1] = activity_flows

                num_places_added += num_places_to_add

            # person_place_data = np.array(list(zip(person_global_place_ids, person_place_flows)))
            #
            # # sort by flow value descending so we can take the n places with highest flows
            # person_place_data = person_place_data[person_place_data[:, 1].argsort()[::-1]]
            # # person_place_data = person_place_data[:places_to_keep_per_person]
            #
            # num_places = person_place_data.shape[0]
            # people_place_ids[people_id][0:num_places] = person_place_data[:, 0]
            # people_flows[people_id][0:num_places] = person_place_data[:, 1]

        # Sort by flow magnitude
        # people_place_flows = people_place_flows[:, people_place_flows[:, :, 1].argsort()]

        # truncate to maximum places per person to keep
        places_to_keep_per_person = 16
        truncated_people_place_ids = people_place_flows[:, 0:places_to_keep_per_person, 0].astype(np.uint32)
        truncated_people_flows = people_place_flows[:, 0:places_to_keep_per_person, 1]

        return truncated_people_place_ids, truncated_people_flows

    def get_place_data(self):
        place_type_enum = np.zeros(len(self.activity_names), dtype=object)
        place_types = np.zeros(self.num_places, dtype=np.uint32)
        place_coordinates = np.full((self.num_places, 2), np.nan, dtype=np.float32)

        for activity_index, activity_name in enumerate(self.activity_names):
            place_type_enum[activity_index] = activity_name
            activity_locations_df = self.locations[activity_name]
            activity_locations_df = activity_locations_df.rename(columns={"bng_e": "Easting", "bng_n": "Northing"})

            for row_index, location_row in tqdm(activity_locations_df.iterrows(), total=self.num_places,
                                                desc=f"Processing data for {activity_name} locations"):
                local_place_id = np.uint32(location_row["ID"])
                global_place_id = self.get_global_place_id(activity_name, local_place_id)

                place_types[global_place_id] = activity_index

                easting = getattr(location_row, "Easting", None)
                northing = getattr(location_row, "Northing", None)

                if easting and northing:
                    long_lat = convert_lonlat([easting], [northing])
                    long = long_lat[0][0]
                    lat = long_lat[1][0]
                    place_coordinates[global_place_id] = np.array([lat, long])

        return place_type_enum, place_types, place_coordinates

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

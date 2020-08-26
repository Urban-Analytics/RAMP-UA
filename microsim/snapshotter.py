import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from convertbng.util import convert_lonlat

sentinel_value = (1 << 31) - 1


class Snapshotter:
    """
    Take snapshot of internal model state before running the python model, so this "snapshot" of model
    state can be saved and transferred to the GPU version of the model
    """

    def __init__(self, individuals, activity_locations, snapshot_dir, data_dir, cache_inputs=True):
        self.snapshot_dir = snapshot_dir
        self.data_dir = data_dir

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

        filepath = os.path.join(self.snapshot_dir, 'people.npz')
        print(f"Saving data for {self.num_people} people to {filepath}")
        np.savez(filepath, ages=ages, people_place_ids=people_place_ids, people_baseline_flows=people_flows)

        activity_name_enum, place_activities = self.get_place_data()
        place_coordinates = self.get_place_coordinates()
        filepath = os.path.join(self.snapshot_dir, 'places.npz')
        print(f"Saving data for {self.num_places} people to {filepath}")
        np.savez(filepath,
                 activity_names=activity_name_enum,
                 place_activities=place_activities,
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

    def get_people_place_data(self, max_places_per_person=100, places_to_keep_per_person=16):
        """
        Calculate the "baseline flows" for each person by multiplying flows for each location by duration, then sorting
        these flows and taking the top n so they can fit in a fixed size array. Locations from all activities are contained
        in the same array so the activity specific location ids are mapped to global location ids.

        :param max_places_per_person: upper limit of places per person so we can use a fixed size array
        :param places_to_keep_per_person:
        :return: Numpy arrays of place ids and baseline flows indexed by person id
        """

        people_place_ids = np.full((self.num_people, max_places_per_person), sentinel_value, dtype=np.uint32)
        people_place_flows = np.zeros((self.num_people, max_places_per_person), dtype=np.float32)

        num_places_added = np.zeros(self.num_people, dtype=np.uint32)

        for activity_name in self.activity_names:
            activity_venues = self.individuals.loc[:, activity_name + "_Venues"]
            activity_flows = self.individuals.loc[:, activity_name + "_Flows"]
            activity_durations = self.individuals.loc[:, activity_name + "_Duration"]

            for people_id, (local_place_ids, flows, duration) in tqdm(
                    enumerate(zip(activity_venues, activity_flows, activity_durations)),
                    total=self.num_people,
                    desc=f"Calculating {activity_name} flows for all people"):
                flows = np.array(flows) * duration

                # check dimensions match
                assert len(local_place_ids) == flows.shape[0]

                num_places_to_add = len(local_place_ids)

                start_idx = num_places_added[people_id]
                end_idx = start_idx + num_places_to_add
                people_place_ids[people_id, start_idx:end_idx] = np.array(
                    [self.get_global_place_id(activity_name, local_place_id)
                     for local_place_id in local_place_ids])

                people_place_flows[people_id, start_idx:end_idx] = flows

                num_places_added[people_id] += num_places_to_add

        # Sort by magnitude of flow (reversed)
        sorted_indices = people_place_flows.argsort()[:, ::-1]
        people_place_ids = np.take_along_axis(people_place_ids, sorted_indices, axis=1)
        people_place_flows = np.take_along_axis(people_place_flows, sorted_indices, axis=1)

        # truncate to maximum places per person
        people_place_ids = people_place_ids[:, 0:places_to_keep_per_person]
        people_place_flows = people_place_flows[:, 0:places_to_keep_per_person]

        return people_place_ids, people_place_flows

    def get_place_data(self):
        activity_name_enum = np.zeros(len(self.activity_names), dtype=object)
        place_activities = np.zeros(self.num_places, dtype=np.uint32)

        for activity_index, activity_name in enumerate(self.activity_names):
            activity_name_enum[activity_index] = activity_name
            activity_locations_df = self.locations[activity_name]

            ids = activity_locations_df.loc[:, "ID"]

            # Store global ids
            for local_place_id in tqdm(ids, desc=f"Storing location type for {activity_name}"):
                global_place_id = self.get_global_place_id(activity_name, local_place_id)
                place_activities[global_place_id] = activity_index

        return activity_name_enum, place_activities

    def get_place_coordinates(self):
        place_coordinates = np.zeros((self.num_places, 2), dtype=np.float32)

        for activity_index, activity_name in enumerate(self.activity_names):
            activity_locations_df = self.locations[activity_name]

            # rename OS grid coordinate columns
            activity_locations_df = activity_locations_df.rename(columns={"bng_e": "Easting", "bng_n": "Northing"})

            # for homes: get coordinates of MSOA area
            if activity_name == "Home":
                eastings, northings = self.get_home_coordinates(activity_locations_df)
                activity_locations_df["Easting"] = eastings
                activity_locations_df["Northing"] = northings

            # Convert OS grid coordinates (eastings and northings) to latitude and longitude
            if 'Easting' in activity_locations_df.columns and 'Northing' in activity_locations_df.columns:
                local_ids = activity_locations_df.loc[:, "ID"]
                eastings = activity_locations_df.loc[:, "Easting"]
                northings = activity_locations_df.loc[:, "Northing"]

                for local_place_id, easting, northing in tqdm(zip(local_ids, eastings, northings),
                                                              desc=f"Processing coordinate data for {activity_name}"):
                    global_place_id = self.get_global_place_id(activity_name, local_place_id)

                    long_lat = convert_lonlat([easting], [northing])
                    long = long_lat[0][0]
                    lat = long_lat[1][0]
                    place_coordinates[global_place_id] = np.array([lat, long])

        return place_coordinates

    def get_home_coordinates(self, home_locations_df):
        print("Getting coordinates for home locations")

        # TODO: load msoa building lookup from JSON file

        areas = home_locations_df.loc[:, "area"]

        num_locations = len(home_locations_df.index)
        eastings = np.zeros(num_locations)
        northings = np.zeros(num_locations)

        for i, area in enumerate(areas):
            # TODO: select random building from within area and assign easting and northing
            eastings[i] = 0
            northings[i] = 0

        return eastings, northings

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
    data_dir = os.path.join(base_dir, "devon_data")
    snapshotter = Snapshotter(individuals=None, activity_locations=None, snapshot_dir=snapshot_dir, data_dir=data_dir)
    snapshotter.store_snapshots()


if __name__ == '__main__':
    main()

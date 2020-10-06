import numpy as np
import pandas as pd
import random
import os
import json
import pickle
from tqdm import tqdm
from convertbng.util import convert_lonlat

from microsim.opencl.ramp.snapshot import Snapshot

sentinel_value = (1 << 31) - 1


class SnapshotConvertor:
    """
    Convert dataframe of individuals and activity locations into a Snapshot object that can be used by the OpenCL model
    """

    def __init__(self, individuals, activity_locations, time_activity_multiplier, data_dir, cache_inputs=True):
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

        # TODO: extract lockdown multipliers correctly
        # def load_lockdown_data():
        #     """Load lockdown multipliers for each day from csv file of google mobility data."""
        #     lockdown_mobility_df = pd.read_csv("data/devon_google_mobility_lockdown_daily.csv")
        #     lockdown_multipliers = lockdown_mobility_df.loc[:, "timeout_multiplier"].to_numpy().astype(np.float32)
        #
        #     # cap multipliers to maximum of 1.0
        #     lockdown_multipliers[lockdown_multipliers > 1] = 1.0
        #
        #     return lockdown_multipliers
        self.lockdown_multipliers = time_activity_multiplier

        self.num_people = self.individuals['ID'].count()
        self.global_place_id_lookup, self.num_places = self.create_global_place_ids()

    def generate_snapshot(self):
        people_ages = self.get_people_ages()
        area_codes = self.get_people_area_codes()
        not_home_probs = self.get_not_home_probs()
        people_place_ids, people_flows = self.get_people_place_data()

        place_activities = self.get_place_data()
        place_coordinates = self.get_place_coordinates()
        return Snapshot.from_arrays(people_ages, people_place_ids, people_flows, area_codes, not_home_probs,
                                    place_activities, place_coordinates, self.lockdown_multipliers)

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

    def get_people_area_codes(self):
        return self.individuals['area'].to_numpy(dtype=np.object)

    def get_not_home_probs(self):
        return self.individuals['pnothome'].to_numpy(dtype=np.float32)

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
        place_activities = np.zeros(self.num_places, dtype=np.uint32)

        for activity_index, activity_name in enumerate(self.activity_names):
            activity_locations_df = self.locations[activity_name]

            ids = activity_locations_df.loc[:, "ID"]

            # Store global ids
            for local_place_id in tqdm(ids, desc=f"Storing location type for {activity_name}"):
                global_place_id = self.get_global_place_id(activity_name, local_place_id)
                place_activities[global_place_id] = activity_index

        return place_activities

    def get_place_coordinates(self):
        place_coordinates = np.zeros((self.num_places, 2), dtype=np.float32)

        non_home_activities = list(filter(lambda activity: activity != "Home", self.activity_names))

        for activity_index, activity_name in enumerate(non_home_activities):
            activity_locations_df = self.locations[activity_name]

            # rename OS grid coordinate columns
            activity_locations_df = activity_locations_df.rename(columns={"bng_e": "Easting", "bng_n": "Northing"})

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

        # for homes: assign coordinates of random building inside MSOA area
        home_locations_df = self.locations["Home"]
        lats, lons = self.get_coordinates_from_buildings(home_locations_df)
        local_ids = home_locations_df.loc[:, "ID"]

        for local_place_id, lat, lon in tqdm(zip(local_ids, lats, lons), desc=f"Storing coordinates for homes"):
            global_place_id = self.get_global_place_id("Home", local_place_id)
            place_coordinates[global_place_id] = np.array([lat, lon])

        return place_coordinates

    def get_coordinates_from_buildings(self, home_locations_df):
        # load msoa building lookup from JSON file
        msoa_building_filepath = os.path.join(self.data_dir, "msoa_building_coordinates.json")
        with open(msoa_building_filepath) as f:
            msoa_buildings = json.load(f)

        areas = home_locations_df.loc[:, "area"]

        num_locations = len(home_locations_df.index)

        lats = np.zeros(num_locations)
        lons = np.zeros(num_locations)

        for i, area in enumerate(areas):
            # select random building from within area and assign easting and northing
            area_buildings = msoa_buildings[area]
            building = random.choice(area_buildings)

            lats[i] = building[0]
            lons[i] = building[1]

        return lats, lons

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

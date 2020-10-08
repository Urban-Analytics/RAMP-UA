import pandas as pd
import numpy as np
import os
from microsim.opencl.ramp.snapshot_convertor import SnapshotConvertor

sentinel_value = (1 << 31) - 1


class TestActivityLocation:
    def __init__(self, name: str, locations: pd.DataFrame):
        self.name = name
        self._locations = locations


individuals_df = pd.DataFrame({'ID': [0, 1, 3],
                               'age': [43, 22, 33],
                               'Home_Venues': [[0], [1], [2]],
                               'Home_Flows': [[1.0], [1.0], [1.0]],
                               'Home_Duration': [0.8, 0.7, 0.6],
                               'Retail_Venues': [[0, 2, 4], [1, 2, 3], [0, 3, 4]],
                               'Retail_Flows': [[0.2, 0.5, 0.3], [0.1, 0.6, 0.3], [0.5, 0.1, 0.4]],
                               'Retail_Duration': [0.2, 0.3, 0.4],
                               'pnothome': [0.2, 0.4, 0.5],
                               'area': np.array(["E02004143", "E02004144", "E02004145"]).astype(np.object)
                               })

home_df = pd.DataFrame({'ID': [0, 1, 2], 'area': ['E02004129', 'E02004130', 'E02004131']})

bng_coordinate_a = [533494, 181851]
lat_lon_coordinate_a = [51.519811, -0.077342]  # NB: converted BNG coordinates to lat long using online tool
bng_coordinate_b = [280397, 696841]
lat_lon_coordinate_b = [56.149214, -3.926794]

retail_data = {'ID': [0, 1, 2, 3, 4],
               "Easting": [bng_coordinate_a[0], bng_coordinate_a[0], bng_coordinate_b[0], bng_coordinate_b[0],
                           bng_coordinate_b[0]],
               "Northing": [bng_coordinate_a[1], bng_coordinate_a[1], bng_coordinate_b[1], bng_coordinate_b[1],
                            bng_coordinate_b[1]]}
retail_df = pd.DataFrame(retail_data, columns=['ID', "Easting", "Northing"])

activity_locations = {
    "Home": TestActivityLocation(name="Home", locations=home_df),
    "Retail": TestActivityLocation(name="Retail", locations=retail_df)
}

base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "tests/dummy_data")
opencl_test_dir = os.path.join(base_dir, "tests/opencl")

snapshot_converter = SnapshotConvertor(individuals_df, activity_locations, time_activity_multiplier=None,
                                       data_dir=data_dir)
snapshot = snapshot_converter.generate_snapshot()
snapshot.save(opencl_test_dir + "/test_snapshot.npz")


def test_global_id_lookup():
    home_global_id = snapshot_converter.get_global_place_id("Home", 0)
    retail_global_id = snapshot_converter.get_global_place_id("Retail", 0)
    assert home_global_id != retail_global_id
    assert home_global_id == 0
    assert retail_global_id == 3


def test_processes_people_flows():
    expected_people_place_ids = np.array([[0, 5, 7, 3, sentinel_value, sentinel_value, sentinel_value, sentinel_value],
                                          [1, 5, 6, 4, sentinel_value, sentinel_value, sentinel_value, sentinel_value],
                                          [2, 3, 7, 6, sentinel_value, sentinel_value, sentinel_value, sentinel_value],
                                          ])
    expected_people_flows = np.array([[0.8, 0.1, 0.06, 0.04, 0.0, 0.0, 0.0, 0.0],
                                      [0.7, 0.18, 0.09, 0.03, 0.0, 0.0, 0.0, 0.0],
                                      [0.6, 0.2, 0.16, 0.04, 0.0, 0.0, 0.0, 0.0]
                                      ])

    people_place_ids, people_flows = snapshot_converter.get_people_place_data(max_places_per_person=20,
                                                                              places_to_keep_per_person=8)

    assert np.array_equal(expected_people_place_ids, people_place_ids)
    assert np.all(np.isclose(expected_people_flows, people_flows))


def test_get_place_data():
    expected_place_activities = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    place_activities = snapshot_converter.get_place_data()
    assert np.array_equal(expected_place_activities, place_activities)


def test_get_coordinates():
    # NB: since homes are randomly allocated within the MSOA area, it is hard to assert expected values for homes
    # so only check locations of retail buildings
    expected_non_home_place_coordinates = np.array([lat_lon_coordinate_a,
                                                    lat_lon_coordinate_a,
                                                    lat_lon_coordinate_b,
                                                    lat_lon_coordinate_b,
                                                    lat_lon_coordinate_b])

    place_coordinates = snapshot_converter.get_place_coordinates()

    # select non-home coordinates
    non_home_place_coordinates = place_coordinates[3:]

    assert np.all(np.isclose(expected_non_home_place_coordinates, non_home_place_coordinates, atol=0.0001,
                             equal_nan=True))

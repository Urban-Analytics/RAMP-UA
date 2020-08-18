import pandas as pd
import numpy as np
from microsim.snapshotter import Snapshotter


class TestActivityLocation:
    def __init__(self, name: str, locations: pd.DataFrame):
        self.name = name
        self._locations = locations


people_data = {'ID': [0, 1, 3],
               'age': [43, 22, 33],
               'Home_Venues': [[0], [1], [2]],
               'Home_Flows': [[1.0], [1.0], [1.0]],
               'Home_Duration': [0.8, 0.7, 0.6],
               'Retail_Venues': [[0, 2, 4], [1, 2, 3], [0, 3, 4]],
               'Retail_Flows': [[0.2, 0.5, 0.3], [0.1, 0.6, 0.3], [0.5, 0.1, 0.4]],
               'Retail_Duration': [0.2, 0.3, 0.4]
               }

individuals_df = pd.DataFrame(people_data,
                              columns=['ID', 'age', 'Home_Venues', 'Home_Duration', 'Home_Flows', 'Retail_Venues',
                                       'Retail_Duration', 'Retail_Flows'])

home_data = {'ID': [0, 1, 2]}
home_df = pd.DataFrame(home_data, columns=['ID'])

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

# pass None in order to load locations from pickle file
snapshotter = Snapshotter(individuals_df, activity_locations, "snapshots", cache_inputs=False)


def test_global_id_lookup():
    home_global_id = snapshotter.get_global_place_id("Home", 0)
    retail_global_id = snapshotter.get_global_place_id("Retail", 0)
    assert home_global_id != retail_global_id
    assert home_global_id == 0
    assert retail_global_id == 3


def test_processes_people_flows():
    expected_people_place_ids = np.array([[0, 5, 7, 3, 0, 0, 0, 0, 0, 0],
                                          [1, 5, 6, 4, 0, 0, 0, 0, 0, 0],
                                          [2, 3, 7, 6, 0, 0, 0, 0, 0, 0]
                                          ])
    expected_people_flows = np.array([[0.8, 0.1, 0.06, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.7, 0.18, 0.09, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.6, 0.2, 0.16, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                                      ])

    people_place_ids, people_flows = snapshotter.get_people_place_data(max_places_per_person=20,
                                                                       places_to_keep_per_person=10)

    assert np.array_equal(expected_people_place_ids, people_place_ids)
    assert np.all(np.isclose(expected_people_flows, people_flows))


def test_get_place_data():
    expected_place_type_enum = np.array(["Home", "Retail"])
    expected_place_types = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    expected_place_coordinates = np.array([[np.nan, np.nan],
                                           [np.nan, np.nan],
                                           [np.nan, np.nan],
                                           lat_lon_coordinate_a,
                                           lat_lon_coordinate_a,
                                           lat_lon_coordinate_b,
                                           lat_lon_coordinate_b,
                                           lat_lon_coordinate_b])

    place_type_enum, place_types, place_coordinates = snapshotter.get_place_data()

    assert np.array_equal(expected_place_type_enum, place_type_enum)
    assert np.array_equal(expected_place_types, place_types)
    assert np.all(np.isclose(expected_place_coordinates, place_coordinates, atol=0.0001, equal_nan=True))

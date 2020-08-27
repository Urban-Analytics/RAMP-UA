import os
import json
import overpy
from tqdm import tqdm
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


def load_osm_buildings():
    api = overpy.Overpass()

    print("Requesting buildings")

    # fetch buildings - small area
    result = api.query("""
        way(50.30745,-3.98185,50.34093,-3.90066) ["building"];
        (._;>;);
        out body;
        """)

    # fetch buildings - All of Devon
    # result = api.query("""
    #     way(50.07156,-4.70709,50.39431,-4.14253) ["building"];
    #     (._;>;);
    #     out body;
    #     """)

    print(f"Found {len(result.ways)} buildings")

    building_coords = []
    for way in tqdm(result.ways, desc="extracting building coordinates from ways"):
        # just take the first node as the approximate location of the building
        # no need to calculate the centroid, since this is just approximate anyway
        node = way.nodes[0]
        # building_coords.append(Point(float(node.lon), float(node.lat)))
        building_coords.append(Point(float(node.lon), float(node.lat)))

    return building_coords


def load_devon_msoas(data_dir, msoa_filename="devon_msoas.csv"):
    return pd.read_csv(os.path.join(data_dir, msoa_filename), header=None,
                       names=["Easting", "Northing", "Num", "Code", "Desc"])


def load_msoa_shapes(data_dir, visualize=False):
    shape_dir = os.path.join(data_dir, "MSOAS_shp")
    shape_file = os.path.join(shape_dir, "bcc21fa2-48d2-42ca-b7b7-0d978761069f2020412-1-12serld.j1f7i.shp")

    all_msoa_shapes = gpd.read_file(shape_file)
    all_msoa_shapes = all_msoa_shapes.rename(columns={"msoa11cd": "Code"})
    print(f"Loaded {len(all_msoa_shapes.index)} MSOA shapes with projection {all_msoa_shapes.crs}")

    # re-project coordinates from british national grid to WGS84 (lat/lon)
    all_msoa_shapes = all_msoa_shapes.to_crs("EPSG:4326")

    # Filter to devon MSOAs
    devon_msoas = load_devon_msoas(data_dir)
    print(f"Loaded {len(devon_msoas.index)} devon MSOA codes")

    devon_msoa_shapes = pd.merge(all_msoa_shapes, devon_msoas, on="Code")
    print(f"Filtered {len(devon_msoa_shapes.index)} devon MSOA shapes")

    if visualize:
        devon_msoa_shapes.plot()
        plt.show()

    return devon_msoa_shapes


def calculate_msoa_buildings(building_coordinates, msoa_shapes):
    msoa_buildings = dict()

    msoa_codes = msoa_shapes.loc[:, "Code"]
    msoa_geometries = msoa_shapes.loc[:, "geometry"]

    # for all msoas store the buildings within their shapes
    for code, geometry in tqdm(zip(msoa_codes, msoa_geometries), desc="Finding buildings for all MSOAs"):
        buildings_within_msoa = []
        # iterate through all buildings and append ones within shape
        for building_point in tqdm(building_coordinates, desc=f"Checking building coordinates for MSOA {code}"):
            if building_point.within(geometry):
                building_lat_lon = [building_point.y, building_point.x]
                buildings_within_msoa.append(building_lat_lon)

        msoa_buildings[code] = buildings_within_msoa

    return msoa_buildings


def main():
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "devon_data")

    building_coordinates = load_osm_buildings()
    devon_msoa_shapes = load_msoa_shapes(data_dir, visualize=False)

    msoa_buildings = calculate_msoa_buildings(building_coordinates, devon_msoa_shapes)

    print("Writing MSOA buildings to JSON file")
    output_filepath = os.path.join(data_dir, "msoa_building_coordinates.json")

    with open(output_filepath, 'w') as output_file:
        json.dump(msoa_buildings, output_file)


if __name__ == '__main__':
    main()

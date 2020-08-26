import os
import overpy
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd


def load_osm_buildings():
    api = overpy.Overpass()

    print("Requesting buildings")

    # fetch buildings - small area
    result = api.query("""
        way(50.36453,-4.17754,50.39431,-4.14253) ["building"];
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
    for way in result.ways:
        # just take the first node as the approximate location of the building
        # no need to calculate the centroid, since this is just approximate anyway
        node = way.nodes[0]
        building_coords.append(Point(float(node.lat), float(node.lon)))

    return building_coords


def load_devon_msoas(data_dir, msoa_filename="devon_msoas.csv"):
    return pd.read_csv(os.path.join(data_dir, msoa_filename), header=None,
                       names=["Easting", "Northing", "Num", "Code", "Desc"])


def load_msoa_shapes(data_dir):
    shape_dir = os.path.join(data_dir, "MSOAS_shp")
    shape_file = os.path.join(shape_dir, "bcc21fa2-48d2-42ca-b7b7-0d978761069f2020412-1-12serld.j1f7i.shp")

    all_msoa_shapes = gpd.read_file(shape_file)
    all_msoa_shapes = all_msoa_shapes.rename(columns={"msoa11cd": "Code"})
    print(f"Loaded {len(all_msoa_shapes.index)} MSOA shapes with projection {all_msoa_shapes.crs}")

    # re-project coordinates from british national grid to WGS84 (lat/lon)
    all_msoa_shapes = all_msoa_shapes.to_crs("EPSG:3395")

    # Filter to devon MSOAs
    devon_msoas = load_devon_msoas(data_dir)
    print(f"Loaded {len(devon_msoas.index)} devon MSOA codes")

    devon_msoa_shapes = pd.merge(all_msoa_shapes, devon_msoas, on="Code")
    print(f"Filtered {len(devon_msoa_shapes.index)} devon MSOA shapes")

    return devon_msoa_shapes


def calculate_msoa_buildings(building_coordinates, msoa_shapes):
    msoa_buildings = dict()

    msoa_codes = msoa_shapes.loc[:, "Code"]
    msoa_geometries = msoa_shapes.loc[:, "geometry"]

    # for all msoas store the buildings within their shapes
    for code, geometry in zip(msoa_codes, msoa_geometries):
        buildings_within_msoa = []
        # iterate through all buildings and append ones within shape
        for building_point in building_coordinates:
            if geometry.within(building_point):
                print(f"Found building within msoa: {code}")
                buildings_within_msoa.append(building_point)

        msoa_buildings[code] = buildings_within_msoa

    return msoa_buildings


def main():
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "devon_data")

    building_coordinates = load_osm_buildings()
    devon_msoa_shapes = load_msoa_shapes(data_dir)

    msoa_buildings = calculate_msoa_buildings(building_coordinates, devon_msoa_shapes)

    # TODO: store dict as JSON


if __name__ == '__main__':
    main()

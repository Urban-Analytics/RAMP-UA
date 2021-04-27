import os
# import sys
# sys.path.append("microsim") 
import json
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from code.constants import Constants
from code.initalise.raw_data_handler import RawDataHandler

# Functionality to create a lookup table of MSOA codes to a list of coordinates of all the buildings
# in that MSOA area, this is stored to a JSON file and used in the OpenCL SnapshotConverter to allocate people's
# homes to real building locations.
# This works by using Geopandas to load OSM data for all buildings in the study area from a local file, then loading all MSOA
# shapes in the country and filtering to just the study area ones. Then iterating through all study area MSOAs and finding the
# buildings that lie within the MSOA boundary polygon.

class MapsHandler:
    
    @staticmethod
    def run():
    """
    Class that generates the JSON file for the OpenCL snapshot (final dashboard)
    """
    base_dir = Constants.Paths.PROJECT_FOLDER_ABSOLUTE_PATH # os.getcwd()
    # data_dir = os.path.join(base_dir,
                            Constants.Paths.DATA_FOLDER,
                            Constants.Paths.COMMON_DATA_FOLDER) #"devon_data")
    data_dir = Constants.Paths.DATA.FULL_PATH_FOLDER

    osm_buildings = load_osm_shapefile(RawDataHandler._combined_shp_file) #("/Users/azanchetta/OneDrive - The Alan Turing Institute/Research/projects/EcoTwins2/data/regional_data/WestYorkshire/")

    msoa_shapes = load_msoa_shapes(Constants.Paths.MSOAS_SHP.FULL_PATH_FILE,
                                   visualize=False)

    msoa_buildings = calculate_msoa_buildings(osm_buildings,
                                              msoa_shapes)

    print("Writing MSOA buildings to JSON file")
    # output_filepath = os.path.join(data_dir, "msoa_building_coordinates.json") # correct this
    output_filepath = output #TODO make this consistent with the processed data study area from the main_initalisation
    
    with open(output_filepath, 'w') as output_file:
        json.dump(msoa_buildings, output_file)

    def load_osm_shapefile(shapefile) # (data_dir):
        # Shape file downloaded for devon from https://download.geofabrik.de/europe/great-britain/england/devon.html
        # osm_dir = os.path.join(data_dir, "osm")
        # shape_file = os.path.join(osm_dir, "gis_osm_buildings_a_free_1.shp")
        shape_file = shapefile

        print("Loading OSM buildings shapefile")
        osm_buildings = gpd.read_file(shape_file)
        print(f"Loaded {len(osm_buildings.index)} buildings from shapefile")
        return osm_buildings


    def load_studyarea_msoas(msoa_list_with_path): #(data_dir, msoa_filename="msoas_list.csv"):
        # return pd.read_csv(os.path.join("/Users/azanchetta/OneDrive - The Alan Turing Institute/Research/projects/EcoTwins2/data/regional_data/WestYorkshire/",
        #                                 msoa_filename), header=None,
        #                 names=["Easting", "Northing", "Num", "Code", "Desc"])
        return pd.read_csv(Constants.Paths.LIST_MSOAS.FULL_PATH_FILE,
                           header = None,
                           names = ["MSOA11CD"])

    def load_msoa_shapes(shapefile,
                         visualize=False):
        # shape_dir = os.path.join(data_dir, "MSOAS_shp")
        # shape_file = os.path.join(shape_dir, "bcc21fa2-48d2-42ca-b7b7-0d978761069f2020412-1-12serld.j1f7i.shp")
        shape_file = shapefile
        all_msoa_shapes = gpd.read_file(shape_file)
        all_msoa_shapes = all_msoa_shapes.rename(columns={"msoa11cd": "MSOA11CD"})
        print(f"Loaded {len(all_msoa_shapes.index)} MSOA shapes with projection {all_msoa_shapes.crs}")

        # re-project coordinates from british national grid to WGS84 (lat/lon)
        all_msoa_shapes = all_msoa_shapes.to_crs("EPSG:4326")

        # Filter to devon MSOAs
        studyarea_msoas = load_studyarea_msoas() #(data_dir)
        print(f"Loaded {len(studyarea_msoas.index)} study area MSOA codes")

        studyarea_msoa_shapes = pd.merge(all_msoa_shapes,
                                         studyarea_msoas,
                                         on = "MSOA11CD")
        print(f"Filtered {len(studyarea_msoa_shapes.index)} study area MSOA shapes")

        if visualize:
            studyarea_msoa_shapes.plot()
            plt.show()

        return studyarea_msoa_shapes


    def calculate_msoa_buildings(osm_buildings, msoa_shapes):
        msoa_buildings = dict()

        msoa_codes = msoa_shapes.loc[:, "Code"]
        msoa_geometries = msoa_shapes.loc[:, "geometry"]
        building_geometries = osm_buildings.loc[:, "geometry"]

        # for all msoas store the buildings within their shapes
        for code, msoa_geometry in tqdm(zip(msoa_codes, msoa_geometries), total=len(msoa_shapes.index),
                                        desc="Finding buildings for all MSOAs"):
            buildings_within_msoa = []
            # iterate through all buildings and append ones within shape
            for building_geometry in tqdm(building_geometries, desc=f"Assigning buildings to MSOA {code}"):
                building_point = building_geometry.centroid
                if building_point.within(msoa_geometry):
                    building_lat_lon = [building_point.y, building_point.x]
                    buildings_within_msoa.append(building_lat_lon)

            msoa_buildings[code] = buildings_within_msoa

        return msoa_buildings
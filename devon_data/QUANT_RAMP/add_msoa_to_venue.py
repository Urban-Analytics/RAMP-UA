# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:24:52 2020

@author: Toshiba
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopandas.tools import sjoin
import os

# read in files

base_dir = os.getcwd()  # get current directory (usually RAMP-UA)
quant_dir = os.path.join(base_dir, "data","QUANT_RAMP","model-runs")

PrimaryZones = pd.read_csv(os.path.join(base_dir,"data","QUANT_RAMP","model-runs","primaryZones.csv"))
SecondaryZones = pd.read_csv(os.path.join(base_dir,"data","QUANT_RAMP","model-runs","secondaryZones.csv"))
RetailZones = pd.read_csv(os.path.join(base_dir,"data","QUANT_RAMP","model-runs","retailpointsZones.csv"))

sh_file = os.path.join(base_dir, "devon_data","MSOAS_shp","bcc21fa2-48d2-42ca-b7b7-0d978761069f2020412-1-12serld.j1f7i.shp")
map_df = gpd.read_file(sh_file)
map_df.rename(index=str, columns={'msoa11cd': 'MSOA'},inplace=True)
list_MSOAS = map_df.MSOA


# Converting to a GeoPandas Dataframe 
#merged_data.crs # check coordinate system from underlay (here MSOAs - epsg:27700)

# Use for all:
crs = {'init': 'epsg:27700'}

# For primary schools:
geometry_primary = [Point(xy) for xy in zip(PrimaryZones.east, PrimaryZones.north)]
#PrimaryZones = PrimaryZones.drop(['east', 'north'], axis=1)
gdf_PrimaryZones = gpd.GeoDataFrame(PrimaryZones, crs=crs, geometry=geometry_primary)

# For secondary schools:
geometry_secondary = [Point(xy) for xy in zip(SecondaryZones.east, SecondaryZones.north)]
#SecondaryZones = SecondaryZones.drop(['east', 'north'], axis=1)
gdf_SecondaryZones = gpd.GeoDataFrame(SecondaryZones, crs=crs, geometry=geometry_secondary)

# For retail:
geometry_retail = [Point(xy) for xy in zip(RetailZones.east, RetailZones.north)]
#RetailZones = RetailZones.drop(['east', 'north'], axis=1)
gdf_RetailZones = gpd.GeoDataFrame(RetailZones, crs=crs, geometry=geometry_retail)

# dictionary for looping:
venues = {
  "primary": PrimaryZones,
  "secondary": SecondaryZones,
  "retail": RetailZones
}
gdf_venues = {
  "primary": gdf_PrimaryZones,
  "secondary": gdf_SecondaryZones,
  "retail": gdf_RetailZones
}




for key in gdf_venues:
    # create new column for output
    venues[key] = venues[key].drop('geometry', 1)
    venues[key] = venues[key].drop('Unnamed: 0', 1)
    venues[key]["MSOA"] = np.nan
    for m in list_MSOAS:
        msoa_mask = map_df.loc[map_df['MSOA']==m]
        poly  = gpd.GeoDataFrame(msoa_mask, crs=crs) 
        pointInPolys = sjoin(gdf_venues[key], poly, how='left')
        grouped = pointInPolys.groupby('index_right')
        results = list(grouped)
        if results: # at least one venue
            tmp = results[0][1]
            venues[key].iloc[tmp.zonei,4] = m
    venues[key].to_csv (os.path.join(base_dir,"data","QUANT_RAMP","model-runs",f"test{key}.csv"), index = True, header=True)


# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# map_df.plot(ax=ax, facecolor='gray');
# msoa_mask.plot(ax=ax, facecolor='red');
# gdf_PrimaryZones.plot(ax=ax, color='blue', markersize=1);
# plt.tight_layout();
# plt.show()
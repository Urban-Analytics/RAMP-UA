#!/usr/bin/env python
# coding: utf-8
# %%
import requests
import os
import tarfile
import zipfile
import pandas as pd
import geopandas as gpd
import numpy as np
# %%
"""for my own testing; assumption: msoasList variable read from input provided by the user"""
os.chdir("/Users/hsalat/MiscPython")
msoasList = pd.read_csv("/Users/hsalat/West_Yorkshire/Seeding/msoas.csv")
msoasList = msoasList["area_code"]
# %%
def download_data(folder: str,file : str):
    """Download data utility function
    Args:
        folder (str): can be: nationaldata, countydata or referencedata.
        file (str): name of the file, must include the extension.
    """
    url = "https://ramp0storage.blob.core.windows.net/" + folder + "/" + file
    target_path = os.path.join("data/common_data/",file)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())
    return target_path
def unpack_data(archive : str):
    """unpack tar data archive
    Args:
        archive (str): A string directory path to archive file using
    """
    tar_file = tarfile.open(archive)
    tar_file.extractall("data/common_data/")
# %%
"""TU files"""
if not os.path.isfile("data/common_data/lookUp.csv"):
    lookUp_path = download_data("referencedata","lookUp.csv")
    lookUp = pd.read_csv(lookUp_path)
else:
    lookUp = pd.read_csv("data/common_data/lookUp.csv")
tus_hse_ref = np.unique(lookUp.NewTU[lookUp.MSOA11CD.isin(msoasList)])
tus_hse = pd.DataFrame()
for x in tus_hse_ref:
    if not os.path.isfile("data/common_data/tus_hse_" + x + ".csv"):
        temp_path = download_data("countydata","tus_hse_" + x + ".csv")
        temp = pd.read_csv(temp_path)
    else:
        temp = pd.read_csv("data/common_data/tus_hse_" + x + ".csv")
    temp = temp[temp.MSOA11CD.isin(msoasList)]
    tus_hse = tus_hse.append(temp)
# %%
"""QUANT RAMP"""
if not os.path.isdir("data/common_data/QUANT_RAMP/")
    QUANT_path = download_data("nationaldata","QUANT_RAMP.tar.gz")
    unpack_data(QUANT_path)
# %%
"""commutingOD dl and selection"""
if not os.path.isfile("data/common_data/commutingOD.csv"):
    OD_path = download_data("nationaldata","commutingOD.gz")
    unpack_data(OD_path)
OD = pd.read_csv("data/common_data/commutingOD.csv")
OD = OD[OD.HomeMSOA.isin(msoasList)]
OD = OD[OD.DestinationMSOA.isin(msoasList)]
# %%
"""Lockdown scenario"""
"""In theory: lookUp already loaded before"""
if not os.path.isfile("data/common_data/timeAtHomeIncreaseCTY.csv"):
    lockdown_path = download_data("nationaldata","timeAtHomeIncreaseCTY.csv")
    lockdown = pd.read_csv(lockdown_path)
else:
    lockdown = pd.read_csv("data/common_data/timeAtHomeIncreaseCTY.csv")
if not os.path.isdir("data/common_data/MSOAS_shp/"):
    shp_path = download_data("nationaldata","MSOAS_shp.tar.gz")
    unpack_data(shp_path)
shp = gpd.read_file("data/common_data/MSOAS_shp/msoas.shp")
msoas_pop = shp["pop"]
change_ref = np.unique(lookUp.GoogleMob[lookUp.MSOA11CD.isin(msoasList)])
cty_pop = np.repeat(0,len(change_ref))
change = np.repeat(0,np.max(lockdown.day)+1)
msoas_pop = shp["pop"]
msoas_pop = msoas_pop[shp.MSOA11CD.isin(msoasList)]
for x in range(0,len(change_ref)):
    cty_pop[x] = np.nansum(msoas_pop[lookUp.GoogleMob[lookUp.MSOA11CD.isin(msoasList)] == change_ref[x]])
    change = change + lockdown.change[lockdown.CTY20 == change_ref[x]]*cty_pop[x]
change = change/np.sum(cty_pop)
lockdown = (1 - (np.mean(tus_hse.phome) * change))/np.mean(tus_hse.phome)
#%%
"""Seeding"""
"""In theory: shp already loaded before"""
msoas_risks = shp.risk[shp.MSOA11CD.isin(msoasList)]
# %%
"""Dashboard material"""
"""In theory: msoas.shp already loaded before"""
"""In theory: tus_hse_ref already defined, see above"""
osm_ref = np.unique(lookUp.OSM[lookUp.MSOA11CD.isin(msoasList)])
url = osm_ref[0]
target_path = os.path.join("data/common_data",tus_hse_ref[0] + ".zip")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())
    zip_file = zipfile.ZipFile(target_path)
    zip_file.extractall("data/common_data/" + tus_hse_ref[0])
osmShp = gpd.read_file("data/common_data/" + tus_hse_ref[0] + "/gis_osm_buildings_a_free_1.shp")
if len(osm_ref)>1:
    for x in range(1,len(osm_ref)):
        url = osm_ref[x]
        target_path = os.path.join("data/common_data",tus_hse_ref[x] + ".zip")
        response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())
    zip_file = zipfile.ZipFile(target_path)
    zip_file.extractall("data/common_data/" + tus_hse_ref[x])
    osmShp = pd.concat([
             osmShp,
             gpd.read_file("data/common_data/" + tus_hse_ref[x] + "/gis_osm_buildings_a_free_1.shp")
             ]).pipe(gpd.GeoDataFrame)
    
# TO_DO:
""" -> branch to load "load_msoa_locations.py" code """
""" Find centroid of intersected shp """
"""extract risks from shp dbf"""
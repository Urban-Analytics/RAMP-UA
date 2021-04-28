#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handling the raw data for the model initialisation

Created on Thu Apr 22

@author: Anna, Hadrien
"""

#TO_DO: add columns names to Constants file (example: MSOA11CD, pop from msoa_shp, CTY20)

import os
import pandas as pd
import requests
import tarfile
import zipfile
import geopandas as gpd
import numpy as np

from coding.constants import Constants

class RawDataHandler:
    _combined_TU_file = None
    _combined_shp_file = None

    @staticmethod
    def run():

        """
        Class that handles the data download and unpack
        """
        # (Hadrien's code)

        ### %%
        ### Reading in msoas list

        # assumption: msoasList variable read from input provided by the user
        # os.chdir("/Users/hsalat/MiscPython")
        # msoasList = pd.read_csv("/Users/hsalat/West_Yorkshire/Seeding/msoas.csv")
        # msoasList = msoasList["area_code"]
        
        # Note: this step needs to be improved by creating a formal way of introducing the list and checking its format is correct
        msoasList_file = pd.read_csv(Constants.Paths.LIST_MSOAS.FULL_PATH_FILE)  ### Needs to be put as initial parameter, for now in Constants
        msoasList = msoasList_file["MSOA11CD"]      # should be ["area_code"] but the current test file has column name "MSOA11CD"
        ### %%
        ###  Checking that look-up table exists and reading it in

        if not os.path.isfile(Constants.Paths.LUT.FULL_PATH_FILE):  #("data/common_data/lookUp.csv"):
            print("Downloading Look up table")
            lookUp_path = RawDataHandler.download_data("referencedata", # name of the folder online in Azure
                                         Constants.Paths.LUT.FILE)
            lookUp = pd.read_csv(lookUp_path)
        else:
            print("Reading Look up table")
            lookUp = pd.read_csv(Constants.Paths.LUT.FULL_PATH_FILE) #("data/common_data/lookUp.csv")
            
        ### %%
        ### TU files

        tus_hse_ref = np.unique(lookUp.NewTU[lookUp.MSOA11CD.isin(msoasList)])
        tus_hse = pd.DataFrame()

        # initially only try with the WYtest TUH file
        for x in tus_hse_ref:
            if not os.path.isfile(Constants.Paths.TU.FULL_PATH_FILE + x + ".csv"):
                print("Downloading the TU files")
                temp_path = RawDataHandler.download_data("countydata",
                                           Constants.Paths.TU.FILE + x + ".csv")
                temp = pd.read_csv(temp_path)
            else:
                print("Reading the TU files")
                temp = pd.read_csv(Constants.Paths.TU.FULL_PATH_FILE + x + ".csv")
            temp = temp[temp.MSOA11CD.isin(msoasList)]
            print("Combining TU files")
            tus_hse = tus_hse.append(temp)
        _combined_TU_file = tus_hse
            
        ### %%
        ### QUANT RAMP
        print(Constants.Paths.QUANT.FULL_PATH_FOLDER)
        if not os.path.isdir(Constants.Paths.QUANT.FULL_PATH_FOLDER): #("data/common_data/QUANT_RAMP/")
            print("Downloading the QUANT files")
            QUANT_path = RawDataHandler.download_data("nationaldata",
                                        "QUANT_RAMP.tar.gz")
            print("Unpacking QUANT files")
            RawDataHandler.unpack_data(QUANT_path)
            
        ### %%
        ###  commutingOD dl and selection

        if not os.path.isfile(Constants.Paths.COMMUTING.FULL_PATH_FILE): #("data/common_data/commutingOD.csv"):
            print("Downloading the CommutingOD file")
            OD_path = RawDataHandler.download_data("nationaldata",
                                     Constants.Paths.COMMUTING.FILE) #"commutingOD.gz")
            print("Unpacking the CommutingOD file")
            RawDataHandler.unpack_data(OD_path)
        OD = pd.read_csv(Constants.Paths.COMMUTING.FULL_PATH_FILE) #("data/common_data/commutingOD.csv")
        OD = OD[OD.HomeMSOA.isin(msoasList)]
        OD = OD[OD.DestinationMSOA.isin(msoasList)]

        ### %%
        ### Lockdown scenario

        # Assumption: look-up table already loaded before

        if not os.path.isfile(Constants.Paths.TIME_AT_HOME.FULL_PATH_FILE): #"data/common_data/timeAtHomeIncreaseCTY.csv"):
            print("Downloading the TimeAtHomeIncrease file (lockdown scenario)")
            lockdown_path = RawDataHandler.download_data("nationaldata",
                                           Constants.Paths.TIME_AT_HOME.FILE) #"timeAtHomeIncreaseCTY.csv")
            lockdown = pd.read_csv(lockdown_path)
        else:
            lockdown = pd.read_csv(Constants.Paths.TIME_AT_HOME.FULL_PATH_FILE) #"data/common_data/timeAtHomeIncreaseCTY.csv")
        
        if not os.path.isdir(Constants.Paths.MSOAS_FOLDER.FULL_PATH_FOLDER): #"data/common_data/MSOAS_shp/"):
            print("Downloading MSOAs shp for the GoogleMobility data")
            shp_path = RawDataHandler.download_data("nationaldata",
                                      Constants.Paths.MSOAS_SHP + ".tar.gz") #"MSOAS_shp.tar.gz")
            RawDataHandler.unpack_data(shp_path)
            
        shp = gpd.read_file(Constants.Paths.MSOAS_FOLDER.FULL_PATH_FOLDER) #"data/common_data/MSOAS_shp/msoas.shp")
        msoas_pop = shp["pop"]
        msoas_pop = msoas_pop[shp.MSOA11CD.isin(msoasList)]

        change_ref = np.unique(lookUp.GoogleMob[lookUp.MSOA11CD.isin(msoasList)])

        # average change within study area weighted by MSOA11CD population 
        cty_pop = np.repeat(0,len(change_ref))
        change = np.repeat(0,np.max(lockdown.day)+1)
        for x in range(0,len(change_ref)):
            cty_pop[x] = np.nansum(msoas_pop[lookUp.GoogleMob[lookUp.MSOA11CD.isin(msoasList)] == change_ref[x]])
            change = change + lockdown.change[lockdown.CTY20 == change_ref[x]]*cty_pop[x]
        change = change/np.sum(cty_pop)

        # From extra time at home to less time away from home
        lockdown = (1 - (np.mean(tus_hse.phome) * change))/np.mean(tus_hse.phome)

        ### %%
        ### Seeding

        # Assumption: shp already loaded before

        msoas_risks = shp.risk[shp.MSOA11CD.isin(msoasList)]


        #TODO: move this part to the main_model?
        # ### %%
        # ### Data for the OpenCL dashboard

        # Assumption: msoas.shp already loaded before
        # Assumption: tus_hse_ref already defined, see above

        osm_ref = np.unique(lookUp.OSM[lookUp.MSOA11CD.isin(msoasList)])
        url = osm_ref[0]
        target_path = os.path.join(Constants.Paths.COUNTY_DATA.FULL_PATH_FOLDER,
                                   tus_hse_ref[0] + ".zip") # ("data/common_data",tus_hse_ref[0] + ".zip")
        response = requests.get(url, stream=True)
        if response.status_code == 200: # HTTP status code for "OK"
            with open(target_path, 'wb') as f:
                f.write(response.raw.read())
        zip_file = zipfile.ZipFile(target_path)
        zip_file.extractall(Constants.Paths.OSM_FOLDER.FULL_PATH_FOLDER + tus_hse_ref[0]) # ("data/common_data/" + tus_hse_ref[0])
        
        
        osmShp = gpd.read_file(os.path.join(Constants.Paths.OSM_FOLDER.FULL_PATH_FOLDER, tus_hse_ref[0], Constants.Paths.OSM_FILE))
                               # ("data/common_data/" + tus_hse_ref[0] + "/gis_osm_buildings_a_free_1.shp")

        # If study area accross more than one County, dl other counties and combine shps into one
        if len(osm_ref)>1:
            for x in range(1,len(osm_ref)):
                url = osm_ref[x]
                target_path = os.path.join(Constants.Paths.OSM_FOLDER, tus_hse_ref[x] + ".zip")
                # ("data/common_data",tus_hse_ref[x] + ".zip")
                response = requests.get(url, stream=True)
            if response.status_code == 200: # HTTP status code for "OK"
                with open(target_path, 'wb') as f:
                    f.write(response.raw.read())
            zip_file = zipfile.ZipFile(target_path)
            zip_file.extractall(os.path.join(Constants.Paths.OSM_FOLDER, tus_hse_ref[x])) #("data/common_data/" + tus_hse_ref[x])
            osmShp = pd.concat([
                    osmShp,
                    gpd.read_file(os.path.join(Constants.Paths.OSM_FOLDER, tus_hse_ref[x], Constants.Paths.OSM_FILE))
                    #("data/common_data/" + tus_hse_ref[x] + "/gis_osm_buildings_a_free_1.shp")
                    ]).pipe(gpd.GeoDataFrame)
            
        # TO_DO
        #  -> branch to load "load_msoa_locations.py" code
        # Find centroid of intersected shp
        # extract risks from shp dbf
        return
    
    ### %%
    ### Defining functions to download data from Azure repository and unpack them right after
    @staticmethod
    def download_data(folder: str, file: str):
        """
        Download data utility function
        Args:
            folder (str): can be: nationaldata, countydata or referencedata.
            file (str): name of the file, must include the extension.
        """
        url = os.path.join(Constants.Paths.AZURE_URL + folder, file)  # TO_DO: does this work written like this?
        target_path = os.path.join(Constants.Paths.DATA.FULL_PATH_FOLDER,
                                   file)
        response = requests.get(url, stream=True)
        if response.status_code == 200:  # Ie checking that the HTTP status code is 'OK'
            with open(target_path, 'wb') as f:
                f.write(response.raw.read())
        return target_path

    @staticmethod
    def unpack_data(archive: str):
        """
        Unpack tar data archive
        Args:
            archive (str): A string directory path to archive file using ## ???
        """
        tar_file = tarfile.open(archive)
        tar_file.extractall(archive)  # ("data/common_data/")  ### extract all or not? is it correct 'archive' here??
        return

        
    # @staticmethod
    # def getCombinedTUFile():
    #     if not _combined_TU_file:
    #         raise Exception("TU file hasn't been created")
    #     return _combined_TU_file
    #
    # @staticmethod
    # def getCombinedShpFile():
    #     if not _combined_shp_file:
    #         raise Exception("MSOA shp file hasn't been created")
    #     return _combined_shp_file
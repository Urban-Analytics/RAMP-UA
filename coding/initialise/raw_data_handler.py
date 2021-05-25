#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handling the raw data for the model initialisation

Created on Thu Apr 22

@author: Anna on Hadrien's code
"""

#TO_DO: add columns names to Constants file (example: MSOA11CD, pop from msoa_shp, CTY20)

import os
import pandas as pd
import requests
import tarfile
import zipfile
import geopandas as gpd
import numpy as np
import gzip # to open .gz files

from coding.constants import Constants
from coding.constants import ColumnNames

class RawDataHandler:
    _combined_TU_file = None
    _combined_shp_file = None
    _lockdown_file = None

    def __init__(self,
                 list_of_msoas):

        """
        Class that handles the data download and unpack
        """
        # (Hadrien's code)

        ### %%
        ### Reading in msoas list
        # assumption: msoas_list variable read from input provided by the user
        # os.chdir("/Users/hsalat/MiscPython")
        # msoas_list = pd.read_csv("/Users/hsalat/West_Yorkshire/Seeding/msoas.csv")
        # msoas_list = msoas_list["area_code"]

        # Note: this step needs to be improved by creating a formal way of introducing the list and checking its format is correct
        # msoas_list_file = pd.read_csv(Constants.Paths.LIST_MSOAS.FULL_PATH_FILE)  ### Needs to be put as initial parameter, for now in Constants
        msoas_list_file = pd.read_csv(list_of_msoas)
        msoas_list = msoas_list_file[ColumnNames.MSOAsID]     # some times "area_code" or "area", current test file has column name "MSOA11CD"

        ### %%
        ###  Checking that look-up table exists and reading it in
        lut_file_with_path = Constants.Paths.LUT.FULL_PATH_FILE
        if not os.path.isfile(lut_file_with_path):  #("data/common_data/lookUp.csv"):
            print("Downloading Look up table")
            RawDataHandler.download_data(remote_folder="referencedata", # name of the folder online in Azure
                                         local_folder=Constants.Paths.REFERENCE_DATA.FULL_PATH_FOLDER,
                                         file=Constants.Paths.LUT.FILE)
            # lut = pd.read_csv(lut_file_with_path)
        else:
            print(f"I'm not downloading the look-up table as {lut_file_with_path} already exists")
        print(f"Reading Look up table from {lut_file_with_path}")
        lut = pd.read_csv(lut_file_with_path) #("data/common_data/lookUp.csv")

        ### %%
        ### TU files
        # tus_hse_ref = np.unique(lut.NewTU[lut.MSOA11CD.isin(msoas_list)])
        tus_hse_ref = np.unique(lut.NewTU[lut[ColumnNames.MSOAsID].isin(msoas_list)])
        tus_hse = pd.DataFrame()
        # initially only try with the WYtest TUH file (fake ad-hoc file)
        for x in tus_hse_ref:
            remote_tus_folder = "countydata"
            local_tus_folder = Constants.Paths.COUNTY_DATA.FULL_PATH_FOLDER
            packed_tus_file = Constants.Paths.TU.FILE + x + ".gz"
            unpacked_tus_file_with_path = Constants.Paths.TU.FULL_PATH_FILE + x + ".csv"
            packed_tus_file_with_path = os.path.join(local_tus_folder,
                                                     packed_tus_file)
            if not os.path.isfile(unpacked_tus_file_with_path):
                print("Downloading the TU files...")
                RawDataHandler.download_data(remote_folder=remote_tus_folder,  # name of the folder online in Azure
                                             local_folder=local_tus_folder,
                                             file=packed_tus_file)
                print("... done!")
                print("Unpacking TU files")
                if not os.path.isfile(unpacked_tus_file_with_path):
                    print("Check: there is no csv file, I'm unpacking the gz file now")
                RawDataHandler.unpack_data(packed_file=packed_tus_file_with_path,
                                           destination_folder=local_tus_folder)
                temp = pd.read_csv(unpacked_tus_file_with_path)
            else:
                print(f"I'm not downloading the TUS files as {unpacked_tus_file_with_path} already exists")
                print("Reading the TU files")
                file_name = unpacked_tus_file_with_path
                temp = pd.read_csv(unpacked_tus_file_with_path)
                print(f"File is {file_name}")
            temp = temp[temp[ColumnNames.MSOAsID].isin(msoas_list)]
            print("Combining TU files")
            tus_hse = tus_hse.append(temp)
            self._combined_TU_file = tus_hse

        ### %%
        ### QUANT RAMP
        # print(Constants.Paths.QUANT.FULL_PATH_FOLDER)
        remote_quant_folder = "nationaldata"
        local_quant_folder = Constants.Paths.NATIONAL_DATA.FULL_PATH_FOLDER
        packed_quant_file = "QUANT_RAMP.tar.gz"
        unpacked_quant_folder_with_path = Constants.Paths.QUANT.FULL_PATH_FOLDER
        packed_quant_file_with_path = os.path.join(local_quant_folder,
                                                   packed_quant_file)
        if not os.path.isdir(unpacked_quant_folder_with_path): #("data/common_data/QUANT_RAMP/")
            print("Downloading the QUANT files...")
            RawDataHandler.download_data(remote_folder=remote_quant_folder,
                                         local_folder=local_quant_folder,
                                         file=packed_quant_file)
            print("... done!")
            print("Unpacking QUANT files")
            RawDataHandler.unpack_data(packed_file=packed_quant_file_with_path,
                                       destination_folder=local_quant_folder)
        else:
            print(f"I'm not downloading the QUANT data as {unpacked_quant_folder_with_path} already exists")
        ### %%
        ###  commutingOD dl and selection
        remote_commuting_folder = "nationaldata"
        local_commuting_folder = Constants.Paths.NATIONAL_DATA.FULL_PATH_FOLDER
        packed_commuting_file = "commutingOD.gz"
        unpacked_commuting_file_with_path = Constants.Paths.COMMUTING.FULL_PATH_FILE
        packed_commuting_file_with_path = os.path.join(local_commuting_folder,
                                                   packed_commuting_file)
        if not os.path.isfile(unpacked_commuting_file_with_path): #("data/common_data/commutingOD.csv"):
            print("Downloading the CommutingOD file...")
            RawDataHandler.download_data(remote_folder=remote_commuting_folder,
                                         local_folder=local_commuting_folder,
                                         file=packed_commuting_file)  # "commutingOD.gz")
            print("... done!")
            print("Unpacking the CommutingOD file")
            RawDataHandler.unpack_data(packed_file=packed_commuting_file_with_path,
                                       destination_folder=local_commuting_folder)
        OD = pd.read_csv(unpacked_commuting_file_with_path) #("data/common_data/commutingOD.csv")
        OD = OD[OD.HomeMSOA.isin(msoas_list)]
        OD = OD[OD.DestinationMSOA.isin(msoas_list)]

        # Note: Add method like for other files to be able to get this variable inside the code?
        # (see end of the script, where we do this for the tus and other files)

        ### EDITED UNTIL HERE ### (AZ)


        ### %%
        ### Lockdown scenario
        # Assumption: look-up table already loaded before
        if not os.path.isfile(Constants.Paths.TIME_AT_HOME.FULL_PATH_FILE): #"data/common_data/timeAtHomeIncreaseCTY.csv"):
            print("Downloading the TimeAtHomeIncrease file (lockdown scenario)")
            lockdown_path = RawDataHandler.download_data("nationaldata",
                                                         local_folder="",
                                                         file=Constants.Paths.TIME_AT_HOME.FILE) #"timeAtHomeIncreaseCTY.csv")
            lockdown = pd.read_csv(lockdown_path)
        else:
            print("Reading the TimeAtHomeIncrease file (lockdown scenario)")
            lockdown = pd.read_csv(Constants.Paths.TIME_AT_HOME.FULL_PATH_FILE) #"data/common_data/timeAtHomeIncreaseCTY.csv")

        if not os.path.isdir(Constants.Paths.MSOAS_FOLDER.FULL_PATH_FOLDER): #"data/common_data/MSOAS_shp/"):
            print("Downloading MSOAs shp for the GoogleMobility data")
            shp_path = RawDataHandler.download_data("nationaldata",
                                                    local_folder="",
                                                    file=Constants.Paths.MSOAS_SHP + ".tar.gz") #"MSOAS_shp.tar.gz")
            RawDataHandler.unpack_data(shp_path)

        shp = gpd.read_file(Constants.Paths.MSOAS_FOLDER.FULL_PATH_FOLDER) #"data/common_data/MSOAS_shp/msoas.shp")
        msoas_pop = shp["pop"]
        # msoas_pop = msoas_pop[shp.MSOA11CD.isin(msoas_list)]
        msoas_pop = msoas_pop[shp[ColumnNames.MSOAsID].isin(msoas_list)]

        # change_ref = np.unique(lut.GoogleMob[lut.MSOA11CD.isin(msoas_list)])
        change_ref = np.unique(lut.GoogleMob[lut[ColumnNames.MSOAsID].isin(msoas_list)])

        # average change within study area weighted by msoa population
        cty_pop = np.repeat(0, len(change_ref))
        change = np.repeat(0, np.max(lockdown.day)+1)
        for x in range(0, len(change_ref)):
            cty_pop[x] = np.nansum(msoas_pop[lut.GoogleMob[lut[ColumnNames.MSOAsID].isin(msoas_list)] == change_ref[x]])
            # match = lockdown.change[lockdown.CTY20 == change_ref[x]]   # error: wy repeats 6 times
            # moltip = match * cty_pop[x]
            # verif = change + moltip
            # change = verif
            change = change + lockdown.change[lockdown.CTY20 == change_ref[x]] * cty_pop[x]
        change = change/np.sum(cty_pop)

        # From extra time at home to less time away from home
        lockdown = (1 - (np.mean(tus_hse.phome) * change))/np.mean(tus_hse.phome)
        self._lockdown_file = lockdown

        ### %%
        ### Seeding
        # Assumption: shp already loaded before
        # msoas_risks = shp.risk[shp.MSOA11CD.isin(msoas_list)]
        msoas_risks = shp.risk[shp[ColumnNames.MSOAsID].isin(msoas_list)]
        # ### %%
        # ### Data for the OpenCL dashboard
        # Assumption: msoas.shp already loaded before
        # Assumption: tus_hse_ref already defined, see above
        print("Downloading OSM data")
        osm_ref = np.unique(lut.OSM[lut[ColumnNames.MSOAsID].isin(msoas_list)])
        url = osm_ref[0]
        target_path = os.path.join(Constants.Paths.OSM_FOLDER.FULL_PATH_FOLDER,
                                   tus_hse_ref[0] + ".zip") # ("data/common_data",tus_hse_ref[0] + ".zip")
        response = requests.get(url, stream=True)

        if not response.ok: # HTTP status code for "OK"
            raise Exception("Error downloading OSM data")

        with open(target_path, 'wb') as f:
            f.write(response.raw.read())
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                raise Exception("Error downloading OSM data: file is empty")
        zip_file = zipfile.ZipFile(target_path)
        print("Downloaded file, will now extract it...")
        zip_file.extractall(os.path.join(Constants.Paths.OSM_FOLDER.FULL_PATH_FOLDER,
                                         tus_hse_ref[0])) # ("data/common_data/" + tus_hse_ref[0])
        print("extracted!")

        osmShp = gpd.read_file(os.path.join(Constants.Paths.OSM_FOLDER.FULL_PATH_FOLDER,
                                            tus_hse_ref[0],
                                            Constants.Paths.OSM_FILE.FILE))
                               # ("data/common_data/" + tus_hse_ref[0] + "/gis_osm_buildings_a_free_1.shp")

        # If study area across more than one County, dl other counties and combine shps into one
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
            print("Combining OSM shapefiles together")
            osmShp = pd.concat([
                    osmShp,
                    gpd.read_file(os.path.join(Constants.Paths.OSM_FOLDER, tus_hse_ref[x], Constants.Paths.OSM_FILE))
                    #("data/common_data/" + tus_hse_ref[x] + "/gis_osm_buildings_a_free_1.shp")
                    ]).pipe(gpd.GeoDataFrame)
            self._combined_shp_file = osmShp

        # TO_DO
        #  branch to load "load_msoa_locations.py" code -> DONE
        # Find centroid of intersected shp -> DONE
        # extract risks from shp dbf
        return


    ### %%
    ### Defining functions to download data from Azure repository and unpack them right after
    @staticmethod
    def download_data(remote_folder: str,
                      local_folder:str,
                      file: str):
        """
        Download data utility function
        Args:
            folder (str): can be: nationaldata, countydata or referencedata.
            file (str): name of the file, must include the extension.
        """
        url = os.path.join(Constants.Paths.AZURE_URL + remote_folder, file)
        target_path = os.path.join(local_folder,#Constants.Paths.COUNTY_DATA.FULL_PATH_FOLDER,
                                   file)
        response = requests.get(url, stream=True)
        if not response.ok: # Ie checking that the HTTP status code is 'OK'
            raise Exception(f"Error downloading file {file}")

        with open(target_path, 'wb') as f:
            f.write(response.raw.read())
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                raise Exception(f"Error downloading {file}: file is empty")
        return

    @staticmethod
    # def unpack_data(archive: str):
    def unpack_data(packed_file: str, destination_folder: str):
        """
        Unpack tar data archive
        Args:
            archive (str): A string directory path to archive file using ## ???
        """
        tar_file = tarfile.open(packed_file)
        tar_file.extractall(destination_folder)  # ("data/common_data/")  ### extract all or not? is it correct 'archive' here??
        return


    def getCombinedTUFile(self):
        if self._combined_TU_file is None:
            raise Exception("TU file hasn't been created")
        return self._combined_TU_file

    def getCombinedShpFile(self):
        if self._combined_shp_file is None:
            raise Exception("MSOA shp file hasn't been created")
        return self._combined_shp_file

    def getLockdownFile(self):
        if self._lockdown_file is None:
            raise Exception("Lockdown file not found")
        return self._lockdown_file
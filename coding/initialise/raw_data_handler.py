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

        """
        MSOAs list
        """
        # Reading in msoas list
        # assumption: msoas_list variable read from input provided by the user
        # os.chdir("/Users/hsalat/MiscPython")
        # msoas_list = pd.read_csv("/Users/hsalat/West_Yorkshire/Seeding/msoas.csv")
        # msoas_list = msoas_list["area_code"]

        # Note: this step needs to be improved by creating a formal way of introducing the list and checking its format is correct
        # msoas_list_file = pd.read_csv(Constants.Paths.LIST_MSOAS.FULL_PATH_FILE)  ### Needs to be put as initial parameter, for now in Constants
        msoas_list_file = pd.read_csv(list_of_msoas)
        msoas_list = msoas_list_file[ColumnNames.MSOAsID]     # some times "area_code" or "area", current test file has column name "MSOA11CD"

        """
        Look-up table
        """
        # Checking that look-up table exists and reading it in
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

        """
        TUS files
        """
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
                print("... finished unpacking!")
                temp = pd.read_csv(unpacked_tus_file_with_path)
            else:
                print(f"I'm not downloading the TUS files as {unpacked_tus_file_with_path} already exists")
                print("Reading the TU files")
                temp = pd.read_csv(unpacked_tus_file_with_path)
                print("...done!")
            temp = temp[temp[ColumnNames.MSOAsID].isin(msoas_list)]
            print("Combining TU files")
            tus_hse = tus_hse.append(temp)
            self._combined_TU_file = tus_hse

        """
        QUANT RAMP
        """
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
            print("... finished unpacking!")
        else:
            print(f"I'm not downloading the QUANT data as {unpacked_quant_folder_with_path} already exists")

        """
        CommutingOD
        """
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
            print("... finished unpacking!")
        print(f"I'm not downloading the commutingOD file as {unpacked_commuting_file_with_path} already exists")
        print("Reading the commutingOD file...")
        OD = pd.read_csv(unpacked_commuting_file_with_path) #("data/common_data/commutingOD.csv")
        print("... done!")
        OD = OD[OD.HomeMSOA.isin(msoas_list)]
        OD = OD[OD.DestinationMSOA.isin(msoas_list)]
        self._origindestination_file = OD
        # Note: Added method similar to other variables to be able to get this variable inside the code
        # (see end of the script, where we do this for the tus and other files)

        """
        Lockdown scenario
        """
        # Assumption: look-up table already loaded before
        remote_lockdown_folder = "nationaldata"
        local_lockdown_folder = Constants.Paths.NATIONAL_DATA.FULL_PATH_FOLDER
        unpacked_lockdown_file = Constants.Paths.TIME_AT_HOME.FILE
        unpacked_lockdown_file_with_path = Constants.Paths.TIME_AT_HOME.FULL_PATH_FILE
        if not os.path.isfile(unpacked_lockdown_file_with_path): #"data/common_data/timeAtHomeIncreaseCTY.csv"):
            print("Downloading the TimeAtHomeIncrease file (lockdown scenario)...")
            RawDataHandler.download_data(remote_folder=remote_lockdown_folder,
                                         local_folder=local_lockdown_folder,
                                         file=unpacked_lockdown_file) #"timeAtHomeIncreaseCTY.csv")
            print("... done!")
        else:
            print(f"I'm not downloading the TimeAtHomeIncrease file as {unpacked_lockdown_file_with_path} already exists")
        print("Reading the TimeAtHomeIncrease file (lockdown scenario)")
        lockdown = pd.read_csv(unpacked_lockdown_file_with_path) #"data/common_data/timeAtHomeIncreaseCTY.csv")
        self._lockdown_file = lockdown
        # Note: Added method similar to other variables to be able to get this variable inside the code
        # (see end of the script, where we do this for the tus and other files)

        remote_shp_folder = "nationaldata"
        local_shp_folder = Constants.Paths.NATIONAL_DATA.FULL_PATH_FOLDER
        packed_shp_file = "MSOAS_shp.tar.gz"
        unpacked_shp_folder_with_path = Constants.Paths.MSOAS_FOLDER.FULL_PATH_FOLDER
        packed_shp_file_with_path = os.path.join(local_shp_folder,
                                                 packed_shp_file)
        unpacked_shp_file_with_path = Constants.Paths.MSOAS_SHP.FULL_PATH_FILE

        if not os.path.isdir(unpacked_shp_folder_with_path): #"data/common_data/MSOAS_shp/"):
            print("Downloading MSOAs shp for the GoogleMobility data...")
            RawDataHandler.download_data(remote_folder=remote_shp_folder,
                                         local_folder=local_shp_folder,
                                         file=packed_shp_file) #"MSOAS_shp.tar.gz")
            print("... done!")
            print("Unpacking the MSOAs shapefile")
            RawDataHandler.unpack_data(packed_file=packed_shp_file_with_path,
                                       destination_folder=local_shp_folder)
            print("... finished unpacking!")
        else:
            print(f"I'm not downloading the MSOAs shapefile as {unpacked_shp_folder_with_path} already exists")
        print("Dealing with the TimeAtHomeIncrease data...")
        shp = gpd.read_file(unpacked_shp_file_with_path) #"data/common_data/MSOAS_shp/msoas.shp")
        msoas_pop = shp[ColumnNames.MSOAS_SHP_POP]  # "pop"
        # msoas_pop = msoas_pop[shp.MSOA11CD.isin(msoas_list)]
        msoas_pop = msoas_pop[shp[ColumnNames.MSOAsID].isin(msoas_list)]  # MSOA11CD

        # change_ref = np.unique(lut.GoogleMob[lut.MSOA11CD.isin(msoas_list)])
        change_ref = np.unique(lut.GoogleMob[lut[ColumnNames.MSOAsID].isin(msoas_list)])

        # average change within study area weighted by msoa population
        cty_pop = np.repeat(0, len(change_ref))
        change = np.repeat(0, np.max(lockdown.day)+1)
        for x in range(0, len(change_ref)):
            cty_pop[x] = np.nansum(msoas_pop[lut.GoogleMob[lut[ColumnNames.MSOAsID].isin(msoas_list)] == change_ref[x]])
            # match = lockdown.change[lockdown[ColumnNames.LOCKDOWN_CTY_NAME] == change_ref[x]]   # CTY20
            # moltip = match * cty_pop[x]
            # verif = change + moltip
            # change = verif
            change = change + lockdown.change[lockdown[ColumnNames.LOCKDOWN_CTY_NAME] == change_ref[x]] * cty_pop[x]  # CTY20
        change = change/np.sum(cty_pop)
        print("... done!")
        # From extra time at home to less time away from home
        lockdown = (1 - (np.mean(tus_hse.phome) * change))/np.mean(tus_hse.phome)
        self._lockdown_file = lockdown

        """
        Seeding
        """
        # Assumption: shp already loaded before
        # msoas_risks = shp.risk[shp.MSOA11CD.isin(msoas_list)]
        msoas_risks = shp.risk[shp[ColumnNames.MSOAsID].isin(msoas_list)]
        self._msoas_risk_list = msoas_risks
        # Note: Added method similar to other variables to be able to get this variable inside the code
        # (see end of the script, where we do this for the tus and other files)

        """
        Data for the OpenCL dashboard
        """
        # Assumption: msoas.shp already loaded before
        # Assumption: tus_hse_ref already defined, see above
        osm_ref = np.unique(lut.OSM[lut[ColumnNames.MSOAsID].isin(msoas_list)])
        url = osm_ref[0]
        local_osm_folder = Constants.Paths.OSM_FOLDER.FULL_PATH_FOLDER
        packed_osm_file = tus_hse_ref[0] + ".zip"
        packed_osm_folder_with_path = os.path.join(local_osm_folder,
                                                   packed_osm_file)
        unpacked_osm_folder_with_path = os.path.join(local_osm_folder,
                                                     tus_hse_ref[0])
        if not os.path.isdir(unpacked_osm_folder_with_path):
            print("Downloading OSM data...")
            target_path = packed_osm_folder_with_path # ("data/common_data",tus_hse_ref[0] + ".zip")
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
            zip_file.extractall(unpacked_osm_folder_with_path) # ("data/common_data/" + tus_hse_ref[0])
            print("extracted!")

        osm_shp = gpd.read_file(os.path.join(unpacked_osm_folder_with_path,
                                             Constants.Paths.OSM_FILE.FILE))
                               # ("data/common_data/" + tus_hse_ref[0] + "/gis_osm_buildings_a_free_1.shp")

        # If study area across more than one County, download other counties and combine shapefiles into one
        if len(osm_ref) > 1:
            for x in range(1, len(osm_ref)):
                url = osm_ref[x]
                packed_osm_file = tus_hse_ref[x] + ".zip"
                local_osm_folder = Constants.Paths.OSM_FOLDER.FULL_PATH_FOLDER
                packed_osm_file = tus_hse_ref[0] + ".zip"
                packed_osm_folder_with_path = os.path.join(local_osm_folder,
                                                           packed_osm_file)
                unpacked_osm_folder_with_path = os.path.join(local_osm_folder,
                                                             tus_hse_ref[0])

                target_path = packed_osm_folder_with_path
                # ("data/common_data",tus_hse_ref[x] + ".zip")
                response = requests.get(url, stream=True)
                if response.ok: #status_code == 200: # HTTP status code for "OK"
                    with open(target_path, 'wb') as f:
                        f.write(response.raw.read())
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        if size == 0:
                            raise Exception("Error downloading OSM data: file is empty")

                zip_file = zipfile.ZipFile(target_path)
                zip_file.extractall(unpacked_osm_folder_with_path) #("data/common_data/" + tus_hse_ref[x])
            print("Combining OSM shapefiles together")
            osm_shp = pd.concat([
                    osm_shp,
                    gpd.read_file(os.path.join(unpacked_osm_folder_with_path,
                                               Constants.Paths.OSM_FILE.FILE))
                    #("data/common_data/" + tus_hse_ref[x] + "/gis_osm_buildings_a_free_1.shp")
                    ]).pipe(gpd.GeoDataFrame)
            self._combined_shp_file = osm_shp

        # TO_DO
        #  branch to load "load_msoa_locations.py" code -> DONE
        # Find centroid of intersected shp -> DONE
        # extract risks from shp dbf -> DONE
        # add the "get..." variables (se bottom of the script) in the rest of code when they are called
        return

    # Defining functions to download data from Azure repository and unpack them right after
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

    # Store variables internally to be able to call them later
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

    def getOriginDestinationFile(self):
        if self._origindestination_file is None:
            raise Exception("Origin-Destination commuting file not found")
        return self._origindestination_file

    def getMsoasRiskList(self):
        if self._msoas_risk_list is None:
            raise Exception("MSOAs risk file not found")
        return self._msoas_risk_list

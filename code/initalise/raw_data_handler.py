from code.constants import Constants

class RawDataHandler:        

    @staticmethod
    def run():
        """
        Data download and unpack
        """
        # (Hadrien's code)

        ### %%
        ### Reading in msoas list

        # assumption: msoasList variable read from input provided by the user
        # os.chdir("/Users/hsalat/MiscPython")
        # msoasList = pd.read_csv("/Users/hsalat/West_Yorkshire/Seeding/msoas.csv")
        # msoasList = msoasList["area_code"]
        # Note: this step needs to be improved by creating a formal way of introducing the list and checking its format is correct
        msoasList = pd.read_csv(os.path.join(home_dir,
                                            Constants.Paths.LIST_MSOAS_FILE))  ### Needs to be put as initial parameter, for now in Constants
        ### %%
        ###  Checking that look-up table exists and reading it in

        if not os.path.isfile(Constants.Paths.LUT.FULL_PATH_FILE):  #("data/common_data/lookUp.csv"):
            lookUp_path = _download_data("referencedata",Constants.Paths.LUT.FILE)
            lookUp = pd.read_csv(lookUp_path)
        else:
            lookUp = pd.read_csv(Constants.Paths.LUT.FULL_PATH_FILE)) #("data/common_data/lookUp.csv")
            
        ### %%
        ### TU files

        tus_hse_ref = np.unique(lookUp.NewTU[lookUp.MSOA11CD.isin(msoasList)])
        tus_hse = pd.DataFrame()

        # initially only try with the WYtest TUH file
        for x in tus_hse_ref:
            if not os.path.isfile("data/common_data/tus_hse_" + x + ".csv"):
                temp_path = _download_data("countydata","tus_hse_" + x + ".csv")
                temp = pd.read_csv(temp_path)
            else:
                temp = pd.read_csv("data/common_data/tus_hse_" + x + ".csv")
            temp = temp[temp.MSOA11CD.isin(msoasList)]
            tus_hse = tus_hse.append(temp)
            
        ### %%
        ### QUANT RAMP

        if not os.path.isdir("data/common_data/QUANT_RAMP/")
            QUANT_path = _download_data("nationaldata","QUANT_RAMP.tar.gz")
            _unpack_data(QUANT_path)
            
        ### %%
        ###  commutingOD dl and selection

        if not os.path.isfile("data/common_data/commutingOD.csv"):
            OD_path = _download_data("nationaldata","commutingOD.gz")
            _unpack_data(OD_path)
        OD = pd.read_csv("data/common_data/commutingOD.csv")
        OD = OD[OD.HomeMSOA.isin(msoasList)]
        OD = OD[OD.DestinationMSOA.isin(msoasList)]

        ### %%
        ### Lockdown scenario

        # In theory: lookUp already loaded before

        if not os.path.isfile("data/common_data/timeAtHomeIncreaseCTY.csv"):
            lockdown_path = _download_data("nationaldata","timeAtHomeIncreaseCTY.csv")
            lockdown = pd.read_csv(lockdown_path)
        else:
            lockdown = pd.read_csv("data/common_data/timeAtHomeIncreaseCTY.csv")
        if not os.path.isdir("data/common_data/MSOAS_shp/"):
            shp_path = _download_data("nationaldata","MSOAS_shp.tar.gz")
            _unpack_data(shp_path)
            
        shp = gpd.read_file("data/common_data/MSOAS_shp/msoas.shp")
        msoas_pop = shp["pop"]
        msoas_pop = msoas_pop[shp.MSOA11CD.isin(msoasList)]

        change_ref = np.unique(lookUp.GoogleMob[lookUp.MSOA11CD.isin(msoasList)])

        # average change within studied area weighted by MSOA11CD population 
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

        # In theory: shp already loaded before

        msoas_risks = shp.risk[shp.MSOA11CD.isin(msoasList)]

        ### %%
        ### Dashboard material

        # In theory: msoas.shp already loaded before
        # In theory: tus_hse_ref already defined, see above

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

        # If study area accross more than one County, dl other counties and combine shps into one
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
            
        # TO_DO
        #  -> branch to load "load_msoa_locations.py" code
        # Find centroid of intersected shp
        # extract risks from shp dbf
    
    ### %%
    ### Defining functions to download data from Azure repository and unpack them right after

    def _download_data(folder: str,file : str):
        """
        Download data utility function
        Args:
            folder (str): can be: nationaldata, countydata or referencedata.
            file (str): name of the file, must include the extension.
        """
        url = Constants.Paths.AZURE_URL + folder + "/" + file   # TO_DO: does this work written like this?
        target_path = os.path.join(,
                                file)
        response = requests.get(url, stream=True)
        if response.status_code == 200:  # Ie checking that the HTTP status code is 'OK'
            with open(target_path, 'wb') as f:
                f.write(response.raw.read())
        return target_path

    def _unpack_data(archive : str):
        """
        Unpack tar data archive
        Args:
            archive (str): A string directory path to archive file using
        """
        tar_file = tarfile.open(archive)
        tar_file.extractall(data_dir) # ("data/common_data/")  ### extract all or not?
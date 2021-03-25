import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import warnings

class QuantRampAPI:
    """
    Class that handles integration of QUANT data into the RAMP microsim model
    QUANT spatial interaction data include probabilities of trips from MSOA 
    or IZ origins to primary schools, secondary schools and retail locations.
    Based on QUANTRampAPI.py provided by UCL
    """

    def __init__(self,
                 quant_dir: str = "QUANT_RAMP",
                 test_mode: bool = False
                 ):
        """
        Initialiser for QuantRampAPI This reads all of the necessary data.
        ----------
        :param quant_dir: Full path to QUANT files
        :param test_mode: Mode used running code tests.
    
        """
        self.QUANT_DIR = quant_dir

        if test_mode:
            warnings.warn("IMPORTANT! QUANT is running in test mode. This should only be used when running tests.",
                          UserWarning)

        # read in and store data 
        QuantRampAPI.read_data(self.QUANT_DIR, test_mode)
     

    @classmethod
    def read_data(cls,QUANT_DIR, test_mode):
        """
        reads in all data in provided data directory and creates series of class object attributes
        """
        # In test mode create small array to represent retail flows (8*8 matrix of small numbers)
        test_matrix = np.tile(np.array([0.001] * 8), (8, 1))

        cls.dfPrimaryPopulation = pd.read_csv(os.path.join(QUANT_DIR,'primaryPopulation.csv'))
        cls.dfPrimaryZones = pd.read_csv(os.path.join(QUANT_DIR,'primaryZones.csv'))
        if test_mode:
            cls.primary_probPij = np.ndarray.copy(test_matrix)
        else:
            cls.primary_probPij = pickle.load( open(os.path.join(QUANT_DIR,'primaryProbPij.bin'), 'rb'))
        
        cls.dfSecondaryPopulation = pd.read_csv(os.path.join(QUANT_DIR,'secondaryPopulation.csv'))
        cls.dfSecondaryZones = pd.read_csv(os.path.join(QUANT_DIR,'secondaryZones.csv'))
        if test_mode:
            cls.secondary_probPij  = np.ndarray.copy(test_matrix)
        else:
            cls.secondary_probPij = pickle.load( open(os.path.join(QUANT_DIR,'secondaryProbPij.bin'), 'rb'))
        
        cls.dfRetailPointsPopulation = pd.read_csv(os.path.join(QUANT_DIR,'retailpointsPopulation.csv'))
        cls.dfRetailPointsZones = pd.read_csv(os.path.join(QUANT_DIR,'retailpointsZones.csv'))
        if test_mode:
            cls.retailpoints_probSij  = np.ndarray.copy(test_matrix)
        else:
            cls.retailpoints_probSij = pickle.load( open(os.path.join(QUANT_DIR,'retailpointsProbSij.bin'), 'rb'))
        
        cls.dfHospitalPopulation = pd.read_csv(os.path.join(QUANT_DIR,'hospitalPopulation.csv'))
        cls.dfHospitalZones = pd.read_csv(os.path.join(QUANT_DIR,'hospitalZones.csv'))
        if test_mode:
            cls.hospital_probHij  = np.ndarray.copy(test_matrix)
        else:
            cls.hospital_probHij = pickle.load( open(os.path.join(QUANT_DIR,'hospitalProbHij.bin'), 'rb'))




    @staticmethod
    def getProbablePrimarySchoolsByMSOAIZ(dfPrimaryPopulation,dfPrimaryZones,primary_probPij,msoa_iz,threshold):
        """
        getProbablePrimarySchoolsByMSOAIZ
        Given an MSOA area code (England and Wales) or an Intermediate Zone (IZ) 2001 code (Scotland), return
        a list of all the surrounding primary schools whose probabilty of being visited by the MSOA_IZ is
        greater than or equal to the threshold.
        School ids are taken from the Edubase list of URN
        NOTE: code identical to the secondary school version, only with switched lookup tables
        @param msoa_iz An MSOA code (England/Wales e.g. E02000001) or an IZ2001 code (Scotland e.g. S02000001)
        @param threshold Probability threshold e.g. 0.5 means return all possible schools with probability>=0.5
        @returns a list of probabilities in the same order as the venues
        """
        result = []
        zonei = int(dfPrimaryPopulation.loc[dfPrimaryPopulation['msoaiz'] == msoa_iz,'zonei'])
        m,n = primary_probPij.shape
        for j in range(n):
            p = primary_probPij[zonei,j]
            if p>=threshold:
                row2 = dfPrimaryZones.loc[dfPrimaryZones['zonei'] == j] #yes, zonei==j is correct, they're always called 'zonei'
                id = row2['URN'].values[0]
                result.append(p)
            #end if
        #end for
        return result


    @staticmethod
    def getProbableSecondarySchoolsByMSOAIZ(dfSecondaryPopulation,dfSecondaryZones,secondary_probPij,msoa_iz,threshold):
        """
        getProbableSecondarySchoolsByMSOAIZ
        Given an MSOA area code (England and Wales) or an Intermediate Zone (IZ) 2001 code (Scotland), return
        a list of all the surrounding secondary schools whose probabilty of being visited by the MSOA_IZ is
        greater than or equal to the threshold.
        School ids are taken from the Edubase list of URN
        NOTE: code identical to the primary school version, only with switched lookup tables
        @param msoa_iz An MSOA code (England/Wales e.g. E02000001) or an IZ2001 code (Scotland e.g. S02000001)
        @param threshold Probability threshold e.g. 0.5 means return all possible schools with probability>=0.5
        @returns a list of probabilities in the same order as the venues
        """
        result = []
        zonei = int(dfSecondaryPopulation.loc[dfSecondaryPopulation['msoaiz'] == msoa_iz, 'zonei'])
        m,n = secondary_probPij.shape
        for j in range(n):
            p = secondary_probPij[zonei,j]
            if p>=threshold:
                row2 = dfSecondaryZones.loc[dfSecondaryZones['zonei'] == j] #yes, zonei==j is correct, they're always called 'zonei'
                id = row2['URN'].values[0]
                result.append(p)
            #end if
        #end for
        return result
    

    @staticmethod
    def getProbableRetailByMSOAIZ(dfRetailPointsPopulation,dfRetailPointsZones,retailpoints_probSij,msoa_iz,threshold):
        """
        getProbableRetailByMSOAIZ
        Given an MSOA area code (England and Wales) or an Intermediate Zone (IZ) 2001 code (Scotland), return
        a list of all the surrounding retail points whose probabilty of being visited by the MSOA_IZ is
        greater than or equal to the threshold.
        Retail ids are from ????
        @param msoa_iz An MSOA code (England/Wales e.g. E02000001) or an IZ2001 code (Scotland e.g. S02000001)
        @param threshold Probability threshold e.g. 0.5 means return all possible retail points with probability>=0.5
        @returns a list of probabilities in the same order as the venues
        """
        result = []
        zonei = int(dfRetailPointsPopulation.loc[dfRetailPointsPopulation['msoaiz'] == msoa_iz, 'zonei'])
        m,n = retailpoints_probSij.shape
        for j in range(n):
            p = retailpoints_probSij[zonei,j]
            if p>=threshold:
                row2 = dfRetailPointsZones.loc[dfRetailPointsZones['zonei'] == j] #yes, zonei==j is correct, they're always called 'zonei'
                id = row2['id'].values[0]
                result.append(p)
            #end if
        #end for
        return result
    

    @staticmethod
    def getProbableHospitalByMSOAIZ(dfHospitalPopulation,dfHospitalZones,hospital_probHijmsoa_iz,threshold):
        """
        getProbableHospitalByMSOAIZ
        Given an MSOA area code (England and Wales) or an Intermediate Zone (IZ) 2001 code (Scotland), return
        a list of all the surrounding hospitals whose probabilty of being visited by the MSOA_IZ is
        greater than or equal to the threshold.
        Hospital ids are taken from the NHS England export of "location" - see hospitalZones for ids and names (and east/north)
        NOTE: code identical to the primary school version, only with switched lookup tables
        @param msoa_iz An MSOA code (England/Wales e.g. E02000001) or an IZ2001 code (Scotland e.g. S02000001)
        @param threshold Probability threshold e.g. 0.5 means return all possible hospital points with probability>=0.5
        @returns a list of [ {id: 'hospitalid1', p: 0.5}, {id: 'hospitalid2', p:0.6}, ... etc] (NOTE: not sorted in any particular order)
        """
        result = []
        zonei = int(dfHospitalPopulation.loc[dfHospitalPopulation['msoaiz'] == msoa_iz, 'zonei'])
        m,n = hospital_probHij.shape
        for j in range(n):
            p = hospital_probHij[zonei,j]
            if p>=threshold:
                row2 = dfHospitalZones.loc[dfHospitalZones['zonei'] == j] #yes, zonei==j is correct, they're always called 'zonei'
                id = row2['id'].values[0]
                result.append(p)
            #end if
        #end for
        return result
    

    
    
    @classmethod
    def get_flows(cls,venue, msoa_list, threshold, thresholdtype):
        """
        Prepare RAMP UA compatible data
        """
        # get all probabilities so they sum to at least threshold value
        dic = {} # appending to dictionary is faster than dataframe
        for m in tqdm(msoa_list, desc=f"Reading {venue} MSOA flows"):
            # get all probabilities for this MSOA (threshold set to 0)
            if venue == "PrimarySchool":
                result_tmp = QuantRampAPI.getProbablePrimarySchoolsByMSOAIZ(cls.dfPrimaryPopulation, cls.dfPrimaryZones, cls.primary_probPij,m,0)
            elif venue == "SecondarySchool":
                result_tmp = QuantRampAPI.getProbableSecondarySchoolsByMSOAIZ(cls.dfSecondaryPopulation, cls.dfSecondaryZones, cls.secondary_probPij,m,0)
            elif venue == "Retail":
                result_tmp = QuantRampAPI.getProbableRetailByMSOAIZ(cls.dfRetailPointsPopulation, cls.dfRetailPointsZones, cls.retailpoints_probSij ,m,0)
            else:
                raise Exception("unknown venue type")
            # keep only values that sum to at least the specified threshold
            sort_index = np.argsort(result_tmp) # index from lowest to highest value
            result = [0.0] * len(result_tmp) # initialise
            i = len(result_tmp)-1 # start with last of sorted (highest prob)
            if thresholdtype == "prob":
                sum_p = 0 # initialise
                while sum_p < threshold:
                  result[sort_index[i]] = result_tmp[sort_index[i]]
                  sum_p = sum_p + result_tmp[sort_index[i]]
                  #print(sum_p)
                  i = i - 1
            elif thresholdtype == "nr":
                for t in range(0,threshold):
                    result[sort_index[i]] = result_tmp[sort_index[i]]
                    i = i - 1
            else:
                 raise Exception("unknown threshold type")
            dic[m] = result
        
        # now turn this into a dataframe with the right columns etc compatible with _flows variable
        nr_venues = len(dic[msoa_list[0]])
        col_names = []
        for n in range(0,nr_venues):
            col_names.append(f"Loc_{n}")
        df = pd.DataFrame.from_dict(dic,orient='index')
        df.columns = col_names   
        df.insert(loc=0, column='Area_ID', value=[*range(1, len(msoa_list)+1, 1)])
        df.insert(loc=1, column='Area_Code', value=df.index)
        df.reset_index(drop=True, inplace=True)
        return df







# # to test:
# microsim_data_dir = os.getcwd()
# quant_user_dir = os.path.join("QUANT_RAMP", "model-runs")

# qa = QuantRampAPI(microsim_data_dir, quant_user_dir)

# threshold = 10 # top 10
# thresholdtype = "nr" # threshold based on nr venues
# study_msoas = ['E02002559', 'E02002560']
# flow_matrix = qa.get_flows("Retail", study_msoas,threshold,thresholdtype)


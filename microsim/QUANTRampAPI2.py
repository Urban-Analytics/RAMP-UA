"""
Public interface to the data
Contains accessor functions for getting probabilities of trips from MSOA or IZ origins to
primary schools, secondary schools and retail locations.
Further information is contained in each function description.
Changes made to read in data in format suitable for RAMP UA.
"""

import pandas as pd
import pickle
import os
import numpy as np

################################################################################
# Utilities
################################################################################


"""
Load a numpy matrix from a file
"""
def loadMatrix(filename):
    with open(filename,'rb') as f:
        matrix = pickle.load(f)
    return matrix

################################################################################
# Globals
################################################################################

os.chdir("..")
quant_dir = 'data/QUANT_RAMP/model-runs/'
dfPrimaryPopulation = pd.read_csv(os.path.join(quant_dir,'primaryPopulation.csv'))
dfPrimaryZones = pd.read_csv(os.path.join(quant_dir,'primaryZones.csv'))
primary_probPij = loadMatrix(os.path.join(quant_dir,'primaryProbPij.bin'))
dfSecondaryPopulation = pd.read_csv(os.path.join(quant_dir,'secondaryPopulation.csv'))
dfSecondaryZones = pd.read_csv(os.path.join(quant_dir,'secondaryZones.csv'))
secondary_probPij = loadMatrix(os.path.join(quant_dir,'secondaryProbPij.bin'))
dfRetailPointsPopulation = pd.read_csv(os.path.join(quant_dir,'retailpointsPopulation.csv'))
dfRetailPointsZones = pd.read_csv(os.path.join(quant_dir,'retailpointsZones.csv'))
retailpoints_probSij = loadMatrix(os.path.join(quant_dir,'retailpointsProbSij.bin'))
dfHospitalPopulation = pd.read_csv(os.path.join(quant_dir,'hospitalPopulation.csv'))
dfHospitalZones = pd.read_csv(os.path.join(quant_dir,'hospitalZones.csv'))
hospital_probHij = loadMatrix(os.path.join(quant_dir,'hospitalProbHij.bin'))


################################################################################
# Interface
################################################################################


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
def getProbablePrimarySchoolsByMSOAIZ(msoa_iz,threshold):
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

################################################################################

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
def getProbableSecondarySchoolsByMSOAIZ(msoa_iz,threshold):
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

################################################################################

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
def getProbableRetailByMSOAIZ(msoa_iz,threshold):
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

################################################################################

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

def getProbableHospitalByMSOAIZ(msoa_iz,threshold):
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

################################################################################



################################################################################
# Prepare RAMP UA compatible data
################################################################################

def get_flows(venue, msoa_list, threshold):
    
    # get all probabilities so they sum to at least threshold value
    dic = {} # appending to dictionary is faster than dataframe
    for m in msoa_list:
        print(m)
        # get all probabilities for this MSOA (threshold set to 0)
        if venue == "PrimarySchool":
            result_tmp = getProbablePrimarySchoolsByMSOAIZ(m,0)
        elif venue == "SecondarySchool":
            result_tmp = getProbableSecondarySchoolsByMSOAIZ(m,0)
        elif venue == "Retail":
            result_tmp = getProbableRetailByMSOAIZ(m,0)
        else:
            sys.exit("unknown venue type") 
        # keep only values that sum to at least the specified threshold
        sort_index = np.argsort(result_tmp) # index from lowest to highest value
        sum_p = 0 # initialise
        result = [0.0] * len(result_tmp) # initialise
        i = len(result_tmp)-1 # start with last of sorted (highest prob)
        while sum_p < threshold:
          result[sort_index[i]] = result_tmp[sort_index[i]]
          sum_p = sum_p + result_tmp[sort_index[i]]
          #print(sum_p)
          i = i - 1
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





# # to call, use something like:
# venue = "SecondarySchool" #PrimarySchool, SecondarySchool, Retail
# msoa_list = ['E02002559', 'E02002560']
# threshold = 0.2 # explain 20%
# df = get_flows(venue, msoa_list,threshold)

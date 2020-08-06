# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:01:02 2020

@author: Natalie
"""


import pytest
import os
import pickle
import pandas as pd
import numpy as np
import microsim.dashboard as dash



# os.chdir("microsim")
# import dashboard as dash
# os.chdir(base_dir)




# right wd - does everything exist
# Do the directories and data files exist

# Do the input files contain the right information eg nr days/ranges/msoas/disease conditions etc

# Are the counts correct - cross checks
    
    # # sanity check: sum across MSOAs should be same as nrs_*
    # assert (msoa_counts_S.iloc[:,d].sum() == nrs_S[d])
    # assert (msoa_counts_E.iloc[:,d].sum() == nrs_E[d])
    # assert (msoa_counts_I.iloc[:,d].sum() == nrs_I[d])
    # assert (msoa_counts_R.iloc[:,d].sum() == nrs_R[d])

#    assert(total_day == nrs_S[d] + nrs_E[d] + nrs_I[d] + nrs_R[d])
    
# check parameter file input


# # code to create pickle files from csv
# retail = pd.read_csv("Retail1.csv")
# pickle_out = open("Retail.pickle", "wb")
# pickle.dump(retail, pickle_out)
# pickle_out.close()
# individuals = pd.read_csv("Individuals1.csv")
# pickle_out = open("Individuals.pickle", "wb")
# pickle.dump(individuals, pickle_out)
# pickle_out.close()


# create parameters used in later tests
@pytest.fixture
def example_input_params():
    params = {}
    params["base_dir"] = os.getcwd()  # get current directory (usually RAMP-UA)
    params["data_dir"] = "dummy_data"
    params["start_day"] = 0
    params["end_day"] = 10
    params["start_run"] = 0
    params["end_run"] = 1               
    params["sc_dir"] = "output"
    params["sc_nam"] = "Test scenario"
    
    params["conditions_dict"] = {
      "susceptible": 0,
      "exposed": 1,
      "presymptomatic": 2,
      "symptomatic": 3,
      "asymptomatic": 4,
      "recovered": 5,
      "dead": 6,
    }
    
    params["locations_dict"] = {
      "Retail": "Retail",
    }
    params["data_dir"] = os.path.join(params["base_dir"], params["data_dir"]) # update data dir
    params["nr_days"] = params["end_day"] - params["start_day"] + 1
    params["nr_runs"] = params["end_run"] - params["start_run"] + 1
    params["r_range"] = range(params["start_run"], params["end_run"]+1)
    return params


# just to check pytest
# def test_always_passes():
#     assert True
    
# def test_always_fails():
#     assert False


# check input parameters
def test_check_input(example_input_params):
    assert example_input_params["nr_days"] == 11
    
def test_calc_nr_days(example_input_params):
    nr_days = dash.calc_nr_days(os.path.join(example_input_params["data_dir"], example_input_params["sc_dir"],"0","Retail.pickle"))
    assert nr_days == example_input_params["nr_days"]

@pytest.fixture
def test_create_venue_dangers_dict(example_input_params):
    dangers_dict, dangers_dict_std, dangers_dict_3d = dash.create_venue_dangers_dict(example_input_params["locations_dict"],example_input_params["r_range"],os.path.join(example_input_params["data_dir"],example_input_params["sc_dir"]),example_input_params["start_day"],example_input_params["end_day"],example_input_params["start_run"],example_input_params["nr_runs"])

    # check there are no missing data
    assert dangers_dict_3d['Retail'].shape == (20, 11, 2)
    assert dangers_dict_std['Retail'].shape == (20, 11)
    assert dangers_dict['Retail'].shape == (20, 11)
    
    # check mean and standard deviation for 1 element
    assert dangers_dict['Retail'].iloc[0,10] == (dangers_dict_3d['Retail'][0,10,0]  + dangers_dict_3d['Retail'][0,10,1]) / 2
    assert dangers_dict_std['Retail'].iloc[0,10] == np.array([dangers_dict_3d['Retail'][0,10,0], dangers_dict_3d['Retail'][0,10,1]]).std()
    
    return dangers_dict
    
def test_create_difference_dict(example_input_params, test_create_venue_dangers_dict):
    # subtract from itself to give 0's
    result = dash.create_difference_dict(test_create_venue_dangers_dict,test_create_venue_dangers_dict,example_input_params["locations_dict"])
    assert result["Retail"].sum(axis=1).sum(axis=0) == 0
    # create some dummy data
    dict1 = {"Retail": pd.DataFrame ({'1':  [1, 2],'2': [1, 2]}, columns = ['1','2'])}
    dict2 = {"Retail": pd.DataFrame ({'1':  [2, 3],'2': [2, 4]}, columns = ['1','2'])}
    result2 = dash.create_difference_dict(dict1,dict2,example_input_params["locations_dict"])
    assert result2["Retail"].sum(axis=1).sum(axis=0) == 5
    assert result2["Retail"].iloc[0,0] == 1


def test_create_msoa_dangers_dict(example_input_params, test_create_venue_dangers_dict):

    msoa_codes = pd.Series(["E02004164","E02004164","E02004164","E02004164","E02004164","E02004164","E02004165","E02004165","E02004165","E02004165","E02004165","E02004165","E02004165","E02004169","E02004169","E02004169","E02004169", "E02004169","E02004169","E02004169"])

    dangers_msoa_dict = dash.create_msoa_dangers_dict(test_create_venue_dangers_dict,["Retail"],[msoa_codes])
    assert dangers_msoa_dict['Retail'].shape == (3, 11)
    assert dangers_msoa_dict['Retail'].iloc[2,10] == test_create_venue_dangers_dict['Retail'].iloc[13:20,10].mean()



def test_create_msoa_dangers_dict(example_input_params):
    
    age_cat = np.array([[0, 19], [20, 29], [30,44], [45,59], [60,74], [75,200]])   
    msoas, totalcounts_dict, cumcounts_dict, agecounts_dict,  msoacounts_dict, cumcounts_dict_3d, totalcounts_dict_std, cumcounts_dict_std, agecounts_dict_std, msoacounts_dict_std, totalcounts_dict_3d, agecounts_dict_3d, msoacounts_dict_3d, uniquecounts_dict_3d, uniquecounts_dict_std, uniquecounts_dict = dash.create_counts_dict(example_input_params["conditions_dict"],example_input_params["r_range"],os.path.join(example_input_params["data_dir"], example_input_params["sc_dir"]),example_input_params["start_day"],example_input_params["end_day"],example_input_params["start_run"],example_input_params["nr_runs"],age_cat)




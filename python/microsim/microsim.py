#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core RAMP-UA model.

Created on Wed Apr 29 19:59:25 2020

@author: nick
"""

import pandas as pd
import glob
import os

DATA_DIR = "../../data/"

class Microsim:
    
    def __init__(self):
        self.households, self.individuals = Microsim._read_msm_data()
        self.iteration = 0
    
    @classmethod
    def _read_msm_data(cls):
        """Read the csv files that have the indivduls and households"""
        
        msm_dir = os.path.join(DATA_DIR,"msm_data")
    
        # households
        house_dfs = []
        for f in glob.glob(msm_dir+'/ass_hh_*_OA11_2020.csv'):
            house_dfs.append(pd.read_csv(f))
        if len(house_dfs)==0:
            raise Exception(f"No household csv files found in {msm_dir}.",
                            f"Have you downloaded and extracted the necessary data? (see {DATA_DIR} README)")
        households = pd.concat(house_dfs)
        
        # individuals
        indiv_dfs= []
        for f in glob.glob(msm_dir+'/ass_*_MSOA11_2020.csv'):
            indiv_dfs.append(pd.read_csv(f))
        if len(indiv_dfs) == 0:
            raise Exception(f"No individual csv files found in {msm_dir}.",
                            f"Have you downloaded and extracted the necessary data? (see {DATA_DIR} README)")
        individuals = pd.concat(indiv_dfs)
        
        # THE FOLLOWING SHOULD BE DONE AS PART OF A TEST SUITE
        #TODO: check that correct numbers of rows have been read.    
        #TODO: check that each individual has a household    
        #TODO: graph number of people per household just to sense check
        
        
        
        print("Have read files:",
              f"\n\tHouseholds:  {len(house_dfs)} files with {len(households)}",
              f"households in {len(households.Area.unique())} areas",
              f"\n\tIndividuals: {len(indiv_dfs)} files with {len(individuals)}",
              f"individuals in {len(individuals.Area.unique())} areas")
        
        return (households, individuals)
    
    
    def step(self):
        """Step (iterate) the model"""
        self.iteration +=1 
        print(f"Iteration: {self.iteration}")
        # XXXX HERE
        

    
    

# PROGRAM ENTRY POINT
        
if __name__=="__main__":
    m = Microsim()
    for i in range(10):
        m.step()
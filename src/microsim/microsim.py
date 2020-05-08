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
import random
import time
import typing

import click # command-line interface

DATA_DIR = "../../data/"

class Microsim:
    """
    A class used to represent the microsimulation model.

    TODO: Document class. Please use reStructuredText format:
    https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
    ...

    Attributes
    ----------
    XXXX

    Methods
    -------
    XXXX
    """

    def __init__(self, random_seed:float = None):
        """
        Microsim constructor.
        ----------
        :param random_seed: A optional random seed to use when creating the class instance. If None then
            the current time is used.
        """

        # This is the main population of individuals and their households
        self.individuals, self.households = Microsim.read_msm_data()

        # Attach a load of health attributes to each individual
        self.individuals = Microsim.attach_health_data(self.individuals)

        # Attach a load of transport attributes to each individual
        self.individuals = Microsim.attach_transport_data(self.individuals)

        # Read the locations of schools, workplaces, etc.
        #self.schools = Microsim.read_school_data()
        #self.shops = Microsim.read_shop_data()
        #self.workplaces = Microsim.read_workplace_data()

        # Assign probabilities that each individual will go to each location (most of these will be 0!)
        # Do this by adding two new columns, both storing lists. One lists ids of the locations that
        # the individual may visit, the other has the probability of them visitting those places



        # Other variables that need to be defined
        self.iteration = 0
        self.random = random.Random(time.time() if random_seed is None else random_seed)
    
    @staticmethod
    def read_msm_data():
        """Read the csv files that have the indivduls and households

        :return a tuple with two pandas dataframes representing individuls (0) and households (1)
        """
        
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

        # TODO Join individuls to households (create lookup columns in each)

        return (individuals, households)

    @staticmethod
    def attach_health_data(individuals):
        """

        :param individuals:
        :return:
        """
        print("Attaching health data ... ",)
        pass
        print("... finished.")
        return individuals

    @staticmethod
    def attach_transport_data(individuals):
        print("Attaching health data ... ",)
        pass
        print("... finished.")
        return individuals


    def step(self):
        """Step (iterate) the model"""
        self.iteration +=1 
        print(f"Iteration: {self.iteration}")
        # XXXX HERE
        

    
    

# PROGRAM ENTRY POINT
# Uses 'click' library so that it can be run from the command line
@click.command()
@click.option('--iterations', default=10, help='Number of model iterations')
def run(iterations):
    num_iter=iterations
    m = Microsim()
    for i in range(num_iter):
        m.step()
    print("End of program")

if __name__=="__main__":
    run()
    print("End of program")

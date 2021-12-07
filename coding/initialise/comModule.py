#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commuting module for the RAMP-UA model.

Created on Wed Sep 01 2021

@author: Hadrien
"""

import pandas as pd
import numpy as np
from numpy.random import choice
from coding.constants import Constants
from sklearn.metrics.pairwise import haversine_distances

class Commuting:

    def __init__(self,
                 population,
                 threshold):

        br_file_with_path = Constants.Paths.BUSINESSREGISTRY.FULL_PATH_FILE
        print(f"Reading business registry from {br_file_with_path}.")
        business_registry = pd.read_csv(br_file_with_path)

        [reg,pop,useSic] = Commuting.trimData(self,business_registry,population,threshold)
        [origIndiv,destWork] = Commuting.getCommuting(self,reg,pop,useSic)

        self.reg = reg
        self.origIndiv = origIndiv
        self.destWork = destWork

        return


    def trimData(self,
                 business_registry,
                 population,
                 threshold):

        useSic = True

        print("Preparing commuting data.")

        population = population[population['pwork'] > 0]
        popLoc = population['MSOA11CD'].unique()
        business_registry = business_registry[business_registry['MSOA11CD'].isin(popLoc)]

        ref = list(set(business_registry['sic1d07'].unique()) & set(population['sic1d07'].unique()))

        business_registry_temp = business_registry[business_registry['sic1d07'] == ref[0]]
        business_registry_conc = business_registry_temp.loc[business_registry_temp.index.repeat(business_registry_temp['size'])]
        population_conc = population[(population['sic1d07'] == ref[0])]

        total_job = len(business_registry_conc)
        total_population = len(population_conc)

        if total_job < total_population:
            population_conc = population_conc.sample(n=total_job)

        if total_job > total_population:
            business_registry_conc = business_registry_conc.sample(n=total_population)

        if len(ref) > 1:
            for i in ref[1:len(ref)]:
                business_registry_temp = business_registry[business_registry['sic1d07'] == i]
                business_registry_temp = business_registry_temp.loc[business_registry_temp.index.repeat(business_registry_temp['size'])]
                population_temp = population[(population['sic1d07'] == i)]
                total_job = len(business_registry_temp)
                total_population = len(population_temp)

                if total_job < total_population:
                    population_temp = population_temp.sample(n=total_job)

                if total_job > total_population:
                    business_registry_temp = business_registry_temp.sample(n=total_population)

                business_registry_conc = business_registry_conc.append(business_registry_temp)
                population_conc = population_conc.append(population_temp)

        if len(population_conc)/len(population[population['pwork'] > 0]) < threshold:
            useSic = False

            business_registry_conc = business_registry.loc[business_registry.index.repeat(business_registry['size'])]
            population_conc = population

            total_job = len(business_registry_conc)
            total_population = len(population_conc)

            if(total_job < total_population):
                population_conc = population_conc.sample(n=total_job)

            if(total_job > total_population):
                business_registry_conc = business_registry_conc.sample(n=total_population)

        business_registry_conc.loc[:,'size'] = 1
        business_registry_conc = pd.merge(business_registry_conc[['id','size']].groupby('id').sum(), business_registry_conc.drop('size',axis=1), how="left", on=["id"])
        business_registry_conc = business_registry_conc.drop_duplicates()
        business_registry_conc.index = range(len(business_registry_conc))

        return [business_registry_conc,population_conc,useSic]


    def commutingDistance(self,
                          reg,
                          pop
                          ):

        regCoords = [np.radians(reg['lat']),np.radians(reg['lng'])]
        latrad = np.radians(pop['lat'])
        lngrad = np.radians(pop['lng'])
        dist = [haversine_distances([regCoords],[[latrad[_],lngrad[_]]])[0][0] for _ in latrad.index] # Warning: the distance unit is the earth radius

        return(dist)


    def getCommuting(self,
                     reg,
                     pop,
                     useSic
                     ):

        print("Calculating commuting flows...")

        origIndiv = []
        destWork = []

        if useSic:
            ref = list(set(reg['sic1d07'].unique()) & set(pop['sic1d07'].unique()))

            for i in ref:
                currentReg = reg[reg['sic1d07'] == i]
                currentPop = pop[pop['sic1d07'] == i]

                for j in currentReg.index:
                    dist = Commuting.commutingDistance(self,currentReg.loc[j,:],currentPop)
                    probDistrib = np.ones(len(dist)) / dist / dist

                    size = currentReg.loc[j,'size']
                    draw = choice(currentPop['idp'],size,p=probDistrib/sum(probDistrib),replace = False)
                    origIndiv += list(draw)
                    destWork += list(np.repeat(currentReg.loc[j,'id'],size))

                    currentPop = currentPop[~currentPop['idp'].isin(draw)]

            return [origIndiv,destWork]

        else:
            for j in range(len(reg)):
                dist = Commuting.commutingDistance(self,reg.loc[j,:],pop)

                probDistrib = np.ones(len(dist)) / dist / dist

                size = reg.loc[j,'size']
                draw = choice(pop['idp'],size,p=probDistrib/sum(probDistrib),replace = False)
                origIndiv += list(draw)
                destWork += list(np.repeat(reg.loc[j,'id'],size))

                pop = pop[~pop['idp'].isin(draw)]

            return [origIndiv,destWork]

    def getCommutingData(self):
        if self.reg is None:
            raise Exception("Failed")
        if self.origIndiv is None:
            raise Exception("Failed")
        if self.destWork is None:
            raise Exception("Failed")
        return [self.reg,self.origIndiv,self.destWork]



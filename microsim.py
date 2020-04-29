#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core RAMP-UA model.

Created on Wed Apr 29 19:59:25 2020

@author: nick
"""

import pandas as pd
import glob

# households
df_list = []
for f in glob.glob('data/msm_data_v0.1/ass_hh_*_OA11_2020.csv'):
    df_list.append(pd.read_csv(f))
households = pd.concat(df_list)

# individuals
df_list = []
for f in glob.glob('data/msm_data_v0.1/ass_*_MSOA11_2020.csv'):
    df_list.append(pd.read_csv(f))
individuals = pd.concat(df_list)

#TODO: check that correct numbers of rows have been read.

#TODO: check that each individual has a household

#TODO: graph number of people per household just to sense check



# First test: Age each person.

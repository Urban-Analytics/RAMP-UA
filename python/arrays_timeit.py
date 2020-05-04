#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test to compare the speed of numpy and pandas arrays

Created on Mon May  4 15:25:32 2020

@author: nick
"""

import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import timeit

age = []
print("Creating initial array)")
for _ in tqdm(range(1000000)):
    age.append(int(random.uniform(0,100)))


age_pd = pd.DataFrame({'age':age})
age_np = np.array(age)

def increment_age_np():
    for i in range(len(age)):
        age[i] += 1

def increment_age_pd1():
    for i in range(len(age)):
        age_pd.iloc[ i,:] =+ 1
        
def increment_age_pd2():
    return age_pd.apply(lambda x: x+1)
    
print(f"Time using numpy:")
print(timeit.timeit( increment_age_np, number=10))

print(f"Time using pd (normal)")
print(timeit.timeit(increment_age_pd1, number=10))
      
print(f"Time using pd (apply):")
print(timeit.timeit(increment_age_pd2, number=10))

# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:01:02 2020

@author: Natalie
"""


# test version, needs adapting to work across platform

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import geopandas as gpd
import os
import pickle


# To fix file path issues, use absolute/full path at all times
# Pick either: get working directory (if user starts this script in place, or set working directory
# Option A: copy current working directory:
os.chdir("..") # assume microsim subdir so need to go up one level
base_dir = os.getcwd()  # get current directory
data_dir = os.path.join(base_dir, "data") # go to output dir
# Option B: specific directory
data_dir = 'C:\\Users\\Toshiba\\git_repos\\RAMP-UA\\dummy_data'

# read in details about venues
data_file = os.path.join(data_dir, "devon-schools","exeter schools.csv")
schools = pd.read_csv(data_file)
data_file = os.path.join(data_dir, "devon-retail","devon smkt.csv")
retail = pd.read_csv(data_file)

# read in pickle files
data_file = os.path.join(data_dir, "output","Individuals.pickle")
pickle_in = open(data_file,"rb")
individuals = pickle.load(pickle_in)
pickle_in.close()

data_file = os.path.join(data_dir, "output","PrimarySchool.pickle")
pickle_in = open(data_file,"rb")
primaryschool_dangers = pickle.load(pickle_in)
pickle_in.close()

data_file = os.path.join(data_dir, "output","SecondarySchool.pickle")
pickle_in = open(data_file,"rb")
secondaryschool_dangers = pickle.load(pickle_in)
pickle_in.close()

data_file = os.path.join(data_dir, "output","Retail.pickle")
pickle_in = open(data_file,"rb")
retail_dangers = pickle.load(pickle_in)
pickle_in.close()

data_file = os.path.join(data_dir, "output","Work.pickle")
pickle_in = open(data_file,"rb")
work_dangers = pickle.load(pickle_in)
pickle_in.close()

# just do schools and retail for now
# merge
primaryschools = pd.merge(schools, primaryschool_dangers, left_index=True, right_index=True)
secondaryschools = pd.merge(schools, secondaryschool_dangers, left_index=True, right_index=True)
retail = pd.merge(retail, retail_dangers, left_index=True, right_index=True)

# how many days have we got
nr_days = retail_dangers.shape[1] - 1
days = [i for i in range(0,nr_days)]

# create list of summed nrs of E, I and R
nrs_S = [0 for i in range(nr_days)] #initialise with zeros
nrs_E = [0 for i in range(nr_days)] #initialise with zeros
nrs_I = [0 for i in range(nr_days)] #initialise with zeros
nrs_R = [0 for i in range(nr_days)] #initialise with zeros
for d in range(0, nr_days):
    nrs = individuals.iloc[:,d+24].value_counts() 
    nrs_S[d] = nrs.get(0)  # susceptible?
    nrs_E[d] = nrs.get(1)  # exposed?
    nrs_I[d] = nrs.get(2)  # infectious?
    nrs_R[d] = nrs.get(3)  # recovered/removed?
    
# total E,I,R across time (summed over all areas)
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(days, nrs_E, label='Exposed')  # Plot some data on the axes.
ax.plot(days, nrs_I, label='Infectious')  # Plot more data on the axes...
ax.plot(days, nrs_R, label='Recovered')  # ... and some more.
ax.set_xlabel('Days')  # Add an x-label to the axes.
ax.set_ylabel('Number of people')  # Add a y-label to the axes.
ax.set_title("Infections over time")  # Add a title to the axes.
ax.legend()  # Add a legend.   

# total infections per area across time
counts_tmp = individuals.groupby(['Area', 'DiseaseStatus0']).agg({'DiseaseStatus0': ['count']})



# this breaks down if at least one category doesn't appear in counts_tmp
ind_0 = [i for i in range(0, len(counts_tmp), 4)]
ind_1 = [i for i in range(1, len(counts_tmp), 4)]
ind_2 = [i for i in range(2, len(counts_tmp), 4)]
ind_3 = [i for i in range(3, len(counts_tmp), 4)]
counts_1 = counts_tmp.iloc[ind_1, 0]
counts_2 = counts_tmp.iloc[ind_2, 0]
counts_3 = counts_tmp.iloc[ind_3, 0]

# loop around remaining days
for d in range(1, nr_days):
    counts_tmp = individuals.groupby(['Area', individuals.columns[d+24]]).agg({individuals.columns[d+24]: ['count']})  # first column = day 0 = index 23 - we already have this one, so start from column 24
    ind_1 = [i for i in range(1, len(counts_tmp), 4)]
    ind_2 = [i for i in range(2, len(counts_tmp), 4)]
    ind_3 = [i for i in range(3, len(counts_tmp), 4)]
    counts_1_tmp = counts_tmp.iloc[ind_1, 0]
    counts_1 = pd. concat([counts_1, counts_1_tmp], axis=1) 
    counts_2_tmp = counts_tmp.iloc[ind_2, 0]
    counts_2 = pd. concat([counts_2, counts_2_tmp], axis=1) 
    counts_3_tmp = counts_tmp.iloc[ind_3, 0]
    counts_3 = pd. concat([counts_3, counts_3_tmp], axis=1) 
    
# heatmap
xticklabels=days
xticks = xticklabels
xticks = [x +1 - 0.5 for x in xticks] # to get the tick in the centre of a heatmap grid rectangle
# pick one colourmap from below
#cmap = sns.color_palette("coolwarm", 128)  
cmap = 'RdYlGn_r'  
plt.figure(figsize=(30, 10))
ax1 = sns.heatmap(counts_1, annot=False, cmap=cmap, xticklabels=xticklabels)
ax1.set_xticks(xticks)
plt.title("NR exposed")
plt.ylabel("MSOA")
plt.xlabel("Day")
plt.show()

# plot of infections per MSOA at a given day
# ask user to pick a day
day2plot = int(input("Please type the number of the day you want to plot (0 to "+str(nr_days-1)+"): "))
msoas = [i for i in range(0,len(counts_1))]

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(msoas, counts_1.iloc[:,day2plot].tolist(), label='Exposed')  
ax.plot(msoas, counts_2.iloc[:,day2plot].tolist(), label='Infectious')
ax.plot(msoas, counts_3.iloc[:,day2plot].tolist(), label='Recovered')
ax.set_xlabel('MSOA')  # Add an x-label to the axes.
ax.set_ylabel('Number of people')  # Add a y-label to the axes.
ax.set_title("Infections across MSOAs, day "+str(day2plot))  # Add a title to the axes.
ax.legend()  # Add a legend.












# !!!! Adapt to work with real data from here on instead of reading var4plottest.csv

# load in a shapefile
fp = 'MSOAS_shp/bcc21fa2-48d2-42ca-b7b7-0d978761069f2020412-1-12serld.j1f7i.shp'
map_df = gpd.read_file(fp)
# check
#map_df.plot()
# rename
map_df.rename(index=str, columns={'msoa11cd': 'Area'},inplace=True)

# load in the data to plot
individuals = pd.read_csv('var4plottest.csv', header=0)


# total infections per area across time
# loop around days
counts_E_days = pd.DataFrame() 
for d in range(0, 30):
    counts_tmp = individuals.groupby(['Area', individuals.columns[d+6]]).agg({individuals.columns[d+6]: ['count']})  
    ind_1 = [i for i in range(1, len(counts_tmp), 4)]
    counts_1 = counts_tmp.iloc[ind_1, 0]
    if d == 0:
        counts_E_days = counts_1
        counts_E_days.rename("Day1",inplace = True)
    else:
        counts_E_days = pd.concat([counts_E_days, counts_1], axis=1) 
        counts_E_days.rename(columns={ counts_E_days.columns[d]: 'Day'+str(d+1) }, inplace = True)
    
# bit sloppy but works for now, could try stuff like obj = obj._drop_axis(labels, axis, level=level, errors=errors)
counts_E_days.reset_index(level=1, drop=True, inplace=True)
counts_E_days.reset_index(inplace=True)
    
# merge spatial and data
merged_data = pd.merge(map_df,counts_E_days,on='Area')

# set the range for the choropleth
vmin = 0
vmax = counts_E_days.iloc[:,1:31].max().max()  # find max to scale (or set max number eg if using %)


# option 1
# create individual plots and save each as image
for d in range(0, 30):
    # set a variable that will call whatever column we want to visualise on the map
    variable = "Day"+str(d+1)
    
    # create figure and axes for Matplotlib
    fig, ax = plt.subplots(1, figsize=(10, 6))
    # create map
    merged_data.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')
    # remove the axis
    ax.axis('off')
    # add a title
    ax.set_title('Exposed cases day'+str(d+1), fontdict={'fontsize': '25', 'fontweight' : '3'})
    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # empty array for the data range
    sm._A = []
    # add the colorbar to the figure
    cbar = fig.colorbar(sm)
    
    fig.savefig('map_day'+str(d+1)+'.png', dpi=300)
    
# put all the images created above together using imageio
import imageio
with imageio.get_writer('map_movie.gif', mode='I', duration=0.5) as writer:
    for d in range(0, 30):
        filename = "map_day"+str(d+1)+".png"
        image = imageio.imread(filename)
        writer.append_data(image)
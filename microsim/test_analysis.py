# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:01:02 2020

@author: Natalie
"""


# test version, needs adapting to work across platform

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
import pickle
from shapely.geometry import Point


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
msoas = sorted(individuals.area.unique())
msoa_counts_S = pd.DataFrame(index=msoas)
msoa_counts_E = pd.DataFrame(index=msoas)
msoa_counts_I = pd.DataFrame(index=msoas)
msoa_counts_R = pd.DataFrame(index=msoas)

for d in range(0, nr_days):
    # counts_tmp = individuals.groupby(['Area', individuals.columns[d+73]]).agg({individuals.columns[d+6]: ['count']})  
    msoa_count_temp = individuals[individuals.iloc[:, 73+d] == 0].groupby(['Area']).agg({individuals.columns[73+d]: ['count']})  
    msoa_counts_S = pd.merge(msoa_counts_S,msoa_count_temp,left_index = True, right_index=True)
    msoa_counts_S.rename(columns={ msoa_counts_S.columns[d]: 'Day'+str(d) }, inplace = True)
    
    msoa_count_temp = individuals[individuals.iloc[:, 73+d] == 1].groupby(['Area']).agg({individuals.columns[73+d]: ['count']})  
    msoa_counts_E = pd.merge(msoa_counts_E,msoa_count_temp,left_index = True, right_index=True)
    msoa_counts_E.rename(columns={ msoa_counts_E.columns[d]: 'Day'+str(d) }, inplace = True)
    
    msoa_count_temp = individuals[individuals.iloc[:, 73+d] == 2].groupby(['Area']).agg({individuals.columns[73+d]: ['count']})  
    msoa_counts_I = pd.merge(msoa_counts_I,msoa_count_temp,left_index = True, right_index=True)
    msoa_counts_I.rename(columns={ msoa_counts_I.columns[d]: 'Day'+str(d) }, inplace = True)
    
    msoa_count_temp = individuals[individuals.iloc[:, 73+d] == 3].groupby(['Area']).agg({individuals.columns[73+d]: ['count']})  
    msoa_counts_R = pd.merge(msoa_counts_R,msoa_count_temp,left_index = True, right_index=True)
    msoa_counts_R.rename(columns={ msoa_counts_R.columns[d]: 'Day'+str(d) }, inplace = True)

    
# heatmap
xticklabels=days
xticks = xticklabels
xticks = [x +1 - 0.5 for x in xticks] # to get the tick in the centre of a heatmap grid rectangle
# pick one colourmap from below
#cmap = sns.color_palette("coolwarm", 128)  
cmap = 'RdYlGn_r'  
plt.figure(figsize=(30, 10))
ax1 = sns.heatmap(msoa_counts_S, annot=False, cmap=cmap, xticklabels=xticklabels)
ax1.set_xticks(xticks)
plt.title("NR exposed")
plt.ylabel("MSOA")
plt.xlabel("Day")
plt.show()

# plot of infections per MSOA at a given day
# ask user to pick a day
day2plot = int(input("Please type the number of the day you want to plot (0 to "+str(nr_days-1)+"): "))
msoas_nr = [i for i in range(0,len(msoas))]

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(msoas_nr, msoa_counts_E.iloc[:,day2plot].tolist(), label='Exposed')  
ax.plot(msoas_nr, msoa_counts_I.iloc[:,day2plot].tolist(), label='Infectious')
ax.plot(msoas_nr, msoa_counts_R.iloc[:,day2plot].tolist(), label='Recovered')
ax.set_xlabel('MSOA')  # Add an x-label to the axes.
ax.set_ylabel('Number of people')  # Add a y-label to the axes.
ax.set_title("Infections across MSOAs, day "+str(day2plot))  # Add a title to the axes.
ax.legend()  # Add a legend.



# geographical plots


# choropleth

# load in a shapefile
sh_file = os.path.join(data_dir, "MSOAS_shp","bcc21fa2-48d2-42ca-b7b7-0d978761069f2020412-1-12serld.j1f7i.shp")
map_df = gpd.read_file(sh_file)
# check
#map_df.plot()
# rename
map_df.rename(index=str, columns={'msoa11cd': 'Area'},inplace=True)

# merge spatial data and counts (created above)
msoa_counts_I['Area'] = msoa_counts_I.index
merged_data = pd.merge(map_df,msoa_counts_I,on='Area')

# set the range for the choropleth
vmin = 0
vmax = msoa_counts_I.iloc[:,0:nr_days].max().max()  # find max to scale (or set max number eg if using %)


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
    ax.set_title('Infected cases day'+str(d+1), fontdict={'fontsize': '25', 'fontweight' : '3'})
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
        
# dots on map
#merged_data.crs # check coordinate system from underlay (here MSOAs - epsg:27700)

#Converting a Pandas object (Dataframe) to a GeoPandas object (Dataframe)


retail_test = retail
geometry = [Point(xy) for xy in zip(retail_test.bng_e, retail_test.bng_n)]
retail_test = retail_test.drop(['bng_e', 'bng_n'], axis=1)
crs = {'init': 'epsg:27700'}
gdf_retail = gpd.GeoDataFrame(retail_test, crs=crs, geometry=geometry)

# plot all retail locations
base = map_df.plot(color='white', edgecolor='black')
gdf_retail.plot(ax=base, marker='o', color='red', markersize=5)

# plot only those with certain level of danger
base = map_df.plot(color='white', edgecolor='black')
gdf_retail[gdf_retail.Danger0 > 0].plot(ax=base, marker='o', color='red', markersize=5)







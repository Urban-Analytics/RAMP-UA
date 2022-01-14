#### Import modules required
import multiprocessing as mp
import numpy as np
import yaml # pyyaml library for reading the parameters.yml file
import os
import itertools
import pandas as pd
import unittest
import pickle
import copy
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import datetime
import matplotlib.cm as cm
import geopandas as gpd
from matplotlib.animation import FuncAnimation, PillowWriter
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

import matplotlib as mpl 
mpl.rcParams['animation.ffmpeg_path'] ='/usr/bin/ffmpeg'

##########################################################################
# Read case data and spatial data
##########################################################################
##########################################################################
from microsim.load_msoa_locations import load_osm_shapefile, load_msoa_shapes

# Directory where spatial data is stored
gis_data_dir = ("../../devon_data")
#osm_buildings = load_osm_shapefile(gis_data_dir)
devon_msoa_shapes = load_msoa_shapes(gis_data_dir, visualize=False)
devon_msoa_shapes = devon_msoa_shapes.set_index('Code', drop=True, verify_integrity=True)

# Observed cases data
# These were prepared by Hadrien and made available on the RAMP blob storage (see the observation data README).
cases_msoa = pd.read_csv(os.path.join("observation_data", "england_initial_cases_MSOAs.csv")).set_index('MSOA11CD', drop=True, verify_integrity=True)

# Merge them to the GIS data for convenience
cases_msoa = cases_msoa.join(other = devon_msoa_shapes, how="inner")  # Joins on the indices (both indices are MSOA code)
assert len(cases_msoa) == len(devon_msoa_shapes)  # Check we don't use any areas in the join

# For some reason we lose the index name when joining
cases_msoa.index.name = "MSOA11CD"

##########################################################################
#########################################################################
# Inspect observations
##########################################################################
#########################################################################
# Extract the daily case data
observations_df = cases_msoa.iloc[:, 0:405]
# save to csv
observations_df.to_csv("observation_data/devon_daily_cases.csv", index = False)

# Create a transposed version, with each MSOA as a column
observations_df_t = observations_df.transpose()
# Remove the D from the start of the Day number
observations_df_t.index = observations_df_t.index.str[1:]
# save to csv
observations_df_t['Day'] = observations_df_t.index
observations_df_t.to_csv("observation_data/devon_daily_cases_t.csv", index = False)

##########################################################################
#########################################################################
# Plots
##########################################################################
#########################################################################
#ax = plt.gca()
fig, ax = plt.subplots(figsize=(20,6))
ax = observations_df_t.iloc[0:14, 0:104].plot.area(figsize=(20,20), subplots=True, legend = None, sharey = True)
#ax.get_legend().remove()
plt.xticks(fontsize = 14)
plt.yticks([])
plt.show()


### All in one plot
days=404
for days in [14, 28, 42, 56, 404]:
    fig, ax = plt.subplots()
    observations_df_t.iloc[0:days:, 0:107].plot.line(figsize = (20,20),ax=ax, subplots=False, legend = None, sharey = True,
                                                        linewidth= 5, colormap='Paired')
    plt.xticks(fontsize =40)
    plt.yticks(fontsize = 40)
    plt.xlabel("Days", fontsize = 30)
    plt.ylabel("Positive cases", fontsize = 30)
    fig.savefig("{}days_allMSOAs.png".format(days))
    plt.show()


days14 =observations_df_t.iloc[0:14:, 0:107]

###### Each MSOA as a subplot
days = 404
fig, ax = plt.subplots(figsize=(40,40))
observations_df_t.iloc[0:days:, 0:107].plot.line(subplots=True, legend = None,
                                             layout=(22,5),ax=ax, color= 'black',
                                             xticks =[], yticks = [], sharey= True)
fig.subplots_adjust(top=0.95)
fig.suptitle('{} days'.format(days), fontsize = 60)
fig.savefig("{}days_commonaxis.png".format(days))

## Check which MSOA has the value in the first 14 days
for col in days14.columns:
    msoa = days14[col]
    if msoa.max() >0 :
        print(col, msoa.max())

##########################################################################
#########################################################################
# Spatial plots
##########################################################################
#########################################################################
# Create a column containing the MSOA code 
observations_df.reset_index(level=0, inplace=True)
# rename this to match the name in the Devon shapefile
observations_df.rename(columns={'MSOA11CD':'msoa11cd'}, inplace=True)
# read in outlines of MSOA's in England and Wales
england_msoas = gpd.read_file("C:/Users/gy17m2a/OneDrive - University of Leeds/Project/MSOA_EngWal_Dec_2011_Generalised_ClippedEW_0/Middle_Layer_Super_Output_Areas_December_2011_Generalised_Clipped_Boundaries_in_England_and_Wales.shp")
# keep only those with msoa codes matching those in the observations
devon_shapefile = england_msoas.loc[england_msoas['msoa11cd'].isin(observations_df['msoa11cd'])]
# reset index, so back to starting at 0
devon_shapefile=devon_shapefile.reset_index()
# Join shapefile to case data
observations_gdf = devon_shapefile.merge(observations_df, on='msoa11cd')

# Find the maximum value across all days (for a consistent scaling)
max=0
for col in observations_df.columns[1:]:
    if observations_df[col].max() > max:
        max = observations_df[col].max()
        
######################
frames = 404
fig = plt.figure(figsize=(36, 28)) 
ax = fig.add_subplot(1,1,1)

def draw(frame):
    ax.clear()
    #plt.clf()
    # Clear the previous figure
    day = 'D' + str(frame)
    #vmax = observations_df[day].max()
    vmax = max
    observations_gdf.plot(day, cmap="Blues",legend = False, edgecolor = 'black',
                    linewidth=2.4, ax=ax)
    #ax.legend(loc="best")
    ax.set_axis_off()
    ax.set_title('Positive cases in Devon - {}, vmax = {}'.format(day, vmax), fontdict={'fontsize':40})
    #plt.title('Observations')
    #return plot
    fig.tight_layout()
    plt.show()
    
def init():
    return draw(0)

def animate(frame):
    return draw(frame)

# Not sure what, if anything, this does
from matplotlib import rc, animation
rc('animation', html='html5')

ani = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=40, blit = False)
ani.save('devon_observations_consistentmax.gif', writer=PillowWriter(fps=8))

#ani.save('sine_wave.mp4', writer=animation.FFMpegWriter(fps=8))#


#################################################################
#################################################################
# Smoothing
#################################################################
#################################################################
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines

E02004151 = pd.DataFrame({'Day': observations_df_t.index,
                          'Cases':observations_df_t["E02004151"]})
fig, ax = plt.subplots(figsize=(20,10))
E02004151.plot(legend = None, layout=(22,5),ax=ax,
               color= 'black',kind = 'scatter',
               x='Day', y ='Cases')
fig, ax = plt.subplots(figsize=(20,10))
E02004151.plot.line(legend = None, layout=(22,5),ax=ax,
               color= 'black')



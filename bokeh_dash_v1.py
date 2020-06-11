# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:22:57 2020

@author: Natalie
"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import geopandas as gpd
import os
import pickle
import imageio
from shapely.geometry import Point
import numpy as np
from scipy import ndimage
import pickle

from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.models import (BasicTicker, CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter, 
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, PrintfTickFormatter, Slider)
from bokeh.layouts import row, column, gridplot, grid, widgetbox
from bokeh.models.widgets import Tabs, Panel
from bokeh.palettes import brewer
from bokeh.transform import transform

import json



# Read in data
# ------------
base_dir = os.getcwd()  # get current directory (assume RAMP-UA)
data_dir = os.path.join(base_dir, "data") # go to data dir

# read in details about venues
data_file = os.path.join(data_dir, "devon-schools","exeter schools.csv")
schools = pd.read_csv(data_file)
data_file = os.path.join(data_dir, "devon-retail","devon smkt.csv")
retail = pd.read_csv(data_file)

# load in a shapefile
sh_file = os.path.join(data_dir, "MSOAS_shp","bcc21fa2-48d2-42ca-b7b7-0d978761069f2020412-1-12serld.j1f7i.shp")
map_df = gpd.read_file(sh_file)
# check
#map_df.plot()
# rename column to get ready for merging
map_df.rename(index=str, columns={'msoa11cd': 'Area'},inplace=True)

# read in pickle file individuals (disease status)
data_file = os.path.join(data_dir, "output","Individuals.pickle")
pickle_in = open(data_file,"rb")
individuals = pickle.load(pickle_in)
pickle_in.close()

# read in pickle files location dangers - loop around locations
# redefining names here so script works as standalone, could refer to ActivityLocations instead
locations_dict = {
  "PrimarySchool": "PrimarySchool",
  "SecondarySchool": "SecondarySchool",
  "Retail": "Retail",
  "Work": "Work",
  "Home": "Home",
}
dangers_dict = {}  # empty dictionary to store results
for key, value in locations_dict.items():
    #print(f"{locations_dict[key]}.pickle")
    data_file = os.path.join(data_dir, "output",f"{locations_dict[key]}.pickle")
    pickle_in = open(data_file,"rb")
    dangers_dict[key] = pickle.load(pickle_in)
    pickle_in.close()
    
    
# Add additional info about schools and retail including spatial coordinates
# merge
primaryschools = pd.merge(schools, dangers_dict["PrimarySchool"], left_index=True, right_index=True)
secondaryschools = pd.merge(schools, dangers_dict["SecondarySchool"], left_index=True, right_index=True)
retail = pd.merge(retail, dangers_dict["Retail"], left_index=True, right_index=True)



# Get nr people per condition, day and msoa
# -----------------------------------------

# how many days have we got
nr_days = dangers_dict["Retail"].shape[1] - 1
days = [i for i in range(0,nr_days)]

# initialise variables
conditions_dict = {
  "susceptible": 0,
  "presymptomatic": 1,
  "symptomatic": 2,
  "recovered": 3,
  "dead": 4,
}
msoacounts_dict = {}  # empty dictionary to store results: nr per msoa and day
msoas = sorted(individuals.area.unique())
msoas_nr = [i for i in range(0,len(msoas))]
totalcounts_dict = {}  # empty dictionary to store results: nr per day

for key, value in conditions_dict.items():
    #msoacounts_dict[key] = pd.DataFrame (msoas, columns = ['Area'])
    msoacounts_dict[key] = pd.DataFrame(index=msoas)
    # loop aroud days
    for d in range(0, nr_days):
        print(d)
        # count nr for this condition per area
        msoa_count_temp = individuals[individuals.iloc[:, -nr_days+d] == conditions_dict[key]].groupby(['Area']).agg({individuals.columns[-nr_days+d]: ['count']})  
        # get current count for condition from dict
        msoacounts = msoacounts_dict[key]
        # add new column
        msoacounts = pd.merge(msoacounts,msoa_count_temp,left_index = True, right_index=True)
        msoacounts.rename(columns={ msoacounts.columns[d]: 'Day'+str(d) }, inplace = True)
        # write back to dict
        msoacounts_dict[key] = msoacounts
        
    # prepare data for plots
    # get current count for condition from dict
    msoacounts = msoacounts_dict[key]
    # to get nr of individuals across all msoas
    totalcounts_dict[key] = msoacounts.sum(axis=0)

        
# threshold map to only use MSOAs currently in the study or selection
map_df = map_df[map_df['Area'].isin(msoas)]
        
    
    
    
    
        























# Determine where the visualization will be rendered
output_file('dashboard_v1.html', title='RAMP-UA microsim output') # Render to static HTML
#output_notebook()  # To tender inline in a Jupyter Notebook

# default tools for  plots
tools = "crosshair,hover,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"


# plot 1: heatmap
# for now, let's plot symptomatic (2) - add other tabs later

var2plot = msoacounts_dict["symptomatic"]
var2plot = var2plot.rename_axis(None, axis=1).rename_axis('MSOA', axis=0)
var2plot.columns.name = 'Day'
# reshape to 1D array or rates with a month and year for each row.
df_var2plot = pd.DataFrame(var2plot.stack(), columns=['condition']).reset_index()
source = ColumnDataSource(df_var2plot)

# add better colour 
colors_1 = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper_1 = LinearColorMapper(palette=colors_1, low=0, high=var2plot.max().max())
s1 = figure(title="Heatmap",
           x_range=list(var2plot.columns), y_range=list(var2plot.index), x_axis_location="above")

s1.rect(x="Day", y="MSOA", width=1, height=1, source=source,
       line_color=None, fill_color=transform('condition', mapper_1))
color_bar_1 = ColorBar(color_mapper=mapper_1, location=(0, 0), orientation = 'horizontal', ticker=BasicTicker(desired_num_ticks=len(colors_1)))
s1.add_layout(color_bar_1, 'below')
s1.axis.axis_line_color = None
s1.axis.major_tick_line_color = None
s1.axis.major_label_text_font_size = "7px"
s1.axis.major_label_standoff = 0
s1.xaxis.major_label_orientation = 1.0

# Create hover tool
s1.add_tools(HoverTool(
    tooltips=[
        ( 'Nr symptomatic',   '@condition'),
        ( 'Day',  '@Day' ), 
        ( 'MSOA', '@MSOA'),
    ],
))



# plot 2: disease conditions across time

# build ColumnDataSource
data = {'days': days,
        'symptomatic': totalcounts_dict["symptomatic"],
        'recovered': totalcounts_dict["recovered"],
        'dead': totalcounts_dict["dead"],
        }
source_2 = ColumnDataSource(data=data)

s2 = figure(background_fill_color="#fafafa",title="Time", x_axis_label='Time', y_axis_label='Nr of people')
s2.line(x = 'days', y = 'symptomatic', source = source_2, legend_label="nr symptomatic", line_width=2, line_color="blue")
s2.circle(x = 'days', y = 'symptomatic', source = source_2, fill_color="blue", line_color="blue", size=5)
s2.line(x = 'days', y = 'recovered', source = source_2, legend_label="nr recovered", line_width=2, line_color="red")
s2.circle(x = 'days', y = 'recovered', source = source_2, fill_color="red", line_color="red", size=5)
s2.line(x = 'days', y = 'dead', source = source_2, legend_label="nr dead", line_width=2, line_color="green")
s2.circle(x = 'days', y = 'dead', source = source_2, fill_color="green", line_color="green", size=5)

s2.add_tools(HoverTool(
    tooltips=[
        ( 'Nr symptomatic',   '@symptomatic'),
        ( 'Nr recovered',   '@recovered'),
        ( 'Nr dead',   '@dead'),
        ( 'Day',  '@days' ), 
    ],
))

# plot 3: Conditions across MSOAs

s3 = figure(background_fill_color="#fafafa",title="MSOA", x_axis_label='Nr people', y_axis_label='MSOA')

# s3.circle(msoa_counts_E.sum(axis = 1), msoas_nr, fill_color="blue", line_color="blue", size=5, legend_label="Exposed")
# s3.circle(msoa_counts_I.sum(axis = 1), msoas_nr, fill_color="red", line_color="red", size=5, legend_label="Infectious")
# s3.circle(msoa_counts_R.sum(axis = 1), msoas_nr, fill_color="green", line_color="green", size=5, legend_label="Removed")



# build ColumnDataSource
data_df = { 'symptomatic': msoacounts_dict["symptomatic"].sum(axis = 1), 'recovered': msoacounts_dict["recovered"].sum(axis = 1), 'dead': msoacounts_dict["dead"].sum(axis = 1) ,  'msoa_nr':msoas_nr} 
data_df = pd.DataFrame(data_df) 
data_df['msoa_name'] = data_df.index
source_3 = ColumnDataSource(data_df)

s3.circle(x = 'symptomatic', y = 'msoa_nr', source = source_3, fill_color="blue", line_color="blue", size=5, legend_label="symptomatic")

s3.circle(x = 'recovered', y = 'msoa_nr', source = source_3, fill_color="red", line_color="red", size=5, legend_label="recovered")

s3.circle(x = 'dead', y = 'msoa_nr', source = source_3, fill_color="green", line_color="green", size=5, legend_label="dead")

s3.add_tools(HoverTool(
    tooltips=[
        ( 'Nr symptomatic',   '@symptomatic'),
        ( 'Nr recovered',   '@recovered'),
        ( 'Nr dead',   '@dead'),
        ( 'MSOA',  '@msoa_name' ), 
    ],
))








# plot 4: choropleth

# merge counts with spatial data
merged_data = msoacounts_dict["symptomatic"]
merged_data['Area'] = merged_data.index
merged_data = pd.merge(map_df,merged_data,on='Area')
# Turn data into GeoJSON source
geosource = GeoJSONDataSource(geojson = merged_data.to_json())

# Define color palettes
palette = brewer['BuGn'][8]
palette = palette[::-1] # reverse order of colors so higher values have darker colors
# Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
mapper_4 = LinearColorMapper(palette = palette, low = 0, high = 2000)
# Define custom tick labels for color bar.
tick_labels = {'0': '0', '500': '500', '1000':'1,000', '1500':'1,500',
 '2000':'2,000+'}
# Create color bar.
color_bar_4 = ColorBar(color_mapper = mapper_4, 
                     label_standoff = 8,
                     #"width = 500, height = 20,
                     border_line_color = None,
                     location = (0,0), 
                     orientation = 'horizontal',
                     major_label_overrides = tick_labels)


# Create figure object.
s4 = figure(title = 'Infections Day 2')
s4.xgrid.grid_line_color = None
s4.ygrid.grid_line_color = None
# Add patch renderer to figure.
msoasrender = s4.patches('xs','ys', source = geosource,
                   fill_color = {'field' :'Day2',
                                 'transform' : mapper_4},     
                   #fill_color = None,
                   line_color = 'gray', 
                   line_width = 0.25, 
                   fill_alpha = 1)
# Create hover tool
s4.add_tools(HoverTool(renderers = [msoasrender],
                      tooltips = [('MSOA','@Area'),
                                ('Nr exposed people','@Day2')]))
s4.add_layout(color_bar_4, 'below')


# make a grid
# grid = gridplot([[s1, s2], [s3, None]], plot_width=250, plot_height=250)
# show(grid)


l = grid([
    [s1,s3],
    [s2,s4],
])

show(l)

















# fig = figure(background_fill_color='gray',
#              background_fill_alpha=0.5,
#              border_fill_color='blue',
#              border_fill_alpha=0.25,
#              plot_height=300,
#              plot_width=500,
#              x_axis_label='X Label',
#              x_axis_type='datetime',
#              x_axis_location='above',
#              x_range=('2018-01-01', '2018-06-30'),
#              y_axis_label='Y Label',
#              y_axis_type='linear',
#              y_axis_location='left',
#              y_range=(0, 100),
#              title='Example Figure',
#              title_location='right',
#              toolbar_location='below',
#              tools='save')

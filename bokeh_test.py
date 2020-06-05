# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:22:57 2020

@author: Natalie
"""

"""Bokeh Visualization Template

This template is a general outline for turning your data into a 
visualization using Bokeh.
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


# Read in data
# ------------

# To fix file path issues, use absolute/full path at all times
# Pick either: get working directory (if user starts this script in place, or set working directory
# Option A: copy current working directory:
#base_dir = os.getcwd()  # get current directory (assume RAMP-UA)
#data_dir = os.path.join(base_dir, "data") # go to output dir
# Option B: specific directory
data_dir = 'C:\\Users\\Toshiba\\Google Drive\\WORK\\LIDA\\RAMP\\test_plots\\data'

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

data_file = os.path.join(data_dir, "output","Home.pickle")
pickle_in = open(data_file,"rb")
home_dangers = pickle.load(pickle_in)
pickle_in.close()

# load in a shapefile
sh_file = os.path.join(data_dir, "MSOAS_shp","bcc21fa2-48d2-42ca-b7b7-0d978761069f2020412-1-12serld.j1f7i.shp")
map_df = gpd.read_file(sh_file)
# check
#map_df.plot()
# rename column to get ready for merging
map_df.rename(index=str, columns={'msoa11cd': 'Area'},inplace=True)





# Preprocess data
# ---------------

# how many days have we got
nr_days = retail_dangers.shape[1] - 1
days = [i for i in range(0,nr_days)]

# total cases (SEIR) per area across time, and summed across MSOAs

# initialise variables
nrs_S = [0 for i in range(nr_days)] 
nrs_E = [0 for i in range(nr_days)] 
nrs_I = [0 for i in range(nr_days)] 
nrs_R = [0 for i in range(nr_days)] 
msoas = sorted(individuals.area.unique())
msoa_counts_S = pd.DataFrame(index=msoas)
msoa_counts_E = pd.DataFrame(index=msoas)
msoa_counts_I = pd.DataFrame(index=msoas)
msoa_counts_R = pd.DataFrame(index=msoas)

# loop aroud days
for d in range(0, nr_days):
    nrs = individuals.iloc[:,-nr_days+d].value_counts() 
    nrs_S[d] = nrs.get(0)  # susceptible?
    nrs_E[d] = nrs.get(1)  # exposed?
    nrs_I[d] = nrs.get(2)  # infectious?
    nrs_R[d] = nrs.get(3)  # recovered/removed?
    # S
    msoa_count_temp = individuals[individuals.iloc[:, -nr_days+d] == 0].groupby(['Area']).agg({individuals.columns[-nr_days+d]: ['count']})  
    msoa_counts_S = pd.merge(msoa_counts_S,msoa_count_temp,left_index = True, right_index=True)
    msoa_counts_S.rename(columns={ msoa_counts_S.columns[d]: 'Day'+str(d) }, inplace = True)
    # E
    msoa_count_temp = individuals[individuals.iloc[:, 73+d] == 1].groupby(['Area']).agg({individuals.columns[73+d]: ['count']})  
    msoa_counts_E = pd.merge(msoa_counts_E,msoa_count_temp,left_index = True, right_index=True)
    msoa_counts_E.rename(columns={ msoa_counts_E.columns[d]: 'Day'+str(d) }, inplace = True)
    # I
    msoa_count_temp = individuals[individuals.iloc[:, 73+d] == 2].groupby(['Area']).agg({individuals.columns[73+d]: ['count']})  
    msoa_counts_I = pd.merge(msoa_counts_I,msoa_count_temp,left_index = True, right_index=True)
    msoa_counts_I.rename(columns={ msoa_counts_I.columns[d]: 'Day'+str(d) }, inplace = True)
    # R
    msoa_count_temp = individuals[individuals.iloc[:, 73+d] == 3].groupby(['Area']).agg({individuals.columns[73+d]: ['count']})  
    msoa_counts_R = pd.merge(msoa_counts_R,msoa_count_temp,left_index = True, right_index=True)
    msoa_counts_R.rename(columns={ msoa_counts_R.columns[d]: 'Day'+str(d) }, inplace = True)
    
    # # sanity check: sum across MSOAs should be same as nrs_*
    # assert (msoa_counts_S.iloc[:,d].sum() == nrs_S[d])
    # assert (msoa_counts_E.iloc[:,d].sum() == nrs_E[d])
    # assert (msoa_counts_I.iloc[:,d].sum() == nrs_I[d])
    # assert (msoa_counts_R.iloc[:,d].sum() == nrs_R[d])




# !!! TEMPORARY - DELETE ONCE MICROSIM FIXED
# for now, original script not fully working (everyone is S so randomly make up some nrs and overwrite previous variables)
import random
random.seed()
for d in range(0, nr_days):
    for m in range(0,len(msoas)):
        total_msoa = msoa_counts_S.iloc[m,d]
        msoa_counts_E.iloc[m,d] = random.randrange(int(0.15*total_msoa), int(0.25*total_msoa), 1)
        msoa_counts_I.iloc[m,d] = random.randrange(int(0.1*total_msoa), int(0.2*total_msoa), 1)
        msoa_counts_R.iloc[m,d] = random.randrange(int(0.05*total_msoa), int(0.1*total_msoa), 1)
        msoa_counts_S.iloc[m,d] = total_msoa - msoa_counts_E.iloc[m,d] - msoa_counts_I.iloc[m,d] - msoa_counts_R.iloc[m,d]
        #assert (msoa_counts_S.iloc[m,d] + msoa_counts_E.iloc[m,d] + msoa_counts_I.iloc[m,d] + msoa_counts_R.iloc[m,d] == total_msoa)
    total_day = nrs_S[d]
    nrs_E[d] = msoa_counts_E.iloc[:,d].sum()
    nrs_I[d] = msoa_counts_I.iloc[:,d].sum()
    nrs_R[d] = msoa_counts_R.iloc[:,d].sum()
    nrs_S[d] = msoa_counts_S.iloc[:,d].sum()
    assert(total_day == nrs_S[d] + nrs_E[d] + nrs_I[d] + nrs_R[d])
    


# create (if first run, takes a while) or read in existing
# create
# for d in range(0, nr_days):
#     print(d)
#     for v in range(0, len(home_dangers)):
#         if d == 0:
#             # set seed
#             home_dangers.iloc[v,d+1] = 0
#         else:
#             home_dangers.iloc[v,d+1] = home_dangers.iloc[v,d] + random.randrange(-5, 5, 1)
#             if home_dangers.iloc[v,d+1] < 0:
#                 home_dangers.iloc[v,d+1] = 0
# data_file = os.path.join(data_dir, "output","Fake_home_dangers.pickle")
# pickle_out = open(data_file,"wb")
# pickle.dump(home_dangers, pickle_out)
# pickle_out.close() 
# read
data_file = os.path.join(data_dir, "output","Fake_home_dangers.pickle")
pickle_in = open(data_file,"rb")
home_dangers = pickle.load(pickle_in)
pickle_in.close()
        
for d in range(0, nr_days):
    print(d)
    for v in range(0, len(work_dangers)):
        if d == 0:
            # set seed
            work_dangers.iloc[v,d+1] = 0
        else:
            work_dangers.iloc[v,d+1] = work_dangers.iloc[v,d] + random.randrange(-5, 5, 1)
            if work_dangers.iloc[v,d+1] < 0:
                work_dangers.iloc[v,d+1] = 0
    
    for v in range(0, len(primaryschool_dangers)):
        if d == 0:
            # set seed
            primaryschool_dangers.iloc[v,d+1] = 0
            secondaryschool_dangers.iloc[v,d+1] = 0
        else:
            primaryschool_dangers.iloc[v,d+1] = primaryschool_dangers.iloc[v,d] + random.randrange(-5, 5, 1)
            if primaryschool_dangers.iloc[v,d+1] < 0:
                primaryschool_dangers.iloc[v,d+1] = 0
            secondaryschool_dangers.iloc[v,d+1] = secondaryschool_dangers.iloc[v,d] + random.randrange(-5, 5, 1)
            if secondaryschool_dangers.iloc[v,d+1] < 0:
                secondaryschool_dangers.iloc[v,d+1] = 0
        
    for v in range(0, len(retail_dangers)):
        if d == 0:
            # set seed
            retail_dangers.iloc[v,d+1] = 0
        else:
            retail_dangers.iloc[v,d+1] = retail_dangers.iloc[v,d] + random.randrange(-5, 5, 1)
            if retail_dangers.iloc[v,d+1] < 0:
                retail_dangers.iloc[v,d+1] = 0
        
     

# Add additional info about schools and retail including spatial coordinates
# merge
primaryschools = pd.merge(schools, primaryschool_dangers, left_index=True, right_index=True)
secondaryschools = pd.merge(schools, secondaryschool_dangers, left_index=True, right_index=True)
retail = pd.merge(retail, retail_dangers, left_index=True, right_index=True)




# merge spatial data and counts (created above)
msoa_counts_S['Area'] = msoa_counts_S.index
msoa_counts_E['Area'] = msoa_counts_E.index
msoa_counts_I['Area'] = msoa_counts_I.index
msoa_counts_R['Area'] = msoa_counts_R.index
msoa_counts_S_shp = pd.merge(map_df,msoa_counts_S,on='Area')
msoa_counts_E_shp = pd.merge(map_df,msoa_counts_E,on='Area')
msoa_counts_I_shp = pd.merge(map_df,msoa_counts_I,on='Area')
msoa_counts_R_shp = pd.merge(map_df,msoa_counts_R,on='Area')

merged_data = msoa_counts_E_shp

# Converting a Pandas Dataframe to a GeoPandas Dataframe forplotting points
#merged_data.crs # check coordinate system from underlay (here MSOAs - epsg:27700)
# Use for all:
crs = {'init': 'epsg:27700'}
# For primary schools:
geometry = [Point(xy) for xy in zip(primaryschools.Easting, primaryschools.Northing)]
primaryschools = primaryschools.drop(['Easting', 'Northing'], axis=1)
gdf_primaryschools = gpd.GeoDataFrame(primaryschools, crs=crs, geometry=geometry)
gdf_primaryschools = gdf_primaryschools[gdf_primaryschools['PhaseOfEducation_name']=="Primary"]

# For secondary schools:
geometry = [Point(xy) for xy in zip(secondaryschools.Easting, secondaryschools.Northing)]
secondaryschools = secondaryschools.drop(['Easting', 'Northing'], axis=1)
gdf_secondaryschools = gpd.GeoDataFrame(secondaryschools, crs=crs, geometry=geometry)
gdf_secondaryschools = gdf_secondaryschools[gdf_secondaryschools['PhaseOfEducation_name']=="Secondary"]

# For retail:
geometry = [Point(xy) for xy in zip(retail.bng_e, retail.bng_n)]
retail = retail.drop(['bng_e', 'bng_n'], axis=1)
gdf_retail = gpd.GeoDataFrame(retail, crs=crs, geometry=geometry)

# threshold map to only use MSOAs currently in the study or selection
map_df = map_df[map_df['Area'].isin(msoas)]

msoas_nr = [i for i in range(0,len(msoas))]


# Bokeh libraries
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





# Import reset_output (only needed once) 
from bokeh.plotting import reset_output

# Use reset_output() between subsequent show() calls, as needed
#reset_output()







# Prepare the data - already done
# # Create a ColumnDataSource object for each variable
# rockets_data = west_top_2[west_top_2['teamAbbr'] == 'HOU']
# nrs_E_cds = ColumnDataSource(nrs_E)


# Determine where the visualization will be rendered
output_file('bokeh_test.html', title='Add title here') # Render to static HTML, or 
#output_notebook()  # Render inline in a Jupyter Notebook

# create a new plot with a title and axis labels
tools = "crosshair,hover,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"


# plot 1: heatmap

var2plot = msoa_counts_E.iloc[:,0:nr_days]
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
        ( 'Nr exposed',   '@condition'),
        ( 'Day',  '@Day' ), 
        ( 'MSOA', '@MSOA'),
    ],
))



# plot 2: SEIR across time

# build ColumnDataSource
data = {'days': days,
        'exposed': nrs_E,
        'infectious': nrs_I,
        'removed': nrs_R,
        }
source_2 = ColumnDataSource(data=data)

s2 = figure(background_fill_color="#fafafa",title="Time", x_axis_label='Time', y_axis_label='Nr of people')
s2.line(x = 'days', y = 'exposed', source = source_2, legend_label="nr exposed", line_width=2, line_color="blue")
s2.circle(x = 'days', y = 'exposed', source = source_2, fill_color="blue", line_color="blue", size=5)
s2.line(x = 'days', y = 'infectious', source = source_2, legend_label="nr infectious", line_width=2, line_color="red")
s2.circle(x = 'days', y = 'infectious', source = source_2, fill_color="red", line_color="red", size=5)
s2.line(x = 'days', y = 'removed', source = source_2, legend_label="nr removed", line_width=2, line_color="green")
s2.circle(x = 'days', y = 'removed', source = source_2, fill_color="green", line_color="green", size=5)

s2.add_tools(HoverTool(
    tooltips=[
        ( 'Nr exposed',   '@exposed'),
        ( 'Nr infectious',   '@infectious'),
        ( 'Nr removed',   '@removed'),
        ( 'Day',  '@days' ), 
    ],
))

# plot 3: SEIR across MSOAs

s3 = figure(background_fill_color="#fafafa",title="MSOA", x_axis_label='Nr people', y_axis_label='MSOA')

# s3.circle(msoa_counts_E.sum(axis = 1), msoas_nr, fill_color="blue", line_color="blue", size=5, legend_label="Exposed")
# s3.circle(msoa_counts_I.sum(axis = 1), msoas_nr, fill_color="red", line_color="red", size=5, legend_label="Infectious")
# s3.circle(msoa_counts_R.sum(axis = 1), msoas_nr, fill_color="green", line_color="green", size=5, legend_label="Removed")



# build ColumnDataSource
data_df = { 'exposed': msoa_counts_E.sum(axis = 1), 'infectious': msoa_counts_I.sum(axis = 1), 'removed': msoa_counts_R.sum(axis = 1) ,  'msoa_nr':msoas_nr} 
data_df = pd.DataFrame(data_df) 
data_df['msoa_name'] = data_df.index
source_3 = ColumnDataSource(data_df)

s3.circle(x = 'exposed', y = 'msoa_nr', source = source_3, fill_color="blue", line_color="blue", size=5, legend_label="Exposed")

s3.circle(x = 'infectious', y = 'msoa_nr', source = source_3, fill_color="red", line_color="red", size=5, legend_label="Infectious")

s3.circle(x = 'removed', y = 'msoa_nr', source = source_3, fill_color="green", line_color="green", size=5, legend_label="Removed")

s3.add_tools(HoverTool(
    tooltips=[
        ( 'Nr exposed',   '@exposed'),
        ( 'Nr infectious',   '@infectious'),
        ( 'Nr removed',   '@removed'),
        ( 'MSOA',  '@msoa_name' ), 
    ],
))








# plot 4: choropleth

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

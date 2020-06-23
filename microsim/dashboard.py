# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:22:57 2020

@author: Natalie
"""

import os
import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
import imageio
from shapely.geometry import Point
import json

from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.models import (BasicTicker, CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter, 
                          GeoJSONDataSource, HoverTool, Legend,
                          LinearColorMapper, PrintfTickFormatter, Slider)
from bokeh.layouts import row, column, gridplot, grid, widgetbox
from bokeh.models.widgets import Tabs, Panel
from bokeh.palettes import brewer
from bokeh.transform import transform




# Read in data
# ------------
base_dir = os.getcwd()  # get current directory (usually RAMP-UA)
data_dir = os.path.join(base_dir, "devon_data") # go to data dir

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




# read in pickle files location dangers - loop around locations
# redefining names here so script works as standalone, could refer to ActivityLocations instead
locations_dict = {
  "PrimarySchool": "PrimarySchool",
  "SecondarySchool": "SecondarySchool",
  "Retail": "Retail",
  "Work": "Work",
  "Home": "Home",
}





# multiple runs: create mean and std dev
# we assume all directories in the ouput directory are runs (alternativelywe could ask user to supply number of runs)
nr_runs = len(next(os.walk(os.path.join(data_dir, "output")))[1])  

# venue dangers
# read in pickle files location dangers - loop around locations
dangers_dict = {}  # empty dictionary to store results
dangers_dict_std = {} 
dangers_dict_3d = {}
for key, value in locations_dict.items():
    for r in range(nr_runs):
        data_file = os.path.join(data_dir, "output",f"{r}",f"{locations_dict[key]}.pickle")
        pickle_in = open(data_file,"rb")
        dangers = pickle.load(pickle_in)
        pickle_in.close()
        # set row index to ID
        dangers.set_index('ID', inplace = True)
        dangers_colnames = dangers.columns
        dangers_rownames = dangers.index
        if r == 0:
            dangers_3d = np.zeros((dangers.shape[0],dangers.shape[1],nr_runs))        
        dangers_3d[:,:,r] = dangers.values
    dangers_dict_3d[key] = dangers_3d
    dangers_dict[key] = pd.DataFrame(data=dangers_3d.mean(axis=2), index=dangers_rownames, columns=dangers_colnames)
    dangers_dict_std[key] = pd.DataFrame(data=dangers_3d.std(axis=2), index=dangers_rownames, columns=dangers_colnames)
    
    
# how many days have we got
nr_days = dangers_dict["Retail"].shape[1]
days = [i for i in range(0,nr_days)]

    
# Add additional info about schools and retail including spatial coordinates
# merge
primaryschools = pd.merge(schools, dangers_dict["PrimarySchool"], left_index=True, right_index=True)
secondaryschools = pd.merge(schools, dangers_dict["SecondarySchool"], left_index=True, right_index=True)
retail = pd.merge(retail, dangers_dict["Retail"], left_index=True, right_index=True)


# read in postcode to MSOA conversion doc
data_file = os.path.join(data_dir, "PCD_OA_LSOA_MSOA_LAD_AUG19_UK_LU.csv")
postcode_lu = pd.read_csv(data_file, encoding = "ISO-8859-1", usecols = ["pcds", "msoa11cd"])
# creat LUT
indices = postcode_lu.pcds # your values from 15 to 75
results =  postcode_lu.msoa11cd # this constructs your ranges
lookup = dict(zip(indices, results)) # zip together the lists and make a dict from it
# use LUT and add column to retail variable
msoa_code = [lookup.get(retail.postcode[i]) for i in range(1, len(retail.postcode)+1, 1)]
retail.insert(2, 'MSOA_code', msoa_code)


# normalised danger scores per msoa for schools and retail (for choropleth)
dangers_msoa_dict = {}
for key in ['Retail','PrimarySchool','SecondarySchool']:
    dangers = dangers_dict[key]
    if key == 'Retail':
        msoa_code = retail.MSOA_code
    else:
        msoa_code = schools.MSOA_code
    dangers['MSOA'] = msoa_code
    # count nr for this condition per area
    msoa_sum = dangers.groupby(['MSOA']).agg('sum')  
    msoa_count = dangers.groupby(['MSOA']).agg('count')  
    msoa_avg =  msoa_sum.div(msoa_count, axis='index')
    dangers_msoa_dict[key] = msoa_avg

   
    
# Get nr people per condition, day and msoa
# -----------------------------------------

# initialise variables
conditions_dict = {
  "susceptible": 0,
  "presymptomatic": 1,
  "symptomatic": 2,
  "recovered": 3,
  "dead": 4,
}

msoacounts_dict = {}  # empty dictionary to store results: nr per msoa and day
totalcounts_dict = {}  # empty dictionary to store results: nr per day
cumcounts_dict = {}  # empty dictionary to store results: total nr across time period

# time range for cumulative counts
start_day = 0
end_day = nr_days-1 # last 'named' day - nrs starts from 0


for r in range(nr_runs):
    # read in pickle file individuals (disease status)
    data_file = os.path.join(data_dir, "output", f"{r}", "Individuals.pickle")
    pickle_in = open(data_file,"rb")
    individuals_tmp = pickle.load(pickle_in)
    pickle_in.close()
    # if first ever run, keep copy and initialise 3D frame for aggregating
    if r == 0:
        individuals = individuals_tmp
        msoas = sorted(individuals.area.unique())
        area_individuals = individuals['area']
        start_col = individuals.shape[1] - nr_days + start_day
        end_col = individuals.shape[1] - nr_days + end_day + 1
        uniqcounts = np.zeros((individuals.shape[0],len(conditions_dict),nr_runs))   # empty dictionary to store results: total nr, each person assigned to highest condition: 0<1<2<(3==4)
        
    for key, value in conditions_dict.items():
        if r == 0:
            msoacounts_dict[key] = np.zeros((len(msoas),nr_days,nr_runs))        
            totalcounts_dict[key] = np.zeros((nr_days,nr_runs))  
            cumcounts_dict[key] = np.zeros((len(msoas),nr_runs))
        # cumulative counts
        # select right columns
        tmp = individuals_tmp.iloc[:,start_col:end_col]  
        # find all rows with condition (dict value)
        indices = tmp[tmp.eq(value).any(1)].index
        # create new df of zeros and replace with 1 at indices
        cumcounts_run = pd.DataFrame(np.zeros((tmp.shape[0], 1)))
        cumcounts_run.loc[indices] = 1
        uniqcounts[:,value,r] = cumcounts_run.values[:,0]
        # merge with MSOA df
        cumcounts_run = cumcounts_run.merge(area_individuals, left_index=True, right_index=True)
        cumcounts_msoa_run = cumcounts_run.groupby(['area']).sum()
        cumcounts_msoa_run = cumcounts_msoa_run.values

        # loop aroud days
        msoacounts_run = np.zeros((len(msoas),nr_days))
        for d in range(0, nr_days):
            # count nr for this condition per area
            msoa_count_temp = individuals_tmp[individuals_tmp.iloc[:, -nr_days+d] == conditions_dict[key]].groupby(['Area']).agg({individuals_tmp.columns[-nr_days+d]: ['count']})  
            msoa_count_temp = msoa_count_temp.values
            # add new column
            msoacounts_run[:,d] = msoa_count_temp[:, 0]
            
        # get current values from dict
        msoacounts = msoacounts_dict[key]
        totalcounts = totalcounts_dict[key]
        cumcounts = cumcounts_dict[key]
        # add current run's values
        msoacounts[:,:,r] = msoacounts_run
        totalcounts[:,r] = msoacounts_run.sum(axis=0)
        cumcounts[:,r] = cumcounts_msoa_run[:, 0]
        # write out to dict
        msoacounts_dict[key] = msoacounts
        totalcounts_dict[key] = totalcounts
        cumcounts_dict[key] = cumcounts
   
dict_days = [] # empty list for column names 'Day0' etc
for d in range(0, nr_days):
    dict_days.append(f'Day{d}')
                 
        
# aggregate counts    
msoacounts_dict_std = {}  
totalcounts_dict_std = {}
cumcounts_dict_std = {}

msoacounts_dict_3d = dict(msoacounts_dict) 
cumcounts_dict_3d = dict(cumcounts_dict) 
for key, value in conditions_dict.items():
    # get current values from dict
    msoacounts = msoacounts_dict[key]
    totalcounts = totalcounts_dict[key]
    cumcounts = cumcounts_dict[key]
    # aggregate
    msoacounts_std = msoacounts.std(axis=2)
    msoacounts = msoacounts.mean(axis=2)
    totalcounts_std = totalcounts.std(axis=1)
    totalcounts = totalcounts.mean(axis=1)
    cumcounts_std = cumcounts.std(axis=1)
    cumcounts = cumcounts.mean(axis = 1)
    # write out to dict
    msoacounts_dict[key] = pd.DataFrame(data=msoacounts, index=msoas, columns=dict_days)
    msoacounts_dict_std[key] = pd.DataFrame(data=msoacounts_std, index=msoas, columns=dict_days)
    totalcounts_dict[key] = pd.Series(data=totalcounts, index=dict_days)
    totalcounts_dict_std[key] = pd.Series(data=totalcounts_std, index=dict_days)
    cumcounts_dict[key] = pd.Series(data=cumcounts, index=msoas)
    cumcounts_dict_std[key] = pd.Series(data=cumcounts_std, index=msoas)
 
    
# TO BE COMPLETED - calculate unique nr people per condition 
# uniqcounts_dict_3d = {}
# for r in range(nr_runs):
#     tmp = uniqcounts[:,4,r]
#     result = np.where(tmp == 1)
#     indices = result[0]
#     # set other values on row to zero




msoas_nr = [i for i in range(0,len(msoas))]
        
# threshold map to only use MSOAs currently in the study or selection
map_df = map_df[map_df['Area'].isin(msoas)]

# to normalise counts, we need to know the total nr of people per MSOA
nrpeople_msoa = individuals[['ID','area']].groupby(['area']).count()
nrpeople_msoa.rename(columns={'ID':'nr_people'}, inplace=True)


    





# Determine where the visualization will be rendered
output_file('dashboard.html', title='RAMP-UA microsim output') # Render to static HTML
#output_notebook()  # To tender inline in a Jupyter Notebook

# default tools for  plots
tools = "crosshair,hover,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"

# pick colours for different conditions
colour_dict = {
  "susceptible": "blue",
  "presymptomatic": "orange",
  "symptomatic": "red",
  "recovered": "green",
  "dead": "black",
  "Retail": "blue",
  "PrimarySchool": "orange",
  "SecondarySchool": "red",
  "Work": "black",
  "Home": "green",
}

# colours for heatmaps
colors_1 = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]

# colours for choropleths
palette = brewer['BuGn'][8]
palette = palette[::-1] # reverse order of colors so higher values have darker colors

# basic tool tip for condition plots
tooltips_cond_basic=[]
for key, value in totalcounts_dict.items():
    tooltips_cond_basic.append(tuple(( f"Nr {key}",   f"@{key}")))

tooltips_venue_basic=[]
for key, value in dangers_dict.items():
    tooltips_venue_basic.append(tuple(( f"Danger {key}",   f"@{key}")))
   
    
# empty dictionary to track condition and venue specific plots
plotref_dict = {}

   



    
# plot 1: heatmap

def plot_heatmap_condition(condition2plot):
    """ Create heatmap plot: x axis = time, y axis = MSOAs, colour = nr people with condition = condition2plot. condition2plot is key to conditions_dict."""
    
    var2plot = msoacounts_dict[condition2plot]
    var2plot = var2plot.rename_axis(None, axis=1).rename_axis('MSOA', axis=0)
    var2plot.columns.name = 'Day'
    # reshape to 1D array or rates with a month and year for each row.
    df_var2plot = pd.DataFrame(var2plot.stack(), columns=['condition']).reset_index()
    source = ColumnDataSource(df_var2plot)
    
    # add better colour 
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
            ( f'Nr {condition2plot}',   '@condition'),
            ( 'Day',  '@Day' ), 
            ( 'MSOA', '@MSOA'),
        ],
    ))
    
    plotref_dict[f"hm{condition2plot}"] = s1

for key,value in conditions_dict.items():
    plot_heatmap_condition(key)


# plot 2: disease conditions across time

# build ColumnDataSource
data_s2 = dict(totalcounts_dict)
data_s2["days"] = days
for key, value in totalcounts_dict.items():
    #totalcounts_dict_std[f"{key}_std"] = totalcounts_dict_std.pop(f"{key}")
    data_s2[f"{key}_std_upper"] = totalcounts_dict[key] + totalcounts_dict_std[key]
     #x.add(y, fill_value=0)
    data_s2[f"{key}_std_lower"] = totalcounts_dict[key] - totalcounts_dict_std[key]
#data_s2.update(totalcounts_dict_std)
source_2 = ColumnDataSource(data=data_s2)

s2 = figure(background_fill_color="#fafafa",title="Time", x_axis_label='Time', y_axis_label='Nr of people')
# for key, value in totalcounts_dict.items():
#     s2.line(x = 'days', y = key, source = source_2, legend_label=f"nr {key}", line_width=2, line_color=colour_dict[key])
#     s2.square(x = 'days', y = key, source = source_2, fill_color=colour_dict[key], line_color=colour_dict[key], size=5,legend_label=f"nr {key}")
    
#     s2.rect('days', f"{key}_std_upper", 0.2, 0.01, source = source_2, line_color="black")
#     s2.rect('days', f"{key}_std_lower", 0.2, 0.01, source = source_2, line_color="black")
#     s2.segment('days', f"{key}_std_lower", 'days', f"{key}_std_upper", source = source_2, line_color="black")
#     # # find the outliers
#     # y = data_s2[key]
#     # std_lower = data_s2[f"{key}_std_lower"]
#     # std_upper = data_s2[f"{key}_std_upper"]
#     # outliers = [y[(y > std_upper) | (y < std_lower) for y in y for std_upper in std_upper]
    
#     # [y[(y > std_upper) | (y < std_lower)] for x in my_list]
    
#     # y[(y > std_upper) | (y < std_lower)]
    
#     # map(lambda ro: map(op, ro[0], ro[1]), zip(a,b))
#     # outliers = y[(y > std_upper) | (y < std_lower)]
#     # x = days[0:len(y3)]
#     # s2.circle(x = x3, y = y3, fill_color="black", size=5)
    

# tooltips = tooltips_cond_basic.copy()
# tooltips.append(tuple(( 'Day',  '@days' )))
# s2.add_tools(HoverTool(
#     tooltips=tooltips,
# ))
# s2.legend.location = "top_left"
# s2.legend.click_policy="hide"


legend_it = []
for key, value in totalcounts_dict.items():
    c1 = s2.line(x = 'days', y = key, source = source_2, line_width=2, line_color=colour_dict[key],muted_color="grey", muted_alpha=0.2)
    
    c2 = s2.square(x = 'days', y = key, source = source_2, fill_color=colour_dict[key], line_color=colour_dict[key], size=5, muted_color="grey", muted_alpha=0.2)
    
    legend_it.append((f"nr {key}", [c1,c2]))
    
    s2.rect('days', f"{key}_std_upper", 0.2, 0.01, source = source_2, line_color="black")
    s2.rect('days', f"{key}_std_lower", 0.2, 0.01, source = source_2, line_color="black")
    s2.segment('days', f"{key}_std_lower", 'days', f"{key}_std_upper", source = source_2, line_color="black")
    
legend = Legend(items=legend_it)
legend.click_policy="hide"

tooltips = tooltips_cond_basic.copy()
tooltips.append(tuple(( 'Day',  '@days' )))
s2.add_tools(HoverTool(
    tooltips=tooltips,
))
s2.add_layout(legend, 'right')

    

    





#     y = msoacounts_dict_3d[key][msoa,day,:]
#     x = np.ones(10)
#     x = [i * value for i in x]
#     # s5.circle(x = x, y = y, fill_color=colour_dict[key], size=3,legend_label=f"nr {key}")
#     y2 = y.mean()
#     x2 = value
#     s5.square(x = x2, y = y2, line_color="black", size=5)
#     stdev = y.std()
#     s5.rect(x, y2-stdev, 0.2, 0.01, line_color="black")
#     s5.rect(x, y2+stdev, 0.2, 0.01, line_color="black")
#     s5.segment(x, y2-stdev, x, y2+stdev, line_color="black")
#     # find the outliers
#     y3 = y[(y > y2+stdev) | (y < y2-stdev)]
#     x3 = x[0:len(y3)]
#     s5.circle(x = x3, y = y3, fill_color="black", size=5)



# plot 3: Conditions across MSOAs

# build ColumnDataSource
data_s3 = {}
data_s3["msoa_nr"] = msoas_nr
data_s3["msoa_name"] = msoas
for key, value in cumcounts_dict.items():
    data_s3[key] = cumcounts_dict[key]
source_3 = ColumnDataSource(data=data_s3)

s3 = figure(background_fill_color="#fafafa",title="MSOA", x_axis_label='Nr people', y_axis_label='MSOA',toolbar_location='above')

legend_it = []
for key, value in msoacounts_dict.items():
    c = s3.circle(x = key, y = 'msoa_nr', source = source_3, fill_color=colour_dict[key], line_color=colour_dict[key], size=5,muted_color="grey", muted_alpha=0.2)    
    legend_it.append((key, [c]))
legend = Legend(items=legend_it)
legend.click_policy="hide"
    
tooltips = tooltips_cond_basic.copy()
tooltips.append(tuple(( 'MSOA',  '@msoa_name' )))
s3.add_tools(HoverTool(
    tooltips=tooltips,
))
s3.add_layout(legend, 'right')


# plot 4: choropleth

# merge counts with spatial data
data_s4 = data_s3.copy()
merged_data = pd.DataFrame()
merged_data['Area'] = msoas
for key, value in msoacounts_dict.items():
    #merged_data[key] = msoacounts_dict[key].sum(axis = 1).to_list()
    merged_data[key] = cumcounts_dict[key].to_list()
merged_data = pd.merge(map_df,merged_data,on='Area')
# Turn data into GeoJSON source
geosource = GeoJSONDataSource(geojson = merged_data.to_json())

def plot_choropleth_condition(condition2plot):
    """ Create choropleth: colour = nr people with condition = condition2plot. condition2plot is key to conditions_dict. """

    # Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
    mapper_4 = LinearColorMapper(palette = palette, low = 0, high = merged_data[condition2plot].max())
    # Create color bar.
    color_bar_4 = ColorBar(color_mapper = mapper_4, 
                         label_standoff = 8,
                         #"width = 500, height = 20,
                         border_line_color = None,
                         location = (0,0), 
                         orientation = 'horizontal')#,
                         #major_label_overrides = tick_labels)
    
    # Create figure object.
    s4 = figure(title = f"{condition2plot} total")
    s4.xgrid.grid_line_color = None
    s4.ygrid.grid_line_color = None
    # Add patch renderer to figure.
    msoasrender = s4.patches('xs','ys', source = geosource,
                       fill_color = {'field' :condition2plot,
                                     'transform' : mapper_4},     
                       #fill_color = None,
                       line_color = 'gray', 
                       line_width = 0.25, 
                       fill_alpha = 1)
    # Create hover tool
    s4.add_tools(HoverTool(renderers = [msoasrender],
                          tooltips = [('MSOA','@Area'),
                                    (f'Nr {condition2plot} people',f'@{condition2plot}')]))
    s4.add_layout(color_bar_4, 'below')

    plotref_dict[f"ch{condition2plot}"] = s4

for key,value in conditions_dict.items():
    plot_choropleth_condition(key)





# # plot 5: dangers heatmap - too dense, replace by composite version or remove

# def plot_heatmap_venues(venue2plot):
#     """ Create heatmap plot: x axis = time, y axis = location, colour = dangerscore = venue2plot. venue2plot is key to dangers_dict."""
    
#     var2plot = dangers_dict[venue2plot]
#     #var2plot = var2plot.rename_axis(None, axis=1).rename_axis('MSOA', axis=0)
#     var2plot.columns.name = 'Day'
#     # reshape to 1D array or rates with a month and year for each row.
#     df_var2plot = pd.DataFrame(var2plot.stack(), columns=['danger']).reset_index()
#     source = ColumnDataSource(df_var2plot)
    
#     # add better colour 
#     mapper_1 = LinearColorMapper(palette=colors_1, low=0, high=var2plot.max().max())
    
#     s5 = figure(title="Heatmap", x_range=list(var2plot.columns), y_range=[str(integer) for integer in list(var2plot.index)], x_axis_location="above")
    
#     s5.rect(x="Day", y="ID", width=1, height=1, source=source,
#            line_color=None, fill_color=transform('danger', mapper_1))
#     color_bar_1 = ColorBar(color_mapper=mapper_1, location=(0, 0), orientation = 'horizontal', ticker=BasicTicker(desired_num_ticks=len(colors_1)))
#     s5.add_layout(color_bar_1, 'below')
#     s5.axis.axis_line_color = None
#     s5.axis.major_tick_line_color = None
#     s5.axis.major_label_text_font_size = "7px"
#     s5.axis.major_label_standoff = 0
#     s5.xaxis.major_label_orientation = 1.0
    
#     # Create hover tool
#     s5.add_tools(HoverTool(
#         tooltips=[
#             ( f'Danger score {venue2plot}',   '@danger'),
#             ( 'Day',  '@Day' ), 
#             ( 'Venue', '@ID'),
#         ],
#     ))
    
#     plotref_dict[f"hm{venue2plot}"] = s5

# for key,value in dangers_dict.items():
#     plot_heatmap_venues(key)




# plot 5: danger scores across time per venue type

# build ColumnDataSource
data_s5 = {}
data_s5["days"] = days
for key, value in dangers_dict.items():
    data_s5[key] = value.mean(axis = 0)


source_5 = ColumnDataSource(data=data_s5)

s5 = figure(background_fill_color="#fafafa",title="Time", x_axis_label='Time', y_axis_label='Average danger score')
for key, value in dangers_dict.items():
    s5.line(x = 'days', y = key, source = source_5, legend_label=f"Danger {key}", line_width=2, line_color=colour_dict[key])
    s5.circle(x = 'days', y = key, source = source_5, fill_color=colour_dict[key], line_color=colour_dict[key], size=5,legend_label=f"Danger {key}")

tooltips = tooltips_venue_basic.copy()
tooltips.append(tuple(( 'Day',  '@days' )))
s5.add_tools(HoverTool(
    tooltips=tooltips,
))
s5.legend.location = "top_left"
s5.legend.click_policy="hide"









# # plot 6: disease conditions across time

# # build ColumnDataSource
# data_s6 = dict(totalcounts_dict_std)
# data_s6["days"] = days
# source_6 = ColumnDataSource(data=data_s6)

# s6 = figure(background_fill_color="#fafafa",title="Time", x_axis_label='Time', y_axis_label='Stdev Nr of people')
# for key, value in totalcounts_dict.items():
#     s6.line(x = 'days', y = key, source = source_6, legend_label=f"nr {key}", line_width=2, line_color=colour_dict[key])
#     s6.circle(x = 'days', y = key, source = source_6, fill_color=colour_dict[key], line_color=colour_dict[key], size=5,legend_label=f"nr {key}")

# tooltips = tooltips_cond_basic.copy()
# tooltips.append(tuple(( 'Day',  '@days' )))
# s6.add_tools(HoverTool(
#     tooltips=tooltips,
# ))
# s6.legend.location = "top_left"
# s6.legend.click_policy="hide"

# show(s6)





# plot 7: choropleth dangers



def plot_choropleth_danger(venue2plot):
    """ Create choropleth: colour = average danger score """
    # merge counts with spatial data
    merged_data = pd.DataFrame()
    merged_data["Danger"] = dangers_msoa_dict[venue2plot].sum(axis = 1).to_list()
    merged_data["Area"] = dangers_msoa_dict[venue2plot].index.to_list()
    merged_data = pd.merge(map_df,merged_data,on='Area')
    # Turn data into GeoJSON source
    geosource = GeoJSONDataSource(geojson = merged_data.to_json())
    
    

    # Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
    mapper_4 = LinearColorMapper(palette = palette, low = 0, high = merged_data["Danger"].max())
    # Create color bar.
    color_bar_4 = ColorBar(color_mapper = mapper_4, 
                          label_standoff = 8,
                          #"width = 500, height = 20,
                          border_line_color = None,
                          location = (0,0), 
                          orientation = 'horizontal')#,
                          #major_label_overrides = tick_labels)
    
    # Create figure object.
    s4 = figure(title = f"{venue2plot} total")
    s4.xgrid.grid_line_color = None
    s4.ygrid.grid_line_color = None
    # Add patch renderer to figure.
    msoasrender = s4.patches('xs','ys', source = geosource,
                        fill_color = {'field' : 'Danger',
                                      'transform' : mapper_4},     
                        #fill_color = None,
                        line_color = 'gray', 
                        line_width = 0.25, 
                        fill_alpha = 1)
    # Create hover tool
    s4.add_tools(HoverTool(renderers = [msoasrender],
                          tooltips = [('MSOA','@Area'),
                                    ('Danger', '@Danger')]))
    s4.add_layout(color_bar_4, 'below')

    plotref_dict[f"ch{venue2plot}"] = s4

for key,value in dangers_msoa_dict.items():
    plot_choropleth_danger(key)




# Layout and outpit
# Create a Panel with a title for each tab
tab1 = Panel(child=row(s2, s3), title='Summary conditions')
tab2 = Panel(child=row(plotref_dict["hmsusceptible"],plotref_dict["chsusceptible"]), title='Susceptible')
tab3 = Panel(child=row(plotref_dict["hmpresymptomatic"],plotref_dict["chpresymptomatic"]), title='Presymptomatic')
tab4 = Panel(child=row(plotref_dict["hmsymptomatic"],plotref_dict["chsymptomatic"]), title='Symptomatic')
tab5 = Panel(child=row(plotref_dict["hmrecovered"],plotref_dict["chrecovered"]), title='Recovered')
tab6 = Panel(child=row(plotref_dict["hmdead"],plotref_dict["chdead"]), title='Dead')

# tab7 = Panel(child=row(plotref_dict["hmRetail"],plotref_dict["hmPrimarySchool"]), title='test')
tab7 = Panel(child=row(s5), title='Summary dangers')

tab8 = Panel(child=row(plotref_dict["chRetail"],plotref_dict["chPrimarySchool"],plotref_dict["chSecondarySchool"]), title='Danger choropleths')

# Put the Panels in a Tabs object
tabs = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8])
show(tabs)




# l = grid([
#     [s1,s3],
#     [s2,s4],
# ])

# show(l)

# day = 20
# msoa = 0

# s5 = figure(background_fill_color="#fafafa",title="Variability", x_axis_label='Condition', y_axis_label='Nr people')

# for key, value in conditions_dict.items():
#     y = msoacounts_dict_3d[key][msoa,day,:]
#     x = np.ones(10)
#     x = [i * value for i in x]
#     # s5.circle(x = x, y = y, fill_color=colour_dict[key], size=3,legend_label=f"nr {key}")
#     y2 = y.mean()
#     x2 = value
#     s5.square(x = x2, y = y2, line_color="black", size=5)
#     stdev = y.std()
#     s5.rect(x, y2-stdev, 0.2, 0.01, line_color="black")
#     s5.rect(x, y2+stdev, 0.2, 0.01, line_color="black")
#     s5.segment(x, y2-stdev, x, y2+stdev, line_color="black")
#     # find the outliers
#     y3 = y[(y > y2+stdev) | (y < y2-stdev)]
#     x3 = x[0:len(y3)]
#     s5.circle(x = x3, y = y3, fill_color="black", size=5)

# show(s5)
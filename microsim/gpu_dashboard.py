# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:22:57 2020

@author: Natalie
"""

import os
import click
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
                          CustomJS, CustomJSFilter, FactorRange, 
                          GeoJSONDataSource, HoverTool, Legend,
                          LinearColorMapper, PrintfTickFormatter, Slider)
from bokeh.layouts import row, column, gridplot, grid, widgetbox
from bokeh.models.widgets import Tabs, Panel
from bokeh.palettes import brewer
from bokeh.transform import transform, factor_cmap

import click  # command-line interface
from yaml import load, dump, SafeLoader  # pyyaml library for reading the parameters.yml file






# Functions for preprocessing
# ---------------------------




def calc_nr_days(data_file):
    # figure out nr days by reading in e.g. retail dangers pickle file of run 0      
    pickle_in = open(data_file,"rb")
    dangers = pickle.load(pickle_in)
    pickle_in.close()
            
    filter_col = [col for col in dangers if col.startswith('Danger')]
    # don't use the column simply called 'Danger'
    filter_col = filter_col[1:len(filter_col)]
    nr_days = len(filter_col)
    return nr_days


def create_venue_dangers_dict(locations_dict,r_range,data_dir,start_day,end_day,start_run,nr_runs):
    '''
    Reads in venue pickle files (venues from locations_dict) and populates dangers_dict_3d (raw data: venue, day, run), dangers_dict (mean across runs) and dangers_dict_std (standard deviation across runs)
    Possible output includes:
    dangers_dict       # mean (value to be plotted)
    dangers_dict_std   # standard deviation (could plot as error bars)
    dangers_dict_3d    # full 3D data (for debugging)
    '''
    
    dangers_dict = {} 
    dangers_dict_std = {} 
    dangers_dict_3d = {} 
    
    for key, value in locations_dict.items():
        #for r in range(nr_runs):
        for r in r_range:
            
            data_file = os.path.join(data_dir, f"{r}",f"{locations_dict[key]}.pickle")
            pickle_in = open(data_file,"rb")
            dangers = pickle.load(pickle_in)
            pickle_in.close()
            
            filter_col = [col for col in dangers if col.startswith('Danger')]
            # don't use the column simply called 'Danger'
            filter_col = filter_col[1:len(filter_col)]
            #nr_days = len(filter_col)

            # # set row index to ID
            # dangers.set_index('ID', inplace = True)
            dangers_colnames = filter_col[start_day:end_day+1]
            dangers_rownames = dangers.index
            dangers_values = dangers[filter_col[start_day:end_day+1]]
            
            if r == start_run:
                dangers_3d = np.zeros((dangers.shape[0],dangers_values.shape[1],nr_runs))        
            dangers_3d[:,:,r-start_run] = dangers_values
        dangers_dict_3d[key] = dangers_3d
        dangers_dict[key] = pd.DataFrame(data=dangers_3d.mean(axis=2), index=dangers_rownames, columns=dangers_colnames)
        dangers_dict_std[key] = pd.DataFrame(data=dangers_3d.std(axis=2), index=dangers_rownames, columns=dangers_colnames)
        
    return dangers_dict, dangers_dict_std, dangers_dict_3d
        
        
def create_difference_dict(dict_sc0,dict_sc1,lookup_dict):
    dict_out = {}
    for key, value in lookup_dict.items():
        dict_out[key] = dict_sc1[key].subtract(dict_sc0[key])
    return dict_out



def create_msoa_dangers_dict(dangers_dict,keys,msoa_codes):
    '''
    Converts dangers_dict to MSOA level data for the appropriate venue types. Produces average danger score (sum dangers in MSOA / total nr venues in MSOA)
    Output: dangers_msoa_dict
    '''
    
    dangers_msoa_dict = {}
    for k in range(0,len(keys)):
        dangers = dangers_dict[keys[k]]
        msoa_code = msoa_codes[k]
        dangers['MSOA'] = msoa_code
        # count nr for this condition per area
        msoa_sum = dangers.groupby(['MSOA']).agg('sum')  
        msoa_count = dangers.groupby(['MSOA']).agg('count')  
        msoa_avg =  msoa_sum.div(msoa_count, axis='index')
        dangers_msoa_dict[keys[k]] = msoa_avg
    return dangers_msoa_dict


def create_counts_dict(conditions_dict,r_range,data_dir,start_day,end_day,start_run,nr_runs,age_cat):
    '''
    Counts per condition (3D, mean and standard deviation)
    Produces 5 types of counts:
    msoacounts: nr per msoa and day
    agecounts: nr per age category and day
    totalcounts: nr per day (across all areas)
    cumcounts: nr per MSOA (across given time period)
    uniquecounts: nr with 'final' disease status across time period e.g. someone who is presymptomatic, symptomatic and recoverd is only counted once as recovered
    Output: 
    msoas                   # list of msoas
    totalcounts_dict, cumcounts_dict, agecounts_dict,  msoacounts_dict, cumcounts_dict_3d, totalcounts_dict_std, cumcounts_dict_std, agecounts_dict_std, msoacounts_dict_std, totalcounts_dict_3d, agecounts_dict_3d, msoacounts_dict_3d, uniquecounts_dict_3d, uniquecounts_dict_std, uniquecounts_dict
    '''
    
    # start with empty dictionaries
    msoas = []
    msoacounts_dict_3d = {} 
    totalcounts_dict_3d = {}  
    cumcounts_dict_3d = {}  
    agecounts_dict_3d = {}  
    uniquecounts_dict_3d = {} 
    msoacounts_dict = {}  
    agecounts_dict = {}  
    totalcounts_dict = {}
    cumcounts_dict = {}
    uniquecounts_dict = {}
    msoacounts_dict_std = {}  
    agecounts_dict_std = {}
    totalcounts_dict_std = {}
    cumcounts_dict_std = {}
    uniquecounts_dict_std = {}
    
    nr_days = end_day - start_day + 1
    dict_days = [] # empty list for column names 'Day0' etc
    for d in range(start_day, end_day+1):
        dict_days.append(f'Day{d}')
    age_cat_str = []
    for a in range(age_cat.shape[0]):
        age_cat_str.append(f"{age_cat[a,0]}-{age_cat[a,1]}")
    
    # first, create 3d dictionaries
    for r in r_range:
        # read in pickle file individuals (disease status)
        data_file = os.path.join(data_dir, f"{r}", "Individuals.pickle")
        pickle_in = open(data_file,"rb")
        individuals_tmp = pickle.load(pickle_in)
        pickle_in.close()
        # if first ever run, keep copy and initialise 3D frame for aggregating
        if r == start_run:
            individuals = individuals_tmp.copy()
            msoas.extend(sorted(individuals.area.unique())) # populate list of msoas (previously empty outside this function)
            area_individuals = individuals['area'] # keep area per person to use later
            
            # next bit of code is to restrict to user specified day range
            # first, find all columns starting with disease_status
            filter_col = [col for col in individuals if col.startswith('disease_status')]
            # don't use the column simply called 'disease_status'
            filter_col = filter_col[1:len(filter_col)]
            counts_colnames = filter_col[start_day:end_day+1]
            
            
            # TO BE ADDED SO USER CAN DEFINE AGE BRACKETS - FOR NOW JUST USING PREDEFINED CATEGORIES
            # individuals['age0'] = np.zeros((len(individuals),1))
            # for a in range(age_cat.shape[0]):
            #     individuals['age0'] = np.where((individuals['DC1117EW_C_AGE'] >= age_cat[a,0]) & (individuals['DC1117EW_C_AGE'] <= age_cat[a,1]), a+1, individuals['age0'])   
            # age_cat_col = individuals['age0'].values
            # temporary workaround
            age_cat_col = individuals['Age1'].values 
            
        # add age brackets column to individuals_tmp
        individuals_tmp.insert(8, 'age0', age_cat_col)

        uniquecounts_df = pd.DataFrame()
        
        for key, value in conditions_dict.items():
            #print(key)
            if r == start_run:
                msoacounts_dict_3d[key] = np.zeros((len(msoas),nr_days,nr_runs))        
                agecounts_dict_3d[key] = np.zeros((age_cat.shape[0],nr_days,nr_runs))
                totalcounts_dict_3d[key] = np.zeros((nr_days,nr_runs))  
                cumcounts_dict_3d[key] = np.zeros((len(msoas),nr_runs))
                uniquecounts_dict_3d[key] = np.zeros(nr_runs)
                
            # cumulative counts
            # select right columns
            tmp = individuals_tmp[counts_colnames]
            #tmp = individuals_tmp.iloc[:,start_col:end_col]  
            # find all rows with condition (dict value)
            indices = tmp[tmp.eq(value).any(1)].index
            # create new df of zeros and replace with 1 at indices
            cumcounts_run = pd.DataFrame(np.zeros((tmp.shape[0], 1)))
            cumcounts_run.loc[indices] = 1
            
            uniquecounts_df[key] = cumcounts_run.values[:,0]
            
            #uniqcounts[:,value,r] = cumcounts_run.values[:,0]
            # merge with MSOA df
            cumcounts_run = cumcounts_run.merge(area_individuals, left_index=True, right_index=True)
            cumcounts_msoa_run = cumcounts_run.groupby(['area']).sum()
            cumcounts_msoa_run = cumcounts_msoa_run.values
    
            # loop aroud days
            msoacounts_run = np.zeros((len(msoas),nr_days))
            agecounts_run = np.zeros((age_cat.shape[0],nr_days))
            for day in range(0, nr_days):
                #print(day)
                # count nr for this condition per area
                #msoa_count_temp = individuals_tmp[individuals_tmp.iloc[:, -nr_days+day] == conditions_dict[key]].groupby(['area']).agg({individuals_tmp.columns[-nr_days+day]: ['count']})  
                
                msoa_count_temp = individuals_tmp[tmp.iloc[:,day] == conditions_dict[key]].groupby(['area']).agg({tmp.columns[day]: ['count']})
                
                if msoa_count_temp.shape[0] == len(msoas):
                    msoa_count_temp = msoa_count_temp.values
                    msoacounts_run[:,day] = msoa_count_temp[:, 0]
                elif msoa_count_temp.empty == False:
                    #print('check MSOAs')
                    # in case some entries don't exist
                    # start with empty dataframe
                    tmp_df =  pd.DataFrame(np.zeros(len(msoas)), columns = ['tmp'], index=msoas)   
                    # drop multiindex to prevent warning msg
                    msoa_count_temp.columns = msoa_count_temp.columns.droplevel(0)
                    # merge with obtained counts - NaN will appear
                    tmp_df = pd.merge(tmp_df, msoa_count_temp, how='left', left_index=True,right_index=True)
                    # replace NaN by 0
                    tmp_df = tmp_df.fillna(0)
                    msoacounts_run[:,day] = tmp_df.iloc[:,1].values
                    

                # count nr for this condition per age bracket             
                age_count_temp = individuals_tmp[tmp.iloc[:,day] == conditions_dict[key]].groupby(['age0']).agg({tmp.columns[day]: ['count']})  
                
                if age_count_temp.shape[0] == 6:
                    age_count_temp = age_count_temp.values
                    agecounts_run[:,day] = age_count_temp[:, 0]
                elif age_count_temp.empty == False:
                    # in case some entries don't exist
                    # start with empty dataframe
                    tmp_df =  pd.DataFrame(np.zeros(age_cat.shape[0]), columns = ['tmp'], index=list(range(1,age_cat.shape[0]+1)))   
                    # drop multilevel index to prevent warning msg
                    age_count_temp.columns = age_count_temp.columns.droplevel(0)
                    # merge with obtained counts - NaN will appear
                    tmp_df = pd.merge(tmp_df, age_count_temp, how='left', left_index=True,right_index=True)
                    # replace NaN by 0
                    tmp_df = tmp_df.fillna(0)
                    agecounts_run[:,day] = tmp_df.iloc[:,1].values
                
                    #age_count_temp.loc['2'].count

            # get current values from dict
            msoacounts = msoacounts_dict_3d[key]
            agecounts = agecounts_dict_3d[key]
            totalcounts = totalcounts_dict_3d[key]
            cumcounts = cumcounts_dict_3d[key]
            # add current run's values
            msoacounts[:,:,r-start_run] = msoacounts_run
            agecounts[:,:,r-start_run] = agecounts_run
            totalcounts[:,r-start_run] = msoacounts_run.sum(axis=0)
            cumcounts[:,r-start_run] = cumcounts_msoa_run[:, 0]
            # write out to dict
            msoacounts_dict_3d[key] = msoacounts
            agecounts_dict_3d[key] = agecounts
            totalcounts_dict_3d[key] = totalcounts
            cumcounts_dict_3d[key] = cumcounts
            
            uniquecounts_df[key] = uniquecounts_df[key]*(value+1)
            
        # uniquecounts['presymptomatic'] = uniquecounts['presymptomatic']*2
        # uniquecounts['symptomatic'] = uniquecounts['symptomatic']*3
        # uniquecounts['recovered'] = uniquecounts['recovered']*4
        # uniquecounts['dead'] = uniquecounts['dead']*5
        uniquecounts_df['maxval'] = uniquecounts_df.max(axis = 1)
        
        for key, value in conditions_dict.items():
            
            # get current values from dict
            uniquecounts = uniquecounts_dict_3d[key]
            # add current run's values
            uniquecounts[r-start_run] = uniquecounts_df[uniquecounts_df.maxval == (value+1)].shape[0]
            # write out to dict
            uniquecounts_dict_3d[key] = uniquecounts
    
    
    # next, create mean and std
    for key, value in conditions_dict.items():
        # get current values from dict
        msoacounts = msoacounts_dict_3d[key]
        agecounts = agecounts_dict_3d[key]
        totalcounts = totalcounts_dict_3d[key]
        cumcounts = cumcounts_dict_3d[key]
        uniquecounts = uniquecounts_dict_3d[key]
        # aggregate
        msoacounts_std = msoacounts.std(axis=2)
        msoacounts = msoacounts.mean(axis=2)
        agecounts_std = agecounts.std(axis=2)
        agecounts = agecounts.mean(axis=2)
        totalcounts_std = totalcounts.std(axis=1)
        totalcounts = totalcounts.mean(axis=1)
        cumcounts_std = cumcounts.std(axis=1)
        cumcounts = cumcounts.mean(axis = 1)
        uniquecounts_std = uniquecounts.std()
        uniquecounts = uniquecounts.mean()
        # write out to dict
        msoacounts_dict[key] = pd.DataFrame(data=msoacounts, index=msoas, columns=dict_days)
        msoacounts_dict_std[key] = pd.DataFrame(data=msoacounts_std, index=msoas, columns=dict_days)
        agecounts_dict[key] = pd.DataFrame(data=agecounts, index=age_cat_str, columns=dict_days)
        agecounts_dict_std[key] = pd.DataFrame(data=agecounts_std, index=age_cat_str, columns=dict_days)
        totalcounts_dict[key] = pd.Series(data=totalcounts, index=dict_days)
        totalcounts_dict_std[key] = pd.Series(data=totalcounts_std, index=dict_days)
        cumcounts_dict[key] = pd.Series(data=cumcounts, index=msoas)
        cumcounts_dict_std[key] = pd.Series(data=cumcounts_std, index=msoas)
        uniquecounts_dict[key] = pd.Series(data=uniquecounts, index=["total"])
        uniquecounts_dict_std[key] = pd.Series(data=uniquecounts_std, index=["total"])
    
    
    return msoas, totalcounts_dict, cumcounts_dict, agecounts_dict,  msoacounts_dict, cumcounts_dict_3d, totalcounts_dict_std, cumcounts_dict_std, agecounts_dict_std, msoacounts_dict_std, totalcounts_dict_3d, agecounts_dict_3d, msoacounts_dict_3d, uniquecounts_dict_3d, uniquecounts_dict_std, uniquecounts_dict




# ********
# PROGRAM ENTRY POINT
# Uses 'click' library so that it can be run from the command line
# ********
@click.command()
@click.option('-p', '--parameters_file', default="./model_parameters/default_dashboard.yml", type=click.Path(exists=True),
              help="Parameters file to use to configure the dashboard. Default: ./model_parameters/default_dashboard.yml")
def create_dashboard(parameters_file):
    # FUNCTIONS FOR PLOTTING
    # ----------------------
    
    # plot 1a: heatmap condition

    def plot_heatmap_condition(condition2plot):
        """ Create heatmap plot: x axis = time, y axis = MSOAs, colour = nr people with condition = condition2plot. condition2plot is key to conditions_dict."""
        
        # Prep data
        var2plot = msoacounts_dict[condition2plot]
        var2plot = var2plot.rename_axis(None, axis=1).rename_axis('MSOA', axis=0)
        var2plot.columns.name = 'Day'
        # reshape to 1D array or rates with a month and year for each row.
        df_var2plot = pd.DataFrame(var2plot.stack(), columns=['condition']).reset_index()
        source = ColumnDataSource(df_var2plot)
        # add better colour 
        mapper_1 = LinearColorMapper(palette=colours_ch_cond[condition2plot], low=0, high=var2plot.max().max())
        # create fig
        s1 = figure(title="Heatmap",
                   x_range=list(var2plot.columns), y_range=list(var2plot.index), x_axis_location="above")
        s1.rect(x="Day", y="MSOA", width=1, height=1, source=source,
               line_color=None, fill_color=transform('condition', mapper_1))
        color_bar_1 = ColorBar(color_mapper=mapper_1, location=(0, 0), orientation = 'horizontal', ticker=BasicTicker(desired_num_ticks=len(colours_ch_cond[condition2plot])))
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
        s1.toolbar.autohide = False
        plotref_dict[f"hm{condition2plot}"] = s1    

    # plot 2: disease conditions across time
    def plot_cond_time():
        # build ColumnDataSource
        data_s2 = dict(totalcounts_dict)
        data_s2["days"] = days
        for key, value in totalcounts_dict.items():
            data_s2[f"{key}_std_upper"] = totalcounts_dict[key] + totalcounts_dict_std[key]
            data_s2[f"{key}_std_lower"] = totalcounts_dict[key] - totalcounts_dict_std[key]
        source_2 = ColumnDataSource(data=data_s2)
        # Create fig
        s2 = figure(background_fill_color="#fafafa",title="Time", x_axis_label='Time', y_axis_label='Nr of people',toolbar_location='above')
        legend_it = []
        for key, value in totalcounts_dict.items():
            c1 = s2.line(x = 'days', y = key, source = source_2, line_width=2, line_color=colour_dict[key],muted_color="grey", muted_alpha=0.2)   
            c2 = s2.square(x = 'days', y = key, source = source_2, fill_color=colour_dict[key], line_color=colour_dict[key], size=5, muted_color="grey", muted_alpha=0.2)
            # c3 = s2.rect('days', f"{key}_std_upper", 0.2, 0.01, source = source_2, line_color="black",muted_color="grey", muted_alpha=0.2)
            # c4 = s2.rect('days', f"{key}_std_lower", 0.2, 0.01, source = source_2, line_color="black",muted_color="grey", muted_alpha=0.2)
            c5 = s2.segment('days', f"{key}_std_lower", 'days', f"{key}_std_upper", source = source_2, line_color="black",muted_color="grey", muted_alpha=0.2)
            legend_it.append((f"nr {key}", [c1,c2,c5]))
        legend = Legend(items=legend_it)
        legend.click_policy="hide"
        # Misc
        tooltips = tooltips_cond_basic.copy()
        tooltips.append(tuple(( 'Day',  '@days' )))
        s2.add_tools(HoverTool(
            tooltips=tooltips,
        ))
        s2.add_layout(legend, 'right')
        s2.toolbar.autohide = False
        plotref_dict["cond_time"] = s2    

    # plot 3: Conditions across MSOAs
    def plot_cond_msoas():
        # build ColumnDataSource
        data_s3 = {}
        data_s3["msoa_nr"] = msoas_nr
        data_s3["msoa_name"] = msoas
        for key, value in cumcounts_dict.items():
            data_s3[key] = cumcounts_dict[key]
            data_s3[f"{key}_std_upper"] = cumcounts_dict[key] + cumcounts_dict_std[key]
            data_s3[f"{key}_std_lower"] = cumcounts_dict[key] - cumcounts_dict_std[key]
        source_3 = ColumnDataSource(data=data_s3)
        # Create fig
        s3 = figure(background_fill_color="#fafafa",title="MSOA", x_axis_label='Nr people', y_axis_label='MSOA',toolbar_location='above')
        legend_it = []
        for key, value in msoacounts_dict.items():
            c1 = s3.circle(x = key, y = 'msoa_nr', source = source_3, fill_color=colour_dict[key], line_color=colour_dict[key], size=5,muted_color="grey", muted_alpha=0.2)   
            c2 = s3.segment(f"{key}_std_lower", 'msoa_nr', f"{key}_std_upper", 'msoa_nr', source = source_3, line_color="black",muted_color="grey", muted_alpha=0.2)
            legend_it.append((key, [c1,c2]))
        legend = Legend(items=legend_it)
        legend.click_policy="hide"
        # Misc
        s3.yaxis.ticker = data_s3["msoa_nr"]
        MSOA_dict = dict(zip(data_s3["msoa_nr"], data_s3["msoa_name"]))
        s3.yaxis.major_label_overrides = MSOA_dict
        tooltips = tooltips_cond_basic.copy()
        tooltips.append(tuple(( 'MSOA',  '@msoa_name' )))
        s3.add_tools(HoverTool(
            tooltips=tooltips,
        ))
        s3.add_layout(legend, 'right')
        s3.toolbar.autohide = False
        plotref_dict["cond_msoas"] = s3

    # plot 4a: choropleth
    def plot_choropleth_condition_slider(condition2plot):
        # Prepare data
        max_val = 0
        merged_data = pd.DataFrame()
        merged_data["y"] = msoacounts_dict[condition2plot].iloc[:,0]
        for d in range(0,nr_days):
            merged_data[f"{d}"] = msoacounts_dict[condition2plot].iloc[:,d]
            max_tmp = merged_data[f"{d}"].max()
            if max_tmp > max_val: max_val = max_tmp
        merged_data["Area"] = msoacounts_dict[condition2plot].index.to_list()
        merged_data = pd.merge(map_df,merged_data,on='Area')
        geosource = GeoJSONDataSource(geojson = merged_data.to_json())
        # Create color bar
        mapper_4 = LinearColorMapper(palette = colours_ch_cond[condition2plot], low = 0, high = max_val)
        color_bar_4 = ColorBar(color_mapper = mapper_4, 
                              label_standoff = 8,
                              #"width = 500, height = 20,
                              border_line_color = None,
                              location = (0,0), 
                              orientation = 'horizontal')
        # Create figure object.
        s4 = figure(title = f"{condition2plot} total")
        s4.xgrid.grid_line_color = None
        s4.ygrid.grid_line_color = None
        # Add patch renderer to figure.
        msoasrender = s4.patches('xs','ys', source = geosource,
                            fill_color = {'field' : 'y',
                                          'transform' : mapper_4},     
                            line_color = 'gray', 
                            line_width = 0.25, 
                            fill_alpha = 1)
        # Create hover tool
        s4.add_tools(HoverTool(renderers = [msoasrender],
                               tooltips = [('MSOA','@Area'),
                                            ('Nr people','@y'),
                                             ]))
        s4.add_layout(color_bar_4, 'below')
        s4.axis.visible = False
        s4.toolbar.autohide = True
        # Slider 
        # create dummy data source to store start value slider
        slider_val = {}
        slider_val["s"] = [start_day]
        source_slider = ColumnDataSource(data=slider_val)
        callback = CustomJS(args=dict(source=geosource,sliderval=source_slider), code="""
            var data = source.data;
            var startday = sliderval.data;
            var s = startday['s'];
            var f = cb_obj.value -s;
            console.log(f);
            var y = data['y'];
            var toreplace = data[f];
            for (var i = 0; i < y.length; i++) {
                y[i] = toreplace[i]
            }
            source.change.emit();
        """)
        slider = Slider(start=start_day, end=end_day, value=start_day, step=1, title="Day")
        slider.js_on_change('value', callback)
        plotref_dict[f"chpl{condition2plot}"] = s4
        plotref_dict[f"chsl{condition2plot}"] = slider
    
    def plot_cond_time_age():
        # 1 plot per condition, nr of lines = nr age brackets
        colour_dict_age = {
          0: "red",
          1: "orange",
          2: "yellow",
          3: "green",
          4: "teal",
          5: "blue",
          6: "purple",
          7: "pink",
          8: "gray",
          9: "black",
        }
        
        for key, value in totalcounts_dict.items():
            data_s2= dict()
            data_s2["days"] = days
            tooltips = []
            for a in range(len(age_cat_str)):
                age_cat_str[a]
                data_s2[f"c{a}"] = agecounts_dict[key].iloc[a]
                data_s2[f"{age_cat_str[a]}_std_upper"] = agecounts_dict[key].iloc[a] + agecounts_dict_std[key].iloc[a]
                data_s2[f"{age_cat_str[a]}_std_lower"] = agecounts_dict[key].iloc[a] - agecounts_dict_std[key].iloc[a]
            source_2 = ColumnDataSource(data=data_s2)
            # Create fig
            s2 = figure(background_fill_color="#fafafa",title=f"{key}", x_axis_label='Time', y_axis_label=f'Nr of people - {key}',toolbar_location='above')
            legend_it = []
            for a in range(len(age_cat_str)):
                c1 = s2.line(x = 'days', y = f"c{a}", source = source_2, line_width=2, line_color=colour_dict_age[a],muted_color="grey", muted_alpha=0.2)   
                c2 = s2.square(x = 'days', y = f"c{a}", source = source_2, fill_color=colour_dict_age[a], line_color=colour_dict_age[a], size=5, muted_color="grey", muted_alpha=0.2)
                c5 = s2.segment('days', f"{age_cat_str[a]}_std_lower", 'days', f"{age_cat_str[a]}_std_upper", source = source_2, line_color="black",muted_color="grey", muted_alpha=0.2)
                legend_it.append((f"nr {age_cat_str[a]}", [c1,c2,c5]))
                tooltips.append(tuple(( f"{age_cat_str[a]}",  f"@c{a}" )))
                
            legend = Legend(items=legend_it)
            legend.click_policy="hide"
            # Misc    
            tooltips.append(tuple(( 'Day',  '@days' )))
            s2.add_tools(HoverTool(
                tooltips=tooltips,
            ))
            s2.add_layout(legend, 'right')
            s2.toolbar.autohide = False
            plotref_dict[f"cond_time_age_{key}"] = s2


    # MAIN SCRIPT
    # -----------
    
    # Set parameters (optional to overwrite defaults)
    # -----------------------------------------------
    # Set to None to use defaults
    
    base_dir = os.getcwd()  # get current directory (usually RAMP-UA)
    
    # from file
    # parameters_file = os.path.join(base_dir, "model_parameters","default_dashboard.yml")
    
    # read from file
    with open(parameters_file, 'r') as f:
                parameters = load(f, Loader=SafeLoader)
                dash_params = parameters["dashboard"]  # Parameters for the dashboard
                output_name_user = dash_params["output_name"]
                data_dir_user = dash_params["data_dir"]
                start_day_user = dash_params["start_day"]
                end_day_user = dash_params["end_day"]
                start_run_user = dash_params["start_run"]
                end_run_user = dash_params["end_run"]                
                sc_dir = dash_params["scenario_dir"]
                sc_nam = dash_params["scenario_name"]

    
    # Set parameters (advanced)
    # -------------------------
    
    # dictionaries with condition and venue names
    # conditions are coded as numbers in microsim output
    conditions_dict = {
      "susceptible": 0,
      "exposed": 1,
      "presymptomatic": 2,
      "symptomatic": 3,
      "asymptomatic": 4,
      "recovered": 5,
      "dead": 6,
    }
    # venues are coded as strings - redefined here so script works as standalone, could refer to ActivityLocations instead
    locations_dict = {
      "PrimarySchool": "PrimarySchool",
      "SecondarySchool": "SecondarySchool",
      "Retail": "Retail",
      "Work": "Work",
      "Home": "Home",
    }
    
    # default list of tools for plots
    tools = "crosshair,hover,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"
    
    # colour schemes for plots
    # colours for line plots
    colour_dict = {
      "susceptible": "grey",
      "exposed": "blue",
      "presymptomatic": "orange",
      "symptomatic": "red",
      "asymptomatic": "magenta",
      "recovered": "green",
      "dead": "black",
      "Retail": "blue",
      "PrimarySchool": "orange",
      "SecondarySchool": "red",
      "Work": "black",
      "Home": "green",
    }
    # colours for heatmaps and choropleths for conditions (colours_ch_cond) and venues/danger scores (colours_ch_danger)
    colours_ch_cond = {
      "susceptible": brewer['Blues'][8][::-1],
      "exposed": brewer['YlOrRd'][8][::-1],
      "presymptomatic": brewer['YlOrRd'][8][::-1],
      "symptomatic": brewer['YlOrRd'][8][::-1],
      "asymptomatic": brewer['YlOrRd'][8][::-1],
      "recovered": brewer['Greens'][8][::-1],
      "dead": brewer['YlOrRd'][8][::-1],
    }
    colours_ch_danger = brewer['YlOrRd'][8][::-1]
    # other good palettes / way to define:
    # palette = brewer['BuGn'][8][::-1]    # -1 reverses the order
    # palette = = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    
    # Nr days, runs, scenarios
    # ------------------------    
    # check user input or revert to default
    
    if sc_nam is None: # no scenarios defined, use output directory
        sc_dir = ["output"]
        sc_nam = ["Scenario"]
    else: # add output to each directory
        for d in range(0,len(sc_dir)):
            sc_dir[d] = os.path.join("output", sc_dir[d])
        
    nr_scenarios = len(sc_dir)
    
    # directory to read data from
    data_dir = "data" if (data_dir_user is None) else data_dir_user
    data_dir = os.path.join(base_dir, data_dir) # update data dir

    # base file name
    file_name = "dashboard" if (output_name_user is None) else output_name_user
    
    # start and end day
    start_day = 0 if (start_day_user is None) else start_day_user
    end_day = calc_nr_days(os.path.join(data_dir, sc_dir[0],"0","Retail.pickle"))-1 if (end_day_user is None) else end_day_user
    nr_days = end_day - start_day + 1
    
    # start and end run
    start_run = 0 if (start_run_user is None) else start_run_user
    end_run = len(next(os.walk(os.path.join(data_dir, "output")))[1]) -1 if (end_run_user is None) else end_run_user
    nr_runs = end_run - start_run + 1
    r_range = range(start_run, end_run+1)
    
    
    dict_days = [] # empty list for column names 'Day0' etc
    for d in range(start_day, end_day+1):
        dict_days.append(f'Day{d}')

    # Read in third party data
    # ------------------------
    
    # load in shapefile with England MSOAs for choropleth
    sh_file = os.path.join(data_dir, "MSOAS_shp","bcc21fa2-48d2-42ca-b7b7-0d978761069f2020412-1-12serld.j1f7i.shp")
    map_df = gpd.read_file(sh_file)
    # rename column to get ready for merging
    map_df.rename(index=str, columns={'msoa11cd': 'Area'},inplace=True)
    
    # postcode to MSOA conversion (for retail data)
    data_file = os.path.join(data_dir, "PCD_OA_LSOA_MSOA_LAD_AUG19_UK_LU.csv")
    postcode_lu = pd.read_csv(data_file, encoding = "ISO-8859-1", usecols = ["pcds", "msoa11cd"])
    
    # age brackets
    age_cat = np.array([[0, 19], [20, 29], [30,44], [45,59], [60,74], [75,200]])    
    # label for plotting age categories
    age_cat_str = []
    for a in range(age_cat.shape[0]):
        age_cat_str.append(f"{age_cat[a,0]}-{age_cat[a,1]}")

    # Read in and process pickled output from microsim

    # load pickled data from GPU model
    gpu_data_dir = data_dir + "/output/gpu/"
    with open(gpu_data_dir + "total_counts.p", "rb") as f:
        total_counts = pickle.load(f)
    with open(gpu_data_dir + "age_counts.p", "rb") as f:
        # TODO: replace index with strings of age categories
        age_counts = pickle.load(f)
    with open(gpu_data_dir + "area_counts.p", "rb") as f:
        area_counts = pickle.load(f)

    # counts per condition
    results_tmp = create_counts_dict(conditions_dict,r_range,os.path.join(data_dir, sc_dir[0]),start_day,end_day,start_run,nr_runs,age_cat) # get all
    #only pick results variables needed
    msoas, totalcounts_dict, cumcounts_dict, agecounts_dict,  msoacounts_dict, cumcounts_dict_3d, totalcounts_dict_std, cumcounts_dict_std, agecounts_dict_std = results_tmp[0:9]
    uniquecounts_dict_std, uniquecounts_dict = results_tmp[14:16]


    # Plotting
    # --------
    
    # MSOA nrs (needs nrs not strings to plot)
    msoas_nr = [i for i in range(0,len(msoas))]
    
    # days (needs list to plot)
    days = [i for i in range(start_day,end_day+1)]
        
    # determine where/how the visualization will be rendered
    html_output = os.path.join(data_dir, f'{file_name}.html')
    output_file(html_output, title='RAMP-UA microsim output') # Render to static HTML
    #output_notebook()  # To tender inline in a Jupyter Notebook

    # optional: threshold map to only use MSOAs currently in the study or selection
    map_df = map_df[map_df['Area'].isin(msoas)]

    # basic tool tip for condition plots
    tooltips_cond_basic=[]
    for key, value in totalcounts_dict.items():
        tooltips_cond_basic.append(tuple(( f"Nr {key}",   f"@{key}")))

    # empty dictionary to track condition and venue specific plots
    plotref_dict = {}

    # create heatmaps condition
    for key,value in conditions_dict.items():
        plot_heatmap_condition(key)

    # disease conditions across time
    plot_cond_time()

    # disease conditions across msoas
    plot_cond_msoas()

    # choropleth conditions
    for key,value in conditions_dict.items():
        plot_choropleth_condition_slider(key)

    # conditions across time per age category
    plot_cond_time_age()

    # Layout and output

    tab1 = Panel(child=row(plotref_dict["cond_time"], plotref_dict["cond_msoas"]), title='Summary conditions')
    tab2 = Panel(child=row(plotref_dict["hmsusceptible"],column(plotref_dict["chslsusceptible"],plotref_dict["chplsusceptible"])), title='Susceptible')
    tab3 = Panel(child=row(plotref_dict["hmexposed"],column(plotref_dict["chslexposed"],plotref_dict["chplexposed"])), title='Exposed')
    tab4 = Panel(child=row(plotref_dict["hmpresymptomatic"],column(plotref_dict["chslpresymptomatic"],plotref_dict["chplpresymptomatic"])), title='Presymptomatic')
    tab5 = Panel(child=row(plotref_dict["hmsymptomatic"],column(plotref_dict["chslsymptomatic"],plotref_dict["chplsymptomatic"])), title='Symptomatic')
    tab6 = Panel(child=row(plotref_dict["hmasymptomatic"],column(plotref_dict["chslasymptomatic"],plotref_dict["chplasymptomatic"])), title='Asymptomatic')
    tab7 = Panel(child=row(plotref_dict["hmrecovered"],column(plotref_dict["chslrecovered"],plotref_dict["chplrecovered"])), title='Recovered')
    tab8 = Panel(child=row(plotref_dict["hmdead"],column(plotref_dict["chsldead"],plotref_dict["chpldead"])), title='Dead')
    tab9 = Panel(child=row(plotref_dict["cond_time_age_susceptible"],plotref_dict["cond_time_age_presymptomatic"],plotref_dict["cond_time_age_symptomatic"],plotref_dict["cond_time_age_recovered"],plotref_dict["cond_time_age_dead"]), title='Breakdown by age')

    # Put the Panels in a Tabs object
    tabs = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9])

    show(tabs)


if __name__ == "__main__":
    create_dashboard()
    print("End of program")






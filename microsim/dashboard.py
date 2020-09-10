# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:22:57 2020

@author: Natalie
"""

import os
import sys
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

from microsim.column_names import ColumnNames



# Functions for preprocessing
# ---------------------------




def calc_nr_days(data_file):
    # figure out nr days by reading in e.g. retail dangers pickle file of run 0      
    pickle_in = open(data_file,"rb")
    dangers = pickle.load(pickle_in)
    pickle_in.close()
            
    filter_col = [col for col in dangers if col.startswith(ColumnNames.LOCATION_DANGER)]
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
    cumcounts: nr per MSOA and day
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
            
            
            # User defined age brackets
            individuals.insert(7, 'Age0', np.zeros((len(individuals),1)))
            for a in range(age_cat.shape[0]):
                individuals['Age0'] = np.where((individuals['age'] >= age_cat[a,0]) & (individuals['age'] <= age_cat[a,1]), a+1, individuals['Age0'])   
            age_cat_col = individuals['Age0'].values
            # temporary workaround if no continuous age
            #age_cat_col = individuals['Age1'].values 
            
        # add age brackets column to individuals_tmp
        individuals_tmp.insert(7, 'Age0', age_cat_col)
        
        
        uniquecounts_df = pd.DataFrame()
        # select right columns
        subset = individuals_tmp[counts_colnames]
            
        for key, value in conditions_dict.items():
            #print(key)
            if r == start_run:
                msoacounts_dict_3d[key] = np.zeros((len(msoas),nr_days,nr_runs))    
                cumcounts_dict_3d[key] = np.zeros((len(msoas),nr_days,nr_runs))
                agecounts_dict_3d[key] = np.zeros((age_cat.shape[0],nr_days,nr_runs))
                totalcounts_dict_3d[key] = np.zeros((nr_days,nr_runs))  
                uniquecounts_dict_3d[key] = np.zeros(nr_runs)
                
            # find all rows with condition (dict value)
            indices = subset[subset.eq(value).any(1)].index
            # create new df of zeros and replace with 1 at indices
            cumcounts_end = pd.DataFrame(np.zeros((subset.shape[0], 1)))
            cumcounts_end.loc[indices] = 1
            uniquecounts_df[key] = cumcounts_end.values[:,0]
    
            # loop aroud days
            msoacounts_run = np.zeros((len(msoas),nr_days))
            cumcounts_run = np.zeros((len(msoas),nr_days))
            agecounts_run = np.zeros((age_cat.shape[0],nr_days))
            for day in range(0, nr_days):
                #print(day)
                # count nr for this condition per area               
                msoa_count_temp = individuals_tmp[subset.iloc[:,day] == conditions_dict[key]].groupby(['area']).agg({subset.columns[day]: ['count']})  
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
                
                
                # cumulative counts
                # select right columns
                tmp_cum = subset.iloc[:,0:day+1]
                indices = tmp_cum[tmp_cum.eq(value).any(1)].index
                # create new df of zeros and replace with 1 at indices
                tmp_df = pd.DataFrame(np.zeros((tmp_cum.shape[0], 1)))
                tmp_df.loc[indices] = 1
                # merge with MSOA df
                tmp_df = tmp_df.merge(area_individuals, left_index=True, right_index=True)
                cumcounts_tmp = tmp_df.groupby(['area']).sum()
                if cumcounts_tmp.shape[0] == len(msoas):
                    cumcounts_tmp = cumcounts_tmp.values
                    cumcounts_run[:,day] = cumcounts_tmp[:, 0]
                elif cumcounts_tmp.empty == False:
                    #print('check MSOAs')
                    # in case some entries don't exist
                    # start with empty dataframe
                    tmp_df =  pd.DataFrame(np.zeros(len(msoas)), columns = ['tmp'], index=msoas)   
                    # drop multiindex to prevent warning msg
                    cumcounts_tmp.columns = cumcounts_tmp.columns.droplevel(0)
                    # merge with obtained counts - NaN will appear
                    tmp_df = pd.merge(tmp_df, cumcounts_tmp, how='left', left_index=True,right_index=True)
                    # replace NaN by 0
                    tmp_df = tmp_df.fillna(0)
                    cumcounts_run[:,day] = tmp_df.iloc[:,1].values
  

                # count nr for this condition per age bracket             
                age_count_temp = individuals_tmp[subset.iloc[:,day] == conditions_dict[key]].groupby(['Age0']).agg({subset.columns[day]: ['count']})  
                
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
            cumcounts = cumcounts_dict_3d[key]
            agecounts = agecounts_dict_3d[key]
            totalcounts = totalcounts_dict_3d[key]
            # add current run's values
            msoacounts[:,:,r-start_run] = msoacounts_run
            cumcounts[:,:,r-start_run] = cumcounts_run
            agecounts[:,:,r-start_run] = agecounts_run
            totalcounts[:,r-start_run] = msoacounts_run.sum(axis=0)

            # write out to dict
            msoacounts_dict_3d[key] = msoacounts
            cumcounts_dict_3d[key] = cumcounts
            agecounts_dict_3d[key] = agecounts
            totalcounts_dict_3d[key] = totalcounts
            
            uniquecounts_df[key] = uniquecounts_df[key]*(value+1)
            
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
        cumcounts = cumcounts_dict_3d[key]
        agecounts = agecounts_dict_3d[key]
        totalcounts = totalcounts_dict_3d[key]
        uniquecounts = uniquecounts_dict_3d[key]
        # aggregate
        msoacounts_std = msoacounts.std(axis=2)
        msoacounts = msoacounts.mean(axis=2)
        cumcounts_std = cumcounts.std(axis=2)
        cumcounts = cumcounts.mean(axis=2)
        agecounts_std = agecounts.std(axis=2)
        agecounts = agecounts.mean(axis=2)
        totalcounts_std = totalcounts.std(axis=1)
        totalcounts = totalcounts.mean(axis=1)
        uniquecounts_std = uniquecounts.std()
        uniquecounts = uniquecounts.mean()
        # write out to dict
        msoacounts_dict[key] = pd.DataFrame(data=msoacounts, index=msoas, columns=dict_days)
        msoacounts_dict_std[key] = pd.DataFrame(data=msoacounts_std, index=msoas, columns=dict_days)
        cumcounts_dict[key] = pd.DataFrame(data=cumcounts, index=msoas, columns=dict_days)
        cumcounts_dict_std[key] = pd.DataFrame(data=cumcounts_std, index=msoas, columns=dict_days)
        agecounts_dict[key] = pd.DataFrame(data=agecounts, index=age_cat_str, columns=dict_days)
        agecounts_dict_std[key] = pd.DataFrame(data=agecounts_std, index=age_cat_str, columns=dict_days)
        totalcounts_dict[key] = pd.Series(data=totalcounts, index=dict_days)
        totalcounts_dict_std[key] = pd.Series(data=totalcounts_std, index=dict_days)
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
        
        
    # plot 1b: heatmap venue
    
    def plot_heatmap_danger(venue2plot):
        """ Create heatmap plot: x axis = time, y axis = MSOAs, colour =danger score. """
        
        # Prep data
        var2plot = dangers_msoa_dict[venue2plot]
        var2plot.columns.name = 'Day'
        # reshape to 1D array or rates with a month and year for each row.
        df_var2plot = pd.DataFrame(var2plot.stack(), columns=['venue']).reset_index()
        source = ColumnDataSource(df_var2plot)
        # add better colour 
        mapper_1 = LinearColorMapper(palette=colours_ch_danger, low=0, high=var2plot.max().max())
        # Create fig
        s1 = figure(title="Heatmap",
                   x_range=list(var2plot.columns), y_range=list(var2plot.index), x_axis_location="above")
        s1.rect(x="Day", y="MSOA", width=1, height=1, source=source,
               line_color=None, fill_color=transform('venue', mapper_1))
        color_bar_1 = ColorBar(color_mapper=mapper_1, location=(0, 0), orientation = 'horizontal', ticker=BasicTicker(desired_num_ticks=len(colours_ch_danger)))
        s1.add_layout(color_bar_1, 'below')
        s1.axis.axis_line_color = None
        s1.axis.major_tick_line_color = None
        s1.axis.major_label_text_font_size = "7px"
        s1.axis.major_label_standoff = 0
        s1.xaxis.major_label_orientation = 1.0
        # Create hover tool
        s1.add_tools(HoverTool(
            tooltips=[
                ( 'danger score',   '@venue'),
                ( 'Day',  '@Day' ), 
                ( 'MSOA', '@MSOA'),
            ],
        ))
        s1.toolbar.autohide = False
        plotref_dict[f"hm{venue2plot}"] = s1    
        
        
    # plot 2: disease conditions across time
    
    def plot_cond_time(flag):
        # build ColumnDataSource
        if flag == "daily":
            title_fig = "Daily counts"
            name_plotref = "cond_time_daily"
            data_s2 = dict(totalcounts_dict)
            data_s2["days"] = days
            for key, value in totalcounts_dict.items():
                data_s2[f"{key}_std_upper"] = totalcounts_dict[key] + totalcounts_dict_std[key]
                data_s2[f"{key}_std_lower"] = totalcounts_dict[key] - totalcounts_dict_std[key]
        elif flag == "cumulative": 
            title_fig = "Cumulative counts"
            name_plotref = "cond_time_cumulative"
            data_s2 = {"days": days}
            for key, value in totalcounts_dict.items():
                data_s2[f"{key}"] = cumcounts_dict[key].sum(axis=0)
                data_s2[f"{key}_std_upper"] = cumcounts_dict[key].sum(axis=0) + cumcounts_dict_std[key].sum(axis=0)
                data_s2[f"{key}_std_lower"] = cumcounts_dict[key].sum(axis=0) - cumcounts_dict_std[key].sum(axis=0)
            
        source_2 = ColumnDataSource(data=data_s2)
        # Create fig
        s2 = figure(background_fill_color="#fafafa",title=title_fig, x_axis_label='Time', y_axis_label='Nr of people',toolbar_location='above')
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
        plotref_dict[name_plotref] = s2    
        
        
    # plot 3: Conditions across MSOAs
    
    def plot_cond_msoas():
        # build ColumnDataSource
        data_s3 = {}
        data_s3["msoa_nr"] = msoas_nr
        data_s3["msoa_name"] = msoas
        for key, value in cumcounts_dict.items():
            data_s3[key] = cumcounts_dict[key].iloc[:,nr_days-1]
            data_s3[f"{key}_std_upper"] = cumcounts_dict[key].iloc[:,nr_days-1] + cumcounts_dict_std[key].iloc[:,nr_days-1]
            data_s3[f"{key}_std_lower"] = cumcounts_dict[key].iloc[:,nr_days-1] - cumcounts_dict_std[key].iloc[:,nr_days-1]
            # old
            # data_s3[key] = cumcounts_dict[key]
            # data_s3[f"{key}_std_upper"] = cumcounts_dict[key] + cumcounts_dict_std[key]
            # data_s3[f"{key}_std_lower"] = cumcounts_dict[key] - cumcounts_dict_std[key]
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
        
        
    # plot 4b: choropleth dangers
    
    def plot_choropleth_danger_slider(venue2plot):
        # Prep data
        max_val = 0
        merged_data = pd.DataFrame()
        merged_data["y"] = dangers_msoa_dict[venue2plot].iloc[:,0]
        for d in range(0,nr_days):
            merged_data[f"{d}"] = dangers_msoa_dict[venue2plot].iloc[:,d]
            max_tmp = merged_data[f"{d}"].max()
            if max_tmp > max_val: max_val = max_tmp    
        merged_data["Area"] = dangers_msoa_dict[venue2plot].index.to_list()
        merged_data = pd.merge(map_df,merged_data,on='Area')
        geosource2 = GeoJSONDataSource(geojson = merged_data.to_json())
        # Create color bar 
        mapper_4 = LinearColorMapper(palette = colours_ch_danger, low = 0, high = max_val)
        color_bar_4 = ColorBar(color_mapper = mapper_4, 
                              label_standoff = 8,
                              border_line_color = None,
                              location = (0,0), 
                              orientation = 'horizontal')
        # Create figure object
        s4 = figure(title = f"{venue2plot} total")
        s4.xgrid.grid_line_color = None
        s4.ygrid.grid_line_color = None
        # Add patch renderer to figure.
        msoasrender = s4.patches('xs','ys', source = geosource2,
                            fill_color = {'field' : 'y',
                                          'transform' : mapper_4},     
                            line_color = 'gray', 
                            line_width = 0.25, 
                            fill_alpha = 1)
        # Create hover tool
        s4.add_tools(HoverTool(renderers = [msoasrender],
                               tooltips = [('MSOA','@Area'),
                                            ('Danger score','@y'),
                                             ]))
        s4.add_layout(color_bar_4, 'below')
        s4.axis.visible = False
        s4.toolbar.autohide = True
        # Slider
        # create dummy data source to store start value slider
        slider_val = {}
        slider_val["s"] = [start_day]
        source_slider = ColumnDataSource(data=slider_val)
        callback = CustomJS(args=dict(source=geosource2,sliderval=source_slider), code="""
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
        plotref_dict[f"chpl{venue2plot}"] = s4
        plotref_dict[f"chsl{venue2plot}"] = slider    
    
    
    # plot 5: danger scores across time per venue type
    
    def plot_danger_time():
        # build ColumnDataSource
        data_s5 = {}
        data_s5["days"] = days
        for key, value in dangers_dict.items():
            data_s5[key] = value.mean(axis = 0)
            data_s5[f"{key}_std_upper"] = value.mean(axis = 0) + value.std(axis = 0)
            data_s5[f"{key}_std_lower"] = value.mean(axis = 0) - value.std(axis = 0)
        source_5 = ColumnDataSource(data=data_s5)
        # Build figure
        s5 = figure(background_fill_color="#fafafa",title="Time", x_axis_label='Time', y_axis_label='Average danger score', toolbar_location='above')
        legend_it = []
        for key, value in dangers_dict.items():
            c1 = s5.line(x = 'days', y = key, source = source_5, line_width=2, line_color=colour_dict[key], muted_color="grey", muted_alpha=0.2)
            c2 = s5.circle(x = 'days', y = key, source = source_5, fill_color=colour_dict[key], line_color=colour_dict[key], size=5)
            c3 = s5.segment('days', f"{key}_std_lower", 'days', f"{key}_std_upper", source = source_5, line_color="black",muted_color="grey", muted_alpha=0.2)
            legend_it.append((key, [c1,c2,c3]))
        legend = Legend(items=legend_it)
        legend.click_policy="hide"
        # Misc
        tooltips = tooltips_venue_basic.copy()
        tooltips.append(tuple(( 'Day',  '@days' )))
        s5.add_tools(HoverTool(
            tooltips=tooltips,
        ))
        s5.add_layout(legend, 'right')
        s5.toolbar.autohide = False
        plotref_dict["danger_time"] = s5
    
    
    
    
    
    
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
    
    
    def plot_scenario_hist(scen_hist,category,label):
        max_plot = 0
        for s in sc_nam:
            if max_plot < max(scen_hist[s]):
                max_plot = max(scen_hist[s])
        data = scen_hist
        Categories = scen_hist[category]
        Scenarios = sc_nam    
        x = []
        counts = []
        v = 0
        for c in Categories:
            for s in Scenarios:
                x.append((c,s))
                counts.append(scen_hist[s][v])
            v = v + 1
        counts = tuple(counts)
        source = ColumnDataSource(data=dict(x=x, counts=counts))
        p = figure(x_range=FactorRange(*x), plot_height=350, title=f"{label} across scenarios",y_axis_label=f"{label}")
        p.vbar(x='x', top='counts', width=0.9, source=source, line_color="white",
                   fill_color=factor_cmap('x', palette=scen_palette, factors=Scenarios, start=1, end=2))
        p.y_range.start = 0
        p.x_range.range_padding = 0.1
        p.xaxis.major_label_orientation = 1
        p.xgrid.grid_line_color = None
        
        #plotref_dict[f"cond_time_age_{key}"] = s2   
        tooltips = [(f'{label}','@counts'),]
        p.add_tools(HoverTool(
                tooltips=tooltips,
            ))
        #show(p)
        plotref_dict[f"scen_hist_{category}"] = p
        
        
        
    def plot_scenario_time(scen_time,category,label,dkey):
        # build ColumnDataSource
        data_s5 = scen_time[dkey].copy()
        data_s5["days"] = days
        # add standard dev?
        source_5 = ColumnDataSource(data=data_s5)
        # Build figure
        s5 = figure(background_fill_color="#fafafa",title=f"{dkey}", x_axis_label='Time', y_axis_label=f"{label}", toolbar_location='above')
        legend_it = []
        snr = 0
        tooltips=[]
        for s in sc_nam:  
            c1 = s5.line(x = 'days', y = s, source = source_5, line_width=2, line_color=scen_palette[snr], muted_color="grey", muted_alpha=0.2)
            c2 = s5.circle(x = 'days', y = s, source = source_5, fill_color=scen_palette[snr], line_color=scen_palette[snr], size=5)
            legend_it.append((s, [c1,c2]))
            tooltips.append(tuple(( f"{s}",   f"@{s}")))
            snr = snr + 1
        legend = Legend(items=legend_it)
        legend.click_policy="hide"
        # Misc
        s5.add_tools(HoverTool(
                tooltips=tooltips,
            ))
        s5.add_layout(legend, 'right')
        s5.toolbar.autohide = False
        plotref_dict[f"scen_time_{category}_{dkey}"] = s5 
        


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
      "susceptible": ColumnNames.DiseaseStatuses.SUSCEPTIBLE,
      "exposed": ColumnNames.DiseaseStatuses.EXPOSED,
      "presymptomatic": ColumnNames.DiseaseStatuses.PRESYMPTOMATIC,
      "symptomatic": ColumnNames.DiseaseStatuses.SYMPTOMATIC,
      "asymptomatic": ColumnNames.DiseaseStatuses.ASYMPTOMATIC,
      "recovered": ColumnNames.DiseaseStatuses.RECOVERED,
      "dead": ColumnNames.DiseaseStatuses.DEAD,
    }
    # venues are coded as strings
    # # for backwards compatability
    # locations_dict = {
    #   "PrimarySchool": "PrimarySchool",
    #   "SecondarySchool": "SecondarySchool",
    #   "Retail": "Retail",
    #   "Work": "Work",
    #   "Home": "Home",
    # }
    # new names
    locations_dict = {
      "PrimarySchool": ColumnNames.Activities.PRIMARY,
      "SecondarySchool": ColumnNames.Activities.SECONDARY,
      "Retail": ColumnNames.Activities.RETAIL,
      "Work": ColumnNames.Activities.WORK,
      "Home": ColumnNames.Activities.HOME,
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
    if data_dir == "devon_data":
        flag_QUANT = 0
    else:
        flag_QUANT = 1
    data_dir = os.path.join(base_dir, data_dir) # update data dir
    
    # check if this directory exists
    if os.path.isdir(os.path.join(data_dir, sc_dir[0])) == False:
        sys.exit('Results directory does not exist, please check input file')
    if os.path.isfile(os.path.join(data_dir,sc_dir[0],"0","Individuals.pickle")) == False:
        sys.exit('No data in results directory, please check input file')
        
    

    # base file name
    file_name = "dashboard" if (output_name_user is None) else output_name_user
    
    # start and end day and run
    
    # first, check maximum end day and run available across all scenarios
    end_day_max = 1000000
    end_run_max = 1000000
    for sdir in sc_dir:
        end_day_max_tmp = calc_nr_days(os.path.join(data_dir, sdir,"0","Retail.pickle"))-1
        if end_day_max_tmp < end_day_max:
            end_day_max = end_day_max_tmp
        end_run_max_tmp = len(next(os.walk(os.path.join(data_dir, sdir)))[1]) -1
        if end_run_max_tmp < end_run_max:
            end_run_max = end_run_max_tmp

    # days
    start_day = 0 if (start_day_user is None) else start_day_user
    end_day = end_day_max if (end_day_user is None) else end_day_user
    if end_day_user is not None and end_day_user > end_day_max:
        print("Warning: user specified end day is greater than number of days available, so setting end_day to last day!")
        end_day = end_day_max
    nr_days = end_day - start_day + 1
    
    #runs
    start_run = 0 if (start_run_user is None) else start_run_user
    end_run = end_run_max if (end_run_user is None) else end_run_user
    if end_run_max is not None and end_run_user > end_run_max:
        print("Warning: user specified end run is greater than number of runs available, so setting end_run to last run!")
        end_run = end_run_max
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
    
    if flag_QUANT == 0: # devon data
    
        # read in details about venues
        data_file = os.path.join(data_dir, "devon-schools","exeter schools.csv")
        schools = pd.read_csv(data_file)
        data_file = os.path.join(data_dir, "devon-retail","devon smkt.csv")
        retail = pd.read_csv(data_file)
        
        # postcode to MSOA conversion (for retail data)
        data_file = os.path.join(data_dir, "PCD_OA_LSOA_MSOA_LAD_AUG19_UK_LU.csv")
        postcode_lu = pd.read_csv(data_file, encoding = "ISO-8859-1", usecols = ["pcds", "msoa11cd"])
    
    elif flag_QUANT == 1: # QUANT API data
        
        
    
    
    # Read in and process pickled output from microsim
    # ------------------------------------------------
    
    # read in and process pickle files location/venue dangers
    # create one or more dangers_dict (only using means i.e. first output of function for now)

    # if there is only 1 scenario, plot as usual
    if nr_scenarios == 1:
        dangers_dict = create_venue_dangers_dict(locations_dict,r_range,os.path.join(data_dir,sc_dir[0]),start_day,end_day,start_run,nr_runs)[0]
    
    # if there are 2 scenarios, replace dangers_dict by the difference between scenario 0 and 1
    elif nr_scenarios == 2:
        dangers_dict_sc0 = create_venue_dangers_dict(locations_dict,r_range,os.path.join(data_dir,sc_dir[0]),start_day,end_day,start_run,nr_runs)[0]
        dangers_dict_sc1 = create_venue_dangers_dict(locations_dict,r_range,os.path.join(data_dir,sc_dir[1]),start_day,end_day,start_run,nr_runs)[0]
        # calculate difference between 2 scenarios
        dangers_dict = {} # this is used for plotting
        dangers_dict = create_difference_dict(dangers_dict_sc0,dangers_dict_sc1,locations_dict)
        
    if nr_scenarios >= 2:
        scen_hist_venues = {}
        scen_hist_venues["Venues"] = []
        scen_time_venues = {}
        
        for s in range(0,nr_scenarios):
            dangers_dict_tmp = create_venue_dangers_dict(locations_dict,r_range,os.path.join(data_dir,sc_dir[s]),start_day,end_day,start_run,nr_runs)[0]
            scen_hist_venues[sc_nam[s]] = []
        
            for key, value in dangers_dict.items():
                if s == 0:
                    scen_hist_venues["Venues"].append(key)
                    scen_time_venues[key] =  dict({"Day":dict_days, sc_nam[0]:dangers_dict_tmp[key].mean(axis=0)})
                else:
                    scen_time_venues[key][sc_nam[s]] = dangers_dict_tmp[key].mean(axis=0)
                scen_hist_venues[sc_nam[s]].append(dangers_dict_tmp[key].mean(axis=0).mean())
                

        # retrieve data
        #tmp = scen_time_data["Retail"]

    
    if nr_scenarios <= 2:
        # Add additional info about schools and retail including spatial coordinates
        # merge
        primaryschools = pd.merge(schools, dangers_dict["PrimarySchool"], left_index=True, right_index=True)
        secondaryschools = pd.merge(schools, dangers_dict["SecondarySchool"], left_index=True, right_index=True)
        retail = pd.merge(retail, dangers_dict["Retail"], left_index=True, right_index=True)
        
        # creat LUT
        lookup = dict(zip(postcode_lu.pcds, postcode_lu.msoa11cd)) # zip together the lists and make a dict from it
        # use LUT and add column to retail variable
        msoa_code = [lookup.get(retail.postcode[i]) for i in range(0, len(retail.postcode), 1)]
        retail.insert(2, 'MSOA_code', msoa_code)
        
        # normalised danger scores per msoa for schools and retail (for choropleth)
        dangers_msoa_dict = create_msoa_dangers_dict(dangers_dict,['Retail','PrimarySchool','SecondarySchool'],[retail.MSOA_code,schools.MSOA_code,schools.MSOA_code])
    
    
    # age brackets
    age_cat = np.array([[0, 19], [20, 29], [30,44], [45,59], [60,74], [75,200]])    # original categories (Age1 column)
    #age_cat = np.array([[0, 19], [20, 29], [30,44]])    # original categories (Age1 column)
    
    # label for plotting age categories
    age_cat_str = []
    for a in range(age_cat.shape[0]):
        age_cat_str.append(f"{age_cat[a,0]}-{age_cat[a,1]}")
        

    
    
    
    
    
    # counts per condition
    if nr_scenarios == 1:
        results_tmp = create_counts_dict(conditions_dict,r_range,os.path.join(data_dir, sc_dir[0]),start_day,end_day,start_run,nr_runs,age_cat) # get all
        #only pick results variables needed
        msoas, totalcounts_dict, cumcounts_dict, agecounts_dict,  msoacounts_dict, cumcounts_dict_3d, totalcounts_dict_std, cumcounts_dict_std, agecounts_dict_std = results_tmp[0:9] 
        uniquecounts_dict_std, uniquecounts_dict = results_tmp[14:16] 
        
    elif nr_scenarios == 2:
    # if there is one other scenario, replace the counts dictionaries by the difference between scenario 0 and 1
        results_tmp = create_counts_dict(conditions_dict,r_range,os.path.join(data_dir,sc_dir[0]),start_day,end_day,start_run,nr_runs,age_cat) # get all
        msoas, totalcounts_dict_sc0, cumcounts_dict_sc0, agecounts_dict_sc0,  msoacounts_dict_sc0, cumcounts_dict_3d_sc0 = results_tmp[0:6] 
        results_tmp = create_counts_dict(conditions_dict,r_range,os.path.join(data_dir,sc_dir[1]),start_day,end_day,start_run,nr_runs,age_cat) # get all
        totalcounts_dict_sc1, cumcounts_dict_sc1, agecounts_dict_sc1,  msoacounts_dict_sc1, cumcounts_dict_3d_sc1 = results_tmp[1:6] 
        # calculate differenes
        msoacounts_dict = create_difference_dict(msoacounts_dict_sc0,msoacounts_dict_sc1,conditions_dict)  
        agecounts_dict = create_difference_dict(agecounts_dict_sc0,agecounts_dict_sc1,conditions_dict)   
        totalcounts_dict = create_difference_dict(totalcounts_dict_sc0,totalcounts_dict_sc1,conditions_dict)  
        cumcounts_dict = create_difference_dict(cumcounts_dict_sc0,cumcounts_dict_sc1,conditions_dict)
        # make sure std are zero, subtract from itself
        agecounts_dict_std = create_difference_dict(agecounts_dict_sc0,agecounts_dict_sc0,conditions_dict) 
        totalcounts_dict_std = create_difference_dict(totalcounts_dict_sc0,totalcounts_dict_sc0,conditions_dict) 
        cumcounts_dict_std = create_difference_dict(cumcounts_dict_sc0,cumcounts_dict_sc0,conditions_dict) 
        
        

    if nr_scenarios >= 2:
        scen_hist_conditions = {}
        scen_hist_conditions["Condition"] = []
        scen_time_conditions = {}
        
        for s in range(0,nr_scenarios):
            result_tmp =  create_counts_dict(conditions_dict,r_range,os.path.join(data_dir, sc_dir[s]),start_day,end_day,start_run,nr_runs,age_cat)
            totalcounts_dict_tmp = result_tmp[1]
            uniquecounts_dict_tmp = result_tmp[-1]
 
            scen_hist_conditions[sc_nam[s]] = []
        
            for key, value in conditions_dict.items():
                if s == 0:
                    scen_hist_conditions["Condition"].append(key)
                    scen_time_conditions[key] =  dict({"Day":dict_days, sc_nam[0]:totalcounts_dict_tmp[key]})
                else:
                    scen_time_conditions[key][sc_nam[s]] = totalcounts_dict_tmp[key]
                scen_hist_conditions[sc_nam[s]].append(uniquecounts_dict_tmp[key].values[0])
        
    
    # Plotting
    # --------
    
    # MSOA nrs (needs nrs not strings to plot)
    msoas_nr = [i for i in range(0,len(msoas))]
    
    # days (needs list to plot)
    days = [i for i in range(start_day,end_day+1)]
    
    
    
    if nr_scenarios <= 2:
        
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
        
        tooltips_venue_basic=[]
        for key, value in dangers_dict.items():
            tooltips_venue_basic.append(tuple(( f"Danger {key}",   f"@{key}")))
           
        # empty dictionary to track condition and venue specific plots
        plotref_dict = {}
        
        # create heatmaps condition
        for key,value in conditions_dict.items():
            plot_heatmap_condition(key)
        
        # create heatmaps venue dangers
        for key,value in dangers_msoa_dict.items():
            plot_heatmap_danger(key)
            
        # disease conditions across time
        plot_cond_time("daily")
        plot_cond_time("cumulative")
        
        # disease conditions across msoas
        plot_cond_msoas()    
        
        # choropleth conditions
        for key,value in conditions_dict.items():
            plot_choropleth_condition_slider(key)
        
        # choropleth dangers
        for key,value in dangers_msoa_dict.items():
            plot_choropleth_danger_slider(key)
        
        # danger scores across time per venue type
        plot_danger_time()   
        
        
        # conditions across time per age category
        plot_cond_time_age()
        
        
        # Layout and output
        
        tab1 = Panel(child=row(plotref_dict["cond_time_daily"],plotref_dict["cond_time_cumulative"], plotref_dict["cond_msoas"]), title='Summary conditions')
        
        tab2 = Panel(child=row(plotref_dict["hmsusceptible"],column(plotref_dict["chslsusceptible"],plotref_dict["chplsusceptible"])), title='Susceptible')
        
        tab3 = Panel(child=row(plotref_dict["hmexposed"],column(plotref_dict["chslexposed"],plotref_dict["chplexposed"])), title='Exposed')
        
        tab4 = Panel(child=row(plotref_dict["hmpresymptomatic"],column(plotref_dict["chslpresymptomatic"],plotref_dict["chplpresymptomatic"])), title='Presymptomatic')
        
        tab5 = Panel(child=row(plotref_dict["hmsymptomatic"],column(plotref_dict["chslsymptomatic"],plotref_dict["chplsymptomatic"])), title='Symptomatic')
            
        tab6 = Panel(child=row(plotref_dict["hmasymptomatic"],column(plotref_dict["chslasymptomatic"],plotref_dict["chplasymptomatic"])), title='Asymptomatic')
        
        tab7 = Panel(child=row(plotref_dict["hmrecovered"],column(plotref_dict["chslrecovered"],plotref_dict["chplrecovered"])), title='Recovered')
        
        tab8 = Panel(child=row(plotref_dict["hmdead"],column(plotref_dict["chsldead"],plotref_dict["chpldead"])), title='Dead')
        
        tab9 = Panel(child=row(plotref_dict["danger_time"]), title='Summary dangers')
        
        tab10 = Panel(child=row(plotref_dict["hmRetail"],column(plotref_dict["chslRetail"],plotref_dict["chplRetail"])), title='Danger retail')
        
        tab11 = Panel(child=row(plotref_dict["hmPrimarySchool"],column(plotref_dict["chslPrimarySchool"],plotref_dict["chplPrimarySchool"])), title='Danger primary school')
        
        tab12 = Panel(child=row(plotref_dict["hmSecondarySchool"],column(plotref_dict["chslSecondarySchool"],plotref_dict["chplSecondarySchool"])), title='Danger secondary school')
        
        tab13 = Panel(child=row(plotref_dict["cond_time_age_susceptible"],plotref_dict["cond_time_age_presymptomatic"],plotref_dict["cond_time_age_symptomatic"],plotref_dict["cond_time_age_recovered"],plotref_dict["cond_time_age_dead"]), title='Breakdown by age')
        
        
        # Put the Panels in a Tabs object
        tabs = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13])
        
        show(tabs)
    
    if nr_scenarios >= 2:
        # determine where/how the visualization will be rendered
        html_output = os.path.join(data_dir, f'{file_name}_scenarios.html')
        output_file(html_output, title='RAMP-UA microsim output scenario comparison') # Render to static HTML
        #output_notebook()  # To tender inline in a Jupyter Notebook
        
        scen_palette = ["darkgrey","royalblue","olive","orange","darkviolet","firebrick","tomato","deeppink","lightseagreen","limegreen"]
        
        plot_scenario_hist(scen_hist_venues,"Venues","Danger scores")
        
        plot_scenario_hist(scen_hist_conditions,"Condition","Nr people")
        
        
        for key,value in conditions_dict.items():
            plot_scenario_time(scen_time_conditions,"Conditions","Nr people",key)
        
        for key,value in dangers_dict.items():
            plot_scenario_time(scen_time_venues,"Venues","Danger scores",key)
            
        # Layout and output
        
        tab1 = Panel(child=row(plotref_dict["scen_hist_Condition"], plotref_dict["scen_hist_Venues"]), title='Histograms')
        
        tab2 = Panel(child=row(plotref_dict["scen_time_Conditions_susceptible"], plotref_dict["scen_time_Conditions_exposed"], plotref_dict["scen_time_Conditions_presymptomatic"], plotref_dict["scen_time_Conditions_symptomatic"], plotref_dict["scen_time_Conditions_asymptomatic"], plotref_dict["scen_time_Conditions_recovered"], plotref_dict["scen_time_Conditions_dead"]), title='Conditions')
        
        tab3 = Panel(child=row(plotref_dict["scen_time_Venues_Retail"], plotref_dict["scen_time_Venues_PrimarySchool"], plotref_dict["scen_time_Venues_SecondarySchool"], plotref_dict["scen_time_Venues_Work"], plotref_dict["scen_time_Venues_Home"]), title='Venues')      
        
        # Put the Panels in a Tabs object
        tabs = Tabs(tabs=[tab1, tab2, tab3])
        
        show(tabs)
        


if __name__ == "__main__":
    create_dashboard()
    print("End of program")






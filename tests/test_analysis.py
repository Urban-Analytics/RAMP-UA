# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:01:02 2020

@author: Natalie
"""


import pytest
import os
import pickle
import imageio
from shapely.geometry import Point
import numpy as np



# Do the directories and data files exist

# Do the input files contain the right information eg nr days/ranges/msoas/disease conditions etc

# Are the counts correct - cross checks
    
    # # sanity check: sum across MSOAs should be same as nrs_*
    # assert (msoa_counts_S.iloc[:,d].sum() == nrs_S[d])
    # assert (msoa_counts_E.iloc[:,d].sum() == nrs_E[d])
    # assert (msoa_counts_I.iloc[:,d].sum() == nrs_I[d])
    # assert (msoa_counts_R.iloc[:,d].sum() == nrs_R[d])

#    assert(total_day == nrs_S[d] + nrs_E[d] + nrs_I[d] + nrs_R[d])
    




# Plot data
# ----------
msoas_nr = [i for i in range(0,len(msoas))]

# line plot SEIR: total E,I,R across time (summed over all areas)
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(days, nrs_E, label='Exposed')  # Plot some data on the axes.
ax.plot(days, nrs_I, label='Infectious')  # Plot more data on the axes...
ax.plot(days, nrs_R, label='Recovered')  # ... and some more.
ax.set_xlabel('Days')  # Add an x-label to the axes.
ax.set_ylabel('Number of people')  # Add a y-label to the axes.
ax.set_title("Infections over time")  # Add a title to the axes.
ax.legend()  # Add a legend.   

# Line plot of SEIR per MSOA at a given day
# ask user to pick a day
day2plot = int(input("Please type the number of the day you want to plot (0 to "+str(nr_days-1)+"): "))
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(msoas_nr, msoa_counts_E.iloc[:,day2plot].tolist(), label='Exposed')  
ax.plot(msoas_nr, msoa_counts_I.iloc[:,day2plot].tolist(), label='Infectious')
ax.plot(msoas_nr, msoa_counts_R.iloc[:,day2plot].tolist(), label='Recovered')
ax.set_xlabel('MSOA')  # Add an x-label to the axes.
ax.set_ylabel('Number of people')  # Add a y-label to the axes.
ax.set_title("Infections across MSOAs, day "+str(day2plot))  # Add a title to the axes.
ax.legend()  # Add a legend.

# Line plot of SEIR per MSOA summed across days
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(msoas_nr, msoa_counts_E.sum(axis = 1), label='Exposed') 
ax.plot(msoas_nr, msoa_counts_I.sum(axis = 1), label='Infectious')
ax.plot(msoas_nr, msoa_counts_R.sum(axis = 1), label='Recovered')
ax.set_xlabel('MSOA')  # Add an x-label to the axes.
ax.set_ylabel('Number of people')  # Add a y-label to the axes.
ax.set_title("Infections across MSOAs summed across days")  # Add a title to the axes.
ax.legend()  # Add a legend.
    
# heatmap SEIR
var2plot = msoa_counts_E.iloc[:,0:nr_days]
title4plot = "nr exposed"
xticklabels=days
xticks = xticklabels
xticks = [x +1 - 0.5 for x in xticks] # to get the tick in the centre of a heatmap grid rectangle
# pick one colourmap from below
#cmap = sns.color_palette("coolwarm", 128)  
cmap = 'RdYlGn_r'  
plt.figure(figsize=(30, 10))
ax1 = sns.heatmap(var2plot, annot=False, cmap=cmap, xticklabels=xticklabels)
ax1.set_xticks(xticks)
plt.title(title4plot)
plt.ylabel("MSOA")
plt.xlabel("Day")
plt.show()

    
# line plot dangers across time, summed per venue type
fig, ax = plt.subplots()
#ax.plot(days, home_dangers.iloc[:,1:nr_days+1].sum(axis=0), label='Home') 
ax.plot(days, work_dangers.iloc[:,1:nr_days+1].sum(axis=0), label='Work')
ax.plot(days, retail_dangers.iloc[:,1:nr_days+1].sum(axis=0), label='Retail')
ax.plot(days, primaryschool_dangers.iloc[:,1:nr_days+1].sum(axis=0), label='Primary schools')
ax.plot(days, secondaryschool_dangers.iloc[:,1:nr_days+1].sum(axis=0), label='Secondary schools')
ax.set_xlabel('Days')  # Add an x-label to the axes.
ax.set_ylabel('Danger score')  # Add a y-label to the axes.
ax.set_title("Venue danger scores over time")  # Add a title to the axes.
ax.legend()  # Add a legend   

# heatmap danger scores
var2plot = retail_dangers.iloc[:,1:nr_days+1]
title4plot = "Danger score retail venues"
xticklabels=days
xticks = xticklabels
xticks = [x +1 - 0.5 for x in xticks] # to get the tick in the centre of a heatmap grid rectangle
# pick one colourmap from below
#cmap = sns.color_palette("coolwarm", 128)  
cmap = 'RdYlGn_r'  
plt.figure(figsize=(30, 10))
ax1 = sns.heatmap(var2plot, annot=False, cmap=cmap, xticklabels=xticklabels)
ax1.set_xticks(xticks)
plt.title(title4plot)
plt.ylabel("Retail venue")
plt.xlabel("Day")
plt.show()


# geographical plots

# preprocessing

# load in a shapefile
sh_file = os.path.join(data_dir, "MSOAS_shp","bcc21fa2-48d2-42ca-b7b7-0d978761069f2020412-1-12serld.j1f7i.shp")
map_df = gpd.read_file(sh_file)
# check
#map_df.plot()
# rename
map_df.rename(index=str, columns={'msoa11cd': 'Area'},inplace=True)

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


# choropleth

# set the range for the choropleth
vmin = 0
vmax = msoa_counts_I.iloc[:,0:nr_days].max().max()  # find max to scale (or set max number eg if using %)
# create individual plots and save each as image
for d in range(0, nr_days):
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
with imageio.get_writer('map_movie.gif', mode='I', duration=0.5) as writer:
    for d in range(0, 30):
        filename = "map_day"+str(d+1)+".png"
        image = imageio.imread(filename)
        writer.append_data(image)
        
        
# points on map

# plot all retail locations
# base = map_df.plot(color='white', edgecolor='black')
# gdf_primaryschools.plot(ax=base, marker='o', color='blue', markersize=5, legend = True)
# gdf_secondaryschools.plot(ax=base, marker='o', color='green', markersize=5, legend = True)
# gdf_retail.plot(ax=base, marker='o', color='red', markersize=5, legend = True)

# # plot only those with certain level of danger
# base = map_df.plot(color='white', edgecolor='black')
# gdf_retail[gdf_retail.Danger0 > 0].plot(ax=base, marker='o', color='red', markersize=5,legend = True)



# alternative way of plotting
fig, ax = plt.subplots()
# set aspect to equal (because we are not using *geopandas* plot)
ax.set_aspect('equal')
map_df.plot(ax=ax, color='white', edgecolor='black')
#gdf_retail.plot(ax=ax, marker='o', color='red', markersize=5)
gdf_primaryschools.plot(ax=ax, marker='o', color='blue', markersize=5, label = "Primary Schools", legend = True)
gdf_secondaryschools.plot(ax=ax, marker='o', color='green', markersize=5, label = "Secondary schools", legend = True)
gdf_retail.plot(ax=ax, marker='o', color='red', markersize=5, label = "Retail", legend = True)
ax.legend()
plt.show()

# plot only those with certain level of danger
thres = 10
fig, ax = plt.subplots()
ax.set_aspect('equal')
map_df.plot(ax=ax, color='white', edgecolor='black')
gdf_primaryschools[gdf_primaryschools.Danger19 > thres].plot(ax=ax, marker='o', color='blue', markersize=5, label = "Primary Schools", legend = True)
gdf_secondaryschools[gdf_secondaryschools.Danger19 > thres].plot(ax=ax, marker='o', color='green', markersize=5, label = "Secondary schools", legend = True)
gdf_retail[gdf_retail.Danger19 > thres].plot(ax=ax, marker='o', color='red', markersize=5, label = "Retail", legend = True)
ax.legend()
ax.set_title("Venues with danger score > " +str(thres)+ " at day 20")
ax.axis('off')
plt.show()

# ax = gdf_retail.plot(color='k', zorder=2)
# map_df.plot(ax=ax, zorder=1);

# geopandas heatmap from points
#takes a GeoDataFrame with point geometries and shows a matplotlib plot of heatmap density. This is done using numpy's 2D histogram binning with smoothing from scipy.

# def heatmap(d, bins=(100,100), smoothing=1.3, cmap='jet'):
#     def getx(pt):
#         return pt.coords[0][0]

#     def gety(pt):
#         return pt.coords[0][1]

#     x = list(d.geometry.apply(getx))
#     y = list(d.geometry.apply(gety))
#     heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins)
#     extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]

#     logheatmap = np.log(heatmap)
#     logheatmap[np.isneginf(logheatmap)] = 0
#     logheatmap = ndimage.filters.gaussian_filter(logheatmap, smoothing, mode='nearest')
    
#     plt.imshow(logheatmap, cmap=cmap, extent=extent)
#     plt.colorbar()
#     plt.gca().invert_yaxis()
#     plt.show()
    
# heatmap(gdf_retail, bins=50, smoothing=1.5)


# Heatmap simple version
# create the 'heat' e.g. all retail venues with total danger score > x
gdf_retail['SummedDanger'] = gdf_retail.iloc[:,-nr_days-1:-1].sum(axis=1)
gdf_retail_heat = gdf_retail[gdf_retail.SummedDanger > 50]

bins=(50,50)
smoothing=1.5
cmap='jet'
x = gdf_retail_heat.centroid.map(lambda p: p.x)
y = gdf_retail_heat.centroid.map(lambda p: p.y)
heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins)
extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
logheatmap = np.log(heatmap)
logheatmap[np.isneginf(logheatmap)] = 0
logheatmap = ndimage.filters.gaussian_filter(logheatmap, smoothing, mode='nearest')
plt.imshow(logheatmap, cmap=cmap, extent=extent,alpha = 0.8)
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()


# Heatmap as (thresholded) overlay on MSOAs
bins=(50,50)
smoothing=1.5
cmap='jet'
x = gdf_retail_heat.centroid.map(lambda p: p.x)
y = gdf_retail_heat.centroid.map(lambda p: p.y)
heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins)
extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
logheatmap = np.log(heatmap)
logheatmap[np.isneginf(logheatmap)] = 0
logheatmap = ndimage.filters.gaussian_filter(logheatmap, smoothing, mode='nearest')
logheatmap = np.ma.masked_where(logheatmap < 0.05, logheatmap)

fig, ax = plt.subplots()
ax.set_aspect('equal')
map_df.plot(ax=ax, color='white', edgecolor='black',zorder=1)
plt.imshow(logheatmap, cmap=cmap, extent=extent,alpha = 1, zorder=2,vmin=0)
plt.colorbar()
plt.gca().invert_yaxis()
ax.set_title("Retail summed danger score > 50 heat map")
ax.axis('off')
plt.show()




# from https://towardsdatascience.com/walkthrough-mapping-basics-with-bokeh-and-geopandas-in-python-43f40aa5b7e9

import json
from bokeh.io import show
from bokeh.models import (CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter, 
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider)
from bokeh.layouts import column, row, widgetbox
from bokeh.palettes import brewer
from bokeh.plotting import figure

# Turn data into GeoJSON source
geosource = GeoJSONDataSource(geojson = merged_data.to_json())

# Define color palettes
palette = brewer['BuGn'][8]
palette = palette[::-1] # reverse order of colors so higher values have darker colors
# Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
color_mapper = LinearColorMapper(palette = palette, low = 0, high = 2000)
# Define custom tick labels for color bar.
tick_labels = {'0': '0', '500': '500', '1000':'1,000', '1500':'1,500',
 '2000':'2,000+'}
# Create color bar.
color_bar = ColorBar(color_mapper = color_mapper, 
                     label_standoff = 8,
                     width = 500, height = 20,
                     border_line_color = None,
                     location = (0,0), 
                     orientation = 'horizontal',
                     major_label_overrides = tick_labels)


# Create figure object.
p = figure(title = 'Title goes here', 
           plot_height = 600 ,
           plot_width = 950, 
           toolbar_location = 'below',
           tools = "pan, wheel_zoom, box_zoom, reset")
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
# Add patch renderer to figure.
msoasrender = p.patches('xs','ys', source = geosource,
                   fill_color = {'field' :'Day2',
                                 'transform' : color_mapper},     
                   #fill_color = None,
                   line_color = 'gray', 
                   line_width = 0.25, 
                   fill_alpha = 1)
# Create hover tool
p.add_tools(HoverTool(renderers = [msoasrender],
                      tooltips = [('MSOA','@Area'),
                                ('Nr exposed people','@Day2')]))
p.add_layout(color_bar, 'below')
show(p)







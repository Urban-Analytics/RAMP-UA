## Selecting which MSOAs should initially be seeded with COVID: ##


In order to initialise the model we needed to seed COVID cases in the population. We seeded these infections in areas and individuals we thought would most likely to have been the first individuals in an area to be exposed to the disease. 

We ranked each of the MSOAs in Devon according to their:
	
* Population density (high population density = greater exposure)
* Index of multiple deprivation (less deprived = greater exposure)
* Transport connectivity (more connected = greater exposure)

MSOAs in the top tertile for all three categories were considered to be high risk areas where COVID initially found in the region. Within these MSOAs we filter out individuals that spend <30% time outside their home and then randomly seed the infection on the first day to a given number of the remaining individuals.

Code for calculating the risk levels of MSOAs is found in R/py_int/msoa_high_risk.R . Prior to using this code the code in R/py_int/msoa_pop_density.R and R/py_int/msoa_connectedness.R must be run in order to create the population density and connectedness scores for each MSOA.


### Population density: ###
	
Code in R/py_int/msoa_pop_density.R

Resulting data in devon_data/population_density_msoas.csv

In order to calculate population density we downloaded MSOA shapefile from [here](https://opendata.arcgis.com/datasets/826dc85fb600440889480f4d9dbb1a24_0.zip?outSR=%7B%22latestWkid%22%3A27700%2C%22wkid%22%3A27700%7D) and LSOA population data from [here](https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/lowersuperoutputareamidyearpopulationestimates/mid2018sape21dt1a/sape21dt1amid2018on2019lalsoasyoaestimatesformatted.zip).

We aggregated the population data to MSOA level using and then calculated the areas of each MSOA in the shapefile, and then used these values to calculate population density.	

### Connectedness: ###
	

Code in R/py_int/msoa_connectedness.R

Resulting data in devon_data/msoa_connectedness.csv

To calculate the 'connectedness' of each MSOA we use the following data:
* Airport and train station locations
* Annual number of passengers passing through each airport and train station
* The distance of each MSOA (centroid) to their three nearest train stations and airport

For each MSOA centroid we calculated the inverse distance to each train station and airport and multiplied this by the annual number of passengers passing through each train station or airport. This gives a connectedness score for each MSOA-transport hub pair.

For each MSOA we summed the connectedness score of the three closest transport hubs to give the MSOA a connectedness score.

### Index of Multiple Deprivation: ###

Code in R/py_int/msoa_high_risk.R

Resulting data in init_data/msoa_danger_fn.csv

We downloaded the Index of Multiple Deprivation Rank data from [here](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/833970/File_1_-_IMD2019_Index_of_Multiple_Deprivation.xlsx)

This was available at LSOA level, we averaged the LSOA ranks to give an MSOA average rank. 








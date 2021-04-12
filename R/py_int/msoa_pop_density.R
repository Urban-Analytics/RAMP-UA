library(tidyr)
library(dplyr)
library(rgdal)
library(janitor)

### Downloading MSOA shapefiles and population estimate data - data won't be downloaded if they don't already exist.
if(!file.exists("devon_data/MSOAs_December_2011_Full_Clipped_Boundaries_in_England_and_Wales.shp")){
  download.file("https://opendata.arcgis.com/datasets/826dc85fb600440889480f4d9dbb1a24_0.zip?outSR=%7B%22latestWkid%22%3A27700%2C%22wkid%22%3A27700%7D", "devon_data/MSOAs_December_2011_Full_Clipped_Boundaries_in_England_and_Wales.zip")
  unzip("devon_data/MSOAs_December_2011_Full_Clipped_Boundaries_in_England_and_Wales.zip", exdir = "devon_data/msoas")
}

if(!file.exists("devon_data/SAPE21DT1a-mid-2018-on-2019-LA-lsoa-syoa-estimates-formatted.xlsx")){
  download.file("https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/lowersuperoutputareamidyearpopulationestimates/mid2018sape21dt1a/sape21dt1amid2018on2019lalsoasyoaestimatesformatted.zip", destfile = "devon_data/sape21dt1amid2018on2019lalsoasyoaestimatesformatted.zip")
  unzip("devon_data/sape21dt1amid2018on2019lalsoasyoaestimatesformatted.zip", exdir = "devon_data")
  }

if(!file.exists("devon_data/lsoa_to_msoa.csv")){
  download.file("https://opendata.arcgis.com/datasets/fe6c55f0924b4734adf1cf7104a0173e_0.csv", "devon_data/lsoa_to_msoa.csv")
}

msoa <- readOGR(
  "devon_data/msoas/Middle_Layer_Super_Output_Areas_(December_2011)_Boundaries.shp")

#calculating the area of each of the MSOAs
msoa_area <- data.frame(msoa_area_codes = msoa$msoa11cd, msoa_km2 = msoa$st_areasha/100000)
msoa_area$msoa_area_codes <- as.character(msoa_area$msoa_area_codes)

# The cross walk table linking LSOAs to MSOAs
lsoa_all <- clean_names(readxl::read_xlsx("devon_data/SAPE21DT1a-mid-2018-on-2019-LA-lsoa-syoa-estimates-formatted.xlsx", sheet = "Mid-2018 Persons", skip = 4)) %>% 
  filter(!is.na(lsoa)) %>% 
  dplyr::select(lsoa_area_codes = area_codes,lsoa, population = all_ages,-la_2019_boundaries, -starts_with("x"))

# Joining the MSOA area and LSOA population data sets
ls_ms <- clean_names(read_csv("devon_data/lsoa_to_msoa.csv")) %>% 
  dplyr::select(lsoa_area_codes = lsoa11cd, msoa_area_codes = msoa11cd) %>% 
  left_join(., lsoa_all, by = "lsoa_area_codes") %>% 
  left_join(., msoa_area, by = "msoa_area_codes") %>% 
  distinct()

# Calculating population density of each MSOA
ls_ms_pd <- ls_ms %>% 
  group_by(msoa_area_codes) %>% 
  summarise(population = sum(population), msoa_km2 = msoa_km2) %>% 
  distinct() %>% 
  mutate(pop_dens_km2 = population/msoa_km2)


write.csv(ls_ms_pd, "devon_data/population_density_msoas.csv", row.names = FALSE)






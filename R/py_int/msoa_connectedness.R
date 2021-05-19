library(readr)
library(readxl)
library(sp)
library(janitor)
library(tidyr)
library(dplyr)
library(ggplot2)
library(sf)
library(readODS)

if(!file.exists("devon_data/MSOAs_December_2011_Full_Clipped_Boundaries_in_England_and_Wales.shp")){
  download.file("https://opendata.arcgis.com/datasets/826dc85fb600440889480f4d9dbb1a24_0.zip?outSR=%7B%22latestWkid%22%3A27700%2C%22wkid%22%3A27700%7D", "devon_data/MSOAs_December_2011_Full_Clipped_Boundaries_in_England_and_Wales.zip")
  unzip("devon_data/MSOAs_December_2011_Full_Clipped_Boundaries_in_England_and_Wales.zip", exdir = "devon_data/msoas")
}


if(!file.exists("devon_data/Table_10_1_EU_and_Other_Intl_Pax_Traffic.csv")){
  download.file("https://www.caa.co.uk/uploadedFiles/CAA/Content/Standard_Content/Data_and_analysis/Datasets/Airport_stats/Airport_data_2020_01/Table_10_1_EU_and_Other_Intl_Pax_Traffic.csv", destfile = "devon_data/Table_10_1_EU_and_Other_Intl_Pax_Traffic.csv")
}


if(!file.exists("devon_data/station_usage.ods")){
  download.file("https://dataportal.orr.gov.uk/media/1667/table-1410-estimates-of-station-usage-2018-19.ods", destfile = "devon_data/station_usage.ods")
}


msoa <- st_read("devon_data/msoas/Middle_Layer_Super_Output_Areas_(December_2011)_Boundaries.shp") %>% 
  st_transform(.,crs = 4326)

# calculating the centroid of each MSOA as this will be the point we calculate distances to transport hubs from.
centroids <- st_centroid(msoa)
xy <- st_coordinates(centroids)
st_geometry(msoa) <- NULL

# adding the coordinates of the MSOA centroids to the MSOA df
msoa <- msoa %>% 
  dplyr::select(msoa11cd ,msoa11nm) %>% 
  mutate( lat = xy[,"Y"], lon = xy[,"X"])

# Reading in the airport location data - collated by Fiona Spooner
airp_xy <- clean_names(read_csv("devon_data/uk_airports.csv")) %>%
  dplyr::select(transport_hub = airport, long, lat)

# Reading in and formatting the airport passenger data for 2019
airp <- clean_names(read_csv("devon_data/Table_10_1_EU_and_Other_Intl_Pax_Traffic.csv")) %>% 
  dplyr::select(transport_hub = rpt_apt_name, annual_passengers = total_pax_tp) %>% #https://www.caa.co.uk/Data-and-analysis/UK-aviation-market/Airports/Datasets/UK-Airport-data/Airport-data-2020-01/
  left_join(., airp_xy, by = "transport_hub") %>% 
  dplyr::select(transport_hub, long, lat, annual_passengers) %>% 
  mutate(transport_type = "Airport", transport_hub = gsub(" ", "_", transport_hub))

#### Loading in train station data - annual passengers from 2018-2019 - source: https://dataportal.orr.gov.uk/statistics/usage/estimates-of-station-usage/
train <- clean_names(readODS::read_ods("devon_data/station_usage.ods", sheet = "Estimates_of_Station_Usage")) %>% 
  dplyr::select(transport_hub = station_name, os_grid_easting, os_grid_northing, annual_passengers = x1819_entries_exits) %>% 
  mutate(transport_type = "Train")

#### Transforming the train station coordinates from os grid to wgs84
#coords <- cbind(train$os_grid_easting, train$os_grid_northing)
latlong <- "+init=epsg:4326"

coordinates(train) <- c("os_grid_easting", "os_grid_northing")
proj4string(train) <- CRS("+init=epsg:27700") # WGS 84
CRS.new <- CRS(latlong)
train_new <- data.frame(spTransform(train, CRS.new))

train_new <- train_new %>% 
                dplyr::select(transport_hub, 
                  long = os_grid_easting, 
                  lat = os_grid_northing,
                  annual_passengers,
                  transport_type) 

#### Combining the train and airport data
trair <- rbind(train_new, airp)
trair$transport_hub <- janitor::make_clean_names(trair$transport_hub) # making the transport hub names consistent

trair_xy <- cbind(trair$long, trair$lat)
msoa_xy <- cbind(msoa$lon, msoa$lat)

## Calculating the distance from each transport hub to each MSOA centroid
dist_out <- apply(trair_xy,1, function(x) spDistsN1(msoa_xy, x, longlat = TRUE))
colnames(dist_out)<- trair$transport_hub

ua_airp_df <- data.frame( msoa11cd = msoa$msoa11cd, msoa11nm = msoa$msoa11nm ,dist_out)

ua_airp_lng <- tidyr::pivot_longer(ua_airp_df, cols = -c( msoa11cd, msoa11nm) ) 
colnames(ua_airp_lng)[3:4] <- c("transport_hub", "distance_km")

ua_airp_lng <- ua_airp_lng %>% 
  left_join(., trair, by = "transport_hub") 

ua_airp_lng$annual_passengers <- as.numeric(ua_airp_lng$annual_passengers)

connect <- ua_airp_lng %>% 
  mutate(connectedness_ann = (1/distance_km)*annual_passengers) %>% # calculating the connectedness metric for each transport hub*msoa 
  group_by( msoa11cd,  msoa11nm, transport_type) %>%  # grouping to calculate connectedness by MSOA
  top_n(n= -3, wt = distance_km) %>%    #closest three airports and train stations - realistically the transport stations people would use.
  ungroup() %>% 
  group_by(msoa11cd,  msoa11nm) %>% 
  summarise(msoa_sum_connected_ann = sum(connectedness_ann),
            log_msoa_sum_connected_ann = log10(msoa_sum_connected_ann))


connect %>% 
  arrange(msoa_sum_connected_ann)


write.csv(connect, "devon_data/msoa_connectedness.csv", row.names = FALSE)


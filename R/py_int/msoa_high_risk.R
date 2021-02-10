##### Code sort data to send Nick
library(tidyverse)
library(janitor)

source("Code/covid_infection_functions.R")

############## MSOA ranking data

population <- clean_names(read_delim("data/devon-tu_health/Devon_simulated_TU_keyworker_health.csv", ","))
pop_dens <- read_csv("devon_data/population_density_msoas.csv") #msoa_pop_density.R

connectivity <- janitor::clean_names(read_csv("devon_data/msoa_connectedness.csv")) %>% 
  filter(!is.na(msoa_sum_connected_ann))
colnames(connectivity)[3:4] <- c("connectedness", "log_connectedness") 


if(!file.exists("devon_data/income_ahc.csv")){
  download.file("https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/smallareaincomeestimatesformiddlelayersuperoutputareasenglandandwales/financialyearending2018/netannualincomeafterhousingcosts2018.csv", "devon_data/income_ahc.csv")
}


if(!file.exists("devon_data/lsoa_to_msoa.csv")){
  download.file("https://opendata.arcgis.com/datasets/fe6c55f0924b4734adf1cf7104a0173e_0.csv", "devon_data/lsoa_to_msoa.csv")
}

lsoa_to_msoa <- clean_names(read_csv("devon_data/lsoa_to_msoa.csv")) 

if(!file.exists("devon_data/msoa_imd.xlsx")){
  download.file("https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/833970/File_1_-_IMD2019_Index_of_Multiple_Deprivation.xlsx","devon_data/msoa_imd.xlsx")
}


qimd <- clean_names(readxl::read_xlsx("devon_data/msoa_imd.xlsx", sheet = "IMD2019")) %>% 
  left_join(., lsoa_to_msoa, by = c("lsoa_code_2011" = "lsoa11cd")) %>% 
  group_by(msoa11cd) %>% 
  summarise(mean_imd_rank = mean(index_of_multiple_deprivation_imd_rank))


population_in <- population %>% 
  left_join(., pop_dens, by = c("area" = "msoa_area_codes")) %>% 
  left_join(., connectivity, by = c("area" = "msoa11cd")) %>% 
  left_join(., qimd, by = c("area" = "msoa11cd")) %>% 
  select(area, pop_dens = pop_dens_km2, connectivity = connectedness, qimd = mean_imd_rank) %>% 
  distinct()


groups <- 3

population_in$pop_dens_ntile <- ntile(population_in$pop_dens,groups)
population_in$connectivity_ntile <- ntile(population_in$connectivity, groups)
population_in$qimd_ntile <- ntile(population_in$qimd, groups)


haz_msoas <- population_in %>%
  mutate(
    risk = case_when(
      pop_dens_ntile == groups & connectivity_ntile == groups & qimd_ntile == groups ~ "High",
      pop_dens_ntile == 1 & connectivity_ntile == 1 & qimd_ntile == 1 ~ "Low"
    ), risk = ifelse(is.na(risk), "Medium", risk)) %>% 
  dplyr::select(area, risk)


write.csv(haz_msoas, "init_data/msoa_danger_fn.csv", row.names = FALSE)

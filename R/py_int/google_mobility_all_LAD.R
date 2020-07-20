library(readr)
library(dplyr)
library(tidyr)
library(rampuaR)
library(stringr)

### download google mobility data and local authority codes 
gm_file_download()
lad_file_download()

lad_codes <- read_csv("lad_codes.csv") %>% 
  distinct()

gm <- read_csv("Google_Global_Mobility_Report.csv") %>% 
  filter(country_region == "United Kingdom" & !is.na(sub_region_1))

all_lad <- list.files("devon_data/FullMatchedData_LAD", full.names = TRUE)

for(i in 80:length(all_lad)){
  print(i)
  lad <- read_csv(all_lad[i])
  
  lad_df <- msoa_lad_code_matcher(pop = lad)
  print(lad_df$lad_name[1])
  
  gm_out <- gm_filter(lad_df)
  
  if(length(gm_out) > 1){
    smth_res <- residential_smoother(gm_out)
    
    lockdown_out <- lockdown_multiplier(smth_res = smth_res, pop = lad_df)
    
    write.csv(lockdown_out, paste0("devon_data/GoogleMobilityLAD/", gsub(".txt", ".csv",basename(all_lad[i]))))
    }
   
}



lad <- read_csv("~/Documents/ECEHH/COVID/RAMP-UA//devon_data/FullMatchedData_LAD/lad_tus_hse_100.txt")






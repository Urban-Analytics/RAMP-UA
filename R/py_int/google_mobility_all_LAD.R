devtools::install_github("Urban-Analytics/rampuaR", force = TRUE)

library(readr)
library(dplyr)
library(tidyr)
library(rampuaR)
library(stringr)

### download google mobility data and local authority codes 
gm_file_download(force_gm = TRUE)
lad_file_download()
county_file_download()
county_codes <- read_csv("lad_county_codes.csv")

lad_codes <- read_csv("lad_codes.csv") %>% 
  distinct() %>% 
  left_join(., county_codes, by = "OA11CD")

gm <- read_csv("Google_Global_Mobility_Report.csv") %>% 
  filter(country_region == "United Kingdom" & !is.na(sub_region_1))

all_lad <- list.files("devon_data/FullMatchedData_LAD", full.names = TRUE)

seq_lad <- (1:length(all_lad))[-79] #skipping rutland for now

for (i in seq_lad){
  print(i)
  lad <- read_csv(all_lad[i], col_types = cols())
  
  lad_df <- msoa_lad_code_matcher(pop = lad, lad_codes = lad_codes)
  print(lad_df$lad_name[1])
  
  gm_filt <- gm_filter(gm,lad_name = unique(pop$lad_name), county_name = unique(pop$county))
  print(gm_out$sub_region_1[1])
  
  print(paste(lad_df$lad_name[1], gm_out$sub_region_1[1]))
  
  if(length(gm_filt) > 1){
    gm_formatted <- format_gm(gm_filt)
    smth_res <- residential_smoother(gm_formatted)
    lockdown_out <- lockdown_multiplier(smth_res = smth_res, pop = lad_df)
    
    write.csv(lockdown_out, paste0("devon_data/GoogleMobilityLAD/", gsub(".txt", ".csv",basename(all_lad[i]))))
    }
   
}


#Stroud
#Rutland 80









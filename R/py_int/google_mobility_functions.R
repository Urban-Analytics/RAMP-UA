library(readr)
library(dplyr)
library(tidyr)

gm_file_download <- function(force_gm = FALSE){
  
  if((force_gm == FALSE &
       !file.exists("Google_Global_Mobility_Report.csv")) | force_gm == TRUE){
    download.file(
      "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv",
      destfile = "Google_Global_Mobility_Report.csv"
    )
  }
}


lad_file_download <- function(force_lad = TRUE){
  
  if((force_lad == FALSE &
       !file.exists("lad_codes.csv")) | force_lad == TRUE){
    download.file(
      "http://geoportal1-ons.opendata.arcgis.com/datasets/fe6c55f0924b4734adf1cf7104a0173e_0.csv",
      "lad_codes.csv")
  }
}


county_file_download <- function(force_county = TRUE){
  
  if((force_county == FALSE &
      !file.exists("lad_county_codes.csv")) | force_county == TRUE){
    download.file(
      "https://opendata.arcgis.com/datasets/79c993a10398400bb025a00849a43dc0_0.csv",
      "lad_county_codes.csv")
  }
}



lad <- read_csv("devon_data/FullMatchedData_LAD/lad_tus_hse_1.txt")

gm <- read_csv("Google_Global_Mobility_Report.csv") %>% 
  filter(country_region == "United Kingdom" & !is.na(sub_region_1))

county_codes <- read_csv("lad_county_codes.csv")

lad_codes <- read_csv("lad_codes.csv") %>% 
  dplyr::select(MSOA11CD,LAD17CD,LAD17NM) %>% 
  distinct() %>% 
  left_join(., county_codes, by = c("LAD17CD" = "LAD19CD"))


msoa_lad_code_matcher <- function(pop){
  
  code_match <- lad_codes %>% 
    filter(MSOA11CD %in% pop$area) %>% 
    select(lad_name = LAD17NM, lad_code = LAD17CD, county = CTY19NM) %>% 
    distinct() 
  
  pop <- data.frame(pop, code_match)
  
  return(pop)
}

lad_df <- msoa_lad_code_matcher(pop = lad)



gm_filter <- function(pop){
  
  name <- unique(pop$lad_name)

  gm_filt <- gm %>% 
    filter(str_detect(name, sub_region_1))
  
  if(nrow(gm_filt) == 0){
    name <- unique(pop$county)
    gm_filt <- gm %>% 
      filter(str_detect(name, sub_region_1))
  }
  
  if(nrow(gm_filt) == 0){
    gm_filt <- paste0("No matching Google Mobility data for ", name)
  }
  
  return(gm_filt)  
}


gm_out <- gm_filter(lad_df)


residential_smoother <- function(gm_out){
  
  residential_pcnt <- gm_out %>% 
    pivot_longer(., contains("percent")) %>% 
    filter(
      name == "residential_percent_change_from_baseline"
    ) %>% 
    mutate(day = as.numeric(date) - 18306) %>% 
    select(day, value) 
  
  new_data <- data.frame(day = 1:nrow(residential_pcnt), value = 0)
  
  smooth_residential <- mgcv::gam(value ~ s(day, bs = "cr"), fx = TRUE, data = residential_pcnt)
  sr <- predict(smooth_residential, new_data, type = "response")
  sr <- (sr/100) + 1
  
  return(sr)
  }


smth_res <- residential_smoother(gm_out)

lockdown_multiplier <- function(smth_res, pop){
  
  new_out <- 1 - (mean(pop$phome, na.rm = TRUE) * smth_res) # During the first two weeks of lockdown people spent 21% more time at home - the average person from devon spent ~ 75% of their time at home before lockdow, during lockdown this increases to 92% - meaning about 8% of time outside the home on average
  
  lock_down_reducer <-  new_out/mean(pop$pnothome) # percentage of time spent outside of compared to pre-lockdown
  lock_down_reducer 
  
  daily_lock_down_multiplier <-  data.frame(lad_name = unique(pop$lad_name), lad_code = unique(pop$lad_code), day = 1:length(lock_down_reducer), timeout_multiplier = lock_down_reducer)
  
  return(daily_lock_down_multiplier)
}

lockdown_out <- lockdown_multiplier(smth_res = smth_res, pop = pop)




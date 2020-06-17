library(tidyr)
library(janitor)
library(readr)
library(mixdist)
#library(arrow)
#library(dplyr)
#library(ggplot2)
#library(sf)
#library(viridisLite)

# set wd is done by parent python class
#setwd("/Users/JA610/Documents/GitHub/RAMP-UA/")

# Working directory set automatically by python
#setwd("/Users/JA610/Documents/GitHub/RAMP-UA/")

#setwd("/Users/JA610/Documents/GitHub/RAMP-UA/")

#source("R/py_int/covid_status_functions.R")
#source("R/py_int/initialize_and_helper_functions.R")

#beta1 <- current_risk /  danger <- 0.55
#pop <- read.csv("~/Downloads/input_population100917.csv")

run_status <- function(pop, timestep=1) {
  
  if(sum(pop$disease_status) == 0){
    seeds <- sample(1:nrow(pop), size = 20)
    pop$disease_status[seeds] <- 1
  }
  
  population <- clean_names(pop)
  
  num_sample <- nrow(population)
  
  #print(num_sample)
  
  # the stuff below here should be loaded only once in python i guess and
  # passed as columns in the dataframe
  # ive removed them for now because im not sure if we want to keep
  # msoa and population density in here since it might be accounted for
  # in nics code. for now we are just including age, sex, and nics "risk"
  
  #pop_dens <- read_csv("~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Population_Data_Devon/msoa_population_density.csv")
  
  #connectivity <- janitor::clean_names(read_csv("~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Transport_Data/msoa_connectedness_closest_three.csv")) %>% 
  #  filter(!is.na(msoa_sum_connected_ann))
  #colnames(connectivity)[3:4] <- c("connectedness", "log_connectedness") 
  #connectivity$connectivity_index <- normalizer(connectivity$log_connectedness, 0.01, 1, min(connectivity$log_connectedness), max(connectivity$log_connectedness))
  
  area <- "area"
  hid <- "house_id"
  #pid <- "pid"
  age <- "age1"
  sex <- "sex"
  id <- "id"
  
  population_in <- population #%>% 
  #left_join(., pop_dens, by =  c("area" = "msoa_area_codes")) %>% 
  #dplyr::left_join(.,connectivity, by = c("area" = "msoa11cd")) %>% 
  #mutate(log_pop_dens = log10(pop_dens_km2)) 
  
  population_in$cases_per_area <- 0
  #population_in$disease_status <- 0
  
  #print("c")
  
  df_cr_in <-create_input(micro_sim_pop  = population_in,
                          num_sample = num_sample,
                          pnothome_multiplier = 0.6,   # 0.1 = a 60% reduction in time people not home
                          vars = c(area,   # must match columns in the population data.frame
                                   hid,
                                   #pid,
                                   id,
                                   age,
                                   sex,
                                   "current_risk"))
  
  df_in <- as_betas_devon(population_sample = df_cr_in, 
                          id = id,
                          age = age, 
                          sex = sex, 
                          beta0_fixed = -9, #0.19, #-9.5, 
                          divider = 4)  # adding in the age/sex betas 
  
  #print("e")
  
  pnothome <-  0.25 #0.35
  connectivity_index <- 0.25#0.3 doesn't work
  log_pop_dens <- 0#0.2#0.4#0.3 #0.175
  cases_per_area <- 10 #2.5
  current_risk <- 1.5 #0.55
  
  origin <- factor(c(0,0,0,0,0))
  names(origin) <- c("1", "2", "3", "4", "5") #1 = white, 2 = black, 3 = asian, 4 = mixed, 5 = other
  qimd1 <- factor(c(0,0,0,0,0))
  names(qimd1) <- c("1", "2", "3", "4", "5")# 1  = Least Deprived ... 5 = Most Deprived
  underlining <- factor(c(0,0))
  names(underlining) <- c("0","1") #1 = has underlying health conditions
  hid_infected <- 0

  ### any betas included must link to columns/data.frames in df_in 
  
  other_betas <- list(current_risk = current_risk)
  
  df_msoa <- df_in
  df_risk <- list()
  
  #print("f")
  tmp.dir <- paste(getwd(),"/output/",Sys.time(),sep="")
  
  df_prob <- covid_prob(df = df_msoa, betas = other_betas, risk_cap=FALSE, risk_cap_val=100)
  df_ass <- case_assign(df = df_prob, with_optimiser = FALSE,timestep=timestep,tmp.dir=tmp.dir)
  df_inf <- infection_length(df = df_ass,
                             presymp_dist = "weibull",
                             presymp_mean = 6.4,
                             presymp_sd = 2.3,
                             infection_dist = "normal",
                             infection_mean =  14,
                             infection_sd = 2,
                             timestep=timestep,
                             tmp.dir=tmp.dir)
  df_rec <- removed(df = df_inf, chance_recovery = 0.95)
  df_msoa <- df_rec #area_cov(df = df_rec, area = area, hid = hid)
  
  #print("h")
  
  #colSums(df_msoa$had_covid)
  #colMeans(df_msoa$cases_per_area)
  
  df_out <- data.frame(area=df_msoa$area,
                       ID=df_msoa$id,
                       house_id=df_msoa$house_id,
                       disease_status=df_msoa$new_status,
                       presymp_days=df_msoa$presymp_days,
                       symp_days=df_msoa$symp_days)
  
  #print("new disease status calculated")
  
  if(timestep==1) {
    stat <- df_out$disease_status
  } else {
    tmp3 <- df_out$disease_status
    stat <- cbind(stat,tmp3)
  }
  #ncase <- as.data.frame(ncase)
  write.csv(stat, paste("disease_status.csv",Sys.time(),".csv",sep=""))

  return(df_out)
}


#out <- run_status(pop)


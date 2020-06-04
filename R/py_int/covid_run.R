library(dplyr)
library(ggplot2)
library(tidyr)
library(janitor)
library(readr)
library(sf)
library(arrow)
library(mixdist)
library(viridisLite)
library(reticulate)

source("Code/covid_status_functions.R")
source("Code/initialize_and_helper_functions.R")

run_status <- function(pop_df) {
  
  population <- clean_names(pop)
  
  # the stuff below here should be loaded only once in python i guess and
  # passed as columns in the dataframe
  pop_dens <- read_csv("~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Population_Data_Devon/msoa_population_density.csv")
  
  connectivity <- janitor::clean_names(read_csv("~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Transport_Data/msoa_connectedness_closest_three.csv")) %>% 
    filter(!is.na(msoa_sum_connected_ann))
  colnames(connectivity)[3:4] <- c("connectedness", "log_connectedness") 
  connectivity$connectivity_index <- normalizer(connectivity$log_connectedness, 0.01, 1, min(connectivity$log_connectedness), max(connectivity$log_connectedness))
  
  population_in$cases_per_area <- 0
  
  area <- "area"
  hid <- "hid"
  pid <- "pid"
  age <- "age1"
  sex <- "sex"
  
  population_in <- population %>% 
    left_join(., pop_dens, by =  c("area" = "msoa_area_codes")) %>% 
    dplyr::left_join(.,connectivity, by = c("area" = "msoa11cd")) %>% 
    mutate(log_pop_dens = log10(pop_dens_km2)) 
  
  df_cr_in <-create_input(micro_sim_pop  = population_in,
                          pnothome_multiplier = 0.6,   # 0.1 = a 60% reduction in time people not home
                          fixed_vars = c(area,   # must match columns in the population data.frame
                                         hid,
                                         pid,
                                         age,
                                         sex,
                                         "msoa_area",
                                         "connectivity_index"))
  
  df_in <- as_betas_devon(population_sample = df_cr_in, 
                          pid = pid,
                          age = age, 
                          sex = sex, 
                          beta0_fixed = -9.5, 
                          divider = 4)  # adding in the age/sex betas 
  
  pnothome <-  0.25 #0.35
  connectivity_index <- 0.25#0.3 doesn't work
  log_pop_dens <- 0#0.2#0.4#0.3 #0.175
  cases_per_area <- 10 #2.5
  
  origin <- factor(c(0,0,0,0,0))
  names(origin) <- c("1", "2", "3", "4", "5") #1 = white, 2 = black, 3 = asian, 4 = mixed, 5 = other
  qimd1 <- factor(c(0,0,0,0,0))
  names(qimd1) <- c("1", "2", "3", "4", "5")# 1  = Least Deprived ... 5 = Most Deprived
  underlining <- factor(c(0,0))
  names(underlining) <- c("0","1") #1 = has underlying health conditions
  hid_infected <- 0
  current_risk <- 0
  
  ### any betas included must link to columns/data.frames in df_in 
  
  other_betas <- list(pnothome = pnothome,
                      cases_per_area = cases_per_area,
                      connectivity_index = connectivity_index)
  
  df_msoa <- df_in
  df_risk <- list()
  
  df_prob <- covid_prob(df = df_msoa, betas = other_betas,timestep=i)
  df_ass <- case_assign(df = df_prob, timestep=i, with_optimiser = FALSE)
  df_inf <- infection_length(df = df_ass,
                             presymp_dist = "weibull",
                             presymp_mean = 6.4,
                             presymp_sd = 2.3,
                             infection_dist = "normal",
                             infection_mean =  14,
                             infection_sd = 2,
                             timestep=i)
  df_rec <- removed(df = df_inf, chance_recovery = 0.95,timestep=i)
  df_msoa <- area_cov(df = df_rec, timestep = i, area = area, hid = hid)
  
  #colSums(df_msoa$had_covid)
  #colMeans(df_msoa$cases_per_area)
  
  df_out <- data.frame(pid=df_msoa$pid,
                       status=df_msoa$status,
                       infected_time=df_msoa$infected_time,
                       infected_time=df_msoa$infected_time)
  
  print("new disease status calculated")
  
  return(tmp)
}





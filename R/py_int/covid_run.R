library(tidyr)
library(janitor)
library(readr)
library(mixdist)
library(dplyr)
#library(mgcv)
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
# 
# devon_cases <- readRDS(paste0(getwd(),"/devon_cases.RDS"))
# devon_cases$cumulative_cases[84] <- 812 #type here I think
# devon_cases$new_cases <- c(0,diff(devon_cases$cumulative_cases))
# devon_cases$devon_date <- as.numeric(devon_cases$date)
# devon_cases <- as.data.frame(devon_cases)
# 
# 
# gam_Devon <- mgcv::gam(new_cases ~ s(devon_date, bs = "cr"), data = devon_cases,family = nb())
# plot(devon_cases$new_cases*20,  ylab="Cases", xlab = "Day")
# points(round(fitted.values(gam_Devon)*20), col = "red")
# abline(v = 38, lty = "dashed")
# # gam_cases <- round(fitted.values(gam_Devon)*20)
# 
gam_cases <- readRDS(paste0(getwd(),"/gam_fitted_PHE_cases.RDS"))
# new_cases[new_cases == 0]<-1
# new_cases <- new_cases*20
#baseline_risk <- readRDS(paste0(getwd(),"/baseline_risk.RDS")) # needs replacing with days beyond 40
#baseline_risk <- readRDS(paste0(getwd(), "/baseline_risk2.RDS"))
risk_per_case <- 509.7954

w <- NULL
nick_cases <- NULL
run_status <- function(pop, timestep=1, current_risk = 0.0042) {
  
  opt_switch <- FALSE
  output_switch <- FALSE
  beta0_fixed <- 0
  # current_risk <- 0.01 #0.004
  rank_assign <- FALSE
  seed_cases <- TRUE
  seed_days <- 10
  normalizer_on <- TRUE
  lockdown_scenario <- FALSE # at the moment need to tell nick's model this separately which isn't ideal  
  risk_cap_on <- TRUE
  risk_cap <- 5
  
  
  print(paste("R timestep:", timestep))
  
  #pop <- vroom::vroom("R/py_int/input_pop_02.csv")
  
  
  #  if(output_switch==TRUE) {
  if(timestep==1) {
    tmp.dir <<- paste(getwd(),"/output/",Sys.time(),sep="")
    if(!dir.exists(tmp.dir)){
      dir.create(tmp.dir, recursive = TRUE)
    }
  }
  #write.csv(pop, paste0(tmp.dir,"/input_pop_", stringr::str_pad(timestep, 2, pad = "0"), ".csv"), row.names = FALSE)
  #}
  
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
                          beta0_fixed = beta0_fixed, #-9, #0.19, #-9.5,
                          divider = 4)  # adding in the age/sex betas 
  
  #print("e")
  
  #pnothome <-  0.25 #0.35
  #connectivity_index <- 0.25#0.3 doesn't work
  #log_pop_dens <- 0#0.2#0.4#0.3 #0.175
  #cases_per_area <- 10 #2.5
  
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
  
  
  #### seeding the first day in high risk MSOAs
  if(timestep==1){
    msoas <- read.csv(paste0(getwd(),"/msoa_danger_fn.csv"))
    msoas <- msoas[msoas$risk == "High",]
    pop_hr <- pop %>% filter(area %in% msoas$area & pnothome > 0.3)
    seeds <- sample(1:nrow(pop_hr), size = gam_cases[timestep])
    seeds_id <- pop_hr$id[seeds]
    df_msoa$new_status[df_msoa$id %in% seeds_id] <- 1
    #  df_msoa$new_status[seeds] <- 1
  }
  
  df_prob <- covid_prob(df = df_msoa,
                        betas = other_betas,
                        risk_cap=risk_cap_on,
                        risk_cap_val=risk_cap,
                        include_age_sex = FALSE,
                        normalizer_on = normalizer_on)
  print("probabilities calculated")
  
  if(opt_switch==TRUE) {
    df_prob <- new_beta0_probs(df = df_prob, daily_case = gam_cases[timestep])
  }
  
  
  if(timestep > 1){
    df_ass <- case_assign(df = df_prob,
                          with_optimiser = opt_switch, 
                          timestep=timestep,
                          tmp.dir=tmp.dir, 
                          save_output = output_switch)
  } else {
    df_ass <- df_prob
  }
  print("cases assigned")
  
  
  print(paste0("PHE cases ", gam_cases[timestep]))
  
  if(lockdown_scenario == TRUE){
    sum_risk <- sum(df_prob$current_risk)
    risk_base_cases <- sum_risk/risk_per_case
    gam_cases[timestep] <-  round(risk_base_cases)
  }
  
  nick_cases[timestep] <- (sum(df_prob$new_status == 0) - sum(df_ass$new_status == 0))
  print(paste0("model cases ", nick_cases[timestep]))
  print(paste0("Adjusted PHE cases ", gam_cases[timestep]))
  
  w[timestep] <- (sum(df_prob$new_status == 0) - sum(df_ass$new_status == 0))/gam_cases[timestep]
  print(paste0("w is ", w[timestep]))
  
  if(!is.finite(w[timestep])){
    w[timestep] <- 0
  }
  
  if(timestep > 1 & timestep <= seed_days & seed_cases == TRUE){
    df_ass <- rank_assign(df = df_prob, daily_case = gam_cases[timestep], timestep=timestep)
    print(paste0((sum(df_prob$new_status == 0) - sum(df_ass$new_status == 0))," cases reassigned"))
  }
  
  
  if((rank_assign == TRUE & seed_cases == FALSE) | (rank_assign == TRUE & seed_cases == TRUE & timestep > seed_days)){
    if(timestep > 1 & (w[timestep] <= 0.9 | w[timestep] >= 1.1)){
      df_ass <- rank_assign(df = df_prob, daily_case = gam_cases[timestep], timestep=timestep)
      print(paste0((sum(df_prob$new_status == 0) - sum(df_ass$new_status == 0))," cases reassigned"))
    }
  }
  
  
  
  df_inf <- infection_length(df = df_ass,
                             presymp_dist = "weibull",
                             presymp_mean = 6.4,
                             presymp_sd = 2.3,
                             infection_dist = "normal",
                             infection_mean =  14,
                             infection_sd = 2,
                             timestep=timestep,
                             tmp.dir=tmp.dir,
                             save_output = output_switch)
  print("infection and recovery lengths assigned")
  
  df_rec <- removed(df = df_inf, chance_recovery = 0.95)
  print("recoveries and deaths assigned")
  
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
  
  # if(output_switch==TRUE) {
  if(timestep==1) {
    stat <<- df_out$disease_status
    #    nb0 <<- unique(df_msoa$new_beta0)
    wo <<- w[timestep[1]]
    #   prob <<- df_msoa$probability
  } else {
    tmp3 <- df_out$disease_status
    # tmp4 <- unique(df_msoa$new_beta0)
    #  tmp5 <- df_msoa$probability
    # stat <<- cbind(stat,tmp3)
    #    nb0 <<- cbind(nb0, tmp4)
    wo <<- rbind(wo, w[timestep])
    #   prob <<- cbind(prob, tmp5)
  }
  #ncase <- as.data.frame(ncase)
  write.csv(stat, paste(tmp.dir,"/disease_status.csv",sep=""))
  #   write.csv(nb0, paste(tmp.dir,"/optim_b0.csv",sep=""))
  write.csv(wo, paste(tmp.dir,"/w_out.csv",sep=""))
  # write.csv(prob, paste(tmp.dir, "/probabilities.csv", sep = ""))
  # }
  
  return(df_out)
}


#out <- run_status(pop)


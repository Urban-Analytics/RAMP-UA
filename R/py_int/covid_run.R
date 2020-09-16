
load_rpackages <- function() {
  list.of.packages <- c("rampuaR")
  new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

  if(length(new.packages)) devtools::install_github("Urban-Analytics/rampuaR", dependencies = F)

  library(rvcheck)

  rampr_version <- check_github("Urban-Analytics/rampuaR")
  if(!rampr_version$up_to_date) devtools::install_github("Urban-Analytics/rampuaR", dependencies = F)

  #devtools::install_github("Urban-Analytics/rampuaR", dependencies = F)

  library(tidyr)
  library(readr)
  library(mixdist)
  library(dplyr)
  library(rampuaR)
}

load_init_data <- function() {
  data(gam_cases)
  data(msoas)
  w <<- NULL
  model_cases <<- NULL
}

initialize_r <- function() {
  load_init_data()
  load_rpackages()
}

run_status <- function(pop,
                       timestep = 1,
                       current_risk_beta = 0.008,
                       risk_cap = 5,
                       seed_days = 10,
                       exposed_dist = "weibull",
                       exposed_mean = 2.56,
                       exposed_sd = 0.72,
                       presymp_dist = "weibull",
                       presymp_mean = 2.3,
                       presymp_sd = 0.35,
                       infection_dist = "normal",
                       infection_mean =  16,
                       infection_sd = 3,
                       asymp_rate = 0.7,
                       chance_recovery = 0.95,
                       output_switch = TRUE,
                       rank_assign = FALSE,
                       local_outbreak_timestep = 0,
                       local_outbreak = FALSE,
                       msoa_infect="E02004152",
                       number_people_local=100,
                       local_prob_increase=0.75) {

  seed_cases <- ifelse(seed_days > 0, TRUE, FALSE)

  print(paste("R timestep:", timestep))

  if(timestep==1) {
    # windows does not allow colons in folder names so substitute sys.time() to hyphen
    tmp.dir <<- paste0(getwd(), "/output/", gsub(":","-", gsub(" ","-",Sys.time())))

    if(!dir.exists(tmp.dir)){
      dir.create(tmp.dir, recursive = TRUE)
    }
  }

  if(output_switch){write.csv(pop, paste0( tmp.dir,"/daily_", timestep, ".csv"))}

  df_cr_in <-create_input(micro_sim_pop  = pop,
                          vars = c("area",   # must match columns in the population data.frame
                                   "house_id",
                                   "id",
                                   "current_risk"))

  other_betas <- list(current_risk = current_risk_beta)

  df_msoa <- df_cr_in

  #### seeding the first day in high risk MSOAs
  if(timestep==1){
  #  msoas <- read.csv(paste0(getwd(),"/msoa_danger_fn.csv"))
    msoas <- msoas[msoas$risk == "High",]
    pop_hr <- pop %>% filter(area %in% msoas$area & pnothome > 0.3)
    seeds <- sample(1:nrow(pop_hr), size = gam_cases[timestep])
    seeds_id <- pop_hr$id[seeds]
    df_msoa$new_status[df_msoa$id %in% seeds_id] <- 1
    print("First day seeded")
  }

  df_sum_betas <- sum_betas(df = df_msoa,
                            betas = other_betas,
                            risk_cap_val = risk_cap)
  print("betas calculated")

  df_prob <- covid_prob(df = df_sum_betas)

  print("probabilities calculated")

  if(local_outbreak == TRUE & timestep == local_outbreak_timestep){
    print("Local outbreak - super spreader event!")
    df_prob <- local_outbreak(df=df_prob,
                              msoa_infect=msoa_infect,
                              number_people=number_people_local,
                              risk_prob=local_prob_increase)
  }
  
  
  if(timestep > 1){
    df_ass <- case_assign(df = df_prob,
                          tmp.dir=tmp.dir,
                          save_output = output_switch)
  } else {
    df_ass <- df_prob
  }

  print("cases assigned")
  print(paste0("PHE cases ", gam_cases[timestep]))

  model_cases[timestep] <- (sum(df_prob$new_status == 0) - sum(df_ass$new_status == 0))
  print(paste0("model cases ", model_cases[timestep]))
  print(paste0("Adjusted PHE cases ", gam_cases[timestep]))

  w[timestep] <- (sum(df_prob$new_status == 0) - sum(df_ass$new_status == 0))/gam_cases[timestep]
  print(paste0("w is ", w[timestep]))

  if(!is.finite(w[timestep])){
    w[timestep] <- 0
  }

  if(timestep > 1 & timestep <= seed_days & seed_cases == TRUE){
    df_ass <- rank_assign(df = df_prob, daily_case = gam_cases[timestep])
    print(paste0((sum(df_prob$new_status == 0) - sum(df_ass$new_status == 0))," cases reassigned"))
  }


  if((rank_assign == TRUE & seed_cases == FALSE) | (rank_assign == TRUE & seed_cases == TRUE & timestep > seed_days)){
    if(timestep > 1 & (w[timestep] <= 0.9 | w[timestep] >= 1.1)){
      df_ass <- rank_assign(df = df_prob, daily_case = gam_cases[timestep])
      print(paste0((sum(df_prob$new_status == 0) - sum(df_ass$new_status == 0))," cases reassigned"))
    }
  }

  df_inf <- infection_length(df = df_ass,
                             exposed_dist = exposed_dist,
                             exposed_mean = exposed_mean,
                             exposed_sd = exposed_sd,
                             presymp_dist = presymp_dist,
                             presymp_mean = presymp_mean,
                             presymp_sd = presymp_sd,
                             infection_dist = infection_dist,
                             infection_mean =  infection_mean,
                             infection_sd = infection_sd,
                             asymp_rate = asymp_rate)

  print("infection and recovery lengths assigned")

  df_rem <- removed_age(df_inf)
  print("individuals removed")

  df_rec <- recalc_sympdays(df_rem)
  print("updating infection lengths")

  df_msoa <- df_rec #area_cov(df = df_rec, area = area, hid = hid)

  df_out <- data.frame(area=df_msoa$area,
                       ID=df_msoa$id,
                       house_id=df_msoa$house_id,
                       disease_status=df_msoa$new_status,
                       exposed_days = df_msoa$exposed_days,
                       presymp_days = df_msoa$presymp_days,
                       symp_days = df_msoa$symp_days)

  if(output_switch){write.csv(df_out, paste0(tmp.dir, "/daily_out_", timestep, ".csv"))}

  return(df_out)
}

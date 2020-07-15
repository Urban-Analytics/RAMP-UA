library(tidyr)
library(readr)
library(mixdist)
library(dplyr)

#source("R/py_int/covid_status_functions.R")
#source("R/py_int/initialize_and_helper_functions.R")
 
gam_cases <- readRDS(paste0(getwd(),"/gam_fitted_PHE_cases.RDS"))

w <- NULL
nick_cases <- NULL
run_status <- function(pop, timestep=1, current_risk_beta = 0.0042, sympt_length = 14) {
  
  output_switch <- FALSE
  rank_assign <- FALSE
  seed_cases <- TRUE
  seed_days <- 10
  risk_cap <- 5 #set to NA or omit if no cap

  print(paste("R timestep:", timestep))

  if(timestep==1) {
    tmp.dir <<- paste(getwd(),"/output/",Sys.time(),sep="")
    if(!dir.exists(tmp.dir)){
      dir.create(tmp.dir, recursive = TRUE)
    }
  }

  df_cr_in <-create_input(micro_sim_pop  = pop,
                          vars = c("area",   # must match columns in the population data.frame
                                   "house_id",
                                   "id",
                                   "current_risk"))

  other_betas <- list(current_risk = current_risk_beta)
  
  df_msoa <- df_cr_in
  
  #### seeding the first day in high risk MSOAs
  if(timestep==1){
    msoas <- read.csv(paste0(getwd(),"/msoa_danger_fn.csv"))
    msoas <- msoas[msoas$risk == "High",]
    pop_hr <- pop %>% filter(area %in% msoas$area & pnothome > 0.3)
    seeds <- sample(1:nrow(pop_hr), size = gam_cases[timestep])
    seeds_id <- pop_hr$id[seeds]
    df_msoa$new_status[df_msoa$id %in% seeds_id] <- 1
    print("First day seeded")
  }
  
  df_prob <- covid_prob(df = df_msoa,
                        betas = other_betas,
                        risk_cap_val=risk_cap)
  
  print("probabilities calculated")
  
  if(timestep > 1){
    df_ass <- case_assign(df = df_prob,
                          tmp.dir=tmp.dir, 
                          save_output = output_switch)
  } else {
    df_ass <- df_prob
  }
  
  print("cases assigned")
  print(paste0("PHE cases ", gam_cases[timestep]))
  
  nick_cases[timestep] <- (sum(df_prob$new_status == 0) - sum(df_ass$new_status == 0))
  print(paste0("model cases ", nick_cases[timestep]))
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
                             presymp_dist = "weibull",
                             presymp_mean = 6.4,
                             presymp_sd = 2.3,
                             infection_dist = "normal",
                             infection_mean =  sympt_length,
                             infection_sd = 2,
                             timestep = timestep,
                             tmp.dir=tmp.dir,
                             save_output = output_switch)
  print("infection and recovery lengths assigned")
  
  df_rec <- removed(df = df_inf, chance_recovery = 0.95)
  print("recoveries and deaths assigned")
  
  df_msoa <- df_rec #area_cov(df = df_rec, area = area, hid = hid)
  
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

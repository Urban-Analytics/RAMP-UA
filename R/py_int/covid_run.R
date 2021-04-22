####################################################################################################################
####################################################################################################################
####################################################################################################################
######################## R counterpart to the microsim/main.py spatial interaction model.   ########################       
######################## This code converts 'risk' received from microsim/main.py into a    ########################       
######################## probability of being infected with COVID and then uses a Bernoulli ########################       
######################## draw to assign COVID cases. This code also assigns timings to each ########################       
######################## disease stage (SEIR) for each individual; and  a mortality risk to ########################       
######################## each individual based on age and health variables.                 ########################  
####################################################################################################################
####################################################################################################################
####################################################################################################################

#############################################################
################# Loading packages and data ################# 
#############################################################

load_rpackages <- function() {
  list.of.packages <- c("rampuaR")
  new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

  if(length(new.packages)) devtools::install_github("Urban-Analytics/rampuaR", dependencies = F)

  library(rvcheck)

  rampr_version <- check_github("Urban-Analytics/rampuaR")
  if(!rampr_version$up_to_date) devtools::install_github("Urban-Analytics/rampuaR", dependencies = F) #If there is a newer version of rampuaR package available this will download it

  library(tidyr)
  library(readr)
  library(mixdist)
  library(dplyr)
  library(rampuaR) #Contains all the key functions used below.
  library(withr)
}

load_init_data <- function() {
  #both msoas and gam_cases are stored in the rampuaR package
  data(gam_cases) #Estimated number of infections by day, based on smoothed PHE data. In the first outbreak PHE data estimated to only make up about 5% of total cases (NHS pers comms), so we amplify PHE cases by 20 and smooth.
  data(msoas) #List of Devon MSOAs and their risk level, cases will be seeded in High risk MSOAS. High risk MSOAs are those which are well connected by train and plane, high pop density and rich 
  w <<- NULL
  model_cases <<- NULL
}

initialize_r <- function() {
  load_rpackages()
  load_init_data()
}

run_status <- function(pop,
                       timestep = 1,
                       rep = NULL,
                       current_risk_beta = 0.008,
                       risk_cap = NA,
                       seed_days = 10,
                       exposed_dist = "weibull",
                       exposed_mean = 2.56,
                       exposed_sd = 0.72,
                       presymp_dist = "weibull",
                       presymp_mean = 2.3,
                       presymp_sd = 0.35,
                       infection_dist = "lognormal",
                       infection_mean =  18,
                       infection_sd = 1.1,
                       output_switch = TRUE,
                       rank_assign = FALSE,
                       local_outbreak_timestep = 0,
                       local_outbreak = FALSE,
                       msoa_infect="E02004152",
                       number_people_local=100,
                       local_prob_increase=0.75,
                       overweight_sympt_mplier = 1.46,
                       overweight = 1,
                       obesity_30 = 1,
                       obesity_35 = 1.4,
                       obesity_40 = 1.9,
                       cvd = 1,
                       diabetes = 1,
                       bloodpressure = 1,
                       improve_health = FALSE,
                       set_seed = TRUE) {

  # option to set the seed of the model run, but resolved to be too complicated to actually set the seed due to different pathways each indidividual can take. Redundant at this stage
  if(set_seed == TRUE) {
    seed <- rep
    set.seed(seed)
  } else {
    seed <- NULL
  }
  
  print(paste0("the seed number is ",seed))

  seed_cases <- ifelse(seed_days > 0, TRUE, FALSE)
  
  print(paste("R timestep:", timestep))
  
  ## Creating a temp directory to store the output.
  if(timestep==1) {
    # windows does not allow colons in folder names so substitute sys.time() to hyphen
    tmp.dir <<- paste0('R/py_int', "/output/", gsub(":","-", gsub(" ","-",Sys.time())))
    
    if(!dir.exists(tmp.dir)){
      dir.create(tmp.dir, recursive = TRUE)
    }
  }
 
  #If true the population gets switched to a healthier version, according to BMI. There are three obese classes, 
  # Obese I, Obese II and Obese III. In the healthier population, individuals in these classes move down a level, 
  # e.g. Obese III becomes Obese II etc. Obese I move to Overweight class. Currently obesity is associated with 
  # increased mortality risk, so we expect fewer deaths in a healthier population. 
  
  if(improve_health == TRUE){
    pop$BMIvg6  <- pop$BMI_healthier
  }
 
  if(output_switch){write.csv(pop, paste0( tmp.dir,"/daily_", timestep, ".csv"))} # Saves a copy of the population at the start of each day - quite heavy but good for understanding what is going.  
  
  ### Function for formatting the data for use in the rest of the code. There is much more data available for each individual, I think this needs to be passed in in the python code.
  
  df_cr_in <- rampuaR::create_input(micro_sim_pop  = pop,
                                    vars = c("area",   # must match columns in the population data.frame
                                             "house_id",
                                             "id",
                                             "current_risk",
                                             "BMIvg6",
                                             "cvd",
                                             "diabetes",
                                             "bloodpressure"))
  
  
 ### Calculating the mortality risk for each individual. Mortality risk is essentially based on age (which is baked in in the rampuaR::mortality_risk() function. 
 ### Values here essentially multiply this mortality risk, so obesity_40 = 1.48 meaning that individuals with a BMI > 40 would have their mortality risk increased by 48%.
 ### This would mean older and BMI > 40 individuals have the highest mortality risk.
  
  df_msoa <- rampuaR::mortality_risk(df = df_cr_in, 
                                     obesity_40 = obesity_40,
                                     obesity_35 = obesity_35,
                                     obesity_30 = obesity_30,
                                     overweight = overweight,
                                     cvd = cvd,
                                     diabetes = diabetes,
                                     bloodpressure = bloodpressure)
  
  ### Similarly to above, some individuals are more likely to be symptomatic if infected. This is based on age which is baked into the rampuaR::sympt_risk() function.
  ### Older individuals are more likely to be symptomatic. Values here multiply that risk, e.g. overweight individuals are 46% more likely to be symptomatic. 
  
  
  df_msoa <- rampuaR::sympt_risk(df = df_msoa,
                                 overweight_sympt_mplier = 1.46,
                                 cvd = NULL,
                                 diabetes = NULL,
                                 bloodpressure = NULL)

  
  
  ### On the first day we seed cases in high risk MSOAS in individuals which spend at least 30% of their time outside their home. 
  
  if(timestep==1){
    msoas <- msoas[msoas$risk == "High",]
    pop_hr <- pop %>% filter(area %in% msoas$area & pnothome > 0.3)
    seeds <- withr::with_seed(seed, sample(1:nrow(pop_hr), size = gam_cases[timestep]))
    seeds_id <- pop_hr$id[seeds]
    df_msoa$new_status[df_msoa$id %in% seeds_id] <- 1  #to be exposed their 'new_status' is changed to 1 (from 0)
    print("First day seeded")
  }
  
  
  ### Previously there were other factors that affected a persons 'betas' or their risk of being infected, such as age and gender. 
  #Now we keep it purely down to activities in the spatial interaction model - current_risk.
  other_betas <- list(current_risk = current_risk_beta)
  
  ### This just sums up the betas, was more important when there were other factors and an intercept, which we no longer use. 
  ### We also toyed with capping the risk as we were previously getting some very large values due to the way workplaces were set up.
  ### But we tend to not use this cap anymore.
  df_sum_betas <- rampuaR::sum_betas(df = df_msoa,
                                     betas = other_betas,
                                     risk_cap_val = risk_cap)
  print("betas calculated")
  
  # df_prob <- covid_prob(df = df_sum_betas)
  ### This converts the summed up betas (aka the current_risk) into a probability of being infected, so has to be a value between 0-1. 
  df_prob <- rampuaR::infection_prob(df = df_sum_betas, dt = 1)
  
  print("probabilities calculated")
  
  ### An optional function for implementing a local outbreak of a given size in a given MSOA, at a given timestep.
  if(local_outbreak == TRUE & timestep == local_outbreak_timestep){
    print("Local outbreak - super spreader event!")
    df_prob <- rampuaR::local_outbreak(df=df_prob,
                                       msoa_infect=msoa_infect,
                                       number_people=number_people_local,
                                       risk_prob=local_prob_increase)
  }
  
  ### In this stage we carry out a bernoulli draw for each susceptible individual, 
  ### with their infection probability giving them a probability of being infected with COVID.
  ### After this function a given number of people will have their new_status set to 1, indicating they are infected with COVID and are in the exposed stage.

  if(timestep > 1){
    df_ass <- rampuaR::case_assign(df = df_prob,
                                   tmp.dir=tmp.dir,
                                   save_output = output_switch,
                                   seed = seed)
  } else {
    df_ass <- df_prob
  }
  
  ### just some print out sanity checks
  
  print("cases assigned")
  print(paste0("PHE cases ", gam_cases[timestep]))
  
  model_cases[timestep] <- (sum(df_prob$new_status == 0) - sum(df_ass$new_status == 0))
  print(paste0("model cases ", model_cases[timestep]))
  print(paste0("Adjusted PHE cases ", gam_cases[timestep]))
  
  #### For seeding cases after day 1 we rank individuals by the amount of risk they have and assign a given number of cases (from gam_cases) based on this.
  #### We do it this way so cases don't just stay in the high risk MSOAs for the seeding period.
  
  if(timestep > 1 & timestep <= seed_days & seed_cases == TRUE){
    df_ass <- rank_assign(df = df_prob, daily_case = gam_cases[timestep], seed = seed)
    print(paste0((sum(df_prob$new_status == 0) - sum(df_ass$new_status == 0))," cases reassigned"))
  }
  
  #### It is possible to rank assign cases for the whole model (rather than bernoulli trial approach) but this isn't really what we want, 
  #### these few lines are redundant.
  if((rank_assign == TRUE & seed_cases == FALSE) | (rank_assign == TRUE & seed_cases == TRUE & timestep > seed_days)){
    if(timestep > 1 & (w[timestep] <= 0.9 | w[timestep] >= 1.1)){
      df_ass <- rank_assign(df = df_prob, daily_case = gam_cases[timestep],seed = seed)
      print(paste0((sum(df_prob$new_status == 0) - sum(df_ass$new_status == 0))," cases reassigned"))
    }
  }
  
  #### This function assigns newly infected individuals the number of days they will be in each disease status. 
  #### The shape of the distributions these time periods are taken from can be altered with this function
  df_inf <- rampuaR::infection_length(df = df_ass,
                                      exposed_dist = exposed_dist,
                                      exposed_mean = exposed_mean,
                                      exposed_sd = exposed_sd,
                                      presymp_dist = presymp_dist,
                                      presymp_mean = presymp_mean,
                                      presymp_sd = presymp_sd,
                                      infection_dist = infection_dist,
                                      infection_mean =  infection_mean,
                                      infection_sd = infection_sd,
                                      seed = seed)
  
  print("infection and recovery lengths assigned")
  
  
  #### In this functions, individuals which have reached the end of their 
  #### symptomatic duration will either recover or die based on their mortality risk, through another bernoulli draw.
  df_rem <- rampuaR::removed_age(df_inf, seed = seed)
  print("individuals removed")
  
  
  #### Here the duration of the current disease status is reduced by one day/ individuals move on to their next disease stage.
  df_rec <- rampuaR::recalc_sympdays(df_rem)
  print("updating infection lengths")
  
  df_msoa <- df_rec #area_cov(df = df_rec, area = area, hid = hid)
  
  #### The output is formattted for passing back into the python spatial interaction model.
  df_out <- data.frame(area=df_msoa$area,
                       ID=df_msoa$id,
                       house_id=df_msoa$house_id,
                       disease_status=df_msoa$new_status,
                       exposed_days = df_msoa$exposed_days,
                       presymp_days = df_msoa$presymp_days,
                       symp_days = df_msoa$symp_days)
  
  #if(output_switch){write.csv(df_out, paste0(tmp.dir, "/daily_out_", timestep, ".csv"))}
  return(df_out)
}

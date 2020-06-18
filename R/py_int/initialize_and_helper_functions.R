
##################################
# Functions for Covid simulation code
# Jesse F. Abrams and Fiona Spooner
##################################
##################################


create_input <- function(micro_sim_pop, num_sample,vars = NULL, lockdown_date = NULL,  pnothome_multiplier = 1){
  
  if(!all(vars %in% colnames(micro_sim_pop))){
    print(paste0(vars[!vars %in% colnames(micro_sim_pop)], " not in population column names"))
  }
  
  var_list <- list()
  for (i in 1:length(vars)){
    var_list[[i]] <- micro_sim_pop[,vars[i]]
  }
  
  names(var_list) <- vars
  
  #micro_sim_pop$presymp_days[micro_sim_pop$presymp_days==-1] <- NA
  #micro_sim_pop$symp_days[micro_sim_pop$symp_days==-1] <- NA
  
  constant_list <- list(
    beta0 = rep(0, num_sample),
    betaxs = rep(0, num_sample),
    #new_beta0 = rep(0, num_sample),
    hid_status = rep(0, num_sample),
    presymp_days = micro_sim_pop$presymp_days,
    symp_days = micro_sim_pop$symp_days,
    probability = rep(0, num_sample),
    #optim_probability = matrix(0, nrow = num_sample),
    status = micro_sim_pop$disease_status,
    new_status = micro_sim_pop$disease_status
  )
  
  df <- c(var_list, constant_list)
  
  df$pnothome <- df$pnothome
  
  if(!is.null(lockdown_date)){
    li <- as.numeric(lockdown_date - start_date)
    df$pnothome[, (li+1)] <-  df$pnothome[,1] * pnothome_multiplier
  }
  
  return(df)
}


beta_make <- function(name, betas, df) {
 
  #dynamic betas with linear multiplier
  if (is.matrix(df[[name]]) & length(betas[[name]])== 1){  #dynamic betas will be a matrix from the create_input function
    y <- df[[name]] * betas[[name]]
  } 
  #fixed betas with linear multiplier
  if(!is.matrix(df[[name]]) & length(betas[[name]]) == 1){
    y <- df[[name]] * betas[[name]]
  }
  #dynamic betas with class multiplier 
  if(is.matrix(df[[name]]) & length(betas[[name]]) > 1){

    classes <- names(betas[[name]])
    y<-numeric(length(df[[name]]))
    for (cls in classes){
      y[which(df[[name]]==cls)] <- as.numeric(as.character(betas[[name]][[cls]])) 
    }
  }
 
  #fixed betas with class multiplier 
 if(!is.matrix(df[[name]]) & length(betas[[name]]) > 1){
    
   classes <- names(betas[[name]])
   y<-numeric(length(df[[name]]))
   for (cls in classes){
     y[which(df[[name]]==cls)] <- as.numeric(as.character(betas[[name]][[cls]])) 
   }
  }
  return(y)
}


#########################################
#### for age sex betas - needs making more listy
as_betas_devon <- function(population_sample,id, age, sex, beta0_fixed = NULL, divider = 1){
  
  if (length(unique(population_sample$age1)) == 6){
    
    fixed_risks <- data.frame(id = population_sample[[id]],
                              age=population_sample[[age]],
                              sex=population_sample[[sex]],
                              beta0 = beta0_fixed,#-7.8100663,
                              age_risk = 0,
                              sex_risk = 0,
                              tot_risk=0)
    
    # check that this is correct. males should be more likely
    fixed_risks$sex_risk[fixed_risks$sex %in% c(0)] <- 0.1297575 
    fixed_risks$sex_risk[fixed_risks$sex %in% c(1)] <- 0
    
    fixed_risks$age_risk[fixed_risks$age %in% c(1)] <- -2.2464676 #0-18
    fixed_risks$age_risk[fixed_risks$age %in% c(2)] <- -0.7057833 #19-29
    fixed_risks$age_risk[fixed_risks$age %in% c(3)] <- 0 #30-44
    fixed_risks$age_risk[fixed_risks$age %in% c(4)] <- 0.3134440 ##45-59
    fixed_risks$age_risk[fixed_risks$age %in% c(5)] <- 0.1095025  #60-74
    fixed_risks$age_risk[fixed_risks$age %in% c(6)] <- 0.8089118 #75+
   
  }
  
  if (length(unique(population_sample$age1)) == 21){
    # we can define fixed risks here for things like age and sex because they won't change
    fixed_risks <- data.frame(id = population_sample[[id]],
                              age=population_sample[[age]],
                              sex=population_sample[[sex]],
                              beta0 = -8.79806350,
                              age_risk = 0,
                              sex_risk = 0,
                              tot_risk=0)
    
    fixed_risks$sex_risk[fixed_risks$sex %in% c(0)] <- 0.06342251
    fixed_risks$sex_risk[fixed_risks$sex %in% c(1)] <- 0
    
    fixed_risks$age_risk[fixed_risks$age %in% c(1,2)] <- -2.40598996 #uk_age$`0 to 4`/(sum(pop_proportion[1:2]))
    fixed_risks$age_risk[fixed_risks$age %in% c(3)] <- -3.53175768 #uk_age$`5 to 9`/(sum(pop_proportion[3:4]))
    fixed_risks$age_risk[fixed_risks$age %in% c(4)] <- -3.53175768  #uk_age$`5 to 9`/(sum(pop_proportion[3:4]))
    fixed_risks$age_risk[fixed_risks$age %in% c(5)] <- -3.22625283  #uk_age$`10 to 14`/(sum(pop_proportion[5:6]))
    fixed_risks$age_risk[fixed_risks$age %in% c(6)] <- -3.22625283  #uk_age$`10 to 14`/(sum(pop_proportion[5:6]))
    fixed_risks$age_risk[fixed_risks$age %in% c(7)] <- -2.25605530 #uk_age$`15 to 19`/pop_proportion[7]
    fixed_risks$age_risk[fixed_risks$age %in% c(8)] <- -0.85396607 #uk_age$`20 to 24`/pop_proportion[8]
    fixed_risks$age_risk[fixed_risks$age %in% c(9)] <- -0.03909346 #uk_age$`25 to 29`/pop_proportion[9]
    fixed_risks$age_risk[fixed_risks$age %in% c(10)] <- 0 #uk_age$`30 to 34`/pop_proportion[10]
    fixed_risks$age_risk[fixed_risks$age %in% c(11)] <- -0.10794990 #uk_age$`35 to 39`/pop_proportion[11]
    fixed_risks$age_risk[fixed_risks$age %in% c(12)] <- 0.01160390 #uk_age$`40 to 44`/pop_proportion[12]
    fixed_risks$age_risk[fixed_risks$age %in% c(13)] <- 0.22233288 #uk_age$`45 to 49`/pop_proportion[13]
    fixed_risks$age_risk[fixed_risks$age %in% c(14)] <- 0.34896150  #uk_age$`50 to 54`/pop_proportion[14]
    fixed_risks$age_risk[fixed_risks$age %in% c(15)] <- 0.32741268  #uk_age$`55 to 59`/pop_proportion[15]
    fixed_risks$age_risk[fixed_risks$age %in% c(16)] <- 0.15543211 #uk_age$`60 to 64`/pop_proportion[16]
    fixed_risks$age_risk[fixed_risks$age %in% c(17)] <- -0.02426055 #uk_age$`65 to 69`/pop_proportion[17]
    fixed_risks$age_risk[fixed_risks$age %in% c(18)] <- 0.17828771 #uk_age$`70 to 74`/pop_proportion[18]
    fixed_risks$age_risk[fixed_risks$age %in% c(19)] <- 0.35265420  #uk_age$`75 to 79`/pop_proportion[19]
    fixed_risks$age_risk[fixed_risks$age %in% c(20)] <- 0.52873351   #uk_age$`80 to 84`/pop_proportion[20]
    fixed_risks$age_risk[fixed_risks$age %in% c(21)] <- 1.12327274 #uk_age$`20`/pop_proportion[21]
  
  }
  
  fixed_risks$as_risk <-  (fixed_risks$sex_risk/divider) + (fixed_risks$age_risk/divider)
  
  fixed_risks$tot_risk <- fixed_risks$beta0 + fixed_risks$sex_risk + fixed_risks$age_risk
  fixed_risks$prob_case <- exp(fixed_risks$tot_risk)/(1+exp(fixed_risks$tot_risk))
  
  population_sample$beta0 <- fixed_risks$beta0
  population_sample$as_risk <- fixed_risks$as_risk
  
  return(population_sample)
}


#########################################
beta0_optim <- function(beta0new, n, betaX, Y){ 
  tmp_mu <-  tmp_prob <- rep(NA, n)
  for (i in 1:n) {
    tmp_mu[i] <- beta0new + betaX[i]
    tmp_prob[i] <- exp(tmp_mu[i])/(1+exp(tmp_mu[i]))
  }
  tmp_sum <- abs(Y-sum(tmp_prob))
  tmp_sum
}


#########################################
new_beta0_probs <- function(df,timestep, daily_case){
  
  susceptible <- which(df$status == 0)
  
  new_beta0 <- optim(par = -1, beta0_optim,  n = length(susceptible), 
                     betaX=df$betaxs[susceptible], Y=daily_case, 
                     method="Brent",  lower  =-30, upper = 0)$par
  
  df$new_beta0 <- new_beta0
  tot_risk_new <- df$new_beta0  + df$betaxs
  df$optim_probability <- exp(tot_risk_new)/(1+exp(tot_risk_new))

  case_YN <- rbinom(n=length(df$optim_probability[susceptible]), size=1, 
                    prob =   df$optim_probability[susceptible])
  print(paste0("optim cases ",sum(case_YN)))
  
  return(df)
}


normalizer <- function(x ,lower_bound, upper_bound, xmin, xmax){
  normx <-  (upper_bound - lower_bound)*(x - xmin)/(xmax-xmin) + lower_bound
  return(normx)
}



#### a function to output a dataframe to feedback to Nick
#format_df <- function(df, timestep, area, hid, pid ){
#  
#  PID <- df[[pid]]
#  Area <- df[[area]]
#  HID <- df[[hid]]
#  MSOA_Cases <- df$msoa_infected[,timestep]
#  HID_Cases <- df$hid_infected[,timestep]
#  
#  ds_df <- data.frame(susceptible = df$susceptible[,timestep], 
#                      exposed = 0, 
#                      presymptomatic_infected = df$presymp[,timestep],
#                      symptomatic_infected = df$symp[,timestep],
#                      recovered =  df$recovered[,timestep],
#                      died = df$died[,timestep],
#                      hid_cases = df$hid_infected[,timestep]) 
#                      msoa_cases = df$msoa_infected[,timestep],
  
#  Disease_Status <- ds_df %>%
#    dplyr::mutate(
#      Disease_Status = case_when(
#        susceptible == 1 ~ 0,
#        exposed == 1 ~ 1,
#        presymptomatic_infected == 1 ~ 2,
#        symptomatic_infected ==1 ~ 3,
#        recovered == 1 |
#          died == 1 ~ 4
#      )
#    ) %>%
#    dplyr::select(Disease_Status)
#  
#  Days_With_Status <- 0
#  Current_Risk <- df$current_risk[,timestep-1]
#  
#  df_sum <- data.frame(Area, HID,PID,Disease_Status, Days_With_Status,
#                       Current_Risk, MSOA_Cases, HID_Cases)
#  return(df_sum)
#  
#}


getUKCovidTimeseries <- function (){
  UKregional = readr::read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTzgWD-_-vxj7ljb-iYPckgtV4ctg-SjVJWQzrjwj0CWF2JE9uyLSUwtQFZal3Cdqf-5Mch-_sBPBv2/pub?gid=163112336&single=true&output=csv", 
                               col_types = readr::cols(date = readr::col_date(format = "%Y-%m-%d")))
  UKregional$uk_cumulative_cases[72:nrow(UKregional)] <- UKregional$england_cumulative_cases[72:nrow(UKregional)] + UKregional$scotland_cumulative_cases[72:nrow(UKregional)] + UKregional$wales_cumulative_cases[72:nrow(UKregional)] + UKregional$northern_ireland_cumulative_cases[72:nrow(UKregional)]
  englandNHS = readr::read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTzgWD-_-vxj7ljb-iYPckgtV4ctg-SjVJWQzrjwj0CWF2JE9uyLSUwtQFZal3Cdqf-5Mch-_sBPBv2/pub?gid=0&single=true&output=csv", 
                               col_types = readr::cols(date = readr::col_date(format = "%Y-%m-%d")))
  scotlandHealthBoard = readr::read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQod-HdDk4Nl8BFcunG5P-QA2CuKdIXCfK53HJDxcsaYlOov4FFc-yQciJyQFrqX5_n_ixz56S7uNBh/pub?gid=490497042&single=true&output=csv")
  walesHealthBoard = readr::read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQod-HdDk4Nl8BFcunG5P-QA2CuKdIXCfK53HJDxcsaYlOov4FFc-yQciJyQFrqX5_n_ixz56S7uNBh/pub?gid=762770891&single=true&output=csv")
  northernIreland = readr::read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQod-HdDk4Nl8BFcunG5P-QA2CuKdIXCfK53HJDxcsaYlOov4FFc-yQciJyQFrqX5_n_ixz56S7uNBh/pub?gid=1217212942&single=true&output=csv")
  englandUnitAuth = readr::read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTzgWD-_-vxj7ljb-iYPckgtV4ctg-SjVJWQzrjwj0CWF2JE9uyLSUwtQFZal3Cdqf-5Mch-_sBPBv2/pub?gid=796246456&single=true&output=csv")
  englandUnitAuth2NHSregion = readr::read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQod-HdDk4Nl8BFcunG5P-QA2CuKdIXCfK53HJDxcsaYlOov4FFc-yQciJyQFrqX5_n_ixz56S7uNBh/pub?gid=1933702254&single=true&output=csv")
  tmp = englandUnitAuth %>% tidyr::pivot_longer(cols = starts_with("20"), 
                                                names_to = "date", values_to = "cumulative_cases") %>% 
    mutate(date = as.Date(as.character(date), "%Y-%m-%d"))
  tmp = tmp %>% left_join(UKregional %>% select(date, daily_total = england_cumulative_cases), 
                          by = "date")
  tidyEnglandUnitAuth = tmp %>% group_by(date) %>% mutate(daily_unknown = daily_total - 
                                                            sum(cumulative_cases, na.rm = TRUE)) %>% ungroup() %>% 
    group_by(CTYUA19CD, CTYUA19NM)
  tmp = englandNHS %>% tidyr::pivot_longer(cols = !date, names_to = "england_nhs_region", 
                                           values_to = "cumulative_cases")
  tmp = tmp %>% left_join(UKregional %>% select(date, daily_total = england_cumulative_cases), 
                          by = "date")
  tidyEnglandNHS = tmp %>% group_by(date) %>% mutate(daily_unknown = daily_total - 
                                                       sum(cumulative_cases, na.rm = TRUE)) %>% ungroup() %>% 
    group_by(england_nhs_region)
  tidyUKRegional = UKregional %>% select(date, england_cumulative_cases, 
                                         scotland_cumulative_cases, wales_cumulative_cases, northern_ireland_cumulative_cases, 
                                         daily_total = uk_cumulative_cases) %>% tidyr::pivot_longer(cols = ends_with("cumulative_cases"), 
                                                                                                    names_to = "uk_region", values_to = "cumulative_cases") %>% 
    filter(!is.na(cumulative_cases)) %>% mutate(uk_region = stringr::str_remove(uk_region, 
                                                                                "_cumulative_cases")) %>% mutate(uk_region = stringr::str_replace(uk_region, 
                                                                                                                                                  "_", " ")) %>% group_by(date) %>% mutate(daily_unknown = daily_total - 
                                                                                                                                                                                             sum(cumulative_cases, na.rm = TRUE)) %>% ungroup() %>% 
    group_by(uk_region)
  return(list(UKregional = UKregional, englandNHS = englandNHS, 
              englandUnitAuth = englandUnitAuth, scotlandHealthBoard = scotlandHealthBoard, 
              walesHealthBoard = walesHealthBoard, northernIrelandLocalGovernmentDistrict = northernIreland, 
              englandUnitAuth2NHSregion = englandUnitAuth2NHSregion, 
              tidyUKRegional = tidyUKRegional, tidyEnglandNHS = tidyEnglandNHS, 
              tidyEnglandUnitAuth = tidyEnglandUnitAuth))
}


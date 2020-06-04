
##################################
# Functions for Covid simulation code
# Jesse F. Abrams and Fiona Spooner
##################################
##################################
# There are three main functions that 
# 1. calculate covid probability
# 2. assign covid and take a random draw to determine how long a person is infectius
# 3. take random draw to determine length of sickness and whether the person recovers or dies at the end
##################################


pull_pop <- function(data_dir="data") {
  population <- pop_init(data_dir)
  return(population)
}

create_input <- function(micro_sim_pop,num_sample=10000, start_date = as.Date("2020-03-07 "), end_date = as.Date("2020-04-12"), fixed_vars = NULL, dynamic_vars = NULL, lockdown_date = NULL,  pnothome_multiplier = 1){
  

  if(is.character(start_date)){ start_date <- as.Date(start_date)}
  
  if(is.character(end_date)){ end_date <- as.Date(end_date)}
  
  num_days <- as.numeric(end_date - start_date)
  
  population_sample <- sample_n(micro_sim_pop, size=num_sample, replace=FALSE)
  
  if(!all(fixed_vars %in% colnames(population_sample))){
    print(paste0(fixed_vars[!fixed_vars %in% colnames(population_sample)], " not in population column names"))
  }
  
  fixed_list <- list()
  for (i in 1:length(fixed_vars)){
    #print(i)
    #fixed_list[[i]] <- pull(population_sample[,fixed_vars[i]])
    fixed_list[[i]] <- population_sample[,fixed_vars[i]]
    
  }
  
  names(fixed_list) <- fixed_vars
  
  if (length(dynamic_vars) >0){
    dynamic_list <- list()
    for (i in 1:length(dynamic_vars)){
      dynamic_list[[i]] <- matrix(0, nrow = num_sample, ncol = num_days)
      #dynamic_list[[i]][,1] <- pull(population_sample[,dynamic_vars[i]])
      dynamic_list[[i]][,1] <- population_sample[,dynamic_vars[i]]
    }
    names(dynamic_list) <- dynamic_vars
  } else {
    dynamic_list <- NULL
  }
   
  constant_list <- list(
    beta0 = matrix(0, nrow = num_sample, ncol = num_days),
    betaxs = matrix(0, nrow = num_sample, ncol = num_days),
    new_beta0 = matrix(0, nrow = num_sample, ncol = num_days),
    #new_case = rep(0,num_people),
    hid_presymp = matrix(0, nrow = num_sample, ncol = num_days),
    hid_symp = matrix(0, nrow = num_sample, ncol = num_days),
    hid_infected = matrix(0, nrow = num_sample, ncol = num_days),
    #hid_index = matrix(0, nrow = num_sample, ncol = num_days),
    msoa_presymp = matrix(0, nrow = num_sample, ncol = num_days),
    msoa_symp = matrix(0, nrow = num_sample, ncol = num_days),
    msoa_infected = matrix(0, nrow = num_sample, ncol = num_days),
    msoa_index = matrix(0, nrow = num_sample, ncol = num_days),
    presymp_days = rep(0, num_sample),
    symp_days = rep(0, num_sample),
    probability = matrix(0, nrow = num_sample, ncol = num_days),
    optim_probability = matrix(0, nrow = num_sample, ncol = num_days),
    susceptible = matrix(1, nrow = num_sample, ncol = num_days),
    had_covid = matrix(0, nrow = num_sample, ncol = num_days),
    presymp = matrix(0, nrow = num_sample, ncol = num_days),
    symp = matrix(0, nrow = num_sample, ncol = num_days),
    recovered = matrix(0, nrow = num_sample, ncol = num_days),
    died = matrix(0, nrow = num_sample, ncol = num_days)
  )
  
  df <- c(fixed_list, dynamic_list, constant_list)
  
  df$pnothome[,1:num_days] <- df$pnothome[,1]
  
  if(!is.null(lockdown_date)){
    li <- as.numeric(lockdown_date - start_date)
    df$pnothome[, (li+1):num_days] <-  df$pnothome[,1] * pnothome_multiplier
  }
  
  return(df)
}


#############
## four types of betas we can make currently
## dynamic betas (can change every day) which multiply linearly e.g. no. people in msoa with covid
## fixed betas which multiply linearly e.g. connectedness
## dynamic betas where groups have specific betas - not sure we have any of these atm
## fixed betas where groups have specific betas e.g. origin
#############
beta_make <- function(name, betas, timestep, df) {
 
  #dynamic betas with linear multiplier
  if (is.matrix(df[[name]]) & length(betas[[name]])== 1){  #dynamic betas will be a matrix from the create_input function
    y <- df[[name]][,timestep] * betas[[name]]
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
      y[which(df[[name]][,timestep]==cls)] <- as.numeric(as.character(betas[[name]][[cls]])) 
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
# calculate the probability of becoming infect
# requires a dataframe list, a vector of betas, and a timestep
covid_prob <- function(df, betas, timestep, interaction_terms = NULL) {
  print("assign probabilities")
 # num_infected <- sum(df$infected[,(timestep-1)])
  
  #beta_names <- colnames(betas)
  beta_names <- names(betas)
  
  if (all(!beta_names %in% names(df))) {
    print(paste0(
      beta_names[!beta_names %in% names(df)],
      " missing from df. They are not included in probabilities."
    ))
  }
  
  beta_names <- beta_names[beta_names %in% names(df)]
  
  beta_out <- lapply(X = beta_names, FUN = beta_make, timestep = timestep-1, betas=betas,df=df)
  beta_out <- do.call(cbind, beta_out)
  colnames(beta_out) <- beta_names
  
  if (length(interaction_terms > 0 )){
    lpsi <- df$beta0 + df$as_risk + rowSums(beta_out) + apply(beta_out[,interaction_terms], 1, prod)
  } else{
    lpsi <- df$beta0 + df$as_risk + rowSums(beta_out)
  }
 
  psi <- exp(lpsi) / (exp(lpsi) + 1)
  psi[df$susceptible[,(timestep-1)] %in% 0] <- 0 # if they are not susceptible then their probability is 0 of getting it 
  psi[df$recovered[,(timestep-1)] %in% 1] <- 0 # if they are recovered then their probability is 0 of getting it 
  psi[df$died[,(timestep-1)] %in% 1] <- 0 # if they've died then their probability is 0 of getting it 
  psi[df$presymp[,(timestep-1)] %in% 1] <- 1 # this makes keeping track of who has it easier
  psi[df$symp[,(timestep-1)] %in% 1] <- 1 # this makes keeping track of who has it easier
  df$betaxs[,timestep] <- df$as_risk + rowSums(beta_out)
  df$probability[,timestep] <- psi
  return(df)
}

#########################################
# assigns covid based on probabilities
case_assign <- function(df,timestep, with_optimiser = FALSE) {
  print("assign cases")
  
  
  susceptible <- which(df$susceptible[,timestep] == 1)
  
  
  if (with_optimiser) {
    df$presymp[susceptible,timestep:ncol(df$presymp)] <- rbinom(n = length(susceptible),
                                                       size = 1,
                                                       prob = df$optim_probability[susceptible,timestep])
  } else{
    print("nop")
    df$presymp[susceptible,timestep:ncol(df$presymp)] <- rbinom(n = length(susceptible),
                                                                size = 1,
                                                                prob = df$probability[susceptible,timestep])
  }
 
  #df$had_covid[,timestep:ncol(df$had_covid)] <- df$infected[,timestep:ncol(df$infected)]
  
    
  df$had_covid[susceptible,timestep:ncol(df$had_covid)] <- df$presymp[susceptible,timestep:ncol(df$presymp)]
  
  new_cases <- which(df$had_covid[,timestep]-df$had_covid[,(timestep-1)]==1)
  if (length(new_cases)>0){
    df$susceptible[new_cases,timestep:ncol(df$susceptible)] <- 0
  }
  
  return(df)
}



#########################################
# calculate the infection length of new cases
infection_length <- function(df,  presymp_dist = "weibull",presymp_mean = NULL, presymp_sd = NULL,infection_dist = "normal", infection_mean = NULL, infection_sd = NULL,timestep){
  
  #new_cases <- which(df$infected == 1 & df$infection_days == 0)
  new_cases <- which(df$had_covid[,timestep]-df$had_covid[,(timestep-1)]==1)
  
  
  # current_cases <- which(df$infection_days > 0)
 
  if (presymp_dist == "weibull"){
    wpar <- mixdist::weibullpar(mu = presymp_mean, sigma = presymp_sd, loc = 0) 
    df$presymp_days[new_cases] <- round(rweibull(1:length(new_cases), shape = as.numeric(wpar["shape"]), scale = as.numeric(wpar["scale"])),) 
  }
  
  if (infection_dist == "normal"){
    df$symp_days[new_cases] <- round(rnorm(1:length(new_cases), mean = infection_mean, sd = infection_sd))
  }

 return(df)
 
}


#########################################
# determines if someone has been removed and if that removal is recovery or death
removed <- function(df, chance_recovery = 0.95,timestep){
  
  removed_cases <- which(df$presymp_days == 0 & df$symp_days == 1)
  df$recovered[removed_cases,timestep:ncol(df$recovered)] <- rbinom(n = length(removed_cases),
                                                                     size = 1,
                                                                     prob = chance_recovery)
 
 # df$died[removed_cases,timestep:ncol(df$died)] <- 1 - df$recovered[removed_cases,timestep]
  df$died[removed_cases,timestep:ncol(df$died)] <- 1 - df$recovered[removed_cases,timestep:ncol(df$recovered)]
  
  df$symp[removed_cases,timestep:ncol(df$symp)] <- 0
  df$symp_days[removed_cases] <- 0
  df$presymp_days[df$presymp_days>0] <- df$presymp_days[df$presymp_days>0] - 1
  df$symp_days[df$symp[,timestep] == 1 & df$symp_days > 0] <- df$symp_days[df$symp[,timestep] == 1 & df$symp_days>0] - 1
  
  #switching people from being pre symptomatic to symptomatic and infected
  becoming_sympt <- which(df$presymp[,timestep] == 1 & df$presymp_days == 0)
  df$presymp[becoming_sympt,timestep:ncol(df$presymp)] <- 0
  df$symp[becoming_sympt,timestep:ncol(df$symp)]<- 1
  
  
  return(df)
}



#########################################

# separates by msoa and household
area_cov <- function(df, timestep, area, hid){
  
  df_msoa_hid <- data.frame(msoa = df[[area]],hid = df[[hid]],pop_dens_km2 = df$pop_dens_km2,msoa_area = df$msoa_area,symp =  df$symp[,timestep], presymp = df$presymp[,timestep])
  
  
  df_gr <-  data.frame(df_msoa_hid %>%
                         group_by(msoa) %>%
                         add_count(msoa, name = "msoa_size") %>%
                         mutate(
                           msoa_presymp = sum(presymp == 1),
                           msoa_symp = sum(symp == 1),
                           msoa_infected = msoa_presymp + msoa_symp
                           # perc_covid_msoa = msoa_infected / msoa_size
                         ) #%>% 
                         #ungroup() %>% 
                         #group_by(msoa, hid) %>% 
                        #         add_count(hid, name = "hid_size") %>%
                        # mutate(
                        #   hid_presymp = sum(presymp == 1),
                        #   hid_symp = sum(symp == 1),
                        #   hid_infected = hid_presymp + hid_symp
                           #    perc_covid_hid = hid_infected / hid_size
                        # ) %>% 
                        # ungroup())
                       )
 #df$hid_presymp[,timestep] <- df_gr$hid_presymp
# df$hid_infected[, timestep] <- df_gr$hid_infected
# df$hid_symp[,timestep] <- df_gr$hid_symp
  df$msoa_presymp[,timestep] <- df_gr$msoa_presymp
  df$msoa_symp[,timestep] <- df_gr$msoa_symp
  df$msoa_infected[, timestep] <- df_gr$msoa_infected
  
  if("cases_per_area" %in% colnames(df)){
    df$cases_per_area[,timestep] <- log10(df_gr$msoa_infected)/df_gr$msoa_area
  }
 # df$msoa_index[, timestep] <- normalizer(df_gr$msoa_infected, 0, 1, 0, df_gr$msoa_size*0.02) #will probs need to change this to link to prevalence var current 2% of mean msoa_pop
 # df$cases_per_area[,timestep] <- df_gr$msoa_infected/df_gr$msoa_size
  
  
  #df$cases_per_area[,timestep] <- (df_gr$msoa_infected/df_gr$msoa_size)/df_gr$msoa_area
  #df$hid_index[,timestep] <- normalizer(df_gr$hid_infected, 0, 1, 0, df_gr$hid_size)
  return(df)
}


#########################################
#### for age sex betas - needs making more listy
as_betas_devon <- function(population_sample,pid, age, sex, beta0_fixed = NULL, divider = 1){
  
  
  if (length(unique(population_sample$age1)) == 6){
    
    fixed_risks <- data.frame(pid = population_sample[[pid]],
                              age=population_sample[[age]],
                              sex=population_sample[[sex]],
                              beta0 = beta0_fixed,#-7.8100663,
                              age_risk = 0,
                              sex_risk = 0,
                              tot_risk=0)
    
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
  fixed_risks <- data.frame(pid = population_sample[[pid]],
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
  
  susceptible <- which(df$susceptible[,timestep] == 1)
  
  new_beta0 <- optim(par = -1, beta0_optim,  n = length(susceptible), betaX=df$betaxs[susceptible,timestep], Y=daily_case, 
                     method="Brent",  lower  =-30, upper = 0)$par
  
  df$new_beta0[,timestep] <- new_beta0
  tot_risk_new <- df$new_beta0[,timestep]  + df$betaxs[,timestep]
  df$optim_probability[,timestep] <- exp(tot_risk_new)/(1+exp(tot_risk_new))

  case_YN <- rbinom(n=length(df$optim_probability[susceptible,timestep]), size=1, prob =   df$optim_probability[susceptible,timestep])
  print(paste0("optim cases ",sum(case_YN)))
  
  return(df)
}


#### a function to output a dataframe to feedback to Nick


format_df <- function(df, timestep, area, hid, pid ){
  
  PID <- df[[pid]]
  Area <- df[[area]]
  HID <- df[[hid]]
  MSOA_Cases <- df$msoa_infected[,timestep]
  HID_Cases <- df$hid_infected[,timestep]
  
  ds_df <- data.frame(susceptible = df$susceptible[,timestep], 
                      exposed = 0, 
                      presymptomatic_infected = df$presymp[,timestep],
                      symptomatic_infected = df$symp[,timestep],
                      recovered =  df$recovered[,timestep],
                      died = df$died[,timestep],
                      msoa_cases = df$msoa_infected[,timestep],
                      hid_cases = df$hid_infected[,timestep]) 
 # ds_df <- data.frame(susceptible = df$susceptible[,timestep], exposed = 0, presymptomatic_infected = df$presymp[,timestep],symptomatic_infected = df$symp[,timestep], recovered =  df$recovered[,timestep], died = df$died[,timestep]) 

  Disease_Status <- ds_df %>%
    dplyr::mutate(
      Disease_Status = case_when(
        susceptible == 1 ~ 0,
        exposed == 1 ~ 1,
        presymptomatic_infected == 1 ~ 2,
        symptomatic_infected ==1 ~ 3,
        recovered == 1 |
          died == 1 ~ 4
      )
    ) %>%
    dplyr::select(Disease_Status)
  
  Days_With_Status <- 0
  Current_Risk <- df$current_risk[,timestep-1]
  
  df_sum <- data.frame(Area, HID,PID,Disease_Status, Days_With_Status,
                       Current_Risk, MSOA_Cases, HID_Cases)
  return(df_sum)
  
}
  

normalizer <- function(x ,lower_bound, upper_bound, xmin, xmax){
  
  normx <-  (upper_bound - lower_bound)*(x - xmin)/(xmax-xmin) + lower_bound
  return(normx)
}
  

reticulate::source_python("microsim/microsim_model.py")

get_hazard <- function(df, timestep, msoa_codes, data_dir="data"){
  #Each day we input each individuals disease status (0 – susceptible, 1 – exposed, 2 – pre-symptomatic, 3 – symptomatic, 4-removed)
  #not sure what to do here...presymptomatic and exposed are the same thing arent they?
  status_df <- data.frame(s = df$susceptible[,(timestep-1)],
                          e = df$presymp[,(timestep-1)],
                          i = df$symp[,(timestep-1)],
                          r = df$recovered[,(timestep-1)],
                          d = df$died[,(timestep-1)])
  
  status_df$s[status_df$s==0] <- NA
  status_df$e[status_df$e==0] <- NA
  status_df$i[status_df$i==0] <- NA
  status_df$r[status_df$r==0] <- NA
  status_df$d[status_df$d==0] <- NA
  
  status_df$s[status_df$s==1] <- 0
  status_df$e[status_df$e==1] <- 1
  status_df$i[status_df$i==1] <- 3
  #both death and recover are removed (4)
  status_df$r[status_df$r==1] <- 4
  status_df$d[status_df$d==1] <- 4
  
  iterations <- 1L
  data_dir <- data_dir
  msoa_codes <- msoa_codes
  
  status <- rowSums(status_df,na.rm=TRUE)
  
  hazards <- run(iterations,
                 data_dir,
                 status,
                 msoa_codes)
  
  return(hazards)
}



#testttttt







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


# status column is coded
# 0 = susceptible
# 1 = presymp
# 2 = symp
# 3 = recovered
# 4 = dead

#########################################
# calculate the probability of becoming infect
# requires a dataframe list, a vector of betas, and a timestep
covid_prob <- function(df, betas, timestep, interaction_terms = NULL) {
  print("assign probabilities")

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
  psi[df$status[,(timestep-1)] %ni% c(3,4)] <- 0 # if they are not susceptible then their probability is 0 of getting it 
  psi[df$status[,(timestep-1)] %in% c(1,2)] <- 1 # this makes keeping track of who has it easier
  df$betaxs[,timestep] <- df$as_risk + rowSums(beta_out)
  df$probability[,timestep] <- psi
  return(df)
}

#########################################
# assigns covid based on probabilities
case_assign <- function(df,timestep, with_optimiser = FALSE) {
  print("assign cases")
  
  susceptible <- which(df$status[,timestep] == 0)
  
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
  
  new_cases <- which(df$had_covid[,timestep]-df$had_covid[,(timestep-1)]==1)
  
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
  
  df_msoa_hid <- data.frame(msoa = df[[area]],hid = df[[hid]],
                            pop_dens_km2 = df$pop_dens_km2,
                            msoa_area = df$msoa_area,symp =  df$symp[,timestep], 
                            presymp = df$presymp[,timestep])
  
  df_gr <-  data.frame(df_msoa_hid %>%
                         group_by(msoa) %>%
                         add_count(msoa, name = "msoa_size") %>%
                         mutate(
                           msoa_presymp = sum(presymp == 1),
                           msoa_symp = sum(symp == 1),
                           msoa_infected = msoa_presymp + msoa_symp
                         ) 
                       )

  df$msoa_presymp[,timestep] <- df_gr$msoa_presymp
  df$msoa_symp[,timestep] <- df_gr$msoa_symp
  df$msoa_infected[, timestep] <- df_gr$msoa_infected
  
  if("cases_per_area" %in% colnames(df)){
    df$cases_per_area[,timestep] <- log10(df_gr$msoa_infected)/df_gr$msoa_area
  }

  return(df)
}


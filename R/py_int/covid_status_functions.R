
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

covid_prob <- function(df, betas, interaction_terms = NULL, risk_cap=FALSE, risk_cap_val=100, include_age_sex = FALSE) {
  #print("assign probabilities")
  
  if(risk_cap==TRUE){
    df$current_risk[df$current_risk>risk_cap_val] <- risk_cap_val
  }
  
  beta_names <- names(betas)
  
  if (all(!beta_names %in% names(df))) {
    print(paste0(
      beta_names[!beta_names %in% names(df)],
      " missing from df. They are not included in probabilities."
    ))
  }
  
  beta_names <- beta_names[beta_names %in% names(df)]
  
  if (length(beta_names) > 0 ){
    beta_out <- lapply(X = beta_names, FUN = beta_make, betas=betas, df=df)
    beta_out <- do.call(cbind, beta_out)
    colnames(beta_out) <- beta_names
    beta_out_sums <- rowSums(beta_out)
  } else{
    beta_out_sums <- 0
  }
  
  if(include_age_sex){
    if (length(interaction_terms) > 0 ){
      lpsi <- df$beta0 + df$as_risk + beta_out_sums + apply(beta_out[,interaction_terms], 1, prod)
    } else{
      lpsi <- df$beta0 + df$as_risk + beta_out_sums
    }
  } else{
    if (length(interaction_terms) > 0 ){
      lpsi <- df$beta0 +  beta_out_sums + apply(beta_out[,interaction_terms], 1, prod)
    } else{
      lpsi <- df$beta0 + beta_out_sums
    }
  }


 

  psi <- exp(lpsi) / (exp(lpsi) + 1)
  psi[df$status %in% c(3,4)] <- 0 # if they are not susceptible then their probability is 0 of getting it 
  psi[df$status %in% c(1,2)] <- 1 # this makes keeping track of who has it easier
  df$betaxs <- df$as_risk + beta_out_sums
  df$probability <- psi
  
  write.csv(df, paste0("df_prob_out_",format(Sys.time(), "%H%M%S"), ".csv"), row.names = FALSE)
  
  return(df)
}

#########################################
# assigns covid based on probabilities
case_assign <- function(df, with_optimiser = FALSE,timestep,tmp.dir) {
  #print("assign cases")
  
  susceptible <- which(df$status == 0)
  
  # df$probability[is.na(df$probability)] <- 0
  
  if (with_optimiser) {
    df$new_status[susceptible] <- rbinom(n = length(susceptible),
                                         size = 1,
                                         prob = df$optim_probability[susceptible])
  } else{
    #print("nop")
    df$new_status[susceptible] <- rbinom(n = length(susceptible),
                                         size = 1,
                                         prob = df$probability[susceptible])
  }
  

  #if(file.exists("new_cases.csv")==FALSE) {
  #  ncase <- sum(df$new_status[susceptible])
  #} else {
  #  ncase <- read.csv("new_cases.csv")
  #  ncase$X <- NULL
  #  tmp <- sum(df$new_status[susceptible])
  #  ncase <- rbind(ncase,tmp)
  #  rownames(ncase) <- seq(1,nrow(ncase))
  #}
  #ncase <- as.data.frame(ncase)
  #write.csv(ncase, "new_cases.csv")
  
  if(timestep==1) {
    nsus <- length(susceptible)
    dir.create(tmp.dir)
  } else {
    tmp <- length(susceptible)
    nsus <- rbind(nsus,tmp)
    rownames(nsus) <- seq(1,nrow(nsus))
  }
  #ncase <- as.data.frame(ncase)

  write.csv(nsus, paste(tmp.dir,"/susceptible_cases.csv",sep=""))

  return(df)
}


#########################################
# calculate the infection length of new cases
infection_length <- function(df,presymp_dist = "weibull",presymp_mean = NULL,presymp_sd = NULL,
                             infection_dist = "normal", infection_mean = NULL, infection_sd = NULL,
                             timestep,tmp.dir){
  
  susceptible <- which(df$status == 0)
  
  new_cases <- which((df$new_status-df$status==1) & df$status == 0)
  
  if(timestep==1) {
    ncase <- length(new_cases)
  } else {
    tmp2 <- length(new_cases)
    ncase <- rbind(ncase,tmp2)
    rownames(ncase) <- seq(1,nrow(ncase))
  }
  #ncase <- as.data.frame(ncase)
  write.csv(ncase, paste(tmp.dir,"/new_cases.csv",sep=""))

  #new_cases <- which(df$new_status[susceptible]-df$status[susceptible]==1)
  
  if (presymp_dist == "weibull"){
    wpar <- mixdist::weibullpar(mu = presymp_mean, sigma = presymp_sd, loc = 0) 
    df$presymp_days[new_cases] <- round(rweibull(1:length(new_cases), shape = as.numeric(wpar["shape"]), scale = as.numeric(wpar["scale"])),) 
  }
  
  if (infection_dist == "normal"){
    df$symp_days[new_cases] <- round(rnorm(1:length(new_cases), mean = infection_mean, sd = infection_sd))
  }
  
  #switching people from being pre symptomatic to symptomatic and infected
  becoming_sympt <- which((df$status == 1 | df$new_status == 1) & df$presymp_days == 0) ### maybe should be status rather than new_status
  df$new_status[becoming_sympt] <- 2
  
  return(df)
}


#########################################
# determines if someone has been removed and if that removal is recovery or death
removed <- function(df, chance_recovery = 0.95){
  
  removed_cases <- which(df$presymp_days == 0 & df$symp_days == 1)
  
  df$new_status[removed_cases] <- 3 + rbinom(n = length(removed_cases),
                                             size = 1,
                                             prob = (1-chance_recovery))
  
  df$symp_days[removed_cases] <- 0
  df$presymp_days[df$presymp_days>0 & !is.na(df$presymp_days)] <- df$presymp_days[df$presymp_days>0 & !is.na(df$presymp_days)] - 1
  df$symp_days[df$new_status == 2 & df$symp_days > 0] <- df$symp_days[df$new_status == 2 & df$symp_days>0] - 1
  
  write.csv(df, paste0("df_rem_out_",format(Sys.time(), "%H%M%S"), ".csv"), row.names = FALSE)
  
  return(df)
}



#########################################
# below is commented out because we are not doing this for now
# this will be accounted for in nics code as far as i know

# separates by msoa and household
#area_cov <- function(df, area, hid){
#  
#  df_msoa_hid <- data.frame(msoa = df[[area]],hid = df[[hid]],
#                            pop_dens_km2 = df$pop_dens_km2,
#                            msoa_area = df$msoa_area,symp =  df$symp, 
#                            presymp = df$presymp)
#  
#  df_gr <-  data.frame(df_msoa_hid %>%
#                         group_by(msoa) %>%
#                         add_count(msoa, name = "msoa_size") %>%
#                         mutate(
#                           msoa_presymp = sum(presymp == 1),
#                           msoa_symp = sum(symp == 1),
#                           msoa_infected = msoa_presymp + msoa_symp
#                         ) 
#                       )

#  df$msoa_presymp <- df_gr$msoa_presymp
#  df$msoa_symp <- df_gr$msoa_symp
#  df$msoa_infected <- df_gr$msoa_infected
#  
#  if("cases_per_area" %in% colnames(df)){
#    df$cases_per_area <- log10(df_gr$msoa_infected)/df_gr$msoa_area
#  }
#
#  return(df)
#}

normalizer <- function(x ,lower_bound, upper_bound, xmin, xmax){
  
  normx <-  (upper_bound - lower_bound)*(x - xmin)/(xmax-xmin) + lower_bound
  return(normx)
}


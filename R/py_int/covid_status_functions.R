
##################################
# Functions for Covid simulation code
# Jesse F. Abrams and Fiona Spooner
##################################
# There are three main functions that 
# 1. calculate covid probability
# 2. assign covid and take a random draw to determine how long a person is infectius
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


#' Formatting data for infection model
#' 
#' Formatting the output of the spatial interaction model for use in the 
#' infection model and selecting which variables should be included
#' 
#' @param microm_sim_pop The output of the spatial interaction model
#' @param vars Variables to be kept for use in the infection model
#' @return A list of data to be used in the infection model
#' @export
create_input <-
  function(micro_sim_pop,
           vars = NULL) {
    
    if (!all(vars %in% colnames(micro_sim_pop))) {
      print(paste0(vars[!vars %in% colnames(micro_sim_pop)], " not in population column names"))
    }
    
    var_list <- list()
    for (i in 1:length(vars)) {
      var_list[[i]] <- micro_sim_pop[, vars[i]]
    }
    
    names(var_list) <- vars
    
    constant_list <- list(
      beta0 = rep(0, nrow(micro_sim_pop)),
      betaxs = rep(0, nrow(micro_sim_pop)),
      hid_status = rep(0, nrow(micro_sim_pop)),
      presymp_days = micro_sim_pop$presymp_days,
      symp_days = micro_sim_pop$symp_days,
      probability = rep(0, nrow(micro_sim_pop)),
      status = micro_sim_pop$disease_status,
      new_status = micro_sim_pop$disease_status
    )
    
    df <- c(var_list, constant_list)
    
    return(df)
  }


#' Calculating probabilities of becoming a COVID case
#'
#' Calculating probabilities of becoming a COVID case based on each individuals
#' 'current_risk'
#'
#' @param df The input list - the output from the create_input function
#' @param betas List of betas associated with variables to be used in
#' calculating probability of becoming a COVID case
#' @param risk_cap_val The value at which current_risk will be capped
#' @return An updated version of the input list with the probabilties updated
#' @export
covid_prob <- function(df, betas, risk_cap_val=NA) {
  
  if(!is.na(risk_cap_val)){
    print(paste0(sum(df$current_risk > 5), " individuals with risk above  ", risk_cap_val))
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
  beta_out_sums <- df[[beta_names]] * betas[[beta_names]]
  
  lpsi <- df$beta0 + beta_out_sums
  
  psi <- exp(lpsi) / (exp(lpsi) + 1)
  psi <- normalizer(psi, 0,1,0.5,1)  # stretching out the probabilities to be between 0 and 1 rather than 0.5 and 1
  
  psi[df$status %in% c(3,4)] <- 0 # if they are not susceptible then their probability is 0 of getting it
  psi[df$status %in% c(1,2)] <- 1 # this makes keeping track of who has it easier
  df$betaxs <- beta_out_sums
  df$probability <- psi
  
  return(df)
}

#' Assigns COVID cases based on individual probabilities
#'
#' Susceptible individuals are assigned COVID through a Bernoulli draw (rbinom)
#' based on their probability of becoming a COVID case.
#' 
#' @param df Input list of the model - output of covid_prob function
#' @param tmp.dir Directory for saving a csv recording the number 
#' of new cases each day
#' @save_output Logical. Should the number of new cases be saved as output.
#' @return An updated version of the input list with the new cases assigned
#' @export
case_assign <- function(df, tmp.dir, save_output = TRUE) {
 
  susceptible <- which(df$status == 0)
  
  df$new_status[susceptible] <- rbinom(n = length(susceptible),
                                         size = 1,
                                         prob = df$probability[susceptible])
  
  if(file.exists("new_cases.csv")==FALSE) {
   ncase <- sum(df$new_status[susceptible])
  } else {
   ncase <- read.csv("new_cases.csv")
   ncase$X <- NULL
   tmp <- sum(df$new_status[susceptible])
   ncase <- rbind(ncase,tmp)
   rownames(ncase) <- seq(1,nrow(ncase))
  }
  ncase <- as.data.frame(ncase)
  write.csv(ncase, paste0(tmp.dir, "/new_cases.csv"))
  return(df)
}

#' Assign COVID cases by ranking individuals current risk
#'
#' Used to assign cases for the time period in which the model is being seeded.
#' Individuals are ranked in descending order by current risk and then desired number
#' of cases are assigned to the highest ranking individuals.
#' 
#' Will be run in place of the case_assign function when the model is being seeded.
#'
#' @param df Input list of the model - output of covid_prob function
#' @param daily_case The desired number of cases that should be assigned on this day
#' @return An updated version of the in put list with new rank assigned cases 
#' added
#' @export
rank_assign <- function(df, daily_case){
  
  dfw <- data.frame(id = df$id, current_risk = df$current_risk, status = df$status)
  dfw <- dfw[dfw$status == 0,]
  rank_inf <- dfw[order(-dfw$current_risk),][1:daily_case,"id"]
  inf_ind <- which(df$id %in% rank_inf)
  df$new_status[inf_ind] <- 1 
  return(df)
}

#' Assigns the infection length of new cases
#' 
#' Each new case is assigned a number of days an individual is both 
#' presymptomatic and symptomatic for.
#' 
#' @param df Input list of the function - output of an ____assign function
#' @param presymp_dist The distribution of the length of the presymptomatic stage
#' @param presymp_mean The mean length of the presymptomatic stage
#' @param presymp_sd The standard deviation of the length of the presymptomatic stage
#' @param infection_dist The distribution of the length of the symptomatic stage
#' @param infection_mean The mean length of the symptomatic stage
#' @param infection_sd The standard deviation of the length of the symptomatic stage
#' @param timestep The day counter
#' @param tmp.dir Directory for saving output
#' @param save_output Logical. Should output be saved.
#' @return An updated version of the input list with the new cases having 
#' infection lengths assigned 
#' @export
infection_length <- function(df, presymp_dist = "weibull", presymp_mean = NULL,presymp_sd = NULL,
                             infection_dist = "normal", infection_mean = NULL, infection_sd = NULL,
                             timestep, tmp.dir, save_output = TRUE){
  
  susceptible <- which(df$status == 0)
  
  new_cases <- which((df$new_status-df$status ==1) & df$status == 0)
  
  #if (save_output == TRUE){
    if(timestep==1) {
      ncase <<- length(new_cases)
    } else {
      tmp2 <- length(new_cases)
      ncase <<- rbind(ncase,tmp2)
      rownames(ncase) <<- seq(1,nrow(ncase))
    }
    #ncase <- as.data.frame(ncase)
    write.csv(ncase, paste(tmp.dir,"/new_cases.csv",sep=""))

  #}

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


#' Determines whether removed individuals recover or die
#' 
#' @param df Input list of the function - output of the infection_length function
#' @param chance_recovery Probability of an infected individual recovering
#' @return An updated version of the input list with the status updates for those 
#' days left in stage = 0.
#' @export
removed <- function(df, chance_recovery = 0.95){
  
  removed_cases <- which(df$presymp_days == 0 & df$symp_days == 1)
  
  df$new_status[removed_cases] <- 3 + rbinom(n = length(removed_cases),
                                             size = 1,
                                             prob = (1-chance_recovery))
  
  df$symp_days[removed_cases] <- 0
  df$presymp_days[df$presymp_days>0 & !is.na(df$presymp_days)] <- df$presymp_days[df$presymp_days>0 & !is.na(df$presymp_days)] - 1
  df$symp_days[df$new_status == 2 & df$symp_days > 0] <- df$symp_days[df$new_status == 2 & df$symp_days>0] - 1
  
  return(df)
}


#' Changing the spread of values to be between two set values
#' 
#' For use in the covid_prob function to change the probabilities 
#' to be 0-1 rather than 0.5-1
#' 
#' @param x A number or vector of numbers
#' @param lower_bound Desired lower_bound of values
#' @param upper_bound Desired upper_bound of values
#' @param xmin Expected minimum value of x
#' @param xmax Expected maximum value of x
#' @return A number or vector of numbers between the lower and upper bounds
#' @export
normalizer <- function(x ,lower_bound, upper_bound, xmin, xmax){
  
  normx <-  (upper_bound - lower_bound) * (x - xmin)/(xmax - xmin) + lower_bound
  return(normx)
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
new_beta0_probs <- function(df, daily_case){
  
  susceptible <- which(df$status == 0)
  
  new_beta0 <- optim(par = -1, beta0_optim,  n = length(susceptible), betaX=df$betaxs[susceptible], Y=daily_case, 
                     method="Brent",  lower =-30, upper = 0)$par
  
  df$new_beta0 <- new_beta0
  tot_risk_new <- df$new_beta0  + df$betaxs
  df$optim_probability <- exp(tot_risk_new)/(1+exp(tot_risk_new))
  
  case_YN <- rbinom(n=length(df$optim_probability[susceptible]), size=1, prob =   df$optim_probability[susceptible])
  print(paste0("optim cases ",sum(case_YN)))
  
  return(df)
}




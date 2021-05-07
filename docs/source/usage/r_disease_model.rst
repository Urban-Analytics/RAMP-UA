How the R disease model works
=============================

.. highlight:: R 

Setup 
-------

The R disease model is found in 'R/py_int/covid_run.R' and is called when the 'microsim/main.py' script is run. 

All of the functions used in the R disease model can be found in the rampuaR R package ::

    remotes::install_github("https://github.com/Urban-Analytics/rampuaR")


The R disease model is made up of a series of functions: ::

    create_input() #Takes the daily output from the python spatial interaction model and formats it into a dataframe to be used in R
    mortality_risk() #Calculates each individuals mortality risk if infected, based on age and health
    age_symp_risk() #Calculates each individuals risk of being symptomatically infected (as opposed to asymptomatically), based on age
    sympt_risk() #Calculates each individuals risk of being symptomatically infected (as opposed to asymptomatically), based on age and health
    sum_betas() #An outdated function used to sum all of the components that contribute to an individuals risk of being infected
    exp_cdf() #Exponential distribution cdf, used in scaling hazard (0-inf) to risk (0-1).
    infection_prob() #Calculates the probability of an individual being infected with COVID-19, to be used in a Bernoulli trial
    covid_prob() #The deprecated version for calculating the probability of an individual being infected - normalising (0-Inf) to (0-1)
    case_assign() #Assigns COVID cases to susceptible individuals through Bernoulli trial
    rank_assign() #Alternative method of assigning COVID cases based on ranking the daily accumulated hazard of individuals. Cases are assigned to the top n individuals. Used for seeding the initial cases.
    infection_length() #Calculates the number of days individuals are in each disease state based on information from literature. 
    removed() #Deprecated function - Individuals that have reached the 'Removed' stage of SEIR either recover or die. A fixed 95% recovery rate.
    removed_age() #Individuals that have reached the 'Removed' stage of SEIR either recover or die, based on their individual mortality risk f(age, health).
    recalc_sympdays() #Reduce the number of days that individuals are in their current disease stage by 1
    run_removal_recalc() #Deprecated - wrapper for removed and recacl_sympdays functions
    vaccinate() #Experimental function - Implenting vaccinations in the population - happens overnight
    normalizer() #Changing the spread of values to be between two set values

Parameters    
-----------

The file that runs the disease model (R/py_int/covid_run.R) uses these functions and takes in a number of parameters for use in these functions:

* pop - Must be the population passed from the spatial interaction model
* timestep - Passed from python, iterates by 1
* rep - Redundant, was used when setting a seed for model
* current_risk_beta - Constant to scale hazard by, key parameter for calibrating the model
* risk_cap - Redundant, used for capping individual hazard when we were getting very high values
* seed_days - Number of days for model is seeded with covid cases (gam_cases)
* exposed_dist - Statistical distribution used to estimate number of days individuals spend in the exposed stage
* exposed_mean - Mean number of days individuals stay in the exposed stage
* exposed_sd - Standard deviation of number of days individuals stay in exposed stage
* presymp_dist - Statistical distribution used to estimate number of days individuals spend in the pre-symptomatic stage
* presymp_mean - Mean number of days individuals stay in the pre-symptomatic stage
* presymp_sd - Standard deviation of number of days individuals stay in pre-symptomatic stage
* infection_dist - Statistical distribution used to estimate number of days individuals spend in the symptomatic/asymptomatic stage
* infection_mean - Mean number of days individuals stay in the symptomatic/asymptomatic stage
* infection_sd - Standard deviation of number of days individuals stay in symptomatic/asymptomatic stage
* output_switch - Should the output be saved
* rank_assign - Redundant, assigning COVID cases by ranked hazard
* local_outbreak_timestep - If using the local outbreak scenario what time-step should the outbreak occur
* local_outbreak - Use the local outbreak scenario
* msoa_infect - Where should the local outbreak occur
* number_people_local - How many people to infect in the local outbreak
* local_prob_increase - Not sure - @Jesse
* overweight_sympt_mplier - The increased probability of overweight individuals having a symptomatic infection e.g. 1.5 = 50% increase
* overweight - Increased probability of mortality if 25 < BMI > 30 e.g. 1.5 = 50% increase
* obesity_30 - Increased probability of mortality if 30 < BMI > 35 e.g. 1.5 = 50% increase
* obesity_35 - Increased probability of mortality if 35 < BMI > 40 e.g. 1.5 = 50% increase
* obesity_40 - Increased probability of mortality if 40 < BMI e.g. 1.5 = 50% increase
* cvd - Increased probability of mortality if individual has cardio-vascular disease e.g. 1.5 = 50% increase
* diabetes - Increased probability of mortality if individual has diabetes disease e.g. 1.5 = 50% increase
* bloodpressure - Increased probability of mortality if individual has high blood-pressure e.g. 1.5 = 50% increase
* improve_health - Select TRUE to reduce the BMI class of all obese individuals e.g. BMI 40+ changes to BMI 35-40
* set_seed - Redundant, used to set the seed of the model runs. 

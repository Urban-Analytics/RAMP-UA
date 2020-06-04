library(dplyr)
library(ggplot2)
library(tidyr)
library(janitor)
library(readr)
library(sf)
library(arrow)
library(mixdist)
library(viridisLite)
library(reticulate)

##why is git not working 
setwd('~/Documents/GitHub/xmsmcv/')

source("Code/covid_infection_functions.R")
source("Code/getUKCovidTimeSeries.R")

start_date <- 
  as.Date("2020-03-07")
lockdown_date <- as.Date("2020-03-23")
end_date <- 
  as.Date("2020-05-25")

# load cases csv and get devon cases and england cases
cases <- getUKCovidTimeseries()
ua_cases <- cases$tidyEnglandUnitAuth
devon_cases <- ua_cases[as.character(ua_cases$CTYUA19NM)=="Devon",]
fa_cases <- devon_cases[devon_cases$date >= start_date & devon_cases$date <= end_date,]
# calculate number of cases in devon using adjustment factor
sum_cases_day <- round(diff(fa_cases$cumulative_cases))
sum_cases_day <- ifelse(sum_cases_day<0,0,sum_cases_day)

#pop <- feather::feather("Processed_Data/Data_from_Nick/individuals.feather")
#population <- read_csv("~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Population_Data_Devon/Devon_simulated_v2.csv")

population <- clean_names(pull_pop())

#population <- clean_names(read_delim("~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Population_Data_Devon/Devon_Complete.txt", ","))
#population$current_risk <- runif(nrow(population),0,1)
pop_dens <- read_csv("~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Population_Data_Devon/msoa_population_density.csv")

connectivity <- janitor::clean_names(read_csv("~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Transport_Data/msoa_connectedness_closest_three.csv")) %>% 
  filter(!is.na(msoa_sum_connected_ann))
colnames(connectivity)[3:4] <- c("connectedness", "log_connectedness") 
connectivity$connectivity_index <- normalizer(connectivity$log_connectedness, 0.01, 1, min(connectivity$log_connectedness), max(connectivity$log_connectedness))


# qpd <- quantile(population_in$pop_dens_km2) 
# qc <- quantile(population_in$log_connectedness) 
# 
# population_in %>% 
#   mutate(
#     msoa_risk = case_when(
#       pop_dens_km2 >= qpd[4] & log_connectedness >= qc[4] ~ "High",
#       pop_dens_km2 <= qpd[2] & log_connectedness <= qpd[2] ~ "Low",
#       between(pop_dens_km2, qpd[2],qpd[4]) | between(log_connectedness, qpd[2],qpd[4]) ~ "Medium" 
#   )) %>% 
#   dplyr::select(area,pop_dens_km2, log_connectedness, msoa_risk) %>% 
#   distinct()

#num_sample = 100000
num_sample = 818177
area <- "area"
hid <- "hid"
pid <- "pid"
age <- "age1"
sex <- "sex"


population_in <- population %>% 
  left_join(., pop_dens, by =  c("area" = "msoa_area_codes")) %>% 
  dplyr::left_join(.,connectivity, by = c("area" = "msoa11cd")) %>% 
  mutate(log_pop_dens = log10(pop_dens_km2)) 

population_in$cases_per_area <- 0

df_cr_in <-create_input(micro_sim_pop  = population_in,
                        num_sample = num_sample,
                        start_date = start_date,
                        end_date = end_date,
                        lockdown_date = lockdown_date,
                        pnothome_multiplier = 0.6,   # 0.1 = a 60% reduction in time people not home
                        fixed_vars = c(area,   # must match columns in the population data.frame
                                       hid,
                                       pid,
                                       age,
                                       sex,
                                       "msoa_area",
                                       "pop_dens_km2",
                                       "connectivity_index"),
                        dynamic_vars = c("pnothome", 
                                         "cases_per_area")) # msoa index already baked in




df_in <- as_betas_devon(population_sample = df_cr_in, pid = pid,age = age, sex = sex, beta0_fixed = -9.5, divider = 4)  # adding in the age/sex betas 
#df_in <- as_betas_devon(population_sample = df_cr_in, pid = pid,age = age, sex = sex, beta0_fixed = -8.75, divider = 2)  # adding in the age/sex betas 

#-7.167267
num_days <- as.numeric(end_date - start_date)

# adding in guesses for other (non age/sex) betas
#hid_index <- 1.5
#msoa_index <- 2
pnothome <-  0.25 #0.35
connectivity_index <- 0.25#0.3 doesn't work
log_pop_dens <- 0#0.2#0.4#0.3 #0.175
cases_per_area <- 10 #2.5

origin <- factor(c(0,0,0,0,0))
names(origin) <- c("1", "2", "3", "4", "5") #1 = white, 2 = black, 3 = asian, 4 = mixed, 5 = other
qimd1 <- factor(c(0,0,0,0,0))
names(qimd1) <- c("1", "2", "3", "4", "5")# 1  = Least Deprived ... 5 = Most Deprived
underlining <- factor(c(0,0))
names(underlining) <- c("0","1") #1 = has underlying health conditions
hid_infected <- 0
current_risk <- 0


### any betas included must link to columns/data.frames in df_in 

other_betas <- list(pnothome = pnothome,
                    #hid_index = hid_index,
                    cases_per_area = cases_per_area,
                    connectivity_index = connectivity_index)
#log_pop_dens = log_pop_dens)
# starter_cases <- sample(which(df_in$connectivity_index == max(df_in$connectivity_index)),(10*(1/0.02))/(nrow(population)/num_sample))
# 
# df_in$presymp[starter_cases,1] <- 1
  
df_msoa <- df_in
df_risk <- list()
start <- Sys.time()

msoa_codes <- unique(population$area)

for (i in 2:num_days){
  
  hazard <- get_hazard(df = df_msoa, timestep=i, msoa_codes=msoa_codes,data_dir="data")
  #df_prob <- covid_prob(df = df_msoa, betas = other_betas, timestep=i, hazard=hazard)
  df_msoa$current_risk <- hazard$Current_Risk
  df_prob <- covid_prob(df = df_msoa, betas = other_betas,timestep=i)
  
  #df_probo <- new_beta0_probs(df = df_prob, timestep = i, daily_case = sum_cases_day[i])
  #df_ass <- case_assign(df = df_probo, timestep=i, with_optimiser = FALSE)
  
  df_ass <- case_assign(df = df_prob, timestep=i, with_optimiser = FALSE)
  
  df_inf <- infection_length(df = df_ass,
                             presymp_dist = "weibull",
                             presymp_mean = 6.4,
                             presymp_sd = 2.3,
                             infection_dist = "normal",
                             infection_mean =  14,
                             infection_sd = 2,
                             timestep=i)
  
  df_rec <- removed(df = df_inf, chance_recovery = 0.95,timestep=i)
  
  df_msoa <- area_cov(df = df_rec, timestep = i, area = area, hid = hid)
  
  colSums(df_msoa$had_covid)
  colMeans(df_msoa$cases_per_area)
  
  #df_risk_out <- format_df(df = df_msoa, timestep = i, area = area, hid = hid, pid = pid) something going wrong here
  
}

end <- Sys.time()

end - start

df_out <- df_msoa
##### age distribution 

df_inf <- which(rowSums(df_out$had_covid)>= 1)
length(df_inf)/num_sample
age <- df_out$age[df_inf]
## adjust for age group size - older groups have higher betas but there are less of them
plot(table(age)/table(df_out$age))

table(df_msoa$area[df_inf])

df_msoa$msoa_area[which(df_msoa$area %in% c("E02004154", "E02004156", "E02004158"))]

mult_dev <- nrow(population)/num_sample

had_covid <- colSums(df_out$had_covid[,-1])
plot(had_covid)
had_covid_9 <- c(rep(0,9), had_covid[-((length(had_covid)-8):length(had_covid))])
plot(had_covid*0.05*mult_dev, type = "l")   # assuming that about 5% of total covid cases end up being phe cases
lines(devon_cases$cumulative_cases, col = "red")

plot(seq(start_date+2, end_date, by ="day"),diff(had_covid_9)*0.05*mult_dev, type = "l")
lines(devon_cases$date[-1],diff(devon_cases$cumulative_cases), type = "l", col = "red")

plot(cumsum(colSums(df_out$presymp))+ cumsum(colSums(df_out$symp)))
plot(colSums(df_out$symp))
plot(colMeans(df_out$cases_per_area))
plot(colMeans(df_out$probability))

plot(colMeans(df_out$betaxs))
plot(colMeans(df_out$pnothome))

plot(df_out$as_risk)
plot(df_out$connectivity_index)


plot(df_out$as_risk, df_out$betaxs[,75]- df_out$as_risk)

msoa_inf <- df_out$area[df_inf]
msoa_ic <- data.frame(table(msoa_inf)) %>% arrange(Freq)

msoa <- sf::st_read("~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Raw_Data/MSOA_Shapefiles/Middle_Layer_Super_Output_Areas_December_2011_Full_Clipped_Boundaries_in_England_and_Wales.shp") %>% 
  st_transform(.,crs = 4326) %>% 
  filter(msoa11cd %in% msoa_ic$msoa_inf) %>% 
  left_join(., msoa_ic, by = c("msoa11cd" = "msoa_inf"))


ggplot(data = msoa) +
  geom_sf(data = msoa,aes(fill = factor(Freq)))+
  scale_fill_viridis_d()



## adjust for age group size - older groups have higher betas but there are less of them

plot(df_msoa$pnothome, df_msoa$age1)
points(pnot, age, col ="red")

plot(df_out$pnothome, rowMeans(df_out$probability))
plot(df_out$age1, rowMeans(df_out$probability))
plot(df_out$log_pop_dens, rowMeans(df_out$probability))



df_msoa %>% 
  filter(area == "E02004200") %>% 
  add_count() %>% 
  select(age1, sex, log_connectedness, pop_dens_km2,n )


plot(colMeans(df_out$new_beta0[,-1]), type = "l")

mean(colMeans(df_out$new_beta0[,-1])[7:70])

plot(df_out$pnothome, rowMeans(df_out$optim_probability))
plot(df_out$age1, rowMeans(df_out$optim_probability))
plot(df_out$log_pop_dens, rowMeans(df_out$optim_probability))
#plot(df_out$pop_dens_km2, rowMeans(df_out$optim_probability))



# df_out_opt <- df_out 

susceptible <- colSums(df_out$susceptible)
susceptible
recovered <- colSums(df_out$recovered)
recovered
died <- colSums(df_out$died)
died
presymp <- colSums(df_out$presymp)
presymp
sympinfected <- colSums(df_out$symp)
sympinfected    #infected and had covid the same at the moment - should differentiate so one keeps track of anyone that's had it



# checking the numbers sum to sample size
check <- susceptible + presymp + sympinfected + recovered + died 
check
#susceptible + had_covid

#had_covid == presymp + sympinfected + recovered + died
  
sum_df <- data.frame(date = seq(start_date, end_date, by = "day")[-1],susceptible = susceptible, presymptomatic = presymp, infected = sympinfected, recovered = recovered, died = died) %>% 
  pivot_longer(-date)

ggplot(sum_df, aes(x = date, y = log10(value), group = name, col = name))+
  geom_line()



### socioeconomic class distribution

qimd <- df_out$qimd1[df_inf]

hist(qimd)
plot(table(qimd)/ table(df_out$qimd))


#### pnothome

pnothome <- df_out$pnothome[df_inf]

hist(df_out$pnothome)
hist(pnothome)

#### msoa plots

msoa_inf <- df_out$msoa[df_inf]
msoa_ic <- data.frame(table(msoa_inf)) %>% arrange(Freq)

msoa <- sf::st_read("~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Raw_Data/MSOA_Shapefiles/Middle_Layer_Super_Output_Areas_December_2011_Full_Clipped_Boundaries_in_England_and_Wales.shp") %>% 
  st_transform(.,crs = 4326) %>% 
  filter(msoa11cd %in% msoa_ic$msoa_inf) %>% 
  left_join(., msoa_ic, by = c("msoa11cd" = "msoa_inf"))


ggplot(data = msoa) +
  geom_sf(data = msoa,aes(fill = factor(Freq)))+
  scale_fill_viridis_d()

connectivity <- janitor::clean_names(read_csv("~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Transport_Data/msoa_connectedness_closest_three.csv"))
colnames(connectivity)[3:4] <- c("connectedness", "log_connectedness") 

msoa_dev_con <- connectivity %>% 
  group_by(msoa11cd) %>%
  filter(msoa11cd %in% population$area) %>%
  mutate(mean_con = mean(connectedness)) %>% 
  arrange(-mean_con)

msoa <- msoa %>% 
  left_join(., connectivity, by = "msoa11cd")




#test

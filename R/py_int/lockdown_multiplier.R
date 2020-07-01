library(dplyr)
library(janitor)
library(readr)
library(tidyr)
library(ggplot2)
# load cases csv and get devon cases and england cases

###downloading latest file from google mobility
download.file("https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv", 
              destfile = "Google_Global_Mobility_Report.csv")

lock_down_start <- as.Date("2020-03-23")
lock_down_14 <- lock_down_start + 14 #two weeks after lockdown

gm <- read_csv("Google_Global_Mobility_Report.csv") %>% 
  filter(country_region == "United Kingdom") %>% 
  dplyr::select( -c(country_region_code, country_region,  sub_region_2)) %>% 
  filter(sub_region_1 == "Devon")%>% 
  pivot_longer(., contains("percent"))


# Plotting change from baseline.
# The baseline is the median value, for the corresponding day of the week, 
# during the 5-week period Jan 3–Feb 6, 2020
ggplot(gm, aes(x  = date, y = value, group = name, colour = name))+
  geom_line() +
  geom_vline(xintercept = lock_down_start, linetype = "dashed", col = "black")+
  geom_vline(xintercept = lock_down_14, linetype = "dashed", col = "black")+
  labs(colour = "", y = "Percent change from baseline")


# How much time had people spent at home during the two weeks after lockdown
ld <- gm  %>% 
  mutate(perc_diff_base = 1+ (value/100)) %>% 
  filter(between(date, lock_down_start, lock_down_14)) %>% 
  filter(name == "residential_percent_change_from_baseline") %>% 
  summarise(mean_home = 1+ (mean(value)/100))

population <- clean_names(read_delim("~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Population_Data_Devon/Devon_Complete.txt", ","))
population <- clean_names(read_delim("~/Documents/ECEHH/COVID/RAMP-UA/data/devon-tu_health/Devon_Complete.txt", ","))

new_out <- 1 - (mean(population$phome) * as.numeric(ld)) # During the first two weeks of lockdown people spent 21% more time at home - the average person from devon spent ~ 75% of their time at home before lockdow, during lockdown this increases to 92% - meaning about 8% of time outside the home on average

lock_down_reducer <-  new_out/mean(population$pnothome) # percentage of time spent outside of compared to pre-lockdown
lock_down_reducer   # a reduction of about 2/3




residential_pcnt <- gm %>% 
  filter(
    name == "residential_percent_change_from_baseline"
  ) %>% 
  mutate(day = as.numeric(date) - 18306)


smooth_residential <- mgcv::gam(value ~ s(day, bs = "cr"), data = residential_pcnt)
sr <- fitted.values(smooth_residential)
sr <- (sr/100) + 1


new_out <- 1 - (mean(population$phome) * sr) # During the first two weeks of lockdown people spent 21% more time at home - the average person from devon spent ~ 75% of their time at home before lockdow, during lockdown this increases to 92% - meaning about 8% of time outside the home on average

lock_down_reducer <-  new_out/mean(population$pnothome) # percentage of time spent outside of compared to pre-lockdown
lock_down_reducer   # a reduction of about 2/3

plot(lock_down_reducer, ylab = "Proportion of Time Outside Compared to Baseline", type = "l")

daily_lock_down_multiplier <-  data.frame(day = 1:length(lock_down_reducer), timeout_multiplier = lock_down_reducer)

write.csv(daily_lock_down_multiplier, "google_mobility_lockdown_daily.csv", row.names = FALSE)







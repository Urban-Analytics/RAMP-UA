library(dplyr)
library(janitor)
library(readr)
library(tidyr)
library(ggplot2)
# load cases csv and get devon cases and england cases
population <- clean_names(read_delim("data/devon-tu_health/Devon_Complete.txt", ","))

###downloading latest file from google mobility
download.file("https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv", 
              destfile = "Google_Global_Mobility_Report.csv")

gm <- read_csv("Google_Global_Mobility_Report.csv") %>% 
  filter(country_region == "United Kingdom" & sub_region_1 == "Devon" & iso_3166_2_code == "GB-DEV") %>% 
  dplyr::select( -c(country_region_code,  sub_region_2)) %>% 
  pivot_longer(., contains("percent"))

residential_pcnt <- gm %>% 
  mutate(day = as.numeric(date) - 18306) %>%
  filter(name == "residential_percent_change_from_baseline" & day < 201) 


rv <- (residential_pcnt$value/100) + 1
dates <- seq(min(gm$date), min(gm$date) + length(rv)-1, 1)

ma <- function(x, n = 14){stats::filter(x, rep(1 / n, n), sides = 2)}

n <- 14
ma_rv <- ma(rv, n = n)
ma_rv[1:(n/2)]<- rv[1:(n/2)]

plot(dates, rv, type = "l")
lines(dates, ma_rv, col = "blue")

new_out <- 1 - (mean(population$phome) * rv) # During the first two weeks of lockdown people spent 21% more time at home - the average person from devon spent ~ 75% of their time at home before lockdow, during lockdown this increases to 92% - meaning about 8% of time outside the home on average
new_out_ma <- 1 -  (mean(population$phome) * ma_rv) 

lock_down_reducer_rv <-  new_out/mean(population$pnothome) # percentage of time spent outside of compared to pre-lockdown
lock_down_reducer_ma <- new_out_ma/mean(population$pnothome) 

plot(dates, lock_down_reducer_rv, ylab = "Proportion of Time Outside Compared to Baseline", type = "l", xlab = "Day")
lines(dates,lock_down_reducer_ma, col = "blue")

daily_lock_down_multiplier_ma <-  data.frame(day = 1:length(lock_down_reducer_ma), timeout_multiplier = lock_down_reducer_ma)

#write.csv(daily_lock_down_multiplier_ma, "devon_data/google_mobility_lockdown_daily_14_day_moving_average.csv", row.names = FALSE)

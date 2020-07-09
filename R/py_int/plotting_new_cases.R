library(dplyr)
library(ggplot2)
library(patchwork)
library(mgcv)
library(tidyr)
library(readr)

gam_cases <- readRDS("gam_fitted_PHE_cases.RDS")
gam_cases_df <- data.frame(day = 1:80, cases = gam_cases[1:80])

lfb <- list.dirs("~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Model_Output_New/Baseline/", full.names = TRUE)
lfb <- lfb[grepl("2020", lfb)]

all_reps_base <- NULL
for (i in 1:length(lfb)){
  nc <- read.csv(paste0(lfb[i], "/new_cases.csv"))
  df <- data.frame(run = i, day = nc$X, cases = nc$V1)
  all_reps_base <- rbind(all_reps_base, df)  
  }

all_reps_median_base <- all_reps_base %>% 
  group_by(day) %>% 
  summarise(median_cases = median(cases))


baseline <- ggplot()+
  geom_line(data = all_reps_base%>% filter(day >= 10 & day <= 60), aes(x = day, y = cases, group = run),alpha = 0.3)+
  geom_line(data = all_reps_median_base%>% filter(day >= 10 & day <= 60), aes(x = day, y = median_cases), col = "black", size = 2)+
 # geom_line(data = gam_cases_df, aes(x = day, y = cases), col = "red")+
  ylab("Daily Cases")+
  xlab("Day")+
#  geom_vline(xintercept = 38, linetype = "dashed")+
#  ggtitle("Baseline")+
  theme_bw()+
  ylim(0,750)+
  theme(text = element_text(size=20))+
  ggsave("Baseline.png",path = "~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Model_Output_New/Plots/", width = 12, height = 8)


lf <- list.dirs("~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Model_Output_New/Lockdown_Early/", full.names = TRUE)
lf <- lf[grepl("2020", lf)]

all_reps <- NULL
for (i in 1:length(lf)){
  nc <- read.csv(paste0(lf[i], "/new_cases.csv"))
  df <- data.frame(run = i, day = nc$X, cases = nc$V1)
  all_reps <- rbind(all_reps, df)  
}

lfg <- list.files("~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Model_Output_New/Lockdown_Early/full_output/", full.names = TRUE)
#lfg <- lfg[1:8]
all_reps_gz <- NULL
for (i in 1:length(lfg)){

  lfgz <- vroom::vroom(paste0(lfg[i], "/Individuals.csv.gz"))
  new_casesgz <- lfgz %>% 
    select(ID, starts_with("disease_status0")) %>% 
    pivot_longer(starts_with("disease")) %>% 
    group_by(name, value) %>% 
    tally() %>% 
    ungroup() %>% 
    filter(value == 0) %>% 
    mutate(run = i + length(lf),cases = c(NA, diff(695308-n)), day = as.numeric(gsub("disease_status0", "", name))) %>% 
    filter(day > 0) %>% 
    select(run, day, cases)
  
  all_reps_gz <- rbind(all_reps_gz, new_casesgz)
  
}


ggplot(all_reps_gz, aes(x = day, y = cases, group = run))+geom_line()


all_reps_all <- rbind(all_reps, all_reps_gz)

all_reps_median <- all_reps_all %>% 
  group_by(day) %>% 
  summarise(median_cases = median(cases))



lock_down_early <- ggplot()+
  geom_line(data = all_reps_all %>% filter(day >= 10 & day <= 60), aes(x = day, y = cases, group = run),alpha = 0.3)+
  geom_line(data = all_reps_median%>% filter(day >= 10 & day <= 60), aes(x = day, y = median_cases), col = "black", size = 2)+
  # geom_line(data = gam_cases_df, aes(x = day, y = cases), col = "red")+
  ylab("Daily Cases")+
  xlab("Day")+
  #  geom_vline(xintercept = 31, linetype = "dashed")+
#  ggtitle("Lockdown a Week Earlier")+
  theme_bw()+
  ylim(0, 750)+
  theme(text = element_text(size=20))+
  ggsave("Early_Lockdown.png",path = "~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Model_Output_New/Plots/", width = 12, height = 8)



comparison <- ggplot()+
  geom_line(data = all_reps_median%>% filter(day >= 10), aes(x = day, y = median_cases), col = "black", size = 2, linetype = "dashed")+
  geom_line(data = all_reps_median_base%>% filter(day >= 10), aes(x = day, y = median_cases), col = "black", size = 2)+
  theme_bw()+
  ylim(0, 420)+
  ylab("Daily Cases")+
  xlab("Day")+
  theme(text = element_text(size=20))+
#  ggtitle("Comparison")+
  ggsave("Comparison.png",path = "~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Model_Output_New/Plots/", width = 12, height = 8)


baseline/lock_down_early



##### PHE cases


devon_cases <- readRDS(paste0(getwd(),"/devon_cases.RDS"))
devon_cases$cumulative_cases[84] <- 812 #typo here I think
devon_cases$new_cases <- c(0,diff(devon_cases$cumulative_cases))
devon_cases$devon_date <- as.numeric(devon_cases$date)
devon_cases <- as.data.frame(devon_cases)
devon_df <- data.frame(case_type = "PHE",day = 1:80, cases = devon_cases$new_cases[1:80]*20)


gam_cases <- readRDS(paste0(getwd(),"/gam_fitted_PHE_cases.RDS"))

gam_df <- data.frame(case_type = "Smoothed", day = 1:80, cases = gam_cases[1:80])

all_cases <- rbind(devon_df, gam_df)



ggplot()+
  geom_line(data = gam_df, aes(x = day, y = cases), colour = "red", size = 2)+
    geom_point(data = devon_df, aes(x = day, y = cases))+
  ylab("Daily Cases")+
  xlab("Day")+
  theme_bw()+
  theme(text = element_text(size=20))+
  ggsave("PHE_daily_cases.png",path = "~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Model_Output_New/Plots/", width = 12, height = 8)


ggplot()+
  geom_line(data = gam_df, aes(x = day, y = cumsum(cases)), colour = "red", size = 2)+
   geom_point(data = devon_df, aes(x = day, y = cumsum(cases)))+
  ylab("Total Cases")+
  xlab("Day")+
  theme_bw()+
  theme(text = element_text(size=20))+
  ggsave("PHE_cumulative_cases.png",path = "~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Model_Output_New/Plots/", width = 12, height = 8)


##### Google Mobility Plots

population <- read_csv("devon_data/devon-tu_health/Devon_Complete.txt")

###downloading latest file from google mobility
download.file("https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv", 
              destfile = "Google_Global_Mobility_Report.csv")

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
 # geom_vline(xintercept = lock_down_start, linetype = "dashed", col = "black")+
#  geom_vline(xintercept = lock_down_14, linetype = "dashed", col = "black")+
  labs(colour = "", y = "Percent change from baseline", x = "Date")



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

daily_lock_down_multiplier <-  data.frame(day = seq(as.Date("2020-02-15"), as.Date("2020-02-15")+79, "day"), multiplier = lock_down_reducer[1:80], location = "Outside")

daily_time_at_home <- data.frame(day = seq(as.Date("2020-02-15"), as.Date("2020-02-15")+79, "day"), multiplier = sr[1:80], location = "Home")


home_out <- rbind(daily_lock_down_multiplier, daily_time_at_home)



ggplot(data = home_out, aes(x = day, y = multiplier, group = location, colour = location))+
  geom_line(size = 2)+
  ylab("Time in Location Relative to Baseline")+
  xlab("Day")+
  geom_hline(yintercept = 1)+
  geom_vline(xintercept = as.Date("2020-03-23"), linetype = "dashed")+
  theme_bw()+
  scale_colour_discrete(name = "Location")+
  theme(text = element_text(size=20))+
#  ggtitle("Google Community Mobility - Devon")+
  ggsave("Google_Mobility.png",path = "~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Model_Output_New/Plots/", width= 12, height = 8)










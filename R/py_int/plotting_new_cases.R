library(dplyr)
library(ggplot2)
library(patchwork)
library(mgcv)
library(tidyr)

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
  geom_line(data = all_reps_median_base%>% filter(day >= 10 & day <= 60), aes(x = day, y = median_cases), col = "black", size = 1)+
 # geom_line(data = gam_cases_df, aes(x = day, y = cases), col = "red")+
  ylab("Daily Cases")+
  xlab("Day")+
#  geom_vline(xintercept = 38, linetype = "dashed")+
  ggtitle("Baseline")+
  theme_bw()+
  ylim(0,750)+
  ggsave("Baseline.png",path = "~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Model_Output_New/Plots/")


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
  geom_line(data = all_reps_median%>% filter(day >= 10 & day <= 60), aes(x = day, y = median_cases), col = "black", size = 1)+
  # geom_line(data = gam_cases_df, aes(x = day, y = cases), col = "red")+
  ylab("Daily Cases")+
  xlab("Day")+
  #  geom_vline(xintercept = 31, linetype = "dashed")+
  ggtitle("Lockdown a Week Earlier")+
  theme_bw()+
  ylim(0, 750)+
  ggsave("Early_Lockdown.png",path = "~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Model_Output_New/Plots/")



comparison <- ggplot()+
  geom_line(data = all_reps_median%>% filter(day >= 10), aes(x = day, y = mean_cases), col = "black", size = 1, linetype = "dashed")+
  geom_line(data = all_reps_median_base%>% filter(day >= 10), aes(x = day, y = mean_cases), col = "black", size = 1)+
  theme_bw()+
  ylim(0, 750)+
  ylab("Daily Cases")+
  xlab("Day")+
  ggtitle("Comparison")+
  ggsave("Comparison.png",path = "~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Model_Output_New/Plots/")


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
  geom_point(data = devon_df, aes(x = day, y = cases))+
  geom_line(data = gam_df, aes(x = day, y = cases), colour = "red", size = 1)+
  ylab("Daily Cases")+
  xlab("Day")+
  theme_bw()+
  ggsave("PHE_daily_cases.png",path = "~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Model_Output_New/Plots/")


ggplot()+
  geom_point(data = devon_df, aes(x = day, y = cumsum(cases)))+
  geom_line(data = gam_df, aes(x = day, y = cumsum(cases)), colour = "red", size = 1)+
  ylab("Total Cases")+
  xlab("Day")+
  theme_bw()+
  ggsave("PHE_cumulative_cases.png",path = "~/University of Exeter/COVID19 Modelling - Documents/Micro_Simulation/Data/Processed_Data/Model_Output_New/Plots/")


  







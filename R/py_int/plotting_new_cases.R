gam_cases <- readRDS("gam_fitted_PHE_cases.RDS")
gam_cases_df <- data.frame(day = 1:80, cases = gam_cases[1:80])

lf <- list.dirs("R/py_int/", full.names = TRUE)

lf <- lf[155:length(lf)]

all_reps <- NULL
for (i in 1:length(lf)){
  nc <- read.csv(paste0(lf[i], "/new_cases.csv"))
  df <- data.frame(run = i, day = nc$X, cases = nc$V1)
  all_reps <- rbind(all_reps, df)  
  }

all_reps_mean <- all_reps %>% 
  group_by(day) %>% 
  summarise(mean_cases = mean(cases))


ggplot()+
  geom_line(data = all_reps, aes(x = day, y = cases, group = run),alpha = 0.3)+
  geom_line(data = all_reps_mean, aes(x = day, y = mean_cases), col = "black", size = 1)+
  geom_line(data = gam_cases_df, aes(x = day, y = cases), col = "red")+
  ylab("Daily Cases")+
  xlab("Day")+
  geom_vline(xintercept = 31, linetype = "dashed")+
  ggtitle("Lockdown a Week Early - March 16th")+
  theme_bw()

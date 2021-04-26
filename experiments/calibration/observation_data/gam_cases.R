library(mgcv)

source("Code/getUKCovidTimeSeries.R")

df <- getUKCovidTimeseries() #Retrieve the PHE data for covid cases

devon_cases <- df$tidyEnglandUnitAuth %>% 
  filter(CTYUA19NM == "Devon" & date <= as.Date("2020-06-24")) #Get cases for Devon only and use only cases for first wave

devon_cases$cumulative_cases[84] <- 812 # I think this is a typo because the cumulative cases shouldn't go down.
devon_cases$new_cases <- diff(c(0,(devon_cases$cumulative_cases))) #Get the daily new cases from cumulative cases
devon_cases$devon_date <- as.numeric(as.Date(devon_cases$date))
devon_cases <- as.data.frame(devon_cases)
devon_cases <- devon_cases[-1,] # There are 10 cases in Devon at the start of the outbreak - we remove these as it inflates the number of total cases at the beginning. For seeding we wan't the gradual increase in cases.  


gam_Devon <- mgcv::gam(new_cases ~ s(devon_date, bs = "cr"), data = devon_cases,family = nb()) #Fit a neg-binom gam to cases
plot(devon_cases$new_cases*20,  ylab="Cases", xlab = "Day") # We estimate that the number of PHE cases is approx 5% of total infections based on Pers Comms from Devon NHS.
points(round(fitted.values(gam_Devon)*20), col = "red")
abline(v = 38, lty = "dashed")
gam_cases <- round(fitted.values(gam_Devon)*20)
write.csv(gam_cases, "gam_cases.csv")


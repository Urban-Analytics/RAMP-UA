library(mgcv)

source("Code/getUKCovidTimeSeries.R")

df <- getUKCovidTimeseries()

devon_cases <- df$tidyEnglandUnitAuth %>% 
  filter(CTYUA19NM == "Devon" & date <= as.Date("2020-06-24"))

devon_cases$cumulative_cases[84] <- 812
devon_cases$new_cases <- diff(c(0,(devon_cases$cumulative_cases)))
devon_cases$devon_date <- as.numeric(as.Date(devon_cases$date))
devon_cases <- as.data.frame(devon_cases)


gam_Devon <- mgcv::gam(new_cases ~ s(devon_date, bs = "cr"), data = devon_cases,family = nb())
plot(devon_cases$new_cases*20,  ylab="Cases", xlab = "Day")
points(round(fitted.values(gam_Devon)*20), col = "red")
abline(v = 38, lty = "dashed")
gam_cases <- round(fitted.values(gam_Devon)*20)
write.csv(gam_cases, "gam_cases.csv")


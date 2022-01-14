# Install libraries
library(mgcv)

# Set working directory and import R file for downloading UK covid data
setwd("C://Users/gy17m2a/OneDrive - University of Leeds/Project/RAMP-UA/experiments/calibration/observation_data")
source("getUKCovidTimeSeries.R")


#############################################################################################
#############################################################################################
#### Using old observations data
#############################################################################################
#############################################################################################
# Get dataframe of Covid cases in the UK
df <- getUKCovidTimeseries()

# Filter this to just the Devon cases
devon_cases <- df$tidyEnglandUnitAuth %>% 
  filter(CTYUA19NM == "Devon" & date <= as.Date("2020-06-24"))

# Changing the value for row 84?
devon_cases$cumulative_cases[84] <- 812
# Gets new cases by comparing the cumulative cases on each date
devon_cases$new_cases <- diff(c(0,(devon_cases$cumulative_cases)))

# convert to numeric
devon_cases$devon_date <- as.numeric(as.Date(devon_cases$date))

# convert to dataframe
devon_cases <- as.data.frame(devon_cases)

# s is defning smooths in gam formula. Devon date is the variable to smooth as a function of
# bs is a 2 letter term that specifies the smoothing basis to use
gam_Devon <- mgcv::gam(new_cases ~ s(devon_date, bs = "cr"), data = devon_cases,family = nb())
plot(devon_cases$new_cases*20,  ylab="Cases", xlab = "Day")
points(round(fitted.values(gam_Devon)*20), col = "red")
abline(v = 38, lty = "dashed")
gam_cases <- round(fitted.values(gam_Devon)*20)

# save to file
write.csv(gam_cases, "gam_cases.csv")

#############################################################################################
#############################################################################################
#### Read in our observations data
#############################################################################################
#############################################################################################
# Read in case data for all MSOAs in Devon
devon_cases_msoa <- read.csv("devon_daily_cases_t.csv")

# Create an empty matrix to store the results for each MSOA in Devon
devon_gam_cases_msoa <- data.frame(matrix(NA, nrow = 405, ncol = 0))

# define the multiplier (this is 20 above?)
multiplier = 20

# Create iterator 
iter_i = 1

# For each MSOA, create a dataframe containing the cases on each day for that MSOA
for (msoa_code in colnames(devon_cases_msoa)[1:length(colnames(devon_cases_msoa))-1]){
  print(msoa_code)
  # select just one MSOA
  one_msoa <- devon_cases_msoa[c("Day",msoa_code)]
  # rename column from MSOA name to cases
  colnames(one_msoa)[2] <- "PosCases"
  # Fit 
  gam_Devon <- mgcv::gam(PosCases ~ s(Day, bs = "cr"), data = one_msoa, family = nb())
  # Plot
  plot(one_msoa$PosCases*multiplier,  ylab="Cases", xlab = "Day")
  points(round(fitted.values(gam_Devon)*multiplier), col = "red")
  abline(v = 38, lty = "dashed")
  gam_cases <- round(fitted.values(gam_Devon)*multiplier)
  
  ## Join back to one_msoa
  one_msoa$gam_cases <- gam_cases
  #
  devon_gam_cases_msoa$col <-gam_cases
  
  # rename
  colnames(devon_gam_cases_msoa)[iter_i] <- col
  
  # Increase the iterator
  iter_i = iter_i+1
  }




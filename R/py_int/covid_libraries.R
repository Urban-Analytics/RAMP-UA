# list.of.packages <- c("rampuaR")
# new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
# if(length(new.packages) >0 ){devtools::install_github("Urban-Analytics/rampuaR", dependencies = F)}
#
# library(rvcheck)
#
# rampr_version <- check_github("Urban-Analytics/rampuaR")
#if(!rampr_version$up_to_date){devtools::install_github("Urban-Analytics/rampuaR", dependencies = F)}

devtools::install_github("Urban-Analytics/rampuaR", dependencies = F, force = TRUE, ref = "wrapper")

library(tidyr)
library(readr)
library(mixdist)
library(dplyr)
library(rampuaR)

#pop <- read.csv("R/py_int/output/2020-07-29 16:48:20/daily_7.csv")

# gam_cases <- readRDS(paste0(getwd(),"/gam_fitted_PHE_cases.RDS"))
#
# w <- NULL
# model_cases <- NULL

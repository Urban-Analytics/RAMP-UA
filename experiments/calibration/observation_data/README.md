# Observation data
##  Preparing estimates of daily Covid cases in Devon

### Current method
The observations data being used for seeding the model and in calibration of the model with ABC is created in [`CreatingObservations-Daily.ipynb`](https://github.com/Urban-Analytics/RAMP-UA/blob/Mollys_DA/experiments/calibration/observation_data/CreatingObservations-Daily.ipynb)

A variation of the method used to create the data in this script was also investigated, but was concluded to be less effective.   
The [`TestingMethod`](https://github.com/Urban-Analytics/RAMP-UA/blob/Mollys_DA/experiments/calibration/observation_data/TestingMethod/) directory contains [`CreatingObservations-InterpolateFirst.ipynb`](https://github.com/Urban-Analytics/RAMP-UA/blob/Mollys_DA/experiments/calibration/observation_data/TestingMethod/CreatingObservations-Daily-InterpolateFirst.ipynb), the notebook used for testing this method, as well as [`CreatingObservations-InterpolateSecond.ipynb`](https://github.com/Urban-Analytics/RAMP-UA/blob/Mollys_DA/experiments/calibration/observation_data/TestingMethod/CreatingObservations-Daily-InterpolateSecond.ipynb) which contains the same method as in [`CreatingObservations-Daily.ipynb`](https://github.com/Urban-Analytics/RAMP-UA/blob/Mollys_DA/experiments/calibration/observation_data/CreatingObservations-Daily.ipynb) but with some additional code to compare to the other method. This directory also contains the `csv` files created by both of these scripts.  

The method in [`CreatingObservations-Daily.ipynb`](https://github.com/Urban-Analytics/RAMP-UA/blob/Mollys_DA/experiments/calibration/observation_data/CreatingObservations-Daily.ipynb) converts weekly data showing the number of new positive test results in the last 7 day period into an estimate of the number of new cases each day. This conversion process includes the following steps:

* Shifting the data back in time by 6 days
* Multiplying the data by a month-specific multiplier
* Linearly interpolating from weekly to daily data
* Smoothing the data

More details on these stages are provided in the notebook.

This directory contains the following `csv` files which are produced in the notebook. In each case for both the whole of Devon and for each MSOA individually:
* weekly_cases_*.csv
  * This contains the weekly test result data after shifting and multiplying
* daily_cases_*.csv
  * This contains the shifted, multiplied data, interpolated to daily values and smoothed
* weekly_cases_*_aggregated_from_daily.csv
  * This contains data resulting from reaggregating the daily data from the stage above to weekly values (i.e. the shifted, multiplied and then interpolated to daily values and smoothed). 


### Previous methods
Two methods were previously applied to generate the observations data. 

#### 1. gam_cases.csv
The R script [`getUKCovidTimeSeries.R`](https://github.com/Urban-Analytics/RAMP-UA/blob/master/experiments/calibration/observation_data/getUKCovidTimeSeries.R) retrieves the latest covid case and hospital admissions data, and produced the case data for Devon [`devon_cases.csv`](https://github.com/Urban-Analytics/RAMP-UA/tree/master/experiments/calibration/observation_data/devon_cases.csv). The [`gam_cases.R`](https://github.com/Urban-Analytics/RAMP-UA/blob/master/experiments/calibration/observation_data/gam_cases.R) script was then used to smooth the cases, creating the output `gam_cases.csv`.

#### 2. england_initial_cases.csv
A second approach was based on modelling daily case data from the weekly cases at hospital and bi-weekly infection survey results, using [`createSeddingFiles.R`](https://github.com/Urban-Analytics/RAMP-UA/blob/Ecotwins-withCommuting/lab/createSeedingFiles.R). This creates [`england_initial_cases.csv`](https://github.com/Urban-Analytics/RAMP-UA/tree/master/experiments/calibration/observation_data/england_initial_cases.csv) in which D0 refers to 05/03/20.



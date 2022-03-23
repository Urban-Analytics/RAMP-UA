# Observation data
##  Preparing estimates of daily Covid cases in Devon

### Current method
Data being used in running the models currently is created in
`CreatingObservations-Daily-InterpolateSecond.ipynb`

A variation of the method used to create the data in this script is also investigated in `CreatingObservations-Daily-InterpolateFirst.ipynb`, but it is concluded that the results of this are not as realistic.

### Previous methods
Two methods were previously applied to generate the observations data. 

#### 1. gam_cases.csv
The R script [`getUKCovidTimeSeries.R`](https://github.com/Urban-Analytics/RAMP-UA/blob/master/experiments/calibration/observation_data/getUKCovidTimeSeries.R) retrieves the latest covid case and hospital admissions data, and produced the case data for Devon [`devon_cases.csv`](https://github.com/Urban-Analytics/RAMP-UA/tree/master/experiments/calibration/observation_data/devon_cases.csv). The [`gam_cases.R`](https://github.com/Urban-Analytics/RAMP-UA/blob/master/experiments/calibration/observation_data/gam_cases.R) script was then used to smooth the cases, creating the output `gam_cases.csv`.

#### 2. england_initial_cases.csv
A second approach was based on modelling daily case data from the weekly cases at hospital and bi-weekly infection survey results, using [`createSeddingFiles.R`](https://github.com/Urban-Analytics/RAMP-UA/blob/Ecotwins-withCommuting/lab/createSeedingFiles.R). This creates [`england_initial_cases.csv`](https://github.com/Urban-Analytics/RAMP-UA/tree/master/experiments/calibration/observation_data/england_initial_cases.csv) in which D0 refers to 05/03/20.



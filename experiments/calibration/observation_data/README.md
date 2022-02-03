# Observation data
##  Preparing estimates of daily Covid cases in Devon

### Current method
Data being used in running the models currently is created in
`CreatingObservations-Daily-InterpolateSecond.ipynb`

A variation of the method used to create the data in this script is also investigated in `CreatingObservations-Daily-InterpolateFirst.ipynb`, but it is concluded that the results of this are not as realistic.

### Previous methods
#### 1. gam_cases.csv
James Salter has been maintaining an r script: `getUKCovidTimeSeries.R` that retrieves the latest covid case and hospital admissions data. Run the function in that script to obtain the latest data, then to get cases for devon do (e.g.):

```
x <- getUKCovidTimeseries()
x$tidyEnglandUnitAuth[x$tidyEnglandUnitAuth$CTYUA19NM=="Devon",c("date", "cumulative_cases")]
```

Those data are used to create `devon_cases.csv`.

Fiona has written a script (`gam_cases.R`) to smooth the cases. That script outputs `gam_cases.csv` which is used to seed (and calibrate) the model

#### 2. Hadrien [england_initial_cases.csv](england_initial_cases.csv)
A second approach was based on modelling daily case data from the weekly cases at hospital and biweekly infection survey results.

`https://github.com/Urban-Analytics/RAMP-UA/blob/Ecotwins-withCommuting/lab/createSeedingFiles.R`

This creates "england_initial_cases.csv" in which D0 refers to 05/03/20

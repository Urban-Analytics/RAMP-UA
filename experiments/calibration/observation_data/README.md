# Observation data

## Devon case data for calibration

Files: 

 - [devon_cases.csv] devon_cases.csv)
 - [gam_cases.R](gam_cases.R)
 - [getUKCovidTimeSeries.R](getUKCovidTimeSeries.R)

James Salter has been maintaining an r script: `getUKCovidTimeSeries.R` that retrieves the latest covid case and hospital admissions data. Run the function in that script to obtain the latest data, then to get cases for devon do (e.g.):

```
x <- getUKCovidTimeseries()
x$tidyEnglandUnitAuth[x$tidyEnglandUnitAuth$CTYUA19NM=="Devon",c("date", "cumulative_cases")]
```

Those data are used to create `devon_cases.csv`.

Fiona has written a script (`gam_cases.R`) to smooth the cases. That script outputs `gam_cases.csv` which is used to seed (and calibrate) the model

## National MSOA data

More recently (May 2011) we obtained case data nationally and at MSOA:

 - [england_initial_cases.csv](england_initial_cases.csv)
 - [england_initial_cases_MSOAs.csv](england_initial_cases_MSOAs.csv)



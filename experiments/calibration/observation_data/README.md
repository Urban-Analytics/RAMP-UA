# Observations of daily COVID-19 cases in Devon
## Background
Observations of COVID-19 infection rates are required in order to validate the performance of the model.  
The government publishes data on COVID-19, including weekly rolling sums of positive test results for MSOAs.  
There are two main issues with this recorded positive test result data:  
1.	There is a lag between the time of a positive test result and the time of infection
2.	Tests do not pick up all positive cases

These two issues could be corrected for by:
1.	Shifting positive test results back in time (based on data on lag between infection and positive result)
2.	Using a multiplier on the number of positive cases (based on data on the proportion of positive case results being picked up by tests)

This would assume:
1.	Individuals go for a test on the day that symptoms develops; and that all of those with positive test results went for a test due to developing symptoms (rather than e.g. exposure to a confirmed case)
2.	Depends on what data used:  
 a.	Could use data on proportion of cases that are asymptomatic (which would assume that all positive test results are for symptomatic people, and that tests do not pick up any asymptomatic people)  
 b.	Alternatively use data on estimated detection rates, in which case any assumption made in this research will be carried over

## Evidence

1.	Lag between infection and positive test result  
  
<i>Cheng et al (2021)</i>: "The pooled mean incubation period of COVID-19 was 6.0 days (95% confidence interval [CI] 5.6–6.5) globally, 6.5 days (95% CI 6.1–6.9) in the mainland of China, and 4.6 days (95% CI 4.1–5.1) outside the mainland of China"  
<i>Paul and Lorin (2021)</i>: "The estimated mean incubation period we obtain is 6.74 days"  
<i>McAloon et al (2020)</i>: "The corresponding mean (95% CIs) was 5.8 (95% CI 5.0 to 6.7) days"  
<i>Lauer et al (2020)</i>: "The median incubation period was estimated to be 5.1 days (95% CI, 4.5 to 5.8 days)" (Study based in China)  

<b> Conclusion: use a lag time of 5 days </b>

2.	Proportion of positive cases picked up by testing

Need to account for the underlying proportion of asymptomatic people who are not taking tests.   
The number of positive cases being picked up by testing will change over time as testing capacity develops.

<i>Phipps et al (2020)</i>: Applies a back casting approach to estimate a distribution for the true cumulative number of infections in 15 developed countries. This includes for the UK a graph of an estimated detection rate of Covid-19 for each month. 
<i>Noh and Danuser (2021)</i>: Actual cumulative cases were estimated to be 5–20 times greater than the confirmed cases
<i>Li et al (2021)</i>: This research used data from China in a model to show the proportion of asymptomatic cases among COVID-19 infected individuals was 42%
<i>Ma et al (2021)</i>: Global study showing 40.5% of those with confirmed Covid cases are asymptomatic.

<i>Cheng, C., Zhang, D., Dang, D., Geng, J., Zhu, P., Yuan, M., Liang, R., Yang, H., Jin, Y., Xie, J. and Chen, S., 2021. The incubation period of COVID-19: a global meta-analysis of 53 studies and a Chinese observation study of 11 545 patients. Infectious diseases of poverty, 10(05), pp.1-13.  
Lauer, S.A., Grantz, K.H., Bi, Q., Jones, F.K., Zheng, Q., Meredith, H.R., Azman, A.S., Reich, N.G. and Lessler, J., 2020. The incubation period of coronavirus disease 2019 (COVID-19) from publicly reported confirmed cases: estimation and application. Annals of internal medicine, 172(9), pp.577-582.  
Li, C., Zhu, Y., Qi, C., Liu, L., Zhang, D., Wang, X., She, K., Jia, Y., Liu, T., He, D. and Xiong, M., 2021. Estimating the Prevalence of Asymptomatic COVID-19 Cases and Their Contribution in Transmission-Using Henan Province, China, as an Example. Frontiers in medicine, 8.
Ma, Qiuyue, Jue Liu, Qiao Liu, Liangyu Kang, Runqing Liu, Wenzhan Jing, Yu Wu, and Min Liu. "Global percentage of asymptomatic SARS-CoV-2 infections among the tested population and individuals with confirmed COVID-19 diagnosis: a systematic review and meta-analysis." JAMA network open 4, no. 12 (2021): e2137257-e2137257.
McAloon, C., Collins, Á., Hunt, K., Barber, A., Byrne, A.W., Butler, F., Casey, M., Griffin, J., Lane, E., McEvoy, D. and Wall, P., 2020. Incubation period of COVID-19: a rapid systematic review and meta-analysis of observational research. BMJ open, 10(8), p.e039652.  
Noh, J. and Danuser, G., 2021. Estimation of the fraction of COVID-19 infected people in US states and countries worldwide. PloS one, 16(2), p.e0246772.
Paul, S. and Lorin, E., 2021. Distribution of incubation periods of COVID-19 in the Canadian context. Scientific Reports, 11(1), pp.1-9.  
Phipps, S.J., Grafton, R.Q. and Kompas, T., 2020. Robust estimates of the true (population) infection rate for COVID-19: a backcasting approach. Royal Society Open Science, 7(11), p.200909.</i>

## Method
### Script
The observations data being used for seeding the model and in calibration of the model with ABC is created in [`CreatingObservations-Daily.ipynb`](https://github.com/Urban-Analytics/RAMP-UA/blob/Mollys_DA/experiments/calibration/observation_data/CreatingObservations-Daily.ipynb)

A variation of the method used to create the data in this script was also investigated, but was concluded to be less effective.   
The [`TestingMethod`](https://github.com/Urban-Analytics/RAMP-UA/blob/Mollys_DA/experiments/calibration/observation_data/TestingMethod/) directory contains [`CreatingObservations-InterpolateFirst.ipynb`](https://github.com/Urban-Analytics/RAMP-UA/blob/Mollys_DA/experiments/calibration/observation_data/TestingMethod/CreatingObservations-Daily-InterpolateFirst.ipynb), the notebook used for testing this method, as well as [`CreatingObservations-InterpolateSecond.ipynb`](https://github.com/Urban-Analytics/RAMP-UA/blob/Mollys_DA/experiments/calibration/observation_data/TestingMethod/CreatingObservations-Daily-InterpolateSecond.ipynb) which contains the same method as in [`CreatingObservations-Daily.ipynb`](https://github.com/Urban-Analytics/RAMP-UA/blob/Mollys_DA/experiments/calibration/observation_data/CreatingObservations-Daily.ipynb) but with some additional code to compare to the other method. This directory also contains the `csv` files created by both of these scripts.  

The method in [`CreatingObservations-Daily.ipynb`](https://github.com/Urban-Analytics/RAMP-UA/blob/Mollys_DA/experiments/calibration/observation_data/CreatingObservations-Daily.ipynb) converts weekly data showing the number of new positive test results in the last 7 day period into an estimate of the number of new cases each day. This conversion process includes the following steps:

* Shifting the data back in time by 6 days
* Multiplying the data by a month-specific multiplier
* Linearly interpolating from weekly to daily data
* Smoothing the data

More details on these stages are provided in the notebook.

## Outputs
This directory contains the following `csv` files which are produced in the notebook. In each case for both the whole of Devon and for each MSOA individually:
* weekly_cases_*.csv
  * This is the weekly test result data after shifting and multiplying
  * This data is used in the distance function for evaluating the model performance during ABC
* daily_cases_*.csv
  * This is the shifted, multiplied data, interpolated to daily values and smoothed
  * daily_cases_devon.csv is used for seeding the model. For this purpose it is read from the [`microsim/opencl/data/`](
https://github.com/Urban-Analytics/RAMP-UA/blob/Mollys_DA/microsim/opencl/data) directory.
* weekly_cases_*_aggregated_from_daily.csv
  * This is data resulting from reaggregating the daily data from the stage above to weekly values (i.e. the shifted, multiplied, interpolated to daily,  smoothed data). This data is slightly different to the original shifted, multiplied weekly data. 
  * This data is not currently used for anything

# Previous methods for deriving observations data
Two methods were previously applied to generate the observations data. 

#### 1. gam_cases.csv
The R script [`getUKCovidTimeSeries.R`](https://github.com/Urban-Analytics/RAMP-UA/blob/master/experiments/calibration/observation_data/getUKCovidTimeSeries.R) retrieves the latest covid case and hospital admissions data, and produced the case data for Devon [`devon_cases.csv`](https://github.com/Urban-Analytics/RAMP-UA/tree/master/experiments/calibration/observation_data/devon_cases.csv). The [`gam_cases.R`](https://github.com/Urban-Analytics/RAMP-UA/blob/master/experiments/calibration/observation_data/gam_cases.R) script was then used to smooth the cases, creating the output `gam_cases.csv`.

#### 2. england_initial_cases.csv
A second approach was based on modelling daily case data from the weekly cases at hospital and bi-weekly infection survey results, using [`createSeddingFiles.R`](https://github.com/Urban-Analytics/RAMP-UA/blob/Ecotwins-withCommuting/lab/createSeedingFiles.R). This creates [`england_initial_cases.csv`](https://github.com/Urban-Analytics/RAMP-UA/tree/master/experiments/calibration/observation_data/england_initial_cases.csv) in which D0 refers to 05/03/20.



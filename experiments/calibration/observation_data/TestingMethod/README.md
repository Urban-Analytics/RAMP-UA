## Testing methods for preparing estimates of daily Covid cases in Devon

This directory contains two ipython notebooks exploring two variations on a general method for estimating daily Covid cases in Devon.  
Both the methods use government data on reported weekly positive test results, and attempt to convert this to a true number of daily infections.  
This involves the following steps, more info on which is provided within the notebooks.

* Shifting the data back in time by 6 days
* Multiplying the data by a month-specific multiplier
* Linearly interpolating from weekly to daily data
* Smoothing the data

The main difference between the notebooks is the order in which these steps are performed. 

## Results
The csv's files in this directory are the outputs from these two notebooks:  
* *_IF.csv are from CreatingObservations-Daily-InterpolateFirst.ipynb
* *_IS.csv are from CreatingObservations-Daily-InterpolateSecond.ipynb

In each case, there are csv's with cases for MSOAs in Devon individually, and for the whole of Devon, and at both a daily and weekly time-step.  

## Conclusions
The results from the two notebooks are compared, and it is concluded that the results from [`CreatingObservations-Daily-InterpolateSecond.ipynb.R`](https://github.com/Urban-Analytics/RAMP-UA/tree/Mollys_DA/experiments/calibration/observation_data/TestingMethod/CreatingObservations-Daily-InterpolateSecond.ipynb) are better, and so this script is used to produce the observations used in the model.   
  
These observations are used for seeding the model in the [`microsim/opencl/data/`](
https://github.com/Urban-Analytics/RAMP-UA/blob/Mollys_DA/microsim/opencl/data) directory, and also as the observations for calibrating the model with ABC in the [`experiments/calibration`](
https://github.com/Urban-Analytics/RAMP-UA/blob/Mollys_DA/experiments/calibration) directory.




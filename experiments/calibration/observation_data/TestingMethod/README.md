## Testing methods for preparing estimates of daily Covid cases in Devon

This directory contains two ipython notebooks exploring two variations on a general method for descibing estimates of daily Covid cases in Devon.  
Both the methods use government data on reported weekly positive test results, and attempt to convert this to a true number of daily infections.  
This involves the following steps, more info on which is provided within the notebooks.

* Shifting the data back in time by 6 days
* Multiplying the data by a month-specific multiplier
* Linearly interpolating from weekly to daily data
* Smoothing the data

The main difference between the notebooks is the order in which these steps are performed.  
The results from the two notebooks are compared, and it is concluded that the results from [`CreatingObservations-Daily-InterpolateSecond.ipynb'](https://github.com/Urban-Analytics/RAMP-UA/tree/Mollys_DA/experiments/calibration/observation_data/TestingMethod/CreatingObservations-Daily-InterpolateSecond.ipynb) are better, so 





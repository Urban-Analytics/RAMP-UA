# RAMP Calibration

This directory holds the files used for performing both an initial calibration of the model to establish a value for current_risk_beta (which is a parameter controlling the general transmissability of the disease in the model), and for performing dynamic calibration of the model to allow the parameteristation of the individual and location hazard paramaters to be adapted as the model runs forward in time and the behaviour of the disease evolves.  

## ../opencl_runner.py

Contains useful convenience functions for working with the OpenCL model and extracting useful data from the results. In particular see the `run_model*` functions.

## InitialModelCalibration.ipynb

Contains code which calibrates the location (home, retail, work, school) and individual (symptomatic, asymptomatic, presymptomatic) hazard parameters, as well as the current risk beta parameter. In this script, these parameters are calibrated once using historical data and Approximate Bayesian Computation (ABC). The model is run for 133 days which covers the period from March up until July 2020. Ten populations are used in the ABC process. Prior distributions are initially defined for the parameters based on knowledge from other diseases.

Each ABC population contains 100 particles or parameter vectors. For the final population the best particle is defined as that with the lowest distance value (between the predictions made using the particle's parameter values and the observations data). The current risk beta value from this particle is taken to be the optimal current risk beta value for the model. 

PLOT all the current risk beta values for final population!? to check distribution

## RunModelWithDynamicCalibration.ipynb

Contains code for running the model with dynamic calibration using ABC. This runs the model forward in time, but allowing for emerging data on case numbers to be fed in every two weeks, and for model parameter values to be updated accordingly.  

In this, the optimal current risk beta value from the initial model calibration stage is set as a constant. Priors are provided for the other location and individual hazards. The model is run for 14 days initially with ABC and ten populations. At the end of this period the parameter vectors from the 100 particles in the final population are used to produce parameter value distributions. These are then used as priors and the model is run again, this time for 28 days. This process can be continued, adding on 14 days to the run time each time. 

Q: if we are assuming that parameter values change over time, then surely applying them over the whole time period always starting from day 0, is not optimal??

# RAMP Calibration

This directory holds the files used for performing both an initial calibration of the model to establish a value for current_risk_beta (which is...), and for performing dynamic calibration of the model to allow the parameteristation to be adapted as the model runs forward in time and the beheaviour of the disease may be evolving.  

## ../opencl_runner.py

Contains useful convenience functions for working with the OpenCL model and extracting useful data from the results. In particular see the `run_model*` functions.

## calibration.ipynb

Example of calibration using a basic minimisation algorithm (Nelder-Mead Simplex), Differential Evolution (a genetic algorithm) and Approximate Bayesian Computation on one parameter

## abc-1.ipynb

Multi-parameter calibration of the model using ABC and then make predictions using the posterior of the parameters.

## abc-2.ipynb

Based on `abc-1.py` but calibrates on the location-specific hazard multipliers. 



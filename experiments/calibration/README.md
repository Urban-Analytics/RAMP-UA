# RAMP Calibration

This directory holds various files that have been used to explore different methods for calibrating the RAMP model, and to see whether automatic calibrate might help to teach us about the possible values for parameters that we don't have reliable (real world) estimates for.

## calibration.py

Example of calibration using a basic minimisation algorithm (Nelder-Mead Simplex), Differential Evolution (a genetic algorithm) and Approximate Bayesian Computation on one parameter

## abc-1.py

Multi-parameter calibration of the model using ABC and then make predictions using the posterior of the parameters.

## abc-2.py

Based on `abc-1.py` but calibrates on the location-specific hazard multipliers. 



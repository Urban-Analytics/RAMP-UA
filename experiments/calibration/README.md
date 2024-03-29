# RAMP Calibration

This directory holds the files used for adapting the ABM to to allow it to be optimised in response to data emerging in real time. This involves performing dynamic calibration (i.e. re-calibrating at every model time step) using ABC to allow the parameteristation of the individual and location hazard paramaters to be adapted as the model runs forward in time and the behaviour of the disease evolves. Previously, the ABM was just calibrated once using historical data and ABC. An initial calibration is still required here to establish a value for current_risk_beta (which is a parameter controlling the general transmissability of the disease in the model).

## opencl_runner.py

Contains useful convenience functions for working with the OpenCL model and extracting useful data from the results. In particular see the `run_model*` functions.

## ArbitraryDistribution.py

This contains code to take an abc_history object (output from an ABC run), and to generate a distribution using KDE from the parameter values of the final population from the ABC run. This is required so that the posterior from an ABC run can be re-used as the prior in a new run. 

It also contains the GreaterThanZeroParameterTransition class. This is because when using pyabc's default MultiVariateNormal transition negative values end up being selected for various parameters. As all parameters relate to the risk associated with various locations/individual's disease states, having a negative value is non-sensical.  

## InitialModelCalibration.ipynb

Contains code which calibrates the location (home, retail, work, school) and individual (symptomatic, asymptomatic, presymptomatic) hazard parameters, as well as the current risk beta parameter. In this script, these parameters are calibrated once using historical data and Approximate Bayesian Computation (ABC). The model is run for 133 days which covers the period from March up until July 2020. Ten populations are used in the ABC process. Prior distributions are initially defined for the parameters based on knowledge from other diseases.

Each ABC population contains 100 particles or parameter vectors. For the final population the best particle is defined as that with the lowest distance value (between the predictions made using the particle's parameter values and the observations data). The current risk beta value from this particle is taken to be the optimal current risk beta value for the model. 

## RunModel_OneShotCalibration.py

Contains code for running the model with one-shot calibration using ABC. This runs the model over a set time period, using ABC to calibrate the model parameters over that full time period.  The optimal current risk beta value from the initial model calibration stage is set as a constant. Priors are provided for the other location and individual hazards. The model is ran this way using a variety of time periods (14, 28, 42, 56, 70, 84 and 98 days). This allows a comparison to be made of running the model with dynamic calibration over each of these time periods, with running it with one-shot calibration. 

## RunModel_DynamicCalibration.py

Contains code for running the model with dynamic calibration using ABC. This runs the model forward in time, but allowing for emerging data on case numbers to be fed in every two weeks, and for model parameter values to be updated accordingly.  

In this, the optimal current risk beta value from the initial model calibration stage is set as a constant. Priors are provided for the other location and individual hazards. The model is run for 14 days initially with ABC and ten populations. At the end of this period the parameter vectors from the 100 particles in the final population are used to produce parameter value distributions. These are then used as priors and the model is run again, this time for 28 days. This process can be continued, adding on 14 days to the run time each time.

## AnalyseResults-OneShotCalibration.ipynb

Analyses the results of running the model with a one-shot calibration. Plot the parameter values for each population, the predictions made within each population, and the predictions for the future using the parameter values from the one-shot calibration.

## AnalyseResults-DynamicCalibration_*.ipynb

Analyses the results of running the model with dynamic calibration. Explores the evolution of the parameter values with dynamic calibration over the course of the pandemic; the predictions made by the model within each calibration window; and the predictions made when the parameter values from the final dynamic calibration window are used to run the model forward in time for 105 days.

## Outputs

Contains the .pkl files from running the model (in the two files listed above), as well as the database files that the pickle files draw from.


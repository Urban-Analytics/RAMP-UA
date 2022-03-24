# Outstanding issues

* As the model runs forward in time and the behaviour of the disease evolves, the parameterisation of the model may need to change in order to make accurate predictions. To allow for this, the model is set up to run with dynamic calibration. The model is run with ABC over 14 day intervals. After the first 14 day period the posterior estimate of the parameter values is then used as a prior for the next 14 days. However, currently, instead of the model continuing to run from day 15 in the second interval, it starts again from day 0 and continues up to day 28, and then in the next instance starts from day 0 and continues up to day 42. Given the assumption that parameter values are changing over time, this is not optimal, as we are always calibrating on the whole time period, rather than just the last 14 days. 


However, we can’t do this (why not?). One reason why not doing this is bad is because we are suggesting that the parameters change over time, but then we are using the parameter values for the later days on the early days in later runs. But also reasons why always running it from day 0 is advantageous e.g. if days 0-14 go badly then if we don’t re do them then there is no way to correct for this. 
. In this case, state of particles remains the same as it was at the end of the first window. Problem with this – is that if any of the particles are doing very badly then changing global parameters won’t make that difference (e.g. won’t stop a Covid outbreak happening in the wrong place.) so, instead, will take new parameter distributions and start all over again. Run from day 0-14. And then repeat, days 15-22. 




* Currently, the accuracy of the model is determined by comparing the number of cases over the whole of Devon each week in the model with the observations. Ideally, the model would consider the distribution of cases at the MSOA level, rather than for Devon as a whole. If the model did consider the distribution of the cases in each MSOA, then ideally it would be able to consider the cases in MSOAs that are close together. For instance, if there is an outbreak of Covid in the model in one MSOA, but in the observations this isn't present in this MSOA, but is in a neighbouring MSOA, then this shouldn't be penalised as heavily.  
  * Previously code in OpenCLWrapper.disance() allowed for comparison of case numbers in each MSOA. This code is still in opencl_runner.py so can try and adapt this (but take care as method for comparing case numbers used in this (summing numbers in disease states 1-4 each day) is not correct). 

* Is comparing cases each week a fine enough temporal resolution?

* The case seeding process currently weights the cases across Devon on the basis of the MSOA's risk rating (which is determined by how many people are likely to bring Covid into the area (numbers of students, young people etc)). Ideally, however, this could be based on the actual case distribution amongst MSOAs in the observations. 

* When running the model for the initial parameter calibration some of the parameters do not seem to stabilise towards a consistent distribution even after 10 populations. Why is this? May be that the value for that parameter doesn't actually influence the outcome of the model. A (theoretical) next stage could be to look at the model and why it is not sensitive to these parameters.

* Need to also look at how parameter values change when running model with dynamic calibration over longer period - i.e. do we see a clear change in the parameterisation or not?
•	To know whether we learn anything from the changes to the parameter posteriors? E.g. does the shape of beta change as the parameters evolve?



* •	To know whether the Bayesian updating makes any difference to the quality of the predictions (this will require a model ran without pyabc to compare to (also using observations by MSOA – but Nick said not to worry about this for now, as it should be easy and he will be able to do this) – how would we do this? 
* Assess the improvement in predictive ability offered by the combined model compared to a model based purely on historical data.

* state estimation/correction -- Look at one particle and say e.g. there has been a Covid outbreak in one area, and there wasn’t one in that area, then we dampen it down a bit, so this is a traditional data assimilation approach.  But not doing that here currently, because too hard. Might be next stage once calibration of parameters is working

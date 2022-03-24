# Outstanding issues

* As the model runs forward in time and the behaviour of the disease evolves, the parameterisation of the model may need to change in order to make accurate predictions. To allow for this, the model is set up to run with dynamic calibration. The model is run with ABC over 14 day intervals. After the first 14 day period the posterior estimate of the parameter values is then used as a prior for the next 14 days. However, currently, instead of the model continuing to run from day 15 in the second interval, it starts again from day 0 and continues up to day 28, and then in the next instance starts from day 0 and continues up to day 42. Given the assumption that parameter values are changing over time, this is not optimal, as we are always calibrating on the whole time period, rather than just the last 14 days. 

* Currently, the accuracy of the model is determined by comparing the number of cases over the whole of Devon each week in the model with the observations. Ideally, the model would consider the distribution of cases at the MSOA level, rather than for Devon as a whole. If the model did consider the distribution of the cases in each MSOA, then ideally it would be able to account for MSOAs that are close together maybe not having the cases distributed quite right.
  * Previously code in OpenCLWrapper.disance() allowed for comparison of case numbers in each MSOA. This code is still in opencl_runner.py so can try and adapt this (but take care as method for comparing case numbers used in this (summing numbers in disease states 1-4 each day) is not correct). 

* Is comparing cases each week a fine enough temporal resolution?

* The case seeding process currently weights the cases across Devon on the basis of the MSOA's risk rating (which is determined by how many people are likely to bring Covid into the area (numbers of students, young people etc)). Ideally, however, this could be based on the actual case distribution amongst MSOAs in the observations. 


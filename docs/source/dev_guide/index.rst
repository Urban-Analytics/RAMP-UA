Developers Guide
===================================

This page provides a broad overview of how the model works. It is intended for programmers who want to read and/or edit the code.

.. highlight:: bash


Source Code and Packages
---------------------------------

The main package for the source code is :py:class:`microsim` and the entry-point for the program is the function :py:func:`microsim.main` in ``microsim/main.py``. To run the model and see what arguments are available, execute the following from the root project directory::

    $ python microsim/main.py --help

(for full details about running the model see the :doc:`usage guide<../usage/index>`).

Regardless of whether the Python or OpenCL versions of the model are executed, ``main.py`` wil do the following things:

1. Process the command line arguments

2. Load the data. These can either be read from a cache that was created previously or else the raw files need to be read. The :ref:`population_initialisation.py<population_initialisation-label>` file reads the data. It creates a pandas DataFrame called ``individuals``, which holds the characteristics of every individual in the study area, and a dictionary called ``activity_locations`` which points to DataFrames for the locations that people will visit (e.g. home, school, retail, etc.)

3. ``main.py`` reads a *time activity multiplier* which is a file that is used to change peoples' normal behaviour so that them spend more time at home (e.g. during a lockdown).

4. Finally, ``main.py`` determines whether the OpenCL model or the Python model should be executed and passes the required arguments to the appropriate classes. 

The Python and OpenCL models are introduced in more detail below, but broadly one of two things will happen:

* If the **OpenCL** model is executed: a secondary cache, called a *snapshot* is created or read in if the model has been executed once already. Basically the model takes the pandas DataFrames created earlier and converts them into arrays. The ``microsim/opencl/ramp/run.py`` file is called and a ``Simulator`` class (``microsim/opencl/ramp/simualtor.py``) is created and stepped to run the model.

* If the **Python** model is executed: a ``Microsim`` object, see  :ref:`microsim/microsim_model.py<microsim-label>`, is created to run the model by calling ``Microsim.run()`` Note that if the model is going to be run multiple times with the same parameters (*repetitions*) then a multi-process wrapper calls a few ``Microsim.run()`` functions simultaneously using different processes.


.. _population_initialisation-label:

Population Initialisation (``microsim/population_initialisation.py``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The task of the :py:class:`microsim.population_initialisation.PopulationInitialisation` class is to read the data that represents the synthetic population at the core of the model. This includes csv files with information about each individual, as well as files used to estimate workplace locations, etc. That file is commented quite extensively, so see the documentation there for further details.

The most important thing that the :py:class:`microsim.population_initialisation.PopulationInitialisation` class does is to return the ``individuals`` dataframe and the ``activity_locations`` dictionary (that points to dataframes for the locations where indvidiauals will spend their time).


.. _microsim-label:

Microsim class (``microsim/microsim_model.py``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`microsim.microsim_model.Microsim` class is the main entry point for the python model. It contains the main code to control the spread of the disease. See the :py:func:`microsim.microsim_model.Microsim.step` function for details about what, specifically, happens in each iteration of the model. The OpenCL model follows the same procedures, although they are implemented differently.

The other thing to note is the :py:func:`microsim.microsim_model.Microsim.calculate_new_disease_status` function. That function calls the file ``microsim/r_interface.py`` file which in turn delegates the estimation of an individual's specific disease status.


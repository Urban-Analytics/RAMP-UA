Setting up the python model
=============================

.. highlight:: bash

Setup 
-------

This project currently supports running on Linux and macOS.

To start working with this repository you need to clone it onto your local machine: ::

    $ git clone https://github.com/Urban-Analytics/RAMP-UA.git
    $ cd RAMP-UA

This project requires a specific conda environment in order to run so you will need the 
`conda package manager system <https://docs.anaconda.com/anaconda/install/>`_ installed. 
Once conda has been installed you can create an environment for this project using the provided environment file. ::

    $ conda env create -f environment.yml

To retrieve data to run the mode you will need to use Git Large File Storage to download the input data. 
Git-lfs is installed within the conda environment (you may need to run git lfs install on your first use of git lfs). 

To retrieve the data you run the following commands within the root of the project repository: ::

    $ git lfs fetch
    $ git lfs checkout

Next we install the RAMP-UA package into the environment using setup.py: ::

    # if developing the code base use:
    $ python setup.py develop
    # for using the code base use
    $ python setup.py install

Running the model
--------------------

Both models can be run from the microsim/main.py script, which can be configured with various arguments to choose which model implementation to run.

The Python / R model runs by default, so simply run the main script with no arguments. ::

$ python microsim/main.py 
import os
import pandas as pd
import rpy2.rinterface
import rpy2.robjects.packages as rpackages  # For installing packages
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

class RInterface():
    """
    An RInterface object can be used to create an R session, initialise everything that is needed for the disease
    status estimation function, and then interact with session to calculate the disease status.
    """

    def __init__(self, script_dir):
        """

        :param script_dir: The directory where the scripts can be found
        """
        print(f"Initialising R interface. Loading R scripts in {script_dir}.")
        R = ro.r
        R.setwd(script_dir)
        R.source('covid_status_functions.R')
        R.source("initialize_and_helper_functions.R")
        try:
            R.source("covid_run.R")
        except rpy2.rinterface.RRuntimeError as e:
            # R libraries probably need installing. THe lines below *should* do this, but it's probably better
            # to install them manually.
            #print("\tInstalling required libraries")
            #RInterface._install_libs(R)
            print(f"\tError trying to start R: {str(e)}. Libraries probably need installing. Look in the file"
                  f"'R/py_int/covid_run.R' to see which libraries are needed.")
            raise e

        # Remember the session
        self.R = R

    def calculate_disease_status(self, individuals: pd.DataFrame):
        """
        Call the R 'run_status' function to calculate the new disease status. It will return a new dataframe with
        a few columns, including the new status.
        :param individuals:  The individuals dataframe from which new statuses need to be calculated
        :return: a new dataframe that includes new disease statuses
        """
        print("Calculating new disease status...",)
        out_df = self.R.run_status(individuals)
        print(" .... finished.")
        assert len(out_df) == len(individuals)
        return out_df


    @staticmethod
    def _install_libs(R: rpy2.robjects.r):
        """Install the libraries required to run the script using the given R session.
        See https://rpy2.github.io/doc/latest/html/introduction.html."""
        # TODO (not important) see if this method works properly. Not important because libraries can be pre-loaded
        # and included in the anaconda environment anyway
        # import R's utility package
        utils = rpackages.importr('utils')
        # select a mirror for R packages
        utils.chooseCRANmirror(ind=1) # select the first mirror in the list
        # R package names
        packnames = ("dplyr", "tidyr", "janitor", "readr", "mixdist")
        # R vector of strings
        from rpy2.robjects.vectors import StrVector
        # Selectively install what needs to be install.
        # We are fancy, just because we can.
        names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
        if len(names_to_install) > 0:
            utils.install_packages(StrVector(names_to_install))
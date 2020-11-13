import os
import pandas as pd
import rpy2.rinterface
import rpy2.robjects.packages as rpackages  # For installing packages
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from microsim.column_names import ColumnNames

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
        self.script_dir = script_dir  # Useful to remember for debugging, but not actually needed
        R = ro.r
        try:
            # Read the script (doesn't run any functions)
            R.source(os.path.join(script_dir,"covid_run.R"))
            # Call a function to initialize the needed R packages and data
            R.initialize_r()
        except rpy2.rinterface.embedded.RRuntimeError as e:
            # R libraries probably need installing. THe lines below *should* do this, but it's probably better
            # to install them manually.
            #print("\tInstalling required libraries")
            #RInterface._install_libs(R)
            print(f"\tError trying to start R: {str(e)}. Libraries probably need installing. Look in the file"
                  f"'R/py_int/covid_run.R' to see which libraries are needed.")
            raise e

        # Remember the session
        self.R = R

    def calculate_disease_status(self, individuals: pd.DataFrame, iteration: int, repnr: int, disease_params: dict ):
        """
        Call the R 'run_status' function to calculate the new disease status. It will return a new dataframe with
        a few columns, including the new status.
        :param individuals:  The individuals dataframe from which new statuses need to be calculated
        :param iteration: The iteration number (i.e. number of model steps so far)
        :param repnr: The repetition number of the model. Like a unique ID for the model.
        :param disease_params: A dictionary of disease parameters used in the R model.
        :return: a new dataframe that includes new disease statuses
        """
        print("\tCalculating new disease status...", end='')
        # It's expesive to convert large dataframes, only give the required columns to R.
        cols = ["area", "House_ID", "ID", "age", "Sex", ColumnNames.CURRENT_RISK, "pnothome",
                ColumnNames.DISEASE_STATUS, ColumnNames.DISEASE_PRESYMP, ColumnNames.DISEASE_SYMP_DAYS,
                ColumnNames.DISEASE_EXPOSED_DAYS, "cvd", "diabetes", "bloodpressure", "BMIvg6", "BMI_healthier"]
        individuals_reduced = individuals.loc[:, cols]
        individuals_reduced["area"] = individuals_reduced.area.astype(str)
        individuals_reduced["id"] = individuals_reduced.ID
        del individuals_reduced["ID"]
        individuals_reduced["house_id"] = individuals_reduced.House_ID
        del individuals_reduced["House_ID"]

        # Call the R function. The returned object will be converted to a pandas dataframe implicitly
        r_df = self.R.run_status(individuals_reduced, iteration, repnr, **disease_params)

        assert len(r_df) == len(individuals)
        assert False not in list(r_df.ID.values == individuals.ID.values)  # Check that person IDs are the same

        # Update the individuals dataframe with the new values
        for col in [ColumnNames.DISEASE_STATUS, ColumnNames.DISEASE_PRESYMP, ColumnNames.DISEASE_SYMP_DAYS, ColumnNames.DISEASE_EXPOSED_DAYS]:
            individuals[col] = list(r_df[col])
        assert False not in (individuals.loc[individuals[ColumnNames.DISEASE_STATUS] > 0, ColumnNames.DISEASE_STATUS].values ==
                             r_df.loc[r_df[ColumnNames.DISEASE_STATUS] > 0, ColumnNames.DISEASE_STATUS].values)

        print(" .... finished.")
        return individuals


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


import pandas as pd
from typing import List
import matplotlib.pyplot as plt

#from microsim.microsim_model import ActivityLocation


class MicrosimAnalysis:
    """Does some analysis of the synthetic population"""



    # Not sure what variables should persist yet. Maybe the fig object etc?
    def __init__(self):
        pass

    @classmethod
    def population_distribution(cls, individuals: pd.DataFrame, variables: List[str]):
        """
        Create a frequency distribtion of the population, using the given variable
        :param individuals: The DataFrame of the individual population
        :param variable: The variable(s) (DataFrame column(s)) on which to create a histogram. Either a single
        string variable or a list of variables.
        :return: The matplotlib Figure object.
        """
        if type(variables) == str: # If only one variable provided then make it into a list
            variables = [variables]
        # Check the columns exist in the dataframe
        for var in variables:
            if var not in individuals.columns:
                raise Exception(f"Column {var} is not in the individuals dataframe.\n Possible columns",
                                f"to use are {individuals.columns}")


        # Create main plot object
        fig = plt.figure(figsize=(12, 14))
        fig.tight_layout()

        # Now add 'sub plots', i.e. plots for each of the variables

        # Iterate over each variable, getting a counter (i) and the variable name (var)
        for i, var in enumerate(variables):
            # Create a new subplot. Currently one column with a new row for each plot.
            ax = fig.add_subplot(len(variables), 1, i+1) # (rows, cols, index) (index used to distinguish each subplot)
            # Subset the data using pandas.loc. Keep all rows but only select the column we want
            data = individuals.loc[:, var]
            ax.hist(data)
            #ax.scatter(data.DateTime, data.Count, s=0.02, c="black")
            ax.set_title(f"Histogram of {var}")

        return fig

    #@classmethod
    #def location_danger_distribution(cls, activity_location: ActivityLocation):
    #    """
    #    Do a frequency distribution for the danger of some locations (e.g. shops)
    #    :param activity_location:
    #    :param variable:
    #    :return:
    #    """
    #    ids = activity_location.get_ids() # (not needed?)
    #    dangers = activity_location.get_dangers()
    #    print(dangers)

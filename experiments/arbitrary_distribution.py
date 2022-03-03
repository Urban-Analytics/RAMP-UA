import warnings

import pyabc
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
import itertools

from pyabc import History
from pyabc.transition.multivariatenormal import MultivariateNormalTransition  # For drawing from the posterior
from pyabc.random_variables import Distribution
from pyabc.parameters import Parameter

pyabc.settings.set_figure_params('pyabc')  # for beautified plots
x=1
class ArbitraryDistribution(Distribution):
    """
    Define an arbitrary distribution. Used so that the posterior from an ABC run can be re-used
    as the prior in a new run. Does this by taking the abc_history object (output from an
    ABC run) and generating a MultivariateNormalTransition (KDE) that can be sampled
    """
    def __init__(self, abc_hist: History):
        # Get the dataframe of particles (parameter point estimates) and associated weights
        dist_df, dist_w = abc_hist.get_distribution(m=0, t=abc_hist.max_t)
        
        if 'current_risk_beta' in dist_df.columns:
            dist_df = dist_df.drop(columns = 'current_risk_beta')
        
        # Create a KDE using the particles
        self.kde = MultivariateNormalTransition(scaling=1)
        self.kde.fit(dist_df, dist_w)

        # set as variable on the object
        self.abc_hist = abc_hist

    def display(self):
        print("display")
        # Get the dataframe of particles (parameter point estimates) and associated weights
        dist_df, dist_w = self.abc_hist.get_distribution(m=0, t=self.abc_hist.max_t)

        if 'current_risk_beta' in dist_df.columns:
            dist_df = dist_df.drop(columns = 'current_risk_beta')

        fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
        x = np.linspace(-1, 2, 99)  # (specified so that we have some whole numbers)
        marker = itertools.cycle((',', '+', '.', 'o', '*'))
        i=0
        for variable in dist_df.columns:
            print(variable)
            ax = axes.flatten()[i]
            pyabc.visualization.plot_kde_1d(
                dist_df, dist_w,
                xmin=0, xmax=1.5,
                x=variable, ax=ax)
            #ax.legend()
            #fig.show()
            i=i+1
        #ax.legend()
        #fig.tight_layout()
        fig.suptitle("Priors")
        fig.show()

    def rvs(self) -> Parameter:
        #print("rvs")
        """Sample from the joint distribution, returning a Parameter object.
           Just calls rvs() on the underlying kde"""
        val = self.kde.rvs()
        while val['asymptomatic']<0 or val['secondary_school']<0 or val['primary_school']<0 or val['retail']<0 or val['presymptomatic']<0 or val['symptomatic']<0 or val['work']<0:
            val = self.kde.rvs()
            #print('resampling')
        return val

    def pdf(self, x: Union[Parameter, pd.Series, pd.DataFrame]) -> Union[float, np.ndarray]:
        #print("pdf")
        """Get probability density at point `x` (product of marginals).
        Just calls pdf(x) on the underlying kde"""
        return self.kde.pdf(x)

    def __repr__(self):
        #print("_repr_")
        print(f"<ArbitraryDistribution\n    " + \
        ",\n    ".join(str(p) for p in self.kde.X.columns) + ">")
        return f"<ArbitraryDistribution\n    " + \
               ",\n    ".join(str(p) for p in self.kde.X.columns) + ">"

    def copy(self) -> Distribution:
        print("copy")
        """Copy the distribution.
        Returns
        -------
        copied_distribution: Distribution
            A copy of the distribution.
        """
        return copy.deepcopy(self)

class GreaterThanZeroParameterTransition(MultivariateNormalTransition):
    """A transition that slightly extends MultivariateNormalTransition to only allow new
    distributions for which all parameters have values greater than 0"""

    def rvs_single(self) -> Parameter:
        """Override the default behaviour of the MultivariateNormalTransition class
        to repeatedly create perturbations until all parameters are above 0. Warns if
        it has to try more than 10 times."""
        sample_ind = np.random.choice(self._range, p=self.w, replace=True)
        sample = self.X.iloc[sample_ind]
        perturbed = sample + np.random.multivariate_normal(
            np.zeros(self.cov.shape[0]), self.cov)
        counter = 0
        # Check that all parameter values are > 0 and sample again if not
        while not (perturbed > 0).all():
            perturbed = sample + np.random.multivariate_normal(
                np.zeros(self.cov.shape[0]), self.cov)
            counter += 1
            if counter % 10 == 0:
                warnings.warn(f"NoZeroPatameterTransition has tried {counter} times to draw a sample for "
                              f"which all parameters are greater than 0")
        return Parameter(perturbed)

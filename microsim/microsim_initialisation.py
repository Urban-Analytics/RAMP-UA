#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Used to create data that can be used to parameterise the disease estimates. These data
need to be created so that the coefficients can be calculated.

Created on Wed 39 May 14:30

@author: nick
"""
import sys
sys.path.append("microsim")
import click  # command-line interface
import os
import pandas as pd

from microsim_model import Microsim



class MicrosimInit():
    """
    Create some preliminary data that are needed to estimate coefficients for the disease estimates.

    Process is as follows:
      1. Read a list of the most high risk MSOAs per day for N (=6?) days
      2. Read the total number of incidences per day.
      3. Then iterate for N days:
         - At the start of each day, randomly distribute cases to randomly chosen individuals to
         the high-risk MSOAs
         - Run step the model to calculate a danger score
         - Save the danger score for each individual each day

    Also repeat the above process M times.
    """

    def __init__(self):
        pass





# PROGRAM ENTRY POINT
# Uses 'click' library so that it can be run from the command line
@click.command()
@click.option('--repetitions', default=10, help='Number of times to repeat the initialisation process')
@click.option('--data_dir', default="data", help='Root directory to load main model data from')
@click.option('--init_dir', default="init-data", help="Directory that stores initialisation data, and where outputs are written")
def run(repetitions, data_dir, init_dir):
    reps = repetitions
    # To fix file path issues, use absolute/full path at all times
    base_dir = os.getcwd()  # get current directory
    data_dir = os.path.join(base_dir, data_dir)
    init_dir = os.path.join(base_dir, init_dir)

    print(f"Running initialisation process {reps} times. Using model data in {data_dir} and initialisation data in {init_dir}")
    # TODO CREATE OUTPUT DIR
    results_dir = os.path.join(init_dir, "results/")

    # Temporarily only want to use Devon MSOAs
    devon_msoas = pd.read_csv(os.path.join(data_dir, "devon_msoas.csv"), header=None,
                              names=["x", "y", "Num", "Code", "Desc"])
    m = Microsim(study_msoas=list(devon_msoas.Code), data_dir=data_dir)

    # TODO Start looping for number of repetitions

    # TODO Read init data and randomly assign cases to individuals

    # Step the model
    m.step()

    # TODO write out the individuals (will need a new directory for each repetition)
    # TODO also maintain the current risk per individual per day so can write that
    # out in one go (easier for Fiona?)

    print("End of initialisation")


if __name__ == "__main__":
    run()
    print("End of initialisation")


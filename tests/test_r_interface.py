import os
import pytest
import pandas as pd
import numpy as np
from microsim.microsim_model import Microsim
from microsim.r_interface import RInterface
from microsim.column_names import ColumnNames
from microsim.activity_location import ActivityLocation

# ********************************************************
# These tests run through a whole dummy model process
# ********************************************************

test_dir = os.path.dirname(os.path.abspath(__file__))

R_script_dir = os.path.abspath(os.path.join(test_dir, '..','R','py_int'))

# arguments used when calling the Microsim constructor. Usually these are the same
microsim_args = {"data_dir": os.path.join(test_dir,"dummy_data"), "r_script_dir": "./R/py_int", "testing": True, "debug": True,
                 "disable_disease_status": True, 'lockdown_from_file':False}

@pytest.fixture()
def rInterface():
    """
    Set up R interface for testing
    """

    r_int = RInterface(R_script_dir)

    return r_int

@pytest.fixture()
def microsim_inst():
    """
    Set up function for microsim object
    """
    test_sim = Microsim(**microsim_args)

    return test_sim 

def test_calculate_disease_status(rInterface):
    """
    A series of tests for the calculate_disease_status function
    """

    raw_indiv = pd.from_csv(os.path.join(test_dir,'dummy_data','devon-tu_health','Devon_simulated_TU_keyworker_health.csv'))

    r_updated_frame = rInterface.calculate_disease_status(individuals = raw_indiv, iteration = 1, disease_params = dict())

    assert len(raw_indiv) == len(r_updated_frame)


def test_calculate_disease_status_wMicrosim(rInterface, microsim_inst):
    """
    A series of tests testing instantiated microsim individual dataset on function
    """

    r_updated_frame = rInterface.calculate_disease_status(individuals = microsim_inst.individuals, iteration = 1, disease_params = dict())

    assert len(microsim_inst.individuals) == len(r_updated_frame)

    assert microsim_inst.columns in r_updated_frame.columns
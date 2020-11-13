import os
import pytest
import pandas as pd
import numpy as np
from microsim.microsim_model import Microsim
from microsim.r_interface import RInterface

# ********************************************************
# These tests run through a whole dummy model process
# ********************************************************

test_dir = os.path.dirname(os.path.abspath(__file__))

R_script_dir = os.path.join('R','py_int')

# arguments used when calling the Microsim constructor. Usually these are the same
microsim_args = {"data_dir": os.path.join(test_dir,"dummy_data"), "disable_disease_status": True}


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

def test_calculate_disease_status_onestep(rInterface):
    """
    A series of tests for the calculate_disease_status function
    """

    raw_indiv = pd.read_csv(os.path.join(test_dir,'dummy_data','test_r_int_data.csv'), index_col=None)

    r_updated_frame = rInterface.calculate_disease_status(individuals = raw_indiv, iteration = 1,
                                                          repnr=0, disease_params = dict())

    assert raw_indiv.shape[0] == r_updated_frame.shape[0]

def test_calculate_disease_status_multistep(rInterface):
    """
    A series of tests for the calculate_disease_status function
    """

    raw_indiv = pd.read_csv(os.path.join(test_dir,'dummy_data','test_r_int_data.csv'), index_col=None)

    for i in range(1,4):
        r_updated_frame = rInterface.calculate_disease_status(individuals = raw_indiv, iteration = i, 
                                                              repnr=0, disease_params = dict())

        assert raw_indiv.shape[0] == r_updated_frame.shape[0]

# work in progress test
@pytest.mark.skip()
def test_calculate_disease_status_wMicrosim(rInterface, microsim_inst):
    """
    A series of tests testing instantiated microsim individual dataset on function
    """

    r_updated_frame = rInterface.calculate_disease_status(individuals = microsim_inst.individuals, iteration = 1,
                                                          repnr=0, disease_params = dict())

    assert microsim_inst.individuals.shape[0] == r_updated_frame.shape[0]

    assert set(["area","ID","house_id","disease_status","exposed_days","presymp_days","symp_days"]).issubset(set(r_updated_frame.columns.tolist()))
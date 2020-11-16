import pytest
import os
import numpy as np
import pandas as pd
from experiments.opencl_runner import OpenCLRunner # Some additional notebook-specific functions required (functions.py)
from opencl.ramp.disease_statuses import DiseaseStatus

# ********************************************************
# Tests of the functions in functions.py
# ********************************************************

# This 'fixture' means that other test functions can use the object created here.
# Note: Don't try to run this test, it will be called when running the others that need it,
# like `test_step()`.

@pytest.fixture()
def setup_results():
    # Create some model summary data
    params = OpenCLRunner.create_parameters()
    results = OpenCLRunner.run_opencl_model_multi(
        repetitions=5, iterations=100, params=params)
    yield results
    # Colud put close-down code here


def test_fit_l2():
    """Test the fitness function"""
    f = OpenCLRunner.fit_l2

    assert f([1, 2, 3],[1, 2, 3]) == 0
    assert f([1, 2, 3],[1, 2, 4]) == 1
    assert f([1, 2, 3],[1, 2, 5]) == 2

    with pytest.raises(Exception):
        f([1, 2, 3], [1, 2])
        f(1, [1, 2])
        f("b", "a")

def test_get_mean_total_counts(setup_results):
    """Test that we get the correct mean total count from a numberof model repetitions"""
    # Run it on normal data to check basic stuff
    summaries = [result[0] for result in setup_results]
    mean = OpenCLRunner.get_mean_total_counts(summaries, DiseaseStatus.Exposed.value)
    assert len(mean) == len(summaries[0].total_counts[0]), \
        f"The mean length ({len(mean)}) should be the same as the number of iterations"
    # TODO Create some artificial data and check the mean is being calculated correctly
    # TODO Test for each iteation the mean should be within the min and max


def test_run_model_with_params():
    with pytest.raises(Exception):
        # Running a model should fail if the class hasn't been initialised first
        OpenCLRunner.run_model_with_params([1,2])
    # Run the model to check it works as expected.
    OpenCLRunner.init(
        iterations=100,
        repetitions=5,
        observations=pd.DataFrame({"Day":range(1,120), "Cases":range(1,120)}),
        use_gpu=False,
        store_detailed_counts=False,
        parameters_file=os.path.join("model_parameters", "default.yml"),
        opencl_dir=os.path.join("microsim", "opencl"),
        snapshot_filepath=os.path.join("microsim", "opencl", "snapshots", "cache.npz")
    )

    (fitness, sim, obs, out_params, summaries) = OpenCLRunner.run_model_with_params(np.array([
        0.005,  # current_risk_beta
        0.75,  # infection_log_scale
        7.123,  # infection_mode
        1.0,  # presymptomatic
        0.75,  # asymptomatic
        0.5  #symptomatic
    ]), return_full_details=True)

    # Check things look broadly ok
    assert len(sim) == 100  # One disease count per iteration
    assert len(sim) == len(obs)  # Returned observations should be same length as the simulated number of iterations

    # Check the returned parameters are correct (should be different to the default)
    assert out_params.infection_mode == 7.123

    # Individual hazard multipliers should be correct (I'm not sure why I can't access them with their
    # names, e.g. 'out_params.individual_hazard_multipliers.asymptomatic')
    assert out_params.individual_hazard_multipliers[0] == 1.0
    assert out_params.individual_hazard_multipliers[1] == 0.75
    assert out_params.individual_hazard_multipliers[2] == 0.5

    # TODO: change parameters, run again, and check that the results make some sort of sense



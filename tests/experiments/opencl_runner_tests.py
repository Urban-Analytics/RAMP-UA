import pytest
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
        # Running a model sould fail if the class hasn't been initialised first
        OpenCLRunner.run_model_with_params([1,2])
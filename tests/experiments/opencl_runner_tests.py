import pytest
import yaml
import os
import warnings
import numpy as np
import pandas as pd
from experiments.opencl_runner import OpenCLRunner, OpenCLWrapper # Some additional notebook-specific functions required (functions.py)
from opencl.ramp.disease_statuses import DiseaseStatus

# ********************************************************
# Tests of the functions in functions.py
# ********************************************************

# This 'fixture' means that other test functions can use the object created here.
# Note: Don't try to run this test, it will be called when running the others that need it,
# like `test_step()`.
from opencl.ramp.params import Params


@pytest.fixture()
def setup_results():
    # Create some model summary data
    params = OpenCLRunner.create_parameters()
    results = OpenCLRunner.run_opencl_model_multi(
        repetitions=5, iterations=100, params=params)
    yield results
    # Could put close-down code here


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

    fit1 = f([1,2,3], [2,3,4])
    fit2 = f([1,2,3], [5,6,7])
    assert fit1 < fit2
    fit3 = f([5,6,7], [1,2,3])
    assert fit2 == fit3

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

def test_create_parameters():
    """Check that the create_parameters files correctly returns a Parameters object"""
    # Firstly, using the defaults is not recommended usually as the values should be read from the parameters file
    with pytest.warns(UserWarning):
        Params()

    # Read the default parameters file, for checking
    parameters_file = os.path.join("./", "model_parameters/", "default.yml")
    with open(parameters_file) as f:
        parameters_in_file = yaml.load(f, Loader=yaml.SafeLoader)
    default_params = OpenCLRunner.create_parameters()

    # Default params should match those in default.yml.

    # Check the individual multipliers
    assert parameters_in_file['microsim_calibration']['hazard_individual_multipliers']['presymptomatic'] == \
           default_params.individual_hazard_multipliers[0]  # presymp is second item in the array
    assert parameters_in_file['microsim_calibration']['hazard_individual_multipliers']['asymptomatic'] == \
           default_params.individual_hazard_multipliers[1]  # asymp is second item in the array
    assert parameters_in_file['microsim_calibration']['hazard_individual_multipliers']['symptomatic'] == \
            default_params.individual_hazard_multipliers[2]  # symp is third item in the array

    # And the place multipliers (note that the values in the parameter file are multiplied by the
    # current risk beta
    current_risk_beta = parameters_in_file["disease"]["current_risk_beta"]
    assert np.isclose(default_params.place_hazard_multipliers[0],  # Retail is first in the array
        parameters_in_file['microsim_calibration']['hazard_location_multipliers']['Retail'] * current_risk_beta)
    assert np.isclose( default_params.place_hazard_multipliers[1],
        parameters_in_file['microsim_calibration']['hazard_location_multipliers']['PrimarySchool'] * current_risk_beta)
    assert np.isclose(default_params.place_hazard_multipliers[2],
        parameters_in_file['microsim_calibration']['hazard_location_multipliers']['SecondarySchool'] * current_risk_beta)
    assert np.isclose(default_params.place_hazard_multipliers[3] ,
        parameters_in_file['microsim_calibration']['hazard_location_multipliers']['Home'] * current_risk_beta)
    assert np.isclose(default_params.place_hazard_multipliers[4],
        parameters_in_file['microsim_calibration']['hazard_location_multipliers']['Work'] * current_risk_beta)

    # Now see that if we override the defaults it still works (and others stay the same)
    # Set new value to 0.123 (for no particular reason)
    new_params = OpenCLRunner.create_parameters(presymptomatic=0.123)
    assert np.isclose(new_params.individual_hazard_multipliers[0], 0.123)
    assert np.isclose(new_params.individual_hazard_multipliers[1], default_params.individual_hazard_multipliers[1])
    assert np.isclose(new_params.individual_hazard_multipliers[2], default_params.individual_hazard_multipliers[2])
    new_params = OpenCLRunner.create_parameters(asymptomatic=0.123)
    assert np.isclose(new_params.individual_hazard_multipliers[1], 0.123)
    assert np.isclose(new_params.individual_hazard_multipliers[0], default_params.individual_hazard_multipliers[0])
    assert np.isclose(new_params.individual_hazard_multipliers[2], default_params.individual_hazard_multipliers[2])
    new_params = OpenCLRunner.create_parameters(symptomatic=0.123)
    assert np.isclose(new_params.individual_hazard_multipliers[2], 0.123)
    assert np.isclose(new_params.individual_hazard_multipliers[0], default_params.individual_hazard_multipliers[0])
    assert np.isclose(new_params.individual_hazard_multipliers[1], default_params.individual_hazard_multipliers[1])

    new_params = OpenCLRunner.create_parameters(retail=0.123)
    assert np.isclose(new_params.place_hazard_multipliers[0], 0.123 * current_risk_beta)
    assert np.all([new_params.place_hazard_multipliers[i] == default_params.place_hazard_multipliers[i] for i in [1,2,3,4]])
    new_params = OpenCLRunner.create_parameters(primary_school=0.123)
    assert np.isclose(new_params.place_hazard_multipliers[1], 0.123* current_risk_beta)
    assert np.all([new_params.place_hazard_multipliers[i] == default_params.place_hazard_multipliers[i] for i in [0,2,3,4]])
    new_params = OpenCLRunner.create_parameters(secondary_school=0.123)
    assert np.isclose(new_params.place_hazard_multipliers[2], 0.123* current_risk_beta)
    assert np.all([new_params.place_hazard_multipliers[i] == default_params.place_hazard_multipliers[i] for i in [0,1,3,4]])
    new_params = OpenCLRunner.create_parameters(home=0.123)
    assert np.isclose(new_params.place_hazard_multipliers[3], 0.123* current_risk_beta)
    assert np.all([new_params.place_hazard_multipliers[i] == default_params.place_hazard_multipliers[i] for i in [0,1,2,4]])
    new_params = OpenCLRunner.create_parameters(work=0.123)
    assert np.isclose(new_params.place_hazard_multipliers[4], 0.123* current_risk_beta)
    assert np.all([new_params.place_hazard_multipliers[i] == default_params.place_hazard_multipliers[i] for i in [0,1,2,3]])

    # Check the current_risk_beta works properly (and dictionary unpacking while we're at it)
    params_dict = {"presymptomatic": 0.2, "asymptomatic": 0.3, "symptomatic": 0.4,
                   "retail": 0.1, "primary_school": 0.2, "secondary_school": 0.3, "home": 0.4, "work": 0.5}
    current_risk_beta = 0.5
    new_params = OpenCLRunner.create_parameters(current_risk_beta=current_risk_beta, **params_dict)
    # Individual multipliers should be unchanged
    for index, param_name in zip([0, 1, 2], ["presymptomatic", "asymptomatic", "symptomatic"]):
        assert np.isclose(new_params.individual_hazard_multipliers[index], params_dict[param_name])
    # Location-based ones should be multiplied by current_risk_beta
    for index, param_name in zip([0, 1, 2, 3, 4], ["retail", "primary_school", "secondary_school", "home", "work"]):
        assert np.isclose(new_params.place_hazard_multipliers[index], params_dict[param_name]*current_risk_beta)

    # Check that if parameters are set as constants that it works as expected
    OpenCLRunner.set_constants({"presymptomatic": 0.456})
    new_params = OpenCLRunner.create_parameters()
    # Presymptomatic should be differne,t but this other two should be same as default
    assert np.isclose(new_params.individual_hazard_multipliers[0], 0.456)
    assert np.all([np.isclose(default_params.individual_hazard_multipliers[i], new_params.individual_hazard_multipliers[i]) for i in [1,2] ])
    OpenCLRunner.clear_constants()
    # Now should all be same as the defaults again
    new_params = OpenCLRunner.create_parameters()
    assert np.all([ np.isclose(default_params.individual_hazard_multipliers[i], new_params.individual_hazard_multipliers[i]) for i in [0, 1,2]])

    # Shouldn't be able to set a parameter *and* a constant
    OpenCLRunner.set_constants({"current_risk_beta": 0.789})
    with pytest.raises(Exception):
        new_params = OpenCLRunner.create_parameters(current_risk_beta=0.123)
    OpenCLRunner.clear_constants()
    OpenCLRunner.create_parameters(current_risk_beta=0.123)  # This should be fine now


    # Could check these parameters as well
    #infection_log_scale: float = None,
    #infection_mode: float = None,

def test_get_cumulative_new_infections(setup_results):
    summaries = [x[0] for x in setup_results ]
    cumulative_infections = OpenCLRunner.get_cumulative_new_infections(summaries)
    # Check that the number is correct at the last iteration
    num_infected_at_end = 0
    for d, disease_status in enumerate(DiseaseStatus):
        if disease_status != DiseaseStatus.Susceptible:
            num_infected_at_end += OpenCLRunner.get_mean_total_counts(summaries, d)[-1]
    assert num_infected_at_end == cumulative_infections[-1]

def test_OpenCLWrapper():
    admin_params = {  # Not important, needed to instantiate the class
        "quiet": True, "use_gpu": False, "store_detailed_counts": True, "start_day": 0,
        "run_length": 10,
        "parameters_file": os.path.join("model_parameters", "default.yml"),
        "snapshot_file": os.path.join("microsim", "opencl", "snapshots", "cache.npz"),
        "opencl_dir": os.path.join("microsim", "opencl"),
        "current_particle_pop_df": None}
    # Check parameters assigned correctly (uses OpenCLRunner.create_params() which is already tested anyway)
    const_params = {'current_risk_beta': 1, 'presymptomatic': 2, 'asymptomatic': 3, 'symptomatic': 4}
    m1 = OpenCLWrapper(const_params_dict=const_params, **admin_params)
    for index, param_name in zip([0, 1, 2], ["presymptomatic", "asymptomatic", "symptomatic"]):
        assert np.isclose(m1.params.individual_hazard_multipliers[index], const_params[param_name])

    # Now check constant and random parameters behave correctly (should both be used to create parameters)
    const_params = {'current_risk_beta': 1, 'presymptomatic': 2, }
    rand_params = {'asymptomatic': 3, 'symptomatic': 4}
    merged_params = {**const_params, **rand_params}
    m1 = OpenCLWrapper(const_params_dict=const_params, **admin_params, _random_params_dict=rand_params)
    for index, param_name in zip([0, 1, 2], ["presymptomatic", "asymptomatic", "symptomatic"]):
        assert np.isclose(m1.params.individual_hazard_multipliers[index], merged_params[param_name])

    # If params duplicated then throw error
    with pytest.raises(Exception):
        OpenCLWrapper(const_params_dict={'current_risk_beta': 1, 'presymptomatic': 2},
                      **admin_params,
                      _random_params_dict={'presymptomatic': 3, 'symptomatic': 4})

    # Check model running works as expected.
    template_model = OpenCLWrapper(const_params_dict=const_params, **admin_params)

    # TODO NOTE: this will break the tests on github because it relies on a snapshot already
    # having been created (snapshot_file)
    results = template_model(rand_params)

    # TODO check results are as expected
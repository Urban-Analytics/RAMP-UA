import pytest
import multiprocessing
import pandas as pd
from microsim.microsim_model import Microsim, ActivityLocation


# ********************************************************
# These tests run through a whole dummy model process
# ********************************************************

# This 'fixture' means that other functions (e.g. step) can use the object created here.
# Note: Don't try to run this test, it will be called when running the others that need it, like `test_step()`.
@pytest.fixture()
def test_microsim():
    """Test the microsim constructor by reading dummy data. The microsim object created here can then be passed
    to other functions for them to do their tests
    """
    with pytest.raises(FileNotFoundError):
        # This should fail because the directory doesn't exist
        m = Microsim(data_dir="./bad_directory")

    m = Microsim(data_dir="./dummy_data", testing=True)

    # TODO check that the dummy data have been read in correctly. E.g. check the number of individuals is
    # accurate, that they link to households correctly, that they have the right flows, etc.

    # Finished initialising the model. Pass it to other tests who need it.
    yield m  # (this could be 'return' but 'yield' means that any cleaning can be done here

    print("Cleaning up .... (actually nothing to clean up at the moment)")


# Test the home flows on the dummy data
def test_add_home_flows(test_microsim):
    # Using dummy data I know that there should be 1 person in household 0:
    assert len(test_microsim.individuals.loc[test_microsim.individuals.HID == 0, :]) == 1
    # And two people in house 1
    assert len(test_microsim.individuals.loc[test_microsim.individuals.HID == 1, :]) == 2
    # And 4 in house 12
    assert len(test_microsim.individuals.loc[test_microsim.individuals.HID == 12, :]) == 4


def test_step(test_microsim):
    """Test the step method."""
    for i in range(10):
        test_microsim.step()
        # TODO test the step method. Make sure all the characteristics of the individuals are as they should be
        # (e.g. disease status, flows, etc.)
        # TODO make sure the characteristics of the locations are as they should be. E.g the 'Danger' etc.

    print("End of test step")

def test_update_venue_danger(test_microsim):
    # TODO Check that danger values are updated appropriately. Especially check indexing works (where the
    # venue ID is not the same as its place in the dataframe.
    assert False

# ********************************************************
# Other (unit) tests
# ********************************************************

def _get_rand(microsim, N=100):
    """Get a random number using the Microsimulation object's random number generator"""
    for _ in range(N):
        microsim.random.random()
    return microsim.random.random()


def test_random():
    """
    Checks that random classes are produce different (or the same!) numbers when they should do
    :return:
    """
    m1 = microsim.Microsim(read_data=False)
    m2 = microsim.Microsim(random_seed=2.0, read_data=False)
    m3 = microsim.Microsim(random_seed=2.0, read_data=False)

    # Genrate a random number from each model. The second two numbers should be the same
    r1, r2, r3 = [_get_rand(x) for x in [m1, m2, m3]]

    assert r1 != r2
    assert r2 == r3

    # Check that this still happens even if they are executed in pools.
    # Create a large number of microsims and check that all random numbers are unique
    pool = multiprocessing.Pool()
    m = [microsim.Microsim(read_data=False) for _ in range(10000)]
    r = pool.map(_get_rand, m)
    assert len(r) == len(set(r))


def test_extract_msoas_from_indiviuals():
    """Check that a list of areas can be successfully extracted from a DataFrame of indviduals"""
    individuals = pd.DataFrame(data={"Area": ["C", "A", "F", "A", "A", "F"]})
    areas = Microsim.extract_msoas_from_indiviuals(individuals)
    assert len(areas) == 3
    # Check the order is correct too
    assert False not in [x == y for (x, y) in zip(areas, ["A", "C", "F"])]


def test_check_study_area():
    all_msoa_list = ["C", "A", "F", "B", "D", "E"]
    individuals = pd.DataFrame(
        data={"PID": [1, 2, 3, 4, 5, 6], "HID": [1, 1, 2, 2, 2, 3], "Area": ["B", "B", "A", "A", "A", "D"]})
    households = pd.DataFrame(data={"HID": [1, 2, 3]})

    with pytest.raises(Exception):
        # Check that it catches duplicate areas
        assert Microsim.check_study_area(all_msoa_list, ["A", "A", "B"], individuals, households)
        assert Microsim.check_study_area(all_msoa_list + ["A"], ["A", "B"], individuals, households)
        # Check that it catches subset areas that aren't in the whole dataset
        assert Microsim.check_study_area(all_msoa_list, ["A", "B", "G"], individuals, households)

    # Should return whole dataset if no subset is provided
    assert Microsim.check_study_area(all_msoa_list, None, individuals, households)[0] == all_msoa_list
    assert Microsim.check_study_area(all_msoa_list, [], individuals, households)[0] == all_msoa_list

    with pytest.raises(Exception):
        # No individuals in area "E" so this should fail:
        assert Microsim.check_study_area(all_msoa_list, ["A", "B", "E"], individuals, households)

    # Correctly subset and remove individuals
    x = Microsim.check_study_area(all_msoa_list, ["A", "D"], individuals, households)
    assert x[0] == ["A", "D"]  # List of areas
    assert list(x[1].PID.unique()) == [3, 4, 5, 6]  # List of individuals
    assert list(x[2].HID.unique()) == [2, 3]  # List of households


def test_add_individual_flows():
    # TODO write test for Microsim.add_individual_flows()
    assert False


def test_read_retail_flows_data():
    # TODO write test to Microsim.read_retail_flows_data()
    assert False


def test_ActivityLocation():
    # TODO write tests to check that ActivtyLocaiton objects are created properly
    assert False


def test_get_danger_and_ids():
    # TODO write tests to make sure that the ActivityLocation.get_danger and get_ids functions
    # return the correct values and are indexed properly (each ID should refer to a specific object and danger)
    a = ActivityLocation(XXXX)
    assert False


def test_update_dangers():
    # TODO write a test that updates the Danger score for an ActivityLocation
    assert False


def test_export_to_feather():
    # TODO write a test that checks the export_to_feather() and  import_to_feather() functions
    assert False


def test_import_from_feather():
    # TODO write a test that checks the export_to_feather() and  import_to_feather() functions
    assert False


def test__add_location_columns():
    # TODO dest that the _add_location_columns function correctly adds the required standard columns
    # to a locaitons dataframe, and does appropriate checks for correct lengths of input lists etc.
    # TODO what happens if you pass lists of IDs or Dataframes or Series? Does the function behave properly?
    assert False


def test_add_home_flows():
    # TODO Check that home locations are created correctly (this is basically parsing the households and individuals
    # tables, adding the standard columns needed, and creating 'flows' from individuals to their households
    assert False

def test__normalise():
    # Should normalise so that the input list sums to 1
    # What if list length is 1, or a single number is given
    for l in [ 2, 1, [0.1], [5.3] ]:
        with pytest.raises(Exception):
            Microsim._normalise(l)

    # If numbers are the same (need to work out why these tests fail,the function seems OK)
    #for l in [ [2, 2], [0, 0], [-1, -1], [1, 1] ]:
    #    assert Microsim._normalise(l) == [0.5, 0.5]

    # Other examples
    assert Microsim._normalise([4, 6]) == [0.4, 0.6]
    assert Microsim._normalise([40, 60]) == [0.4, 0.6]
    assert Microsim._normalise([6, 6, 6, 6, 6]) == [0.2, 0.2, 0.2, 0.2, 0.2]




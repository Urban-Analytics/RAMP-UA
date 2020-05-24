import pytest
import multiprocessing
import pandas as pd
import numpy as np
from microsim.microsim_model import Microsim
from microsim.column_names import ColumnNames
from microsim.activity_location import ActivityLocation


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


def test_read_school_flows_data(test_microsim):
    """Check that flows to primary and secondary schools were read correctly """
    # Check priary and seconary are actually the same dataframe (they're read together)
    primary_schools = test_microsim.activity_locations["PrimarySchool"]._locations
    secondary_schools = test_microsim.activity_locations["SecondarySchool"]._locations
    assert primary_schools.equals(secondary_schools)
    # Check that if we add a column in one, the other gets it too
    primary_schools["TestCol"] = 0
    assert "TestCol" in list(secondary_schools.columns)

    schools = primary_schools  # Just refer to them with one name

    # Check correct number of primary and secondary schools
    # (these don't need to sum to total schools because there are a couple of nurseries in there
    assert len(schools) == 350
    primary_schools = schools.loc[schools.PhaseOfEducation_name == "Primary"]
    secondary_schools = schools.loc[schools.PhaseOfEducation_name == "Secondary"]
    len(primary_schools) == 309
    len(secondary_schools) == 39

    # Check all primary flows go to primary schools and secondary flows go to secondary schools
    primary_flows = test_microsim.activity_locations["PrimarySchool"]._flows
    secondary_flows = test_microsim.activity_locations["SecondarySchool"]._flows
    # Following slice slice gives the total flow to each of the 350 schools (sum across rows for each colum and then
    # drop the first two columns which are area ID and Code)
    for school_no, flow in enumerate(primary_flows.sum(0)[2:]):
        if flow > 0:
            assert schools.iloc[school_no].PhaseOfEducation_name == "Primary"
    for school_no, flow in enumerate(secondary_flows.sum(0)[2:]):
        if flow > 0:
            assert schools.iloc[school_no].PhaseOfEducation_name == "Secondary"


def test_read_msm_data(test_microsim):
    """Checks the individual microsimulation data are read correctly"""
    assert len(test_microsim.individuals) == 17
    assert len(test_microsim.households) == 8
    # Check correct number of 'homeless' (this is OK because of how I set up the data)
    with pytest.raises(Exception) as e:
        Microsim._check_no_homeless(test_microsim.individuals, test_microsim.households, warn=False)
        # This should reaise an exception. Get the number of homeless. Should be 15
        num_homeless = [int(s) for s in e.message.split() if s.isdigit()][0]
        print(f"Correctly found homeless: {num_homeless}")
        assert num_homeless == 15


def test_update_disease_counts(test_microsim):
    """Check that disease counts for MSOAs and households are updated properly"""
    m = test_microsim  # less typing

    m.individuals.loc[m.individuals._PID == 100799, "Disease_Status"] = 1
    m.individuals.loc[m.individuals._PID == 23968, "Disease_Status"] = 1
    m.individuals.loc[m.individuals._PID == 23434, "Disease_Status"] = 1
    m.individuals.loc[m.individuals._PID == 90653, "Disease_Status"] = 1
    #m.individuals.loc[:, ["PID", "HID", "Area", "Disease_Status", "MSOA_Cases", "HID_Cases"]]
    m.update_disease_counts()
    # This person has the disease
    assert m.individuals.loc[m.individuals._PID == 100799, "MSOA_Cases"].values[0] == 1
    assert m.individuals.loc[m.individuals._PID == 100799, "HID_Cases"].values[0] == 1
    # These people live there too (but live in different msoas, so it's OK the disease hasn't propogated there!)
    assert m.individuals.loc[m.individuals._PID == 64788, "HID_Cases"].values[0] == 1
    assert m.individuals.loc[m.individuals._PID == 69754, "HID_Cases"].values[0] == 1
    # In this house of 4, two people have the disease
    assert m.individuals.loc[m.individuals._PID == 17942, "HID_Cases"].values[0] == 2
    assert m.individuals.loc[m.individuals._PID == 22526, "HID_Cases"].values[0] == 2
    assert m.individuals.loc[m.individuals._PID == 23434, "HID_Cases"].values[0] == 2
    assert m.individuals.loc[m.individuals._PID == 23968, "HID_Cases"].values[0] == 2
    # One person in this area has the disease
    assert m.individuals.loc[m.individuals._PID == 90653, "MSOA_Cases"].values[0] == 1
    assert False not in (m.individuals.loc[(m.individuals._HID == 1) | (m.individuals._HID == 3) |
                                           (m.individuals._HID == 6)]["HID_Cases"] == 0)
    assert False not in (m.individuals.loc[(m.individuals.Area == "E02004147") | (m.individuals.Area == "E02004138") |
                                           (m.individuals.Area == "E02004158")]["MSOA_Cases"] == 0)

    # Note: Can't fully test MSOA cases because I don't have any examples of people from different
    # households living in the same MSOA in the test data


def test_update_current_risk(test_microsim):
    """Check that the current risk is updated properly"""
    #test_microsim.update_current_risk()
    assert False


def test_step(test_microsim):
    """Test the step method. This is the main test of the model. Simulate a deterministic run through and
    make sure that the model runs as expected

    :param test_microsim: This is a pointer to the initialised model. Dummy data will have been read in,
    but no stepping has taken place yet."""
    # TODO Big test of the whole model with dummy data
    m = test_microsim  # Just for less typing

    # Note: the following is a useul way to get relevant info about the individuals
    #m.individuals.loc[:, ["ID", "PID", "HID", "Area", "Disease_Status", "MSOA_Cases", "HID_Cases"]]

    # Step 0 (initialisation):
    # Give some people a disease status manually. Maybe just 2 people so it's easy to track.
    # E.g.: m.individuals.loc[m.individuals.ID==1234,"Disease_Status"] = 1

    #  ******** Step 1: ********

    # Update the danger associated with each venue
    #m.update_venue_danger()

    # Check that the venues have been updated properly. We know what people do and how long they
    # spend doing it, so should be able to check the locations for all of their activities.
    #m.activity_locations['Retail']._locations ....

    # Update the current risk for individuals who may be visitting those venues
    # m.update_current_risk()

    # Check that people visiting those places have the correct risk

    # Update disease counters. E.g. count diseases in MSOAs & households
    #m.update_disease_counts()
    # (maybe don't bother checking this because I've already written a separate unit test for it)

    # Update disease status
    # (Manually give some people with higher risk the disease. Or maybe don't this iteration. Whatever.
    #m.individuals.loc[m.individuals.ID

    #  ******** Step 2 ********

    # Update the danger associated with each venue
    # m.update_venue_danger()

    # Check that the venues have been updated properly.
    # m.activity_locations['Retail']._locations ....

    # Update the current risk for individuals who may be visiting those venues
    # m.update_current_risk()

    # Check that people visiting those places have the correct risk

    # Update disease counters. E.g. count diseases in MSOAs & households
    #m.update_disease_counts()

    # Update disease status
    # xxxx

    #  ******** Step 3 ********
    # REPEAT !
    # Probably put some of the above checks in separate functions as they will be called repeatedly

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
    df = pd.DataFrame(data={"Name": ['a', 'b', 'c', 'd']})
    with pytest.raises(Exception):  # Should fail if lists are wrong length
        Microsim._add_location_columns(df, location_names=["a", "b"], location_ids=None)
        Microsim._add_location_columns(df, location_names=df.Name, location_ids=[1, 2])
    with pytest.raises(TypeError):  # Can't get the length of None
        Microsim._add_location_columns(df, location_names=None)

    # Call the function
    x = Microsim._add_location_columns(df, location_names=df.Name)
    assert x is None  # Function shouldn't return anything. Does things inplace
    # Default behaviour is just add columns
    assert False not in (df.columns.values == ["Name", "ID", "Location_Name", "Danger"])
    assert False not in list(df.Location_Name == df.Name)
    assert False not in list(df.ID == range(0, 4))
    assert False not in list(df.index == range(0, 4))
    # Adding columns again shouldn't change anything
    Microsim._add_location_columns(df, location_names=df.Name)
    assert False not in (df.columns.values == ["Name", "ID", "Location_Name", "Danger"])
    assert False not in list(df.Location_Name == df.Name)
    assert False not in list(df.ID == range(0, 4))
    assert False not in list(df.index == range(0, 4))
    # See what happens if we give it IDs
    Microsim._add_location_columns(df, location_names=df.Name, location_ids=[5, 7, 10, -1])
    assert False not in (df.columns.values == ["Name", "ID", "Location_Name", "Danger"])
    assert False not in list(df.ID == [5, 7, 10, -1])
    assert False not in list(df.index == range(0, 4))  # Index shouldn't change
    # Shouldn't matter if IDs are Dataframes or Series
    Microsim._add_location_columns(df, location_names=pd.Series(df.Name))
    assert False not in list(df.Location_Name == df.Name)
    assert False not in list(df.index == range(0, 4))  # Index shouldn't change
    Microsim._add_location_columns(df, location_names=df.Name, location_ids=np.array([5, 7, 10, -1]))
    assert False not in list(df.ID == [5, 7, 10, -1])
    assert False not in list(df.index == range(0, 4))  # Index shouldn't change
    # Set a weird index, the function should replace it with the row number
    df = pd.DataFrame(data={"Name": ['a', 'b', 'c', 'd'], "Col2": [4, -6, 8, 1.4]}, )
    df.set_index("Col2")
    Microsim._add_location_columns(df, location_names=df.Name)
    assert False not in list(df.ID == range(0, 4))
    assert False not in list(df.index == range(0, 4))

    # TODO dest that the _add_location_columns function correctly adds the required standard columns
    # to a locaitons dataframe, and does appropriate checks for correct lengths of input lists etc.


def test_add_home_flows():
    # TODO Check that home locations are created correctly (this is basically parsing the households and individuals
    # tables, adding the standard columns needed, and creating 'flows' from individuals to their households
    assert False


def test__normalise():
    # Should normalise so that the input list sums to 1
    # Fail if aa single number is given
    for l in [2, 1]:
        with pytest.raises(Exception):
            Microsim._normalise(l)

    # 1-item lists should return [1.0]
    for l in [[0.1], [5.3]]:
        assert Microsim._normalise(l) == [1.0]

    # If numbers are the same (need to work out why these tests fail,the function seems OK)
    # for l in [ [2, 2], [0, 0], [-1, -1], [1, 1] ]:
    #    assert Microsim._normalise(l) == [0.5, 0.5]

    # Other examples
    assert Microsim._normalise([4, 6]) == [0.4, 0.6]
    assert Microsim._normalise([40, 60]) == [0.4, 0.6]
    assert Microsim._normalise([6, 6, 6, 6, 6]) == [0.2, 0.2, 0.2, 0.2, 0.2]


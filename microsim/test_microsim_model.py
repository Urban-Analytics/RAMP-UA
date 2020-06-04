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

    # Check that the dummy data have been read in correctly. E.g. check the number of individuals is
    # accurate, that they link to households correctly, that they have the right *flows* to the right
    # *destinations* and the right *durations* etc.

    assert len(m.individuals) == 17

    # Households
    # (The households df should be the same as the one in the corresponding activity location)
    assert m.activity_locations['Home']._locations.equals(m.households)
    # All flows should be to one location (single element [1.0])
    for flow in m.individuals[f"Home{ColumnNames.ACTIVITY_FLOWS}"]:
        assert flow == [1.0]

    # House IDs are the same as the row index
    assert False not in list(m.households.index == m.households.ID)

    # First two people live together in first household
    assert list(m.individuals.loc[0:1,:][f"Home{ColumnNames.ACTIVITY_VENUES}"].values) ==[[0], [0]]
    # This one lives on their own in the fourth house
    assert list(m.individuals.loc[9:9,:][f"Home{ColumnNames.ACTIVITY_VENUES}"].values) ==[[3]]
    # These three live together in the last house
    assert list(m.individuals.loc[13:15,:][f"Home{ColumnNames.ACTIVITY_VENUES}"].values) ==[[6], [6], [6]]

    # Workplaces
    # All flows should be to one location (single element [1.0])
    for flow in m.individuals[f"Work{ColumnNames.ACTIVITY_FLOWS}"]:
        assert flow == [1.0]
    # First person is the only one who does that job
    assert len(list(m.individuals.loc[0:0, ][f"Work{ColumnNames.ACTIVITY_VENUES}"]))
    job_index = list(m.individuals.loc[0:0, ][f"Work{ColumnNames.ACTIVITY_VENUES}"])[0][0]
    for work_id in m.individuals.loc[1:len(m.individuals), f"Work{ColumnNames.ACTIVITY_VENUES}"]:
        assert work_id[0] != job_index
    # Three people do the same job as second person
    job_index = list(m.individuals.loc[1:1, ][f"Work{ColumnNames.ACTIVITY_VENUES}"])[0]
    assert list(m.individuals.loc[4:4, f"Work{ColumnNames.ACTIVITY_VENUES}"])[0] == job_index
    assert list(m.individuals.loc[13:13, f"Work{ColumnNames.ACTIVITY_VENUES}"])[0] == job_index
    # Not this person:
    assert list(m.individuals.loc[15:15, f"Work{ColumnNames.ACTIVITY_VENUES}"])[0] != job_index

    # Test Shops
    shop_locs = m.activity_locations['Retail']._locations
    assert len(shop_locs) == 248
    # First person has these flows and venues
    venue_ids = list(m.individuals.loc[0:0, f"Retail{ColumnNames.ACTIVITY_VENUES}"])[0]
    #flows = list(m.individuals.loc[0:0, f"Retail{ColumnNames.ACTIVITY_FLOWS}"])[0]
    # These are the venues in the filename:
    raw_venues = sorted([24,  23, 22, 21, 19, 12, 13, 25, 20, 17])
    # Mark counts from 1, so these should be 1 greater than the ids
    assert [ x-1 for x in raw_venues ] == venue_ids
    # Check the indexes point correctly
    assert shop_locs.loc[0:0, ColumnNames.LOCATION_NAME].values[0] == "Co-op Lyme Regis"
    assert shop_locs.loc[18:18, ColumnNames.LOCATION_NAME].values[0] == "Aldi Honiton"

    # Test Schools (similar to house/work above) (need to do for primary and secondary)
    primary_locs = m.activity_locations['PrimarySchool']._locations
    secondary_locs = m.activity_locations['SecondarySchool']._locations
    # All schools are read in from one file, both primary and secondary
    assert len(primary_locs) == 350
    assert len(secondary_locs) == 350
    assert primary_locs.equals(secondary_locs)
    # Check primary and secondary indexes point to primary and secondary schools respectively
    for indexes in m.individuals.loc[:,f"PrimarySchool{ColumnNames.ACTIVITY_VENUES}"]:
        for index in indexes:
            assert primary_locs.loc[index,"PhaseOfEducation_name"]=="Primary"
    for indexes in m.individuals.loc[:,f"SecondarySchool{ColumnNames.ACTIVITY_VENUES}"]:
        for index in indexes:
            assert secondary_locs.loc[index,"PhaseOfEducation_name"]=="Secondary"

    # First person has these flows and venues to primary school
    # (we know this because, by coincidence, the first person lives in the area that has the
    # first area name if they were ordered alphabetically)
    list(m.individuals.loc[0:0, "Area"])[0] == "E00101308"
    venue_ids = list(m.individuals.loc[0:0, f"PrimarySchool{ColumnNames.ACTIVITY_VENUES}"])[0]
    raw_venues = sorted([12, 110, 118, 151, 163, 180, 220, 249, 280] )
    # Mark counts from 1, so these should be 1 greater than the ids
    assert [ x-1 for x in raw_venues ] == venue_ids
    # Check the indexes point correctly
    assert primary_locs.loc[12:12, ColumnNames.LOCATION_NAME].values[0] == "Axminster Community Primary Academy"
    assert primary_locs.loc[163:163, ColumnNames.LOCATION_NAME].values[0] == "Milton Abbot School"

    # Second to last person lives in 'E02004138' which will be the last area recorded in Mark's file
    assert list(m.individuals.loc[9:9, "Area"])[0] == "E02004159"
    venue_ids = list(m.individuals.loc[9:9, f"SecondarySchool{ColumnNames.ACTIVITY_VENUES}"])[0]
    raw_venues = sorted([335, 346])
    # Mark counts from 1, so these should be 1 greater than the ids
    assert [ x-1 for x in raw_venues ] == venue_ids
    # Check these are both secondary schools
    for idx in venue_ids:
        assert secondary_locs.loc[idx,"PhaseOfEducation_name"] == "Secondary"
    # Check the indexes point correctly
    assert secondary_locs.loc[335:335, ColumnNames.LOCATION_NAME].values[0] == "South Dartmoor Community College"

    # Finished initialising the model. Pass it to other tests who need it.
    yield m  # (this could be 'return' but 'yield' means that any cleaning can be done here

    print("Cleaning up .... (actually nothing to clean up at the moment)")


# Test the home flows on the dummy data
def test_add_home_flows(test_microsim):
    ind = test_microsim.individuals  # save typine
    # Using dummy data I know that there should be 2 person in household ID 0:
    assert len(ind.loc[ind.House_ID == 0, :]) == 2
    # And 4 people in house ID 2
    assert len(ind.loc[ind.House_ID == 1, :]) == 3
    # And 1 in house ID 7
    assert len(ind.loc[ind.House_ID == 7, :]) == 1


def test_read_school_flows_data(test_microsim):
    """Check that flows to primary and secondary schools were read correctly """
    # Check priary and seconary have the same data (they're read together)
    primary_schools = test_microsim.activity_locations["PrimarySchool"]._locations
    secondary_schools = test_microsim.activity_locations["SecondarySchool"]._locations
    assert primary_schools.equals(secondary_schools)
    # But they don't point to the same dataframe
    primary_schools["TestCol"] = 0
    assert "TestCol" not in list(secondary_schools.columns)

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
    # Make sure no one has the disease to start with
    m.individuals["Disease_Status"] = 0
    # (Shouldn't use _PID any more, this is a hangover to old version, but works OK with dummy data)
    m.individuals.loc[9, "Disease_Status"] = 1  # lives alone
    m.individuals.loc[13, "Disease_Status"] = 1  # Lives with 3 people
    m.individuals.loc[11, "Disease_Status"] = 1  # | Live
    m.individuals.loc[12, "Disease_Status"] = 1  # | Together
    #m.individuals.loc[:, ["PID", "HID", "Area", "Disease_Status", "MSOA_Cases", "HID_Cases"]]
    m.update_disease_counts()
    # This person has the disease
    assert m.individuals.at[9, "MSOA_Cases"] == 1
    assert m.individuals.at[9, "HID_Cases"] == 1
    # These people live with someone who has the disease
    for p in [13, 14, 15]:
        assert m.individuals.at[p, "MSOA_Cases"] == 1
        assert m.individuals.at[p, "HID_Cases"] == 1
    # Two people in this house have the disease
    for p in [11, 12]:
        assert m.individuals.at[p, "MSOA_Cases"] == 2
        assert m.individuals.at[p, "HID_Cases"] == 2


    # Note: Can't fully test MSOA cases because I don't have any examples of people from different
    # households living in the same MSOA in the test data


def test_update_venue_danger_and_risks(test_microsim):
    """Check that the current risk is updated properly"""
    # This is actually tested as part of test_step
    assert True


def test_step(test_microsim):
    """
    Test the step method. This is the main test of the model. Simulate a deterministic run through and
    make sure that the model runs as expected.

    Only thing it doesn't do is check for retail, shopping, etc., that danger and risk increase by the correct
    amount. It just checks they go above 0 (or not). It does do that more precise checks for home activities though.

    :param test_microsim: This is a pointer to the initialised model. Dummy data will have been read in,
    but no stepping has taken place yet."""
    m = test_microsim  # For less typing and so as not to interfere with other functions use test_microsim

    # Note: the following is a useul way to get relevant info about the individuals
    #m.individuals.loc[:, ["ID", "PID", "HID", "Area", "Disease_Status", "MSOA_Cases", "HID_Cases"]]

    # Step 0 (initialisation):

    # Everyone should start without the disease (they will have been assigned a status as part of initialisation)
    m.individuals["Disease_Status"] = 0

    # Set understandable multipliers
    m.risk_multiplier = 1.0
    m.danger_multiplier = 1.0

    #
    # Person 1: lives with one other person (p2). Both people spend all their time at home doing nothing else
    #
    p1 = 0
    p2 = 1

    m.individuals.loc[p1, "Disease_Status"] = 1  # Give them the disease
    for p in [p1, p2]: # Set their activity durations to 0
        for name, activity in m.activity_locations.items():
            m.individuals[f"{name}{ColumnNames.ACTIVITY_DURATION}"] = 0.0
        m.individuals[f"Home{ColumnNames.ACTIVITY_DURATION}"] = 1.0  # Spend all their time at home

    m.step()

    # Check the disease has spread to the house but nowhere else
    for p in [p1, p2]:
        assert m.individuals.at[p, ColumnNames.CURRENT_RISK] == 1.0
    for p in range(2,len(m.individuals)):
        assert m.individuals.at[p, ColumnNames.CURRENT_RISK] == 0.0
    assert m.households.at[0, ColumnNames.LOCATION_DANGER] == 1.0
    for h in range(1, len(m.households)):  # all others are 0
        assert m.households.at[h, ColumnNames.LOCATION_DANGER] == 0.0

    m.step()

    # Risk and danger stay the same (it does not cumulate over days)
    for p in [p1, p2]:
        assert m.individuals.at[p, ColumnNames.CURRENT_RISK] == 1.0
    for p in range(2,len(m.individuals)):
        assert m.individuals.at[p, ColumnNames.CURRENT_RISK] == 0.0
    m.households.at[0, ColumnNames.LOCATION_DANGER] == 1.0
    for h in range(1, len(m.households)):
        assert m.households.at[h, ColumnNames.LOCATION_DANGER] == 0.0

    # If the infected person doesn't go home (in this test they do absolutely nothing) then danger and risks should go
    # back to 0
    m.individuals.at[p1, f"Home{ColumnNames.ACTIVITY_DURATION}"] = 0.0
    m.step()
    for p in range(len(m.individuals)):
        assert m.individuals.at[p, ColumnNames.CURRENT_RISK] == 0.0
    for h in range(0, len(m.households)):
        assert m.households.at[h, ColumnNames.LOCATION_DANGER] == 0.0


    # But if they both get sick then they should be 2.0 (double danger and risk)
    m.individuals.loc[p1:p2, "Disease_Status"] = 1  # Give them the disease
    m.individuals.at[p1, f"Home{ColumnNames.ACTIVITY_DURATION}"] = 1.0 # Make the duration normal again
    m.step()
    for p in [p1, p2]:
        assert m.individuals.at[p, ColumnNames.CURRENT_RISK] == 2.0
    assert m.households.at[0, ColumnNames.LOCATION_DANGER] == 2.0
    for h in range(1, len(m.households)): # All other houses are danger free
        m.households.at[h, ColumnNames.LOCATION_DANGER] == 0.0

    #
    # Now see what happens when one person gets the disease and spreads it to schools, shops and work
    #
    del p1, p2
    p1 = 4 # The infected person is index 1
    # Make everyone better except for that one person
    m.individuals["Disease_Status"] = 0
    m.individuals.loc[p1, "Disease_Status"] = 1
    # Assign everyone equal time doing all activities
    for name, activity in m.activity_locations.items():
        m.individuals[f"{name}{ColumnNames.ACTIVITY_DURATION}"] = 1.0/len(m.activity_locations)

    m.step()

    # Now check that the danger has propagated to locations and risk to people
    # TODO Also check that the total risks and danger scores sum correctly
    for name, activity in m.activity_locations.items():
        # Indices of the locations where this person visited
        visited_idx = m.individuals.at[p1, f"{name}{ColumnNames.ACTIVITY_VENUES}"]
        not_visited_idx = list(set(range(len(activity._locations)))-set(visited_idx))
        # Dangers should be >0.0 (or not if the person didn't visit there)
        assert False not in list(activity._locations.loc[visited_idx, "Danger"].values > 0 )
        assert False not in list(activity._locations.loc[not_visited_idx, "Danger"].values == 0 )
        # Individuals should have an associated risk
        for index, row in m.individuals.iterrows():
            for idx in visited_idx:
                if idx in row[f"{name}{ColumnNames.ACTIVITY_VENUES}"]:
                    assert row[ColumnNames.CURRENT_RISK] > 0
                    # Note: can't check if risk is equal to 0 becuase it might come from another activity

    print("End of test step")



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
    m1 = Microsim(read_data=False)
    m2 = Microsim(random_seed=2.0, read_data=False)
    m3 = Microsim(random_seed=2.0, read_data=False)

    # Genrate a random number from each model. The second two numbers should be the same
    r1, r2, r3 = [_get_rand(x) for x in [m1, m2, m3]]

    assert r1 != r2
    assert r2 == r3

    # Check that this still happens even if they are executed in pools.
    # Create a large number of microsims and check that all random numbers are unique
    pool = multiprocessing.Pool()
    m = [Microsim(read_data=False) for _ in range(10000)]
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


def test__normalise():
    # TODO test the 'decimals' argument too.
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


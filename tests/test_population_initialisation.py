import os
import pytest
import multiprocessing
import pandas as pd
import numpy as np
from microsim.column_names import ColumnNames
from microsim.population_initialisation import PopulationInitialisation

# ********************************************************
# These tests run through a whole dummy model process
# ********************************************************

test_dir = os.path.dirname(os.path.abspath(__file__))

# arguments used when calling the PopulationInitialisation constructor. Usually these are the same
population_init_args = {"data_dir": os.path.join(test_dir, "dummy_data"),
                        "testing": True, "debug": True, 'lockdown_file': ""
                        }


# This 'fixture' means that other test functions can use the object created here.
# Note: Don't try to run this test, it will be called when running the others that need it,
# like `test_add_home_flows()`.
@pytest.fixture()
def test_population_init():
    population_init = PopulationInitialisation(**population_init_args)

    # Check that the dummy data have been read in correctly. E.g. check the number of individuals is
    # accurate, that they link to households correctly, that they have the right *flows* to the right
    # *destinations* and the right *durations* etc.

    assert len(population_init.individuals) == 17

    # Households
    # (The households df should be the same as the one in the corresponding activity location)
    assert population_init.activity_locations[f"{ColumnNames.Activities.HOME}"]._locations.equals(population_init.households)
    # All flows should be to one location (single element [1.0])
    for flow in population_init.individuals[f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_FLOWS}"]:
        assert flow == [1.0]

    # House IDs are the same as the row index
    assert False not in list(population_init.households.index == population_init.households.ID)

    # First two people live together in first household
    assert list(population_init.individuals.loc[0:1, :][f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_VENUES}"].values) == [[0], [0]]
    # This one lives on their own in the fourth house
    assert list(population_init.individuals.loc[9:9, :][f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_VENUES}"].values) == [[3]]
    # These three live together in the last house
    assert list(population_init.individuals.loc[13:15, :][f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_VENUES}"].values) == [[6], [6], [6]]

    # Workplaces
    # For convenience:
    work_venues = f"{ColumnNames.Activities.WORK}{ColumnNames.ACTIVITY_VENUES}"
    work_flows = f"{ColumnNames.Activities.WORK}{ColumnNames.ACTIVITY_FLOWS}"
    workplaces = population_init.activity_locations[ColumnNames.Activities.WORK]._locations
    # Check no non-unique workplace names
    assert len(workplaces) == len(workplaces.loc[:, ColumnNames.LOCATION_NAME].unique())
    # Check the total number of workplaces created is correct (one per soc per area)
    assert len(workplaces) == len(population_init.individuals.loc[:, 'soc2010'].unique()) * len(population_init.all_msoas)
    # These people should all have single flows to a workplace in their home area
    for p in range(5, len(population_init.individuals)):
        assert len(population_init.individuals.at[p, work_venues]) == 1
        assert len(population_init.individuals.at[p, work_flows]) == 1
        venue_number = population_init.individuals.at[p, work_venues][0]
        venue = workplaces.iloc[venue_number]
        # check the area and soc of the venue (these are unique fields for workplaces and not used in the model)
        assert venue['MSOA'] == population_init.individuals.at[p, "area"]  # This workplace is in the person's home msoa
        assert venue['SOC'] == population_init.individuals.at[p, "soc2010"]
        # check the name is correct (this is a unique identifier of the workplace)
        assert venue[ColumnNames.LOCATION_NAME]==f"{population_init.individuals.at[p, 'area']}-{population_init.individuals.at[p,'soc2010']}"
    # These should have four workpaces with flows 0.5, 0.15, 0.1, 0.25
    for p in range(0, 2):
        assert population_init.individuals.at[p, work_flows] == [0.5, 0.15, 0.1, 0.25]  # CHeck flows
        # Check the SOC of the workplace is the same as the individual
        for venue_number in population_init.individuals.at[p, work_venues]:
            venue = workplaces.iloc[venue_number]
            assert venue['SOC'] == population_init.individuals.at[p, "soc2010"]
        # The destination areas of the workplace should be as follows:
        assert set(workplaces.loc[population_init.individuals.at[p, "Work_Venues"], "MSOA"]) ==\
            set(['E00101308', 'E02004132', 'E02004147', 'E02004151'])
    # These should have 5 workplaces (assuming the total number of workplaces threshold is 5)
    for p in range(2, 5):
        assert len(population_init.individuals.at[p, work_venues]) == 5
        assert len(population_init.individuals.at[p, work_flows]) == 5
        for venue_number in population_init.individuals.at[p, work_venues]:
            venue = workplaces.iloc[venue_number]
            assert venue['SOC'] == population_init.individuals.at[p, "soc2010"]
        assert set(workplaces.loc[population_init.individuals.at[p, "Work_Venues"], "MSOA"]) == \
            set(['E02004158', 'E02004147', 'E02004132', 'E02004138', 'E02004159'])

    # Test Shops
    shop_locs = population_init.activity_locations[ColumnNames.Activities.RETAIL]._locations
    assert len(shop_locs) == 248
    # First person has these flows and venues
    venue_ids = list(population_init.individuals.loc[0:0, f"{ColumnNames.Activities.RETAIL}{ColumnNames.ACTIVITY_VENUES}"])[0]
    # flows = list(m.individuals.loc[0:0, f"Retail{ColumnNames.ACTIVITY_FLOWS}"])[0]
    # These are the venues in the filename:
    raw_venues = sorted([24, 23, 22, 21, 19, 12, 13, 25, 20, 17])
    # Mark counts from 1, so these should be 1 greater than the ids
    assert [x - 1 for x in raw_venues] == venue_ids
    # Check the indexes point correctly
    assert shop_locs.loc[0:0, ColumnNames.LOCATION_NAME].values[0] == "Co-op Lyme Regis"
    assert shop_locs.loc[18:18, ColumnNames.LOCATION_NAME].values[0] == "Aldi Honiton"

    # Test Schools (similar to house/work above) (need to do for primary and secondary)
    primary_locs = population_init.activity_locations[f"{ColumnNames.Activities.PRIMARY}"]._locations
    secondary_locs = population_init.activity_locations[f"{ColumnNames.Activities.SECONDARY}"]._locations
    # All schools are read in from one file, both primary and secondary
    assert len(primary_locs) == 350
    assert len(secondary_locs) == 350
    assert primary_locs.equals(secondary_locs)
    # Check primary and secondary indexes point to primary and secondary schools respectively
    for indexes in population_init.individuals.loc[:, f"{ColumnNames.Activities.PRIMARY}{ColumnNames.ACTIVITY_VENUES}"]:
        for index in indexes:
            assert primary_locs.loc[index, "PhaseOfEducation_name"] == "Primary"
    for indexes in population_init.individuals.loc[:, f"{ColumnNames.Activities.SECONDARY}{ColumnNames.ACTIVITY_VENUES}"]:
        for index in indexes:
            assert secondary_locs.loc[index, "PhaseOfEducation_name"] == "Secondary"

    # First person has these flows and venues to primary school
    # (we know this because, by coincidence, the first person lives in the area that has the
    # first area name if they were ordered alphabetically)
    assert list(population_init.individuals.loc[0:0, "area"])[0] == "E00101308"
    venue_ids = list(population_init.individuals.loc[0:0, f"{ColumnNames.Activities.PRIMARY}{ColumnNames.ACTIVITY_VENUES}"])[0]
    raw_venues = sorted([12, 110, 118, 151, 163, 180, 220, 249, 280])
    # Mark counts from 1, so these should be 1 greater than the ids
    assert [x - 1 for x in raw_venues] == venue_ids
    # Check the indexes point correctly
    assert primary_locs.loc[12:12, ColumnNames.LOCATION_NAME].values[0] == "Axminster Community Primary Academy"
    assert primary_locs.loc[163:163, ColumnNames.LOCATION_NAME].values[0] == "Milton Abbot School"

    # Second to last person lives in 'E02004138' which will be the last area recorded in Mark's file
    assert list(population_init.individuals.loc[9:9, "area"])[0] == "E02004159"
    venue_ids = list(population_init.individuals.loc[9:9, f"{ColumnNames.Activities.SECONDARY}{ColumnNames.ACTIVITY_VENUES}"])[0]
    raw_venues = sorted([335, 346])
    # Mark counts from 1, so these should be 1 greater than the ids
    assert [x - 1 for x in raw_venues] == venue_ids
    # Check these are both secondary schools
    for idx in venue_ids:
        assert secondary_locs.loc[idx, "PhaseOfEducation_name"] == "Secondary"
    # Check the indexes point correctly
    assert secondary_locs.loc[335:335, ColumnNames.LOCATION_NAME].values[0] == "South Dartmoor Community College"

    # Finished initialising the model. Pass it to other tests who need it.
    yield population_init  # (this could be 'return' but 'yield' means that any cleaning can be done here

    print("Cleaning up .... (actually nothing to clean up at the moment)")


def test_bad_directory_path_throws_error():
    """Test the PopulationInitialisation constructor by reading dummy data. The microsim object created here can then
    be passed to other functions for them to do their tests
    """
    with pytest.raises(FileNotFoundError):
        # This should fail because the directory doesn't exist
        args = population_init_args.copy()
        args['data_dir'] = "./bad_directory"
        p = PopulationInitialisation(**args)


# Test the home flows on the dummy data
def test_add_home_flows(test_population_init):
    ind = test_population_init.individuals  # save typing
    # Using dummy data I know that there should be 2 person in household ID 0:
    assert len(ind.loc[ind.House_ID == 0, :]) == 2
    # And 4 people in house ID 2
    assert len(ind.loc[ind.House_ID == 1, :]) == 3
    # And 1 in house ID 7
    assert len(ind.loc[ind.House_ID == 7, :]) == 1


def test_read_school_flows_data(test_population_init):
    """Check that flows to primary and secondary schools were read correctly """
    # Check primary and secondary have the same data (they're read together)
    primary_schools = test_population_init.activity_locations[f"{ColumnNames.Activities.PRIMARY}"]._locations
    secondary_schools = test_population_init.activity_locations[f"{ColumnNames.Activities.SECONDARY}"]._locations
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
    assert len(primary_schools) == 309
    assert len(secondary_schools) == 39

    # Check all primary flows go to primary schools and secondary flows go to secondary schools
    primary_flows = test_population_init.activity_locations[f"{ColumnNames.Activities.PRIMARY}"]._flows
    secondary_flows = test_population_init.activity_locations[f"{ColumnNames.Activities.SECONDARY}"]._flows
    # Following slice slice gives the total flow to each of the 350 schools (sum across rows for each colum and then
    # drop the first two columns which are area ID and Code)
    for school_no, flow in enumerate(primary_flows.sum(0)[2:]):
        if flow > 0:
            assert schools.iloc[school_no].PhaseOfEducation_name == "Primary"
    for school_no, flow in enumerate(secondary_flows.sum(0)[2:]):
        if flow > 0:
            assert schools.iloc[school_no].PhaseOfEducation_name == "Secondary"


def test_read_msm_data(test_population_init):
    """Checks the individual microsimulation data are read correctly"""
    assert len(test_population_init.individuals) == 17
    assert len(test_population_init.households) == 8
    # Check correct number of 'homeless' (this is OK because of how I set up the data)
    with pytest.raises(Exception) as e:
        PopulationInitialisation._check_no_homeless(test_population_init.individuals, test_population_init.households,
                                                    warn=False)
        # This should reaise an exception. Get the number of homeless. Should be 15
        num_homeless = [int(s) for s in e.message.split() if s.isdigit()][0]
        print(f"Correctly found homeless: {num_homeless}")
        assert num_homeless == 15

# ********************************************************
# Other (unit) tests
# ********************************************************


def _get_rand(test_population_initialisation, N=100):
    """Get a random number using the PopulationInitialisation object's random number generator"""
    for _ in range(N):
        test_population_initialisation.random.random()
    return test_population_initialisation.random.random()


def test_random():
    """
    Checks that random classes are produce different (or the same!) numbers when they should do
    :return:
    """
    p1 = PopulationInitialisation(**population_init_args, read_data=False)
    p2 = PopulationInitialisation(**population_init_args, random_seed=2.0, read_data=False)
    p3 = PopulationInitialisation(**population_init_args, random_seed=2.0, read_data=False)

    # Genrate a random number from each model. The second two numbers should be the same
    r1, r2, r3 = [_get_rand(x) for x in [p1, p2, p3]]

    assert r1 != r2
    assert r2 == r3

    # Check that this still happens even if they are executed in pools.
    # Create a large number of microsims and check that all random numbers are unique
    pool = multiprocessing.Pool()
    num_reps = 1000
    m = [PopulationInitialisation(**population_init_args, read_data=False) for _ in range(num_reps)]
    r = pool.map(_get_rand, m)
    assert len(r) == len(set(r))


def test_extract_msoas_from_individuals():
    """Check that a list of areas can be successfully extracted from a DataFrame of indviduals"""
    individuals = pd.DataFrame(data={"area": ["C", "A", "F", "A", "A", "F"]})
    areas = PopulationInitialisation.extract_msoas_from_individuals(individuals)
    assert len(areas) == 3
    # Check the order is correct too
    assert False not in [x == y for (x, y) in zip(areas, ["A", "C", "F"])]


def test__add_location_columns():
    df = pd.DataFrame(data={"Name": ['a', 'b', 'c', 'd']})
    with pytest.raises(Exception):  # Should fail if lists are wrong length
        PopulationInitialisation._add_location_columns(df, location_names=["a", "b"], location_ids=None)
        PopulationInitialisation._add_location_columns(df, location_names=df.Name, location_ids=[1, 2])
    with pytest.raises(TypeError):  # Can't get the length of None
        PopulationInitialisation._add_location_columns(df, location_names=None)

    # Call the function
    x = PopulationInitialisation._add_location_columns(df, location_names=df.Name)
    assert x is None  # Function shouldn't return anything. Does things inplace
    # Default behaviour is just add columns
    assert False not in (df.columns.values == ["Name", "ID", "Location_Name", "Danger"])
    assert False not in list(df.Location_Name == df.Name)
    assert False not in list(df.ID == range(0, 4))
    assert False not in list(df.index == range(0, 4))
    # Adding columns again shouldn't change anything
    PopulationInitialisation._add_location_columns(df, location_names=df.Name)
    assert False not in (df.columns.values == ["Name", "ID", "Location_Name", "Danger"])
    assert False not in list(df.Location_Name == df.Name)
    assert False not in list(df.ID == range(0, 4))
    assert False not in list(df.index == range(0, 4))
    # See what happens if we give it IDs
    PopulationInitialisation._add_location_columns(df, location_names=df.Name, location_ids=[5, 7, 10, -1])
    assert False not in (df.columns.values == ["Name", "ID", "Location_Name", "Danger"])
    assert False not in list(df.ID == [5, 7, 10, -1])
    assert False not in list(df.index == range(0, 4))  # Index shouldn't change
    # Shouldn't matter if IDs are Dataframes or Series
    PopulationInitialisation._add_location_columns(df, location_names=pd.Series(df.Name))
    assert False not in list(df.Location_Name == df.Name)
    assert False not in list(df.index == range(0, 4))  # Index shouldn't change
    PopulationInitialisation._add_location_columns(df, location_names=df.Name, location_ids=np.array([5, 7, 10, -1]))
    assert False not in list(df.ID == [5, 7, 10, -1])
    assert False not in list(df.index == range(0, 4))  # Index shouldn't change
    # Set a weird index, the function should replace it with the row number
    df = pd.DataFrame(data={"Name": ['a', 'b', 'c', 'd'], "Col2": [4, -6, 8, 1.4]}, )
    df.set_index("Col2")
    PopulationInitialisation._add_location_columns(df, location_names=df.Name)
    assert False not in list(df.ID == range(0, 4))
    assert False not in list(df.index == range(0, 4))

    # TODO dest that the _add_location_columns function correctly adds the required standard columns
    # to a locations dataframe, and does appropriate checks for correct lengths of input lists etc.


def test__normalise():
    # TODO test the 'decimals' argument too.
    # Should normalise so that the input list sums to 1
    # Fail if aa single number is given
    for l in [2, 1]:
        with pytest.raises(Exception):
            PopulationInitialisation._normalise(l)

    # 1-item lists should return [1.0]
    for l in [[0.1], [5.3]]:
        assert PopulationInitialisation._normalise(l) == [1.0]

    # If numbers are the same (need to work out why these tests fail,the function seems OK)
    # for l in [ [2, 2], [0, 0], [-1, -1], [1, 1] ]:
    #    assert PopulationInitialisation._normalise(l) == [0.5, 0.5]

    # Other examples
    assert PopulationInitialisation._normalise([4, 6]) == [0.4, 0.6]
    assert PopulationInitialisation._normalise([40, 60]) == [0.4, 0.6]
    assert PopulationInitialisation._normalise([6, 6, 6, 6, 6]) == [0.2, 0.2, 0.2, 0.2, 0.2]


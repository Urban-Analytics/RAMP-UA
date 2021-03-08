import os
import pytest
import copy
from microsim.microsim_model import Microsim
from microsim.column_names import ColumnNames
from microsim.population_initialisation import PopulationInitialisation
import multiprocessing

# ********************************************************
# These tests run through a whole dummy model process
# ********************************************************
from quant_api import QuantRampAPI

test_dir = os.path.dirname(os.path.abspath(__file__))

# arguments used when calling the PopulationInitialisation constructor.
population_init_args = {"data_dir": os.path.join(test_dir, "dummy_data"),
                        "testing": True, "debug": True,
                        "quant_object": QuantRampAPI(os.path.join("devon_data", "QUANT_RAMP"))
                        }

# arguments used when calling the Microsim constructor.
microsim_args = {"data_dir": os.path.join(test_dir, "dummy_data"),
                 "r_script_dir": os.path.normpath(os.path.join(test_dir, "..", "R/py_int")),
                 "disable_disease_status": True}


# This 'fixture' means that other test functions can use the object created here.
# Note: Don't try to run this test, it will be called when running the others that need it,
# like `test_step()`.
@pytest.fixture()
def test_microsim():
    population_init = PopulationInitialisation(**population_init_args)

    microsim = Microsim(
        individuals=population_init.individuals,
        activity_locations=population_init.activity_locations,
        time_activity_multiplier=None,
        **microsim_args)

    yield microsim


def test_change_behaviour_with_disease(test_microsim):
    """Check that individuals behaviour changed correctly with the disease status"""
    m = copy.deepcopy(test_microsim)  # less typing and so as not to interfere with other tests

    # Give some people the disease (these two chosen because they both spend a bit of time in retail
    p1 = 1
    p2 = 6
    m.individuals.loc[p1, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.SYMPTOMATIC  # Behaviour change
    m.individuals.loc[p2, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.PRESYMPTOMATIC  # No change

    m.step()
    m.change_behaviour_with_disease()  # (this isn't called by default when testing)

    # Nothing should have happened as we hadn't indicated a change in disease status
    for p, act in zip([p1, p1, p2, p2], [ColumnNames.Activities.HOME, ColumnNames.Activities.RETAIL,
                                         ColumnNames.Activities.HOME, ColumnNames.Activities.RETAIL]):
        assert m.individuals.loc[p, f"{act}{ColumnNames.ACTIVITY_DURATION}"] == \
           m.individuals.loc[p, f"{act}{ColumnNames.ACTIVITY_DURATION_INITIAL}"]

    # Mark behaviour changed then try again
    m.individuals.loc[p1, ColumnNames.DISEASE_STATUS_CHANGED] = True
    m.individuals.loc[p2, ColumnNames.DISEASE_STATUS_CHANGED] = True

    m.step()
    m.change_behaviour_with_disease()  # (this isn't called by default when testing)

    # First person should spend more time at home and less at work
    assert m.individuals.loc[p1, f"{ColumnNames.Activities.RETAIL}{ColumnNames.ACTIVITY_DURATION}"] < m.individuals.loc[
        p1, f"{ColumnNames.Activities.RETAIL}{ColumnNames.ACTIVITY_DURATION_INITIAL}"]
    assert m.individuals.loc[p1, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION}"] > m.individuals.loc[
        p1, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION_INITIAL}"]
    # Second person should be unchanged
    assert m.individuals.loc[p2, f"{ColumnNames.Activities.RETAIL}{ColumnNames.ACTIVITY_DURATION}"] == m.individuals.loc[
        p2, f"{ColumnNames.Activities.RETAIL}{ColumnNames.ACTIVITY_DURATION_INITIAL}"]
    assert m.individuals.loc[p2, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION}"] == m.individuals.loc[
        p2, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION_INITIAL}"]

    # Mark behaviour changed then try again
    m.individuals.loc[p1, ColumnNames.DISEASE_STATUS_CHANGED] = True
    m.individuals.loc[p2, ColumnNames.DISEASE_STATUS_CHANGED] = True

    m.step()
    m.change_behaviour_with_disease()  # (this isn't called by default when testing)

    # First person should spend more time at home and less at work
    assert m.individuals.loc[p1, f"{ColumnNames.Activities.RETAIL}{ColumnNames.ACTIVITY_DURATION}"] < m.individuals.loc[
        p1, f"{ColumnNames.Activities.RETAIL}{ColumnNames.ACTIVITY_DURATION_INITIAL}"]
    assert m.individuals.loc[p1, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION}"] > m.individuals.loc[
        p1, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION_INITIAL}"]

    # Second person should be unchanged
    assert m.individuals.loc[p2, f"{ColumnNames.Activities.RETAIL}{ColumnNames.ACTIVITY_DURATION}"] == m.individuals.loc[
        p2, f"{ColumnNames.Activities.RETAIL}{ColumnNames.ACTIVITY_DURATION_INITIAL}"]
    assert m.individuals.loc[p2, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION}"] == m.individuals.loc[
        p2, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION_INITIAL}"]

    # First person no longer infectious, behaviour should go back to normal
    m.individuals.loc[p1, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.RECOVERED
    m.step()
    m.change_behaviour_with_disease()  # (this isn't called by default when testing)
    assert m.individuals.loc[p1, f"{ColumnNames.Activities.RETAIL}{ColumnNames.ACTIVITY_DURATION}"] == m.individuals.loc[
        p1, f"{ColumnNames.Activities.RETAIL}{ColumnNames.ACTIVITY_DURATION_INITIAL}"]
    assert m.individuals.loc[p1, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION}"] == m.individuals.loc[
        p1, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION_INITIAL}"]


def test_update_venue_danger_and_risks(test_microsim):
    """Check that the current risk is updated properly"""
    # This is actually tested as part of test_step
    assert True


def test_hazard_multipliers(test_microsim):
    """
    This tests whether hazards for particular disease statuses or locations are multiplied properly.
    The relevant code is in update_venue_danger_and_risks().

    :param test_microsim: This is a pointer to the initialised model. Dummy data will have been read in,
    but no stepping has taken place yet."""
    m = copy.deepcopy(test_microsim)  # For less typing and so as not to interfere with other functions use microsim
    households = m.activity_locations[f"{ColumnNames.Activities.HOME}"]._locations

    # Note: the following is a useful way to get relevant info about the individuals
    # m.individuals.loc[:, ["ID", "PID", "HID", "area", ColumnNames.DISEASE_STATUS, "MSOA_Cases", "HID_Cases"]]

    # Set the hazard-related parameters.

    # As we don't specify them when the tests are set up, they should be empty dictionaries
    assert not m.hazard_location_multipliers
    assert not m.hazard_individual_multipliers

    # Manually create some hazards for individuals and locations as per the parameters file
    m.hazard_individual_multipliers["presymptomatic"] = 1.0
    m.hazard_individual_multipliers["asymptomatic"] = 2.0
    m.hazard_individual_multipliers["symptomatic"] = 3.0
    for act in ColumnNames.Activities.ALL:
        m.hazard_location_multipliers[act] = 1.0

    # Step 0 (initialisation):

    # Everyone should start without the disease (they will have been assigned a status as part of initialisation)
    m.individuals[ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.SUSCEPTIBLE

    #
    # Person 1: lives with one other person (p2). Both people spend all their time at home doing nothing else
    #
    p1 = 0
    p2 = 1

    m.individuals.loc[p1, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.PRESYMPTOMATIC  # Give p1 the disease
    for p in [p1, p2]:  # Set their activity durations to 0 except for home
        for name, activity in m.activity_locations.items():
            m.individuals.at[p, f"{name}{ColumnNames.ACTIVITY_DURATION}"] = 0.0
        m.individuals.at[p, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION}"] = 1.0

    m.step()

    # Check the disease has spread to the house with a multiplier of 1.0, but nowhere else
    _check_hazard_spread(p1, p2, m.individuals, households, 1.0)

    # If the person is asymptomatic, we said the hazard should be doubled, so the risk should be doubled
    m.individuals.loc[p1, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.ASYMPTOMATIC  # Give p1 the disease
    m.individuals.loc[p2, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.SUSCEPTIBLE  # Make sure p2 is clean

    m.step()
    _check_hazard_spread(p1, p2, m.individuals, households, 2.0)

    # And for symptomatic we said 3.0
    m.individuals.loc[p1, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.SYMPTOMATIC  # Give p1 the disease
    m.individuals.loc[p2, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.SUSCEPTIBLE  # Make sure p2 is clean

    m.step()
    _check_hazard_spread(p1, p2, m.individuals, households, 3.0)


    # But if they both get sick then double danger and risk)
    m.individuals.loc[p1, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.SYMPTOMATIC
    m.individuals.loc[p2, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.SYMPTOMATIC
    m.step()
    _check_hazard_spread(p1, p2, m.individuals, households, 6.0)

    #
    # Now see if the hazards for locations work. Check houses and schools
    #

    # Both people are symptomatic. And double the hazard for home. So in total the new risk should
    # be 3 * 2 * 5 = 30
    m.individuals.loc[p1, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.SYMPTOMATIC
    m.individuals.loc[p2, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.SYMPTOMATIC
    m.hazard_location_multipliers[ColumnNames.Activities.HOME] = 5.0

    m.step()
    _check_hazard_spread(p1, p2, m.individuals, households, 30.0)


    # Check for school as well. Now give durations for home and school as 0.5. Make them asymptomatic so the additional
    # hazard is 2.0 (set above). And make the risks for home 5.35 and for school 2.9.

    # Make sure all *other* individuals go to a different school (school 1), then make p1 and p2 go to the same school
    # (school 0) below
    # (annoying apply is because pandas doesn't like a list being assigned to a value in a cell)
    m.individuals[f"{ColumnNames.Activities.PRIMARY}{ColumnNames.ACTIVITY_VENUES}"] = \
        m.individuals.loc[:, f"{ColumnNames.Activities.PRIMARY}{ColumnNames.ACTIVITY_VENUES}"].apply(lambda x: [1])
    m.individuals.loc[[p1, p2], f"{ColumnNames.Activities.PRIMARY}{ColumnNames.ACTIVITY_VENUES}"] = \
        m.individuals.loc[[p1, p2], f"{ColumnNames.Activities.PRIMARY}{ColumnNames.ACTIVITY_VENUES}"].apply(lambda x: [0])
    # All school flows need to be 1 (don't want the people to go to more than 1 school
    m.individuals[f"{ColumnNames.Activities.PRIMARY}{ColumnNames.ACTIVITY_FLOWS}"] = \
        m.individuals.loc[:, f"{ColumnNames.Activities.PRIMARY}{ColumnNames.ACTIVITY_VENUES}"].apply(lambda x: [1.0])

    for p in [p1, p2]:  # Set their activity durations to 0.5 for home and school
        for name, activity in m.activity_locations.items():
            m.individuals.at[p, f"{name}{ColumnNames.ACTIVITY_DURATION}"] = 0.0
        m.individuals.at[p, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION}"] = 0.5
        m.individuals.at[p, f"{ColumnNames.Activities.PRIMARY}{ColumnNames.ACTIVITY_DURATION}"] = 0.5
    # Make them asymptomatic
    m.individuals.loc[p1, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.ASYMPTOMATIC
    m.individuals.loc[p2, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.ASYMPTOMATIC
    # Set hazards for home and school
    m.hazard_location_multipliers[ColumnNames.Activities.HOME] = 5.35
    m.hazard_location_multipliers[ColumnNames.Activities.PRIMARY] = 2.9

    m.step()

    # Can't use _check_hazard_spread because it assumes only one activity (HOME)
    # Current risks are:
    # For home. 2 people * 2.0 asymptomatic hazard * 0.5 duration * 5.35 HOME risk = 10.7
    # For school. 2 people * 2.0 asymptomatic hazard * 0.5 duration * 2.9 PRIMARY risk = 5.8
    # Total risk for individuals: 10.7*0.5 + 5.8*0.5 = 8.25

    # Individuals
    for p in [p1, p2]:
        assert m.individuals.at[p, ColumnNames.CURRENT_RISK] == 8.25
    for p in range(2, len(m.individuals)):
        assert m.individuals.at[p, ColumnNames.CURRENT_RISK] == 0.0

    # Households
    assert households.at[0, ColumnNames.LOCATION_DANGER] == 10.7
    # (the self.households dataframe should be the same as the one stored in the activity_locations)
    assert m.activity_locations[ColumnNames.Activities.HOME]._locations.at[0, ColumnNames.LOCATION_DANGER] == 10.7
    for h in range(1, len(households)):  # all others are 0
        assert households.at[h, ColumnNames.LOCATION_DANGER] == 0.0

    # Schools
    assert m.activity_locations[ColumnNames.Activities.PRIMARY]._locations.at[0, ColumnNames.LOCATION_DANGER] == 5.8
    for h in range(1, len( m.activity_locations[ColumnNames.Activities.PRIMARY]._locations)):  # all others are 0
        assert m.activity_locations[ColumnNames.Activities.PRIMARY]._locations.at[h, ColumnNames.LOCATION_DANGER] == 0.0

    print("End of test hazard multipliers")


def _check_hazard_spread(p1, p2, individuals, households, risk):
    """Checks how the disease is spreading. To save code repetition in test_hazard_multipliers"""
    for p in [p1, p2]:
        assert individuals.at[p, ColumnNames.CURRENT_RISK] == risk
    for p in range(2, len(individuals)):
        assert individuals.at[p, ColumnNames.CURRENT_RISK] == 0.0
    assert households.at[0, ColumnNames.LOCATION_DANGER] == risk
    for h in range(1, len(households)):  # all others are 0
        assert households.at[h, ColumnNames.LOCATION_DANGER] == 0.0


def test_step(test_microsim):
    """
    Test the step method. This is the main test of the model. Simulate a deterministic run through and
    make sure that the model runs as expected.

    Only thing it doesn't do is check for retail, shopping, etc., that danger and risk increase by the correct
    amount. It just checks they go above 0 (or not). It does do that more precise checks for home activities though.

    :param test_microsim: This is a pointer to the initialised model. Dummy data will have been read in,
    but no stepping has taken place yet."""
    m = copy.deepcopy(test_microsim)  # For less typing and so as not to interfere with other functions use microsim
    households = m.activity_locations[f"{ColumnNames.Activities.HOME}"]._locations

    # Note: the following is a useful way to get relevant info about the individuals
    # m.individuals.loc[:, ["ID", "PID", "HID", "area", ColumnNames.DISEASE_STATUS, "MSOA_Cases", "HID_Cases"]]

    # Step 0 (initialisation):

    # Everyone should start without the disease (they will have been assigned a status as part of initialisation)
    m.individuals[ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.SUSCEPTIBLE

    #
    # Person 1: lives with one other person (p2). Both people spend all their time at home doing nothing else
    #
    p1 = 0
    p2 = 1

    m.individuals.loc[p1, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.SYMPTOMATIC  # Give them the disease
    for p in [p1, p2]:  # Set their activity durations to 0
        for name, activity in m.activity_locations.items():
            m.individuals.at[p, f"{name}{ColumnNames.ACTIVITY_DURATION}"] = 0.0
        m.individuals.at[p, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION}"] = 1.0  # Spend all their time at home

    m.step()

    # Check the disease has spread to the house but nowhere else
    for p in [p1, p2]:
        assert m.individuals.at[p, ColumnNames.CURRENT_RISK] == 1.0
    for p in range(2, len(m.individuals)):
        assert m.individuals.at[p, ColumnNames.CURRENT_RISK] == 0.0

    assert households.at[0, ColumnNames.LOCATION_DANGER] == 1.0
    for h in range(1, len(households)):  # all others are 0
        assert households.at[h, ColumnNames.LOCATION_DANGER] == 0.0

    m.step()

    # Risk and danger stay the same (it does not cumulate over days)
    for p in [p1, p2]:
        assert m.individuals.at[p, ColumnNames.CURRENT_RISK] == 1.0
    for p in range(2, len(m.individuals)):
        assert m.individuals.at[p, ColumnNames.CURRENT_RISK] == 0.0
    assert households.at[0, ColumnNames.LOCATION_DANGER] == 1.0
    for h in range(1, len(households)):
        assert households.at[h, ColumnNames.LOCATION_DANGER] == 0.0

    # If the infected person doesn't go home (in this test they do absolutely nothing) then danger and risks should go
    # back to 0
    m.individuals.at[p1, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION}"] = 0.0
    m.step()
    for p in range(len(m.individuals)):
        assert m.individuals.at[p, ColumnNames.CURRENT_RISK] == 0.0
    for h in range(0, len(households)):
        assert households.at[h, ColumnNames.LOCATION_DANGER] == 0.0

    # But if they both get sick then they should be 2.0 (double danger and risk)
    m.individuals.loc[p1:p2, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.SYMPTOMATIC  # Give them the disease
    m.individuals.at[p1, f"{ColumnNames.Activities.HOME}{ColumnNames.ACTIVITY_DURATION}"] = 1.0  # Make the duration normal again
    m.step()
    for p in [p1, p2]:
        assert m.individuals.at[p, ColumnNames.CURRENT_RISK] == 2.0
    assert households.at[0, ColumnNames.LOCATION_DANGER] == 2.0
    for h in range(1, len(households)):  # All other houses are danger free
        assert households.at[h, ColumnNames.LOCATION_DANGER] == 0.0

    #
    # Now see what happens when one person gets the disease and spreads it to schools, shops and work
    #
    del p1, p2
    p1 = 4  # The infected person is index 1
    # Make everyone better except for that one person
    m.individuals[ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.SUSCEPTIBLE
    m.individuals.loc[p1, ColumnNames.DISEASE_STATUS] = ColumnNames.DiseaseStatuses.SYMPTOMATIC
    # Assign everyone equal time doing all activities
    for name, activity in m.activity_locations.items():
        m.individuals[f"{name}{ColumnNames.ACTIVITY_DURATION}"] = 1.0 / len(m.activity_locations)

    m.step()

    # Now check that the danger has propagated to locations and risk to people
    # TODO Also check that the total risks and danger scores sum correctly
    for name, activity in m.activity_locations.items():
        # Indices of the locations where this person visited
        visited_idx = m.individuals.at[p1, f"{name}{ColumnNames.ACTIVITY_VENUES}"]
        not_visited_idx = list(set(range(len(activity._locations))) - set(visited_idx))
        # Dangers should be >0.0 (or not if the person didn't visit there)
        assert False not in list(activity._locations.loc[visited_idx, "Danger"].values > 0)
        assert False not in list(activity._locations.loc[not_visited_idx, "Danger"].values == 0)
        # Individuals should have an associated risk
        for index, row in m.individuals.iterrows():
            for idx in visited_idx:
                if idx in row[f"{name}{ColumnNames.ACTIVITY_VENUES}"]:
                    assert row[ColumnNames.CURRENT_RISK] > 0
                    # Note: can't check if risk is equal to 0 because it might come from another activity

    print("End of test step")


def _get_rand(microsim_model, N=100):
    """Get a random number using the PopulationInitialisation object's random number generator"""
    for _ in range(N):
        microsim_model.random.random()
    return microsim_model.random.random()


def test_random():
    """
    Checks that random classes are produce different (or the same!) numbers when they should do
    :return:
    """
    population_init = PopulationInitialisation(**population_init_args)

    p1 = Microsim(individuals=population_init.individuals, activity_locations=population_init.activity_locations,
                  **microsim_args)
    p2 = Microsim(individuals=population_init.individuals, activity_locations=population_init.activity_locations,
                  random_seed=2.0, **microsim_args)
    p3 = Microsim(individuals=population_init.individuals, activity_locations=population_init.activity_locations,
                  random_seed=2.0, **microsim_args)

    # Genrate a random number from each model. The second two numbers should be the same
    r1, r2, r3 = [_get_rand(x) for x in [p1, p2, p3]]

    assert r1 != r2
    assert r2 == r3

    # Check that this still happens even if they are executed in pools.
    # Create a large number of microsims and check that all random numbers are unique
    pool = multiprocessing.Pool()
    num_reps = 1000
    m = [Microsim(individuals=population_init.individuals, activity_locations=population_init.activity_locations,
                  **microsim_args) for _ in range(num_reps)]
    r = pool.map(_get_rand, m)
    assert len(r) == len(set(r))
    pool.close()

    # Repeat, this time explicitly passing a None seed
    pool = multiprocessing.Pool()
    num_reps = 50  # (don't do quite as many this time, it takes ages)
    m = [Microsim(individuals=population_init.individuals, activity_locations=population_init.activity_locations,
                  random_seed=None, **microsim_args) for _ in range(num_reps)]
    r = pool.map(_get_rand, m)
    assert len(r) == len(set(r))
    pool.close()



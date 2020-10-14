import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
from microsim.opencl.ramp.params import Params
from microsim.opencl.ramp.simulator import Simulator
from microsim.opencl.ramp.snapshot import Snapshot
from microsim.opencl.ramp.disease_statuses import DiseaseStatus

nplaces = 8
npeople = 50000
nslots = 8


def test_susceptible_become_infected():
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    test_hazard = 0.4

    people_hazards_test_data = np.full(npeople, test_hazard, dtype=np.float32)

    people_statuses_test_data = np.full(npeople, DiseaseStatus.Susceptible.value, dtype=np.uint32)

    people_transition_times_test_data = np.zeros(npeople, dtype=np.uint32)

    snapshot.buffers.people_hazards[:] = people_hazards_test_data
    snapshot.buffers.people_statuses[:] = people_statuses_test_data
    snapshot.buffers.people_transition_times[:] = people_transition_times_test_data

    simulator = Simulator(snapshot, gpu=False)
    simulator.upload_all(snapshot.buffers)

    people_statuses_before = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_before)

    assert np.array_equal(people_statuses_before, people_statuses_test_data)

    simulator.step_kernel("people_update_statuses")

    people_statuses_after = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after)

    num_exposed = np.count_nonzero(people_statuses_after == DiseaseStatus.Exposed.value)
    proportion_exposed = num_exposed / npeople

    expected_proportion_infected = test_hazard / (test_hazard + np.exp(-test_hazard))

    assert np.isclose(expected_proportion_infected, proportion_exposed, atol=0.01)


def test_transmission_times_decremented():
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    people_transition_times_test_data = np.full(npeople, 3, dtype=np.uint32)
    snapshot.buffers.people_transition_times[:] = people_transition_times_test_data

    simulator = Simulator(snapshot, gpu=False)
    simulator.upload_all(snapshot.buffers)

    # check decremented after one step
    simulator.step_kernel("people_update_statuses")

    people_transmission_times_after = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_transition_times", people_transmission_times_after)

    expected_people_transition_times = np.full(npeople, 2, dtype=np.uint32)

    assert np.array_equal(expected_people_transition_times, people_transmission_times_after)

    # check decremented after another step
    simulator.step_kernel("people_update_statuses")

    people_transmission_times_after = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_transition_times", people_transmission_times_after)

    expected_people_transition_times = np.full(npeople, 1, dtype=np.uint32)

    assert np.array_equal(expected_people_transition_times, people_transmission_times_after)


# def test_exposed_become_presymptomatic():
#     snapshot = Snapshot.random(nplaces, npeople, nslots)
#
#     people_statuses_test_data = np.full(npeople, DiseaseStatus.Exposed.value, dtype=np.uint32)
#
#     people_transition_times_test_data = np.full(npeople, 1, dtype=np.uint32)
#
#     snapshot.buffers.people_statuses[:] = people_statuses_test_data
#     snapshot.buffers.people_transition_times[:] = people_transition_times_test_data
#
#     simulator = Simulator(snapshot, gpu=False)
#     simulator.upload_all(snapshot.buffers)
#
#     people_statuses_before = np.zeros(npeople, dtype=np.uint32)
#     simulator.download("people_statuses", people_statuses_before)
#
#     assert np.array_equal(people_statuses_before, people_statuses_test_data)
#
#     simulator.step_kernel("people_update_statuses")
#
#     # check that statuses don't change after one step
#     people_statuses_after_one_step = np.zeros(npeople, dtype=np.uint32)
#     simulator.download("people_statuses", people_statuses_after_one_step)
#
#     assert np.array_equal(people_statuses_after_one_step, people_statuses_test_data)
#
#     # run another timestep, this time statuses should change
#     simulator.step_kernel("people_update_statuses")
#
#     # assert that all statuses change to presymptomatic after two timesteps
#     people_statuses_after_two_steps = np.zeros(npeople, dtype=np.uint32)
#     simulator.download("people_statuses", people_statuses_after_two_steps)
#
#     assert np.all(people_statuses_after_two_steps == DiseaseStatus.Presymptomatic.value)


# def test_presymptomatic_update_status():
#     snapshot = Snapshot.random(nplaces, npeople, nslots)
#
#     people_statuses_test_data = np.full(npeople, DiseaseStatus.Presymptomatic.value, dtype=np.uint32)
#
#     people_transition_times_test_data = np.full(npeople, 1, dtype=np.uint32)
#
#     snapshot.buffers.people_statuses[:] = people_statuses_test_data
#     snapshot.buffers.people_transition_times[:] = people_transition_times_test_data
#
#     simulator = Simulator(snapshot, gpu=False)
#     simulator.upload_all(snapshot.buffers)
#
#     people_statuses_before = np.zeros(npeople, dtype=np.uint32)
#     simulator.download("people_statuses", people_statuses_before)
#
#     assert np.array_equal(people_statuses_before, people_statuses_test_data)
#
#     simulator.step_kernel("people_update_statuses")
#
#     # check that statuses don't change after one step
#     people_statuses_after_one_step = np.zeros(npeople, dtype=np.uint32)
#     simulator.download("people_statuses", people_statuses_after_one_step)
#
#     assert np.array_equal(people_statuses_after_one_step, people_statuses_test_data)
#
#     # run another timestep, this time statuses should change
#     simulator.step_kernel("people_update_statuses")
#
#     # assert that statuses change to either symptomatic or asymptomatic in the correct proportion
#     people_statuses_after_two_steps = np.zeros(npeople, dtype=np.uint32)
#     simulator.download("people_statuses", people_statuses_after_two_steps)
#
#     num_asymptomatic = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Asymptomatic.value)
#     proportion_asymptomatic = num_asymptomatic / npeople
#
#     num_symptomatic = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Symptomatic.value)
#     proportion_symptomatic = num_symptomatic / npeople
#
#     expected_proportion_asymptomatic = 0.4
#     expected_proportion_symptomatic = 1 - expected_proportion_asymptomatic
#
#     assert np.isclose(expected_proportion_asymptomatic, proportion_asymptomatic, atol=0.01)
#     assert np.isclose(expected_proportion_symptomatic, proportion_symptomatic, atol=0.01)

def test_exposed_become_asymptomatic_or_presymptomatic():
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    people_statuses_test_data = np.full(npeople, DiseaseStatus.Exposed.value, dtype=np.uint32)
    people_transition_times_test_data = np.full(npeople, 1, dtype=np.uint32)
    # set all people to obesity=0, corresponding to normal BMI
    people_obesity_test_data = np.full(npeople, 0, dtype=np.uint8)
    people_transition_times_test_data = np.full(npeople, 1, dtype=np.uint32)

    snapshot.buffers.people_statuses[:] = people_statuses_test_data
    snapshot.buffers.people_obesity[:] = people_obesity_test_data
    snapshot.buffers.people_transition_times[:] = people_transition_times_test_data

    params = Params()
    expected_proportion_asymptomatic = 0.5
    params.proportion_asymptomatic = expected_proportion_asymptomatic
    snapshot.update_params(params)

    simulator = Simulator(snapshot, gpu=False)
    simulator.upload_all(snapshot.buffers)

    people_statuses_before = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_before)

    assert np.array_equal(people_statuses_before, people_statuses_test_data)

    simulator.step_kernel("people_update_statuses")

    # check that statuses don't change after one step
    people_statuses_after_one_step = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after_one_step)

    assert np.array_equal(people_statuses_after_one_step, people_statuses_test_data)

    # run another timestep, this time statuses should change
    simulator.step_kernel("people_update_statuses")

    # assert that statuses change to either symptomatic or asymptomatic in the correct proportion
    people_statuses_after_two_steps = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after_two_steps)

    num_asymptomatic = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Asymptomatic.value)
    result_proportion_asymptomatic = num_asymptomatic / npeople

    num_presymptomatic = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Presymptomatic.value)
    result_proportion_presymptomatic = num_presymptomatic / npeople

    expected_proportion_presymptomatic = 1 - expected_proportion_asymptomatic

    assert np.isclose(expected_proportion_asymptomatic, result_proportion_asymptomatic, atol=0.01)
    assert np.isclose(expected_proportion_presymptomatic, result_proportion_presymptomatic, atol=0.01)


def test_more_overweight_become_symptomatic():
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    people_statuses_test_data = np.full(npeople, DiseaseStatus.Exposed.value, dtype=np.uint32)
    people_transition_times_test_data = np.full(npeople, 1, dtype=np.uint32)
    # set all people to obesity=1, corresponding to overweight
    people_obesity_test_data = np.full(npeople, 1, dtype=np.uint8)
    people_transition_times_test_data = np.full(npeople, 1, dtype=np.uint32)

    snapshot.buffers.people_statuses[:] = people_statuses_test_data
    snapshot.buffers.people_obesity[:] = people_obesity_test_data
    snapshot.buffers.people_transition_times[:] = people_transition_times_test_data

    params = Params()
    base_proportion_asymptomatic = 0.5
    params.proportion_asymptomatic = base_proportion_asymptomatic
    overweight_sympt_mplier = 1.2
    params.overweight_sympt_mplier = overweight_sympt_mplier
    snapshot.update_params(params)

    simulator = Simulator(snapshot, gpu=False)
    simulator.upload_all(snapshot.buffers)

    people_statuses_before = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_before)

    assert np.array_equal(people_statuses_before, people_statuses_test_data)

    simulator.step_kernel("people_update_statuses")

    # check that statuses don't change after one step
    people_statuses_after_one_step = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after_one_step)

    assert np.array_equal(people_statuses_after_one_step, people_statuses_test_data)

    # run another timestep, this time statuses should change
    simulator.step_kernel("people_update_statuses")

    # assert that statuses change to either symptomatic or asymptomatic in the correct proportion
    people_statuses_after_two_steps = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after_two_steps)

    num_asymptomatic = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Asymptomatic.value)
    result_proportion_asymptomatic = num_asymptomatic / npeople

    num_presymptomatic = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Presymptomatic.value)
    result_proportion_presymptomatic = num_presymptomatic / npeople

    base_proportion_symptomatic = 1 - base_proportion_asymptomatic
    expected_proportion_presymptomatic = overweight_sympt_mplier * base_proportion_symptomatic
    expected_proportion_asymptomatic = 1 - expected_proportion_presymptomatic

    assert np.isclose(expected_proportion_asymptomatic, result_proportion_asymptomatic, atol=0.01)
    assert np.isclose(expected_proportion_presymptomatic, result_proportion_presymptomatic, atol=0.01)


def test_presymptomatic_become_symptomatic():
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    people_statuses_test_data = np.full(npeople, DiseaseStatus.Presymptomatic.value, dtype=np.uint32)

    people_transition_times_test_data = np.full(npeople, 1, dtype=np.uint32)

    snapshot.buffers.people_statuses[:] = people_statuses_test_data
    snapshot.buffers.people_transition_times[:] = people_transition_times_test_data

    simulator = Simulator(snapshot, gpu=False)
    simulator.upload_all(snapshot.buffers)

    people_statuses_before = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_before)

    assert np.array_equal(people_statuses_before, people_statuses_test_data)

    simulator.step_kernel("people_update_statuses")

    # check that statuses don't change after one step
    people_statuses_after_one_step = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after_one_step)

    assert np.array_equal(people_statuses_after_one_step, people_statuses_test_data)

    # run another timestep, this time statuses should change
    simulator.step_kernel("people_update_statuses")

    # assert that all statuses change to symptomatic after two timesteps
    people_statuses_after_two_steps = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after_two_steps)

    assert np.all(people_statuses_after_two_steps == DiseaseStatus.Symptomatic.value)


def test_symptomatic_become_recovered_or_dead_young_age():
    # NB: run with more people since chance of young people dying is low
    npeople = 500000
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    people_ages_test_data = np.full(npeople, 25, dtype=np.uint16)
    people_statuses_test_data = np.full(npeople, DiseaseStatus.Symptomatic.value, dtype=np.uint32)
    people_transition_times_test_data = np.full(npeople, 1, dtype=np.uint32)

    snapshot.buffers.people_ages[:] = people_ages_test_data
    snapshot.buffers.people_statuses[:] = people_statuses_test_data
    snapshot.buffers.people_transition_times[:] = people_transition_times_test_data

    simulator = Simulator(snapshot, gpu=False)
    simulator.upload_all(snapshot.buffers)

    people_statuses_before = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_before)

    assert np.array_equal(people_statuses_before, people_statuses_test_data)

    simulator.step_kernel("people_update_statuses")

    # check that statuses don't change after one step
    people_statuses_after_one_step = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after_one_step)

    assert np.array_equal(people_statuses_after_one_step, people_statuses_test_data)

    # run another timestep, this time statuses should change
    simulator.step_kernel("people_update_statuses")

    # assert that statuses change to either recovered or dead in the correct proportion
    people_statuses_after_two_steps = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after_two_steps)

    num_recovered = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Recovered.value)
    proportion_recovered = num_recovered / npeople

    num_dead = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Dead.value)
    proportion_dead = num_dead / npeople

    # expected recovery probability for ages 20 to 29
    expected_proportion_dead = 0.000309
    expected_proportion_recovered = 1 - expected_proportion_dead

    assert np.isclose(expected_proportion_recovered, proportion_recovered, atol=0.0001)
    assert np.isclose(expected_proportion_dead, proportion_dead, atol=0.0001)


def test_symptomatic_become_recovered_or_dead_old_age():
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    people_ages_test_data = np.full(npeople, 92, dtype=np.uint16)
    people_statuses_test_data = np.full(npeople, DiseaseStatus.Symptomatic.value, dtype=np.uint32)
    people_transition_times_test_data = np.full(npeople, 1, dtype=np.uint32)

    snapshot.buffers.people_ages[:] = people_ages_test_data
    snapshot.buffers.people_statuses[:] = people_statuses_test_data
    snapshot.buffers.people_transition_times[:] = people_transition_times_test_data

    simulator = Simulator(snapshot, gpu=False)
    simulator.upload_all(snapshot.buffers)

    people_statuses_before = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_before)

    assert np.array_equal(people_statuses_before, people_statuses_test_data)

    # run two timesteps so statuses should change
    simulator.step_kernel("people_update_statuses")
    simulator.step_kernel("people_update_statuses")

    # assert that statuses change to either recovered or dead in the correct proportion
    people_statuses_after_two_steps = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after_two_steps)

    num_recovered = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Recovered.value)
    proportion_recovered = num_recovered / npeople

    num_dead = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Dead.value)
    proportion_dead = num_dead / npeople

    # expected recovery probability for ages 80+
    expected_proportion_dead = 0.078
    expected_proportion_recovered = 1 - expected_proportion_dead

    assert np.isclose(expected_proportion_recovered, proportion_recovered, atol=0.01)
    assert np.isclose(expected_proportion_dead, proportion_dead, atol=0.01)


def test_not_obese_lower_mortality():
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    people_ages_test_data = np.full(npeople, 65, dtype=np.uint16)
    people_statuses_test_data = np.full(npeople, DiseaseStatus.Symptomatic.value, dtype=np.uint32)
    # set all people to obesity=0, corresponding to normal BMI
    people_obesity_test_data = np.full(npeople, 0, dtype=np.uint8)
    people_transition_times_test_data = np.full(npeople, 1, dtype=np.uint32)

    snapshot.buffers.people_ages[:] = people_ages_test_data
    snapshot.buffers.people_statuses[:] = people_statuses_test_data
    snapshot.buffers.people_obesity[:] = people_obesity_test_data
    snapshot.buffers.people_transition_times[:] = people_transition_times_test_data

    params = Params()
    params.obesity_multipliers = [1.0, 1.48, 1.48, 1.9]
    snapshot.update_params(params)

    simulator = Simulator(snapshot, gpu=False)
    simulator.upload_all(snapshot.buffers)

    people_statuses_before = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_before)

    assert np.array_equal(people_statuses_before, people_statuses_test_data)

    # run two timesteps so statuses should change
    simulator.step_kernel("people_update_statuses")
    simulator.step_kernel("people_update_statuses")

    # assert that statuses change to either recovered or dead in the correct proportion
    people_statuses_after_two_steps = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after_two_steps)

    num_recovered = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Recovered.value)
    proportion_recovered = num_recovered / npeople

    num_dead = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Dead.value)
    proportion_dead = num_dead / npeople

    # expected recovery probability for ages 60-70
    expected_proportion_dead = 0.0193
    expected_proportion_recovered = 1 - expected_proportion_dead

    assert np.isclose(expected_proportion_recovered, proportion_recovered, atol=0.01)
    assert np.isclose(expected_proportion_dead, proportion_dead, atol=0.01)


def test_obese_higher_mortality():
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    people_ages_test_data = np.full(npeople, 65, dtype=np.uint16)
    people_statuses_test_data = np.full(npeople, DiseaseStatus.Symptomatic.value, dtype=np.uint32)
    # set all people to obesity=4, corresponding to the highest category
    people_obesity_test_data = np.full(npeople, 4, dtype=np.uint8)
    people_transition_times_test_data = np.full(npeople, 1, dtype=np.uint32)

    snapshot.buffers.people_ages[:] = people_ages_test_data
    snapshot.buffers.people_statuses[:] = people_statuses_test_data
    snapshot.buffers.people_obesity[:] = people_obesity_test_data
    snapshot.buffers.people_transition_times[:] = people_transition_times_test_data

    params = Params()
    high_bmi_mortality = 1.9
    params.obesity_multipliers = [1.0, 1.48, 1.48, high_bmi_mortality]
    snapshot.update_params(params)

    simulator = Simulator(snapshot, gpu=False)
    simulator.upload_all(snapshot.buffers)

    people_statuses_before = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_before)

    assert np.array_equal(people_statuses_before, people_statuses_test_data)

    # run two timesteps so statuses should change
    simulator.step_kernel("people_update_statuses")
    simulator.step_kernel("people_update_statuses")

    # assert that statuses change to either recovered or dead in the correct proportion
    people_statuses_after_two_steps = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after_two_steps)

    num_recovered = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Recovered.value)
    proportion_recovered = num_recovered / npeople

    num_dead = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Dead.value)
    proportion_dead = num_dead / npeople

    # expected recovery probability for ages 60-70
    expected_proportion_dead = 0.0193
    expected_proportion_dead *= high_bmi_mortality
    expected_proportion_recovered = 1 - expected_proportion_dead

    assert np.isclose(expected_proportion_recovered, proportion_recovered, atol=0.01)
    assert np.isclose(expected_proportion_dead, proportion_dead, atol=0.01)


def test_diabetes_higher_mortality():
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    people_ages_test_data = np.full(npeople, 65, dtype=np.uint16)
    people_statuses_test_data = np.full(npeople, DiseaseStatus.Symptomatic.value, dtype=np.uint32)
    # set all people to diabetes=1
    people_diabetes_test_data = np.ones(npeople, dtype=np.uint8)
    people_transition_times_test_data = np.full(npeople, 1, dtype=np.uint32)

    snapshot.buffers.people_ages[:] = people_ages_test_data
    snapshot.buffers.people_statuses[:] = people_statuses_test_data
    snapshot.buffers.people_diabetes[:] = people_diabetes_test_data
    snapshot.buffers.people_transition_times[:] = people_transition_times_test_data

    params = Params()
    diabetes_multiplier = 1.4
    params.diabetes_multiplier = diabetes_multiplier
    snapshot.update_params(params)

    simulator = Simulator(snapshot, gpu=False)
    simulator.upload_all(snapshot.buffers)

    people_statuses_before = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_before)

    assert np.array_equal(people_statuses_before, people_statuses_test_data)

    # run two timesteps so statuses should change
    simulator.step_kernel("people_update_statuses")
    simulator.step_kernel("people_update_statuses")

    # assert that statuses change to either recovered or dead in the correct proportion
    people_statuses_after_two_steps = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after_two_steps)

    num_recovered = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Recovered.value)
    proportion_recovered = num_recovered / npeople

    num_dead = np.count_nonzero(people_statuses_after_two_steps == DiseaseStatus.Dead.value)
    proportion_dead = num_dead / npeople

    # expected recovery probability for ages 60-70
    expected_proportion_dead = 0.0193
    expected_proportion_dead *= diabetes_multiplier
    expected_proportion_recovered = 1 - expected_proportion_dead

    assert np.isclose(expected_proportion_recovered, proportion_recovered, atol=0.01)
    assert np.isclose(expected_proportion_dead, proportion_dead, atol=0.01)


def test_all_asymptomatic_become_recovered():
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    people_statuses_test_data = np.full(npeople, DiseaseStatus.Asymptomatic.value, dtype=np.uint32)

    people_transition_times_test_data = np.full(npeople, 1, dtype=np.uint32)

    snapshot.buffers.people_statuses[:] = people_statuses_test_data
    snapshot.buffers.people_transition_times[:] = people_transition_times_test_data

    simulator = Simulator(snapshot, gpu=False)
    simulator.upload_all(snapshot.buffers)

    people_statuses_before = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_before)

    assert np.array_equal(people_statuses_before, people_statuses_test_data)

    simulator.step_kernel("people_update_statuses")

    # check that statuses don't change after one step
    people_statuses_after_one_step = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after_one_step)

    assert np.array_equal(people_statuses_after_one_step, people_statuses_test_data)

    # run another timestep, this time statuses should change
    simulator.step_kernel("people_update_statuses")

    # assert that all statuses change to presymptomatic after two timesteps
    people_statuses_after_two_steps = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after_two_steps)

    assert np.all(people_statuses_after_two_steps == DiseaseStatus.Recovered.value)


def test_infection_transition_times_distribution(visualize=False):
    npeople = 1000000
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    test_hazard = 0.9

    people_hazards_test_data = np.full(npeople, test_hazard, dtype=np.float32)
    people_statuses_test_data = np.full(npeople, DiseaseStatus.Presymptomatic.value, dtype=np.uint32)
    people_transition_times_test_data = np.zeros(npeople, dtype=np.uint32)

    snapshot.buffers.people_hazards[:] = people_hazards_test_data
    snapshot.buffers.people_statuses[:] = people_statuses_test_data
    snapshot.buffers.people_transition_times[:] = people_transition_times_test_data

    params = Params()
    infection_log_scale = 0.75
    infection_mode = 7.0
    params.infection_log_scale = infection_log_scale
    params.infection_mode = infection_mode
    snapshot.update_params(params)

    simulator = Simulator(snapshot, gpu=False)
    simulator.upload_all(snapshot.buffers)

    simulator.step_kernel("people_update_statuses")

    people_statuses_after = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_statuses", people_statuses_after)

    people_transition_times_after = np.zeros(npeople, dtype=np.uint32)
    simulator.download("people_transition_times", people_transition_times_after)

    # Check that transition times are distributed with a log-normal distribution
    adjusted_transition_times = people_transition_times_after + 1
    mean = adjusted_transition_times.mean()
    std_dev = adjusted_transition_times.std()
    mode = scipy.stats.mode(adjusted_transition_times)[0][0]

    meanlog = infection_log_scale**2 + np.log(infection_mode)
    expected_samples = np.random.lognormal(mean=meanlog, sigma=infection_log_scale, size=npeople)
    # round samples to nearest integer
    expected_samples = np.rint(expected_samples)
    expected_mean = expected_samples.mean()
    expected_std_dev = expected_samples.std()
    expected_mode = scipy.stats.mode(expected_samples)[0][0]

    # Float to integer rounding and clamping at zero makes the original random numbers hard
    # to recover so we have slightly larger tolerances here to avoid false negatives.
    assert np.isclose(expected_mean, mean, atol=0.7)
    assert np.isclose(expected_std_dev, std_dev, atol=0.4)
    assert np.isclose(expected_mode, mode, atol=1.0)

    # check that mode is similar to original mode parameter
    assert np.isclose(infection_mode, mode, atol=1.0)

    if visualize:  # show histogram of distribution
        fig, ax = plt.subplots(1, 1)
        ax.hist(adjusted_transition_times, bins=50, range=[0, 60])
        plt.title("Result Samples")
        plt.show()
        fig, ax = plt.subplots(1, 1)
        ax.hist(expected_samples, bins=50, range=[0, 60])
        plt.title("Expected Samples")
        plt.show()

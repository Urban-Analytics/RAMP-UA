import numpy as np
from ramp.activity import Activity
from ramp.params import Params
from ramp.simulator import Simulator
from ramp.snapshot import Snapshot
from ramp.disease_statuses import DiseaseStatus

sentinel_value = (1 << 31) - 1

nplaces = 8
npeople = 3
nslots = 8


def test_correct_flow_calculation_no_lockdown():
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    people_place_ids_test_data = np.full((npeople, nslots), sentinel_value, dtype=np.uint32)
    people_place_ids_test_data[0][0:4] = [0, 5, 7, 3]
    people_place_ids_test_data[1][0:4] = [1, 5, 6, 4]
    people_place_ids_test_data[2][0:4] = [2, 3, 7, 6]

    people_flows_test_data = np.zeros((npeople, nslots), dtype=np.float32)
    people_flows_test_data[0][0:4] = [0.8, 0.1, 0.06, 0.04]
    people_flows_test_data[1][0:4] = [0.7, 0.18, 0.09, 0.03]
    people_flows_test_data[2][0:4] = [0.6, 0.2, 0.16, 0.04]

    people_statuses_test_data = np.full(npeople, DiseaseStatus.Susceptible.value, dtype=np.uint32)
    symptomatic_person_id = 1
    people_statuses_test_data[symptomatic_person_id] = np.uint32(DiseaseStatus.Symptomatic.value)

    place_activities_test_data = np.full(nplaces, Activity.Retail.value, dtype=np.uint32)
    place_activities_test_data[:3] = Activity.Home.value

    snapshot.buffers.people_baseline_flows[:] = people_flows_test_data.flatten()
    snapshot.buffers.people_flows[:] = people_flows_test_data.flatten()
    snapshot.buffers.people_place_ids[:] = people_place_ids_test_data.flatten()
    snapshot.buffers.people_statuses[:] = people_statuses_test_data
    snapshot.buffers.place_activities[:] = place_activities_test_data

    simulator = Simulator(snapshot)
    simulator.upload_all(snapshot.buffers)

    people_flows_before = np.zeros(npeople*nslots, dtype=np.float32)
    simulator.download("people_flows", people_flows_before)

    expected_people_flows_before = people_flows_test_data.flatten()
    assert np.array_equal(people_flows_before, expected_people_flows_before)

    simulator.step_kernel("people_update_flows")

    people_flows_after = np.zeros(npeople*nslots, dtype=np.float32)
    simulator.download("people_flows", people_flows_after)

    expected_people_flows_after = people_flows_test_data
    expected_people_flows_after[symptomatic_person_id][0:4] = [0.85, 0.09, 0.045, 0.015]
    expected_people_flows_after = expected_people_flows_after.flatten()

    assert not np.array_equal(people_flows_before, people_flows_after)
    assert np.array_equal(expected_people_flows_after, people_flows_after)


def test_correct_flow_calculation_with_lockdown():
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    people_place_ids_test_data = np.full((npeople, nslots), sentinel_value, dtype=np.uint32)
    people_place_ids_test_data[0][0:4] = [0, 5, 7, 3]
    people_place_ids_test_data[1][0:4] = [1, 5, 6, 4]
    people_place_ids_test_data[2][0:4] = [2, 3, 7, 6]

    people_flows_test_data = np.zeros((npeople, nslots), dtype=np.float32)
    people_flows_test_data[0][0:4] = [0.8, 0.1, 0.06, 0.04]
    people_flows_test_data[1][0:4] = [0.7, 0.18, 0.09, 0.03]
    people_flows_test_data[2][0:4] = [0.6, 0.2, 0.16, 0.04]

    people_statuses_test_data = np.full(npeople, DiseaseStatus.Susceptible.value, dtype=np.uint32)
    symptomatic_person_id = 1
    people_statuses_test_data[symptomatic_person_id] = np.uint32(DiseaseStatus.Symptomatic.value)

    place_activities_test_data = np.full(nplaces, Activity.Retail.value, dtype=np.uint32)
    place_activities_test_data[:3] = Activity.Home.value

    snapshot.buffers.people_baseline_flows[:] = people_flows_test_data.flatten()
    snapshot.buffers.people_flows[:] = people_flows_test_data.flatten()
    snapshot.buffers.people_place_ids[:] = people_place_ids_test_data.flatten()
    snapshot.buffers.people_statuses[:] = people_statuses_test_data
    snapshot.buffers.place_activities[:] = place_activities_test_data

    params = Params()
    params.set_lockdown_multiplier(snapshot.lockdown_multipliers, 0)
    snapshot.buffers.params[:] = params.asarray()

    simulator = Simulator(snapshot)
    simulator.upload_all(snapshot.buffers)

    people_flows_before = np.zeros(npeople*nslots, dtype=np.float32)
    simulator.download("people_flows", people_flows_before)

    expected_people_flows_before = people_flows_test_data.flatten()
    assert np.array_equal(people_flows_before, expected_people_flows_before)

    simulator.step_kernel("people_update_flows")

    people_flows_after = np.zeros(npeople*nslots, dtype=np.float32)
    simulator.download("people_flows", people_flows_after)

    assert not np.array_equal(people_flows_before, people_flows_after)

    # assert correct flows for symptomatic person
    expected_symptomatic_flows_after = np.array([0.85, 0.09, 0.045, 0.015])
    symptomatic_person_start_idx = symptomatic_person_id * nslots
    symptomatic_flows_after = people_flows_after[symptomatic_person_start_idx:symptomatic_person_start_idx + 4]

    assert np.allclose(expected_symptomatic_flows_after, symptomatic_flows_after)

    # assert correct flows for person who is not symptomatic
    # adjustments calculated using first lockdown multiplier (approx. 0.930517)
    expected_non_symptomatic_flows_after = np.array([0.6277932, 0.1861034, 0.14888272, 0.03722068])
    non_symptomatic_person_id = 2
    person_start_idx = non_symptomatic_person_id * nslots
    non_symptomatic_flows_after = people_flows_after[person_start_idx:person_start_idx + 4]

    assert np.allclose(expected_non_symptomatic_flows_after, non_symptomatic_flows_after)

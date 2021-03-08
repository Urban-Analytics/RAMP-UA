import numpy as np
import os

from microsim.opencl.ramp.activity import Activity
from microsim.opencl.ramp.params import Params
from microsim.opencl.ramp.simulator import Simulator
from microsim.opencl.ramp.snapshot import Snapshot
from microsim.opencl.ramp.disease_statuses import DiseaseStatus
from microsim.population_initialisation import PopulationInitialisation
from quant_api import QuantRampAPI

sentinel_value = (1 << 31) - 1

nplaces = 8
npeople = 3
nslots = 8

test_dir = "tests/"

# arguments used when calling the PopulationInitialisation constructor. Usually these are the same
population_init_args = {"data_dir": os.path.join(test_dir, "dummy_data"),
                        "testing": True, "debug": True,
                        "quant_object": QuantRampAPI(os.path.join("devon_data", "QUANT_RAMP"))
                        }

population_init = PopulationInitialisation(**population_init_args)
time_activity_multiplier = PopulationInitialisation.read_time_activity_multiplier(
    os.path.join(population_init.DATA_DIR, "google_mobility_lockdown_daily.csv"))
lockdown_multipliers = time_activity_multiplier.loc[:, "timeout_multiplier"]


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
    
    params = Params()
    symptomatic_multiplier = 0.5
    params.symptomatic_multiplier = symptomatic_multiplier
    snapshot.update_params(params)

    simulator = Simulator(snapshot, gpu=False)
    simulator.upload_all(snapshot.buffers)

    people_flows_before = np.zeros(npeople*nslots, dtype=np.float32)
    simulator.download("people_flows", people_flows_before)

    expected_people_flows_before = people_flows_test_data.flatten()
    assert np.array_equal(people_flows_before, expected_people_flows_before)

    simulator.step_kernel("people_update_flows")

    people_flows_after = np.zeros(npeople*nslots, dtype=np.float32)
    simulator.download("people_flows", people_flows_after)

    # adjust symptomatic persons flows according to symptomatic multiplier
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
    params.set_lockdown_multiplier(lockdown_multipliers, 0)
    params.symptomatic_multiplier = 0.5
    snapshot.buffers.params[:] = params.asarray()

    simulator = Simulator(snapshot, gpu=False)
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
    # adjustments calculated using first lockdown multiplier (approx. 0.9084687)
    expected_non_symptomatic_flows_after = np.array([0.63661252, 0.18169374, 0.145354992, 0.036338748])
    non_symptomatic_person_id = 2
    person_start_idx = non_symptomatic_person_id * nslots
    non_symptomatic_flows_after = people_flows_after[person_start_idx:person_start_idx + 4]

    assert np.allclose(expected_non_symptomatic_flows_after, non_symptomatic_flows_after)

import numpy as np
from microsim.opencl.ramp.params import Params
from microsim.opencl.ramp.simulator import Simulator
from microsim.opencl.ramp.snapshot import Snapshot
from microsim.opencl.ramp.disease_statuses import DiseaseStatus

sentinel_value = (1 << 31) - 1
fixed_factor = 8388608.0

nplaces = 8
npeople = 3
nslots = 8


def test_correct_send_hazard():
    # Set up and upload the test data
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    people_place_ids = np.full((npeople, nslots), sentinel_value, dtype=np.uint32)
    people_place_ids[0][0:4] = [0, 5, 7, 3]
    people_place_ids[1][0:4] = [1, 5, 6, 4]
    people_place_ids[2][0:4] = [2, 3, 7, 6]

    people_flows = np.zeros((npeople, nslots), dtype=np.float32)
    people_flows[0][0:4] = [0.8, 0.1, 0.06, 0.04]
    people_flows[1][0:4] = [0.7, 0.18, 0.09, 0.03]
    people_flows[2][0:4] = [0.6, 0.2, 0.16, 0.04]

    people_statuses = np.full(npeople, DiseaseStatus.Symptomatic.value, dtype=np.uint32)

    snapshot.buffers.people_flows[:] = people_flows.flatten()
    snapshot.buffers.people_place_ids[:] = people_place_ids.flatten()
    snapshot.buffers.people_statuses[:] = people_statuses
    snapshot.buffers.place_activities[:] = np.random.randint(4, size=nplaces, dtype=np.uint32)

    params = Params()
    params.place_hazard_multipliers = np.ones(5, dtype=np.float32)
    snapshot.update_params(params)

    simulator = Simulator(snapshot, gpu=False)
    simulator.upload_all(snapshot.buffers)

    # Run the kernel
    simulator.step_kernel("people_send_hazards")

    # Download the result
    place_hazards = np.zeros(nplaces, dtype=np.uint32)
    simulator.download("place_hazards", place_hazards)
    place_counts = np.zeros(nplaces, dtype=np.uint32)
    simulator.download("place_counts", place_counts)

    # Assert expected results
    expected_place_hazards_floats = np.array([0.8, 0.7, 0.6, 0.24, 0.03, 0.28, 0.13, 0.22], dtype=np.float32)
    expected_place_hazards = (fixed_factor * expected_place_hazards_floats).astype(np.uint32)
    expected_place_counts = np.array([1, 1, 1, 2, 1, 2, 2, 2], dtype=np.uint32)

    assert np.allclose(expected_place_hazards, place_hazards)
    assert np.array_equal(expected_place_counts, place_counts)


def test_asymptomatics_add_less_hazard():
    # Set up and upload the test data
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    people_place_ids = np.full((npeople, nslots), sentinel_value, dtype=np.uint32)
    people_place_ids[0][0:4] = [0, 5, 7, 3]
    people_place_ids[1][0:4] = [1, 5, 6, 4]
    people_place_ids[2][0:4] = [2, 3, 7, 6]

    people_flows = np.zeros((npeople, nslots), dtype=np.float32)
    people_flows[0][0:4] = [0.8, 0.1, 0.06, 0.04]
    people_flows[1][0:4] = [0.7, 0.18, 0.09, 0.03]
    people_flows[2][0:4] = [0.6, 0.2, 0.16, 0.04]

    people_statuses_symptomatic = np.full(npeople, DiseaseStatus.Symptomatic.value, dtype=np.uint32)

    snapshot.buffers.people_flows[:] = people_flows.flatten()
    snapshot.buffers.people_place_ids[:] = people_place_ids.flatten()
    snapshot.buffers.people_statuses[:] = people_statuses_symptomatic
    snapshot.buffers.place_activities[:] = np.random.randint(4, size=nplaces, dtype=np.uint32)

    params = Params()
    params.place_hazard_multipliers = np.ones(5, dtype=np.float32)
    asymptomatic_multiplier = 0.75
    params.individual_hazard_multipliers = np.array([1.0, asymptomatic_multiplier, 1.0])
    snapshot.update_params(params)

    simulator = Simulator(snapshot, gpu=False)
    simulator.upload_all(snapshot.buffers)

    # Run the kernel
    simulator.step_kernel("people_send_hazards")

    # Download the result
    place_hazards_symptomatic = np.zeros(nplaces, dtype=np.uint32)
    simulator.download("place_hazards", place_hazards_symptomatic)
    place_counts = np.zeros(nplaces, dtype=np.uint32)
    simulator.download("place_counts", place_counts)

    # Assert expected results
    expected_place_hazards_floats_symptomatic = np.array([0.8, 0.7, 0.6, 0.24, 0.03, 0.28, 0.13, 0.22], dtype=np.float32)
    expected_place_hazards_symptomatic = (fixed_factor * expected_place_hazards_floats_symptomatic).astype(np.uint32)

    assert np.allclose(expected_place_hazards_symptomatic, place_hazards_symptomatic)

    # Run for asymptomatic people
    people_statuses_asymptomatic = np.full(npeople, DiseaseStatus.Asymptomatic.value, dtype=np.uint32)
    simulator.upload("people_statuses", people_statuses_asymptomatic)

    # reset place hazards to zero
    simulator.step_kernel("places_reset")

    # run kernel with asymptomatic population
    simulator.step_kernel("people_send_hazards")

    # Download the result
    place_hazards_asymptomatic = np.zeros(nplaces, dtype=np.uint32)
    simulator.download("place_hazards", place_hazards_asymptomatic)

    # Assert expected results
    expected_place_hazards_floats_asymptomatic = expected_place_hazards_floats_symptomatic * asymptomatic_multiplier
    expected_place_hazards_asymptomatic = (fixed_factor * expected_place_hazards_floats_asymptomatic).astype(np.uint32)

    assert np.allclose(expected_place_hazards_asymptomatic, place_hazards_asymptomatic)

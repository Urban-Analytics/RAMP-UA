import numpy as np
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

    place_hazards_floats = np.array([0.2, 0.0, 0.5, 0.03, 0.0123, 0.22, 0.0001, 0.73], dtype=np.float32)
    place_hazards = (fixed_factor * place_hazards_floats).astype(np.uint32)

    people_place_ids = np.full((npeople, nslots), sentinel_value, dtype=np.uint32)
    people_place_ids[0][0:4] = [1, 6, 0, 4]
    people_place_ids[1][0:4] = [2, 6, 7, 5]
    people_place_ids[2][0:4] = [3, 4, 0, 7]

    people_flows = np.zeros((npeople, nslots), dtype=np.float32)
    people_flows[0][0:4] = [0.6, 0.2, 0.16, 0.04]
    people_flows[1][0:4] = [0.7, 0.18, 0.09, 0.03]
    people_flows[2][0:4] = [0.8, 0.1, 0.06, 0.04]

    people_statuses = np.full(npeople, DiseaseStatus.Susceptible.value, dtype=np.uint32)
    people_statuses[0] = DiseaseStatus.Symptomatic.value

    people_hazards = np.zeros(npeople, dtype=np.float32)

    snapshot.buffers.people_flows[:] = people_flows.flatten()
    snapshot.buffers.people_place_ids[:] = people_place_ids.flatten()
    snapshot.buffers.people_statuses[:] = people_statuses
    snapshot.buffers.place_hazards[:] = place_hazards
    snapshot.buffers.people_hazards[:] = people_hazards

    simulator = Simulator(snapshot)
    simulator.upload_all(snapshot.buffers)

    # Run the kernel
    simulator.step_kernel("people_recv_hazards")

    # Download the result
    people_hazards = np.zeros(npeople, dtype=np.float32)
    simulator.download("people_hazards", people_hazards)

    # Assert expected results
    expected_people_hazards = np.array([0.0, 0.422318, 0.06643], dtype=np.float32)

    assert np.allclose(expected_people_hazards, people_hazards)

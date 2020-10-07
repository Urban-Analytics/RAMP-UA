import numpy as np
from microsim.opencl.ramp.simulator import Simulator
from microsim.opencl.ramp.snapshot import Snapshot

nplaces = 5
npeople = 100
nslots = 8


def test_places_are_reset_to_zero():
    snapshot = Snapshot.random(nplaces, npeople, nslots)

    hazards_test_data = np.array([4, 3, 5, 1, 6], dtype=np.uint32)
    counts_test_data = np.array([1, 4, 8, 10, 2], dtype=np.uint32)

    snapshot.buffers.place_hazards[:] = hazards_test_data
    snapshot.buffers.place_counts[:] = counts_test_data

    simulator = Simulator(snapshot, gpu=True)
    simulator.upload_all(snapshot.buffers)

    place_hazards_before = np.zeros(nplaces, dtype=np.uint32)
    simulator.download("place_hazards", place_hazards_before)

    place_counts_before = np.zeros(nplaces, dtype=np.uint32)
    simulator.download("place_counts", place_counts_before)

    assert np.array_equal(place_hazards_before, hazards_test_data)
    assert np.array_equal(place_counts_before, counts_test_data)

    simulator.step_kernel("places_reset")

    # initialise host buffers randomly so we know they are filled to zeros
    place_hazards_after = np.random.randint(0, 11, nplaces, dtype=np.uint32)
    simulator.download("place_hazards", place_hazards_after)

    place_counts_after = np.random.randint(0, 11, nplaces, dtype=np.uint32)
    simulator.download("place_counts", place_counts_after)

    assert np.array_equal(place_hazards_after, np.zeros(nplaces, dtype=np.uint32))
    assert np.array_equal(place_counts_after, np.zeros(nplaces, dtype=np.uint32))

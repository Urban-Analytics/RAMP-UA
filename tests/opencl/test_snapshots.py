from microsim.opencl.ramp.snapshot import Snapshot
import numpy as np
import os

sentinel_value = (1 << 31) - 1


def test_generate_zeros_snapshot():
    nplaces = 10
    npeople = 100
    nslots = 16
    snapshot = Snapshot.zeros(nplaces=nplaces, npeople=npeople, nslots=nslots)

    assert snapshot.buffers.place_activities.shape[0] == nplaces
    assert snapshot.buffers.people_ages.shape[0] == npeople
    assert snapshot.buffers.people_place_ids.shape[0] == npeople * nslots
    assert snapshot.buffers.place_coords.shape[0] == nplaces * 2

    # check all zeros
    assert not snapshot.buffers.place_activities.any()
    assert not snapshot.buffers.people_ages.any()
    assert not snapshot.buffers.people_place_ids.any()
    assert not snapshot.buffers.place_coords.any()


def test_save_and_load_full_snapshot():
    nplaces = 10
    npeople = 100
    nslots = 16
    generated_snapshot = Snapshot.random(nplaces=nplaces, npeople=npeople, nslots=nslots)

    snapshot_path = "tests/test_snapshots/random.npz"

    generated_snapshot.save(snapshot_path)

    loaded_snapshot = Snapshot.load_full_snapshot(snapshot_path)

    os.remove(snapshot_path)

    assert np.array_equal(generated_snapshot.buffers.people_ages, loaded_snapshot.buffers.people_ages)
    assert np.all(np.isclose(generated_snapshot.buffers.place_coords, loaded_snapshot.buffers.place_coords))

    assert loaded_snapshot.nplaces == nplaces
    assert loaded_snapshot.npeople == npeople
    assert loaded_snapshot.nslots == nslots


def test_load_initial_snapshot():
    # Load initial snapshot generated from the test data in the RAMP-UA repo
    loaded_snapshot = Snapshot.load_initial_snapshot(people_filepath="tests/test_snapshots/people.npz",
                                                     places_filepath="tests/test_snapshots/places.npz")

    expected_nplaces = 8
    expected_npeople = 3
    expected_nslots = 16

    assert loaded_snapshot.nplaces == expected_nplaces
    assert loaded_snapshot.npeople == expected_npeople
    assert loaded_snapshot.nslots == expected_nslots

    expected_people_place_ids = np.full((expected_npeople, expected_nslots), sentinel_value)
    expected_people_place_ids[0][0:4] = [0, 5, 7, 3]
    expected_people_place_ids[1][0:4] = [1, 5, 6, 4]
    expected_people_place_ids[2][0:4] = [2, 3, 7, 6]
    expected_people_place_ids = expected_people_place_ids.flatten()

    expected_people_flows = np.zeros((expected_npeople, expected_nslots))
    expected_people_flows[0][0:4] = [0.8, 0.1, 0.06, 0.04]
    expected_people_flows[1][0:4] = [0.7, 0.18, 0.09, 0.03]
    expected_people_flows[2][0:4] = [0.6, 0.2, 0.16, 0.04]
    expected_people_flows = expected_people_flows.flatten()

    assert np.array_equal(expected_people_place_ids, loaded_snapshot.buffers.people_place_ids)
    assert np.all(np.isclose(expected_people_flows, loaded_snapshot.buffers.people_baseline_flows))


def test_seed_initial_infections():
    # Load initial snapshot generated from the test data in the RAMP-UA repo
    snapshot = Snapshot.load_initial_snapshot(people_filepath="tests/test_snapshots/people.npz",
                                              places_filepath="tests/test_snapshots/places.npz")

    # assert that no people are infected before seeding
    assert not snapshot.buffers.people_statuses.any()

    snapshot.seed_initial_infections(num_seed_days=1)

    expected_num_infections = 1  # since there is 1 person in a high risk area with not_home_prob > 0.3

    num_people_infected = np.count_nonzero(snapshot.buffers.people_statuses)

    assert num_people_infected == expected_num_infections


def test_seed_prngs():
    snapshot = Snapshot.random(nplaces=50, npeople=100, nslots=5)
    prngs_before = np.zeros(400, dtype=np.uint32)
    prngs_before[:] = snapshot.buffers.people_prngs

    snapshot.seed_prngs(46)
    prngs_after = np.zeros(400, dtype=np.uint32)
    prngs_after[:] = snapshot.buffers.people_prngs

    snapshot.seed_prngs(46)
    prngs_after_after = np.zeros(400, dtype=np.uint32)
    prngs_after_after[:] = snapshot.buffers.people_prngs

    assert np.any(prngs_before != prngs_after)
    assert np.all(prngs_after == prngs_after_after)

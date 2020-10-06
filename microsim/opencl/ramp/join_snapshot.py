from ramp.snapshot import Snapshot


def main():
    """
    Combine two "partial" snapshots to create a full snapshot which can be used to initialise the simulation state
    """

    snapshot = Snapshot.load_initial_snapshot("snapshots/partial/people.npz", "snapshots/partial/places.npz")
    snapshot.seed_initial_infections()
    snapshot.save("snapshots/default.npz")


if __name__ == "__main__":
    main()

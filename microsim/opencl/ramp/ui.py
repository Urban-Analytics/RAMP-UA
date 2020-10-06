import argparse

from microsim.opencl.ramp.inspector import Inspector
from microsim.opencl.ramp.simulator import Simulator
from microsim.opencl.ramp.snapshot import Snapshot


def main():
    """
    Entry point for running the OpenCL simulation with the UI.
    This will run until the UI window is closed by the user.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Whether to run on the GPU", action='store_const', const=True, default=False)
    parsed_args = parser.parse_args()
    gpu = parsed_args.gpu

    width = 2560  # Initial window width in pixels
    height = 1440  # Initial window height in pixels
    nlines = 4  # Number of visualised connections per person

    # Load initial simulation state from snapshot
    snapshot = Snapshot.load_full_snapshot("snapshots/devon.npz")
    snapshot.sanitize_coords()

    # Create a simulator and upload the snapshot data to the OpenCL device
    simulator = Simulator(snapshot, gpu)
    simulator.upload_all(snapshot.buffers)

    # Create an inspector and upload static data
    inspector = Inspector(simulator, snapshot, nlines, "Ramp UA", width, height)

    # Main UI loop
    while inspector.is_active():
        inspector.update()


if __name__ == "__main__":
    main()

import argparse
import pickle
from tqdm import tqdm
import pandas as pd

from microsim.opencl.ramp.params import Params
from microsim.opencl.ramp.simulator import Simulator
from microsim.opencl.ramp.snapshot import Snapshot
from microsim.opencl.ramp.summary import Summary
from microsim.opencl.ramp.disease_statuses import DiseaseStatus


def main():
    """
    Entry point for running the OpenCL simulation in "headless" mode, ie. without the UI.
    The results will be stored in a "Summary" object.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", help="Number of timesteps to run", type=int, default=10)
    parser.add_argument("--gpu", help="Whether to run on the GPU", action='store_const', const=True, default=False)
    parser.add_argument("--quiet", help="Suppress printouts", action='store_const', const=True, default=False)
    parser.add_argument("--no-detailed-counts", help="Don't store detailed counts in summary", action='store_const',
                        const=True, default=False)
    parsed_args = parser.parse_args()
    steps = parsed_args.steps
    gpu = parsed_args.gpu
    quiet = parsed_args.quiet
    store_detailed_counts = not parsed_args.no_detailed_counts

    snapshot = Snapshot.load_full_snapshot("snapshots/devon.npz")
    if not quiet:
        print(f"\nSnapshot Size:\t{int(snapshot.num_bytes() / 1000000)} MB\n")

    simulator = Simulator(snapshot, gpu)
    if not quiet:
        print(f"Platform:\t{simulator.platform_name()}\nDevice:\t\t{simulator.device_name()}\n")

    params = Params()
    summary = Summary(snapshot, store_detailed_counts=store_detailed_counts, max_time=steps)

    # Upload the snapshot to OpenCL
    simulator.upload_all(snapshot.buffers)

    for time in tqdm(range(steps), desc="Running simulation"):
        # Update parameters based on lockdown
        params.set_lockdown_multiplier(snapshot.lockdown_multipliers, time)
        simulator.upload("params", params.asarray())

        # Step the simulator
        simulator.step()

        # Update the statuses
        simulator.download("people_statuses", snapshot.buffers.people_statuses)
        summary.update(time, snapshot.buffers.people_statuses)

    if not quiet:
        for i in range(steps):
            print(f"\nDay {i}")
            summary.print_counts(i)

    # Download the snapshot from OpenCL to host memory
    simulator.download_all(snapshot.buffers)
    if not quiet:
        print("\nFinished")

    store_summary_data(summary, store_detailed_counts)


def store_summary_data(summary, store_detailed_counts):
    data_dir = "data/output/"

    # convert total_counts to dict of pandas dataseries
    total_counts_dict = {}
    for status, timeseries in enumerate(summary.total_counts):
        total_counts_dict[DiseaseStatus(status).name.lower()] = pd.Series(timeseries)

    with open(data_dir + "total_counts.p", "wb") as f:
        pickle.dump(total_counts_dict, f)

    if store_detailed_counts:
        # turn 2D arrays into dataframes for ages and areas
        columns = [f"Day{i}" for i in range(summary.max_time)]

        age_counts_dict = {}
        for status, age_count_array in summary.age_counts.items():
            age_counts_dict[status] = pd.DataFrame.from_records(age_count_array, columns=columns)

        area_counts_dict = {}
        for status, area_count_array in summary.area_counts.items():
            area_counts_dict[status] = pd.DataFrame.from_records(area_count_array, columns=columns,
                                                                 index=summary.unique_area_codes)

        # Store pickled summary objects
        with open(data_dir + "age_counts.p", "wb") as f:
            pickle.dump(age_counts_dict, f)
        with open(data_dir + "area_counts.p", "wb") as f:
            pickle.dump(area_counts_dict, f)


if __name__ == "__main__":
    main()

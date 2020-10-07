import pickle
from tqdm import tqdm
import pandas as pd
import os

from microsim.opencl.ramp.inspector import Inspector
from microsim.opencl.ramp.params import Params
from microsim.opencl.ramp.simulator import Simulator
from microsim.opencl.ramp.summary import Summary
from microsim.opencl.ramp.disease_statuses import DiseaseStatus


def run_opencl(snapshot, iterations=100, data_dir="./data", use_gui=True, use_gpu=False, quiet=False):
    """
    Entry point for running the OpenCL simulation either with the UI or headless
    """

    if not quiet:
        print(f"\nSnapshot Size:\t{int(snapshot.num_bytes() / 1000000)} MB\n")

    simulator = Simulator(snapshot, use_gpu)
    if not quiet:
        print(f"Platform:\t{simulator.platform_name()}\nDevice:\t\t{simulator.device_name()}\n")

    # Create a simulator and upload the snapshot data to the OpenCL device
    simulator = Simulator(snapshot, use_gpu)
    simulator.upload_all(snapshot.buffers)

    if use_gui:
        run_with_gui(simulator, snapshot)
    else:
        run_headless(simulator, snapshot, iterations, quiet, data_dir)


def run_with_gui(simulator, snapshot):
    width = 2560  # Initial window width in pixels
    height = 1440  # Initial window height in pixels
    nlines = 4  # Number of visualised connections per person
    # Create an inspector and upload static data
    inspector = Inspector(simulator, snapshot, nlines, "Ramp UA", width, height)

    # Main UI loop
    while inspector.is_active():
        inspector.update()


def run_headless(simulator, snapshot, iterations, quiet, data_dir):
    """Run the simulation in headless mode and store summary data.
    NB: running in this mode is required in order to view output data in the dashboard"""
    params = Params()
    summary = Summary(snapshot, store_detailed_counts=True, max_time=iterations)
    for time in tqdm(range(iterations), desc="Running simulation"):
        # Update parameters based on lockdown
        params.set_lockdown_multiplier(snapshot.lockdown_multipliers, time)
        simulator.upload("params", params.asarray())

        # Step the simulator
        simulator.step()

        # Update the statuses
        simulator.download("people_statuses", snapshot.buffers.people_statuses)
        summary.update(time, snapshot.buffers.people_statuses)

    if not quiet:
        for i in range(iterations):
            print(f"\nDay {i}")
            summary.print_counts(i)

    # Download the snapshot from OpenCL to host memory
    simulator.download_all(snapshot.buffers)
    if not quiet:
        print("\nFinished")

    store_summary_data(summary, store_detailed_counts=True, data_dir=data_dir)


def store_summary_data(summary, store_detailed_counts, data_dir):
    # convert total_counts to dict of pandas dataseries
    total_counts_dict = {}
    for status, timeseries in enumerate(summary.total_counts):
        total_counts_dict[DiseaseStatus(status).name.lower()] = pd.Series(timeseries)

    output_dir = data_dir + "/output/OpenCL/"

    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + "total_counts.pkl", "wb") as f:
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
        with open(output_dir + "age_counts.pkl", "wb") as f:
            pickle.dump(age_counts_dict, f)
        with open(output_dir + "area_counts.pkl", "wb") as f:
            pickle.dump(area_counts_dict, f)

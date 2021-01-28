# Contains some useful utility functionality
import os
import requests
import tarfile
import pandas as pd
from typing import List

from microsim.column_names import ColumnNames

class Optimise:
    """
    Functions to optimise the memory use of pandas dataframes.
    From https://medium.com/bigdatarepublic/advanced-pandas-optimize-speed-and-memory-a654b53be6c2
    """

    @staticmethod
    def optimize(df: pd.DataFrame, datetime_features: List[str] = []):
        return Optimise._optimize_floats(Optimise._optimize_ints(Optimise._optimize_objects(df, datetime_features)))


    @staticmethod
    def _optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
        floats = df.select_dtypes(include=['float64']).columns.tolist()
        df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
        return df


    @staticmethod
    def _optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
        ints = df.select_dtypes(include=['int64']).columns.tolist()
        df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
        return df


    @staticmethod
    def _optimize_objects(df: pd.DataFrame, datetime_features: List[str]) -> pd.DataFrame:
        for col in df.select_dtypes(include=['object']):
            if col not in datetime_features:
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if float(num_unique_values) / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
            else:
                df[col] = pd.to_datetime(df[col])
        return df


def check_durations_sum_to_1(individuals, activities):
    total_duration = [0.0] * len(individuals)  # Add up all the different activity durations
    for activity in activities:
        total_duration = total_duration + individuals.loc[:, f"{activity}{ColumnNames.ACTIVITY_DURATION}"]
    if not (total_duration.apply(lambda x: round(x, 5)) == 1.0).all():
        print("Some activity durations don't sum to 1", flush=True)
        print(total_duration[total_duration != 1.0], flush=True)
        raise Exception("Some activity durations don't sum to 1")


# data fetching functions

def download_data(url : str):
    """Download data utility function

    Args:
        url (str, optional): A url to an archive file. Defaults to "https://ramp0storage.blob.core.windows.net/rampdata/devon_data.tar.gz".
    """
    response = requests.get(url, stream=True)

    # specify target_path as name of tarfile downloaded by splitting url 
    # and retrieving last item
    target_path = os.path.join(url.split('/')[-1])
    
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())

    return target_path

def unpack_data(archive : str):
    """unpack tar data archive

    Args:
        archive (str): A string directory path to archive file using
    """

    tar_file = tarfile.open(archive)

    tar_file.extractall(".")

def data_setup(archive : str, url : str  = "https://ramp0storage.blob.core.windows.net/rampdata/devon_data.tar.gz"):
    """A wrapper function for downloading and unpacking Azure stored devon_data

    Args:
        archive (str): A string directory path to archive file using
        url (str, optional): A url to an archive file. Defaults to "https://ramp0storage.blob.core.windows.net/rampdata/devon_data.tar.gz".
    """
    download_data(url = url)

    unpack_data(archive = archive)

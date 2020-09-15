# Contains some useful utility functionality
import pandas as pd
from typing import List

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




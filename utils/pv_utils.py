import os
import pandas as pd
import logging


def read_files(path: str):
    """
    Reads in hdf5 file and two metadata csvs from local given a path. The csvs are returned as dataframes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError('file path does not exist')

    file_info = {}

    for file in os.listdir(path):

        if file.endswith('.hdf'):
            file_info['hdf_path'] = path + '/' + file

        elif file.endswith('.csv'):
            if 'metadata' in file:
                file_info['metadata'] = pd.read_csv(path + '/' + file)
            else:
                file_info['pv_systems'] = pd.read_csv(path + '/' + file)

    return file_info

def read_hdf(hdf_path, key):

    # with pd.HDFStore(hdf_path) as hdf:
    #     keys = hdf.keys()
    #
    # for counter, key in enumerate(keys):
    #     logging.info(f"Reading {key}: {counter} out of {len(keys)}")

        # if counter == 0:
    df = pd.read_hdf(hdf_path, key)
    df = df.resample("5T").mean()
    print(df.head())
    print(df.tail())


def process_pv_data():
    """ some preprocessing"""
    # if len(df) > 288 * 10:  # more than 10 days of data
    #
    #     print(len(df))
    #
    #     name = key.split("/")[-1]
    #
    #     # resample to 15 minutes
    #     df = df.resample("15T").mean()
    #
    #     # take difference so its ~power not cumulative
    #     if "instantaneous_power_gen_W" in df.columns:
    #         df.rename(columns={"instantaneous_power_gen_W": name}, inplace=True)
    #     elif "cumulative_energy_gen_Wh" in df.columns:
    #         df.rename(columns={"cumulative_energy_gen_Wh": name}, inplace=True)
    #         df = df.diff()
    #         df[df < 0] = 0
    #     elif "energy_gen_Wh" in df.columns:
    #         df.rename(columns={"energy_gen_Wh": name}, inplace=True)
    #     else:
    #         raise Exception('Data does not contain "cumulative_energy_gen_Wh" or "energy_gen_Wh"')
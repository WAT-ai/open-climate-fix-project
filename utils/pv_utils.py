import os
import pandas as pd
import logging


def read_files(path: str):
    """
    Reads in hdf5 file and two metadata csvs from local given a path. The csvs are returned as dataframes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError('file path does not exist')

    file_info = ['', None, None]

    for file in os.listdir(path):

        if file.endswith('.hdf'):
            file_info[0] = path + '/' + file

        elif file.endswith('.csv'):
            if 'metadata' in file:
                file_info[1] = pd.read_csv(path + '/' + file, index_col=1)
            else:
                file_info[2] = pd.read_csv(path + '/' + file, index_col=0)

    return file_info


def open_hdf(path, key):
    """
    :param path:
    :param key:
    :return:
    """

    # with pd.HDFStore(hdf_path) as hdf:
    #     keys = hdf.keys()
    #
    # for counter, key in enumerate(keys):
    #     logging.info(f"Reading {key}: {counter} out of {len(keys)}")

    # if counter == 0:
    return pd.read_hdf(path, key)

    print(df.head())
    print(df.tail())


def process_pv_data():
    """ some preprocessing"""

    df = df.resample("5T").mean()
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

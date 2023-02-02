import os

import zipfile
import xarray as xr
import imagecodecs
import jpeg_xl_float_with_nans
from datetime import datetime
from dateutil.relativedelta import *
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

START_DATE = datetime.strptime("January 1, 2022", "%B %d, %Y")
END_DATE = datetime.strptime("January 1, 2022", "%B %d, %Y")
FEATURES = []
INTERVAL = relativedelta(days=1)

##
# 1. Download from HuggingFace
# 2. Unzip data
# 3. Preprocess
# 4. Upload

if __name__ == '__main__':
    date = START_DATE
    filename = f"data/{'YEAR'}/hrv/{'YEAR'}_000000-of-000056.zarr.zip"

    while date <= END_DATE:
        # 1. Download Data
        partition = filename.replace('YEAR', str(date.year)) \
                            .replace('MONTH', str(date.month).zfill(2)) \
                            .replace('DAY', str(date.day).zfill(2))
        print(f"Downloading {partition}...")

        path = hf_hub_download(repo_id="openclimatefix/eumetsat-rss",
                               filename=partition,
                               repo_type="dataset",
                               token='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')

        # 2. Unzip Data
        data_path = f"./gcp-data/{partition[-30:-4]}"
        print(f"Unzipping to {data_path}")
        # with zipfile.ZipFile(path, "r") as zip_ref:
        #     zip_ref.extractall(data_path)

        # 3. Preprocess data
        print("Preprocessing...")
        dataset = xr.open_dataset(data_path, engine='zarr', chunks='auto')

        time_slice = dataset.indexes['time'][0]
        fig = plt.figure()
        im = plt.imshow(dataset["data"].sel(time=time_slice))

        date += INTERVAL
        print(date)
        break

    print("Done")


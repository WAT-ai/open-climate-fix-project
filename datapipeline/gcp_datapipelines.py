import json

from os import getcwd

from datetime import datetime
from dateutil.relativedelta import *

import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import xarray as xr

# For each chunk
# 1. Download from HuggingFace
# 2. Unzip data
# 3. Preprocess
# 4. Upload to GCP
# 5. Clean up


class GCPPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def unzip(source: str, dest: str) -> None:
        """
        Unzips all files from the source dir and extracts them to the dest dir

        Args:
            source: source file path
            dest: destination file path
        
        Returns:
            None
        """
        pass

    def gcp_upload(source: str, dest: str) -> None:
        """
        Upload data from the source dir to a GCP bucket at dest

        Args:
            source: source file path
            dest: GCP bucket file path
        
        Returns:
            None
        """
        pass

    def cleanup(filepath: str) -> None:
        """
        Deletes all the contents in the directory at filepath
        """
        pass


class SatellitePipeline(GCPPipeline):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def download(self) -> str:
        print("Downloading Data")
        try:
            for chunk in range(self.config['chunk_count'] + 1):
                file = f"data/{self.config['year']}/hrv/{self.config['year']}_0000{str(chunk).zfill(2)}-of-0000{self.config['chunk_count']}.zarr.zip"
                path = hf_hub_download(
                    repo_id=self.config['hf_repo_id'],
                    filename=file,
                    repo_type=self.config['hf_repo_type'],
                    token=self.config['hf_token'],
                    cache_dir=getcwd()
                )
        except Exception as error:
            print(error)

        print("Download Complete")

        return 'THIS SHOULD BE THE PATH WHERE THE DATA WAS DOWNLOADED'

    def preprocess(self) -> None:
        """
        Preprocesses the data as desired
        This function should probably take instructions from the download_configurations JSON file about the crop range, date range and features to drop
        """
        pass

    def execute(self) -> None:
        self.download()
        self.unzip()
        self.prepocess()
        self.gcp_upload()
        self.cleanup()


class NWPPipeline(GCPPipeline):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def download(self, file_name: str) -> str:
        """
            Downloads data from "file_name" location from HugginFace and returns the location of the downloaded data

            Args:
                file_name: HuggingFace data location
            
            Returns:
                returns downloaded location of the data
        """
        

    def preprocess(self: str) -> None:
        """
            Preprocesses the data as desired
            This function should probably take instructions from the download_configurations JSON file about the crop range, date range and features to drop
        """
        pass

    def execute(self: str) -> None:
        """
            Runs the pipeline according to the configuration file

            Args:
                None
            
            Returns:
                None
        """
        START_DATE = datetime.strptime("January 1, 2022", "%B %d, %Y")
        END_DATE = datetime.strptime("January 1, 2022", "%B %d, %Y")
        FEATURES = []
        INTERVAL = relativedelta(days=1)

        date = START_DATE
        filename = f"data/{'YEAR'}/hrv/{'YEAR'}_000000-of-000056.zarr.zip"

        while date <= END_DATE:
            # 1. Download Data
            partition = filename.replace('YEAR', str(date.year)) \
                                .replace('MONTH', str(date.month).zfill(2)) \
                                .replace('DAY', str(date.day).zfill(2))
            print(f"Downloading {partition}...")

            path = hf_hub_download(
                repo_id="openclimatefix/eumetsat-rss",
                filename=partition,
                repo_type="dataset",
                token='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'
            )

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
        self.download()
        self.unzip()
        self.prepocess()
        self.gcp_upload()
        self.cleanup()

if __name__ == '__main__':
    config = json.load(open('datapipeline_config.json'))
    datapipeline = NWPPipeline(config=config)
    datapipeline.execute()


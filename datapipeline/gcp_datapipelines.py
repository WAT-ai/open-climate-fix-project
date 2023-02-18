import json
import zipfile
import shutil
import xarray as xr

from os import getcwd
from google.cloud import storage
from huggingface_hub import hf_hub_download
from datetime import date, datetime, timedelta

# For each chunk
# 1. Download from HuggingFace
# 2. Unzip data
# 3. Preprocess
# 4. Upload to GCP
# abstract methods
# hydra/gin + pydantic to read and validate configurations

class GCPPipeline:
    def __init__(self, config: str) -> None:
        """
        Intialization function for pipeline. Expects a path to JSON configuration file.

        Args:
            config: path to JSON config object
        """
        self.config = json.load(open(config))

    def unzip(self, source: str, dest: str) -> None:
        """
        Unzips all files from the source dir and extracts them to the dest dir

        Args:
            source: source file path
            dest: destination file path
        
        Returns:
            None
        """
        print(f'Unzipping {source}')
        print(f'Extracting to {dest}')
        with zipfile.ZipFile(source, 'r') as zip_ref:
            zip_ref.extractall(dest)

    def gcp_upload(self, source: str, blob_name: str) -> None:
        """
        Uploads source dir to the GCP bucket specified the configuration

        Args:
            source: source file path
        
        Returns:
            None
        """
        storage_client = storage.Client()
        bucket = storage_client.bucket(blob_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(source)
        print(f"File {source} uploaded to {blob_name}.")
        return None

    def teardown(self, filepath: str) -> None:
        """
        Deletes the directory at filepath and all of its contents

        Args:
            filepath: location of directory to delete
        """
        shutil.rmtree(filepath)


class NWPPipeline(GCPPipeline):
    def __init__(self, config: str) -> None:
        super().__init__(config)

    def download(self, filepath: str) -> str:
        """
        Downloads data from "filepath" location from HugginFace and 
        returns the location of the downloaded data. If the filepath 
        is not found, it logs the error in a log file.

        Args:
            filepath: HuggingFace data location
        
        Returns:
            returns filepath of downloaded data
        """
        print(f'\nDownloading: {filepath}...')
        try:
            download_path = hf_hub_download(
                repo_id=self.config['hf_repo_id'],
                filename=filepath,
                repo_type=self.config['hf_repo_type'],
                token=self.config['hf_token'],
                cache_dir=getcwd() + '/cache/downloaded'
            )
            return download_path
        except Exception as error:
            log_file = open(self.config['error_log_path'] + 'error_logs.txt', 'a')
            log_file.write(str(error))
            return error

    def preprocess(self, filepath: str) -> None:
        """
        Preprocesses the zarr file at filepath according to configuration parameters

        Selecting time slice (6 AM to 9 PM)
        Dropping features
        Selecting a region (longitudinal latitudinal coords)

        Args:
            filepath: location of zarr path
        """
        max_lat, min_lat = self.config['preprocess']['latitude']
        min_lon, max_lon = self.config['preprocess']['longitude']
        min_time, max_time = self.config['preprocess']['time_range']

        dataset = xr.open_dataset(filepath, engine='zarr', chunks='auto')
        dataset = dataset[self.config['preprocess']['features']]
        dataset = dataset.sel(latitude=slice(min_lat, max_lat),
                              longitude=slice(min_lon, max_lon),
                              time=slice(dataset['time'][int(min_time)], dataset['time'][int(max_time)]))

        # overwrites zarr file with post-processed data
        self.teardown(filepath)
        dataset.to_zarr('./data/hi.zarr')

    def format_date(self, date_str: str) -> date:
        """
        Takes a date in mm-dd-yyyy format and returns a date object

        Args:
            date_str: data in mm-dd-yyyy string format
        
        Return
            a date object
        """
        return datetime.strptime(date_str, '%m-%d-%Y').date()

    def execute(self) -> None:
        """
        Runs the NWP pipeline according to the configuration file
        """
        assert self.config['data_type'] == 'nwp', 'Configuration Error: Expects "nwp" data_type in configuration'

        START_DATE = self.format_date(self.config['start_date'])
        END_DATE = self.format_date(self.config['end_date'])
        assert START_DATE <= END_DATE, 'Configuration Error: start date must <= end date'

        TEMPLATE_PATH = f"data/surface/{'YEAR'}/{'MONTH'}/{'DATE'}.zarr.zip"
        
        cur_date = START_DATE
        while cur_date <= END_DATE:

            # download file
            huggingface_path = TEMPLATE_PATH.replace('YEAR', str(cur_date.year)) \
                                            .replace('MONTH', str(cur_date.month).zfill(2)) \
                                            .replace('DATE', str(cur_date.strftime("%Y%m%d")))
            download_path = self.download(huggingface_path)

            if type(download_path) != 'str':
                print(download_path)
                cur_date += timedelta(days=1)
                continue

            # unzip file
            unzipped_path = 'cache/unzipped/' + download_path[-30:-4]
            self.unzip(download_path, unzipped_path)

            # preprocess data
            self.preprocess(unzipped_path)
            
            # upload to GCP
            blob_file_name = huggingface_path[5:-4]
            self.gcp_upload(unzipped_path, self.config['gcp_dest_blob'] + blob_file_name)
            self.teardown("cache")

            # increment date
            cur_date += timedelta(days=1)


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
        self.teardown()


if __name__ == '__main__':
    config_path = 'nwp_config.json'
    datapipeline = NWPPipeline(config=config_path)
    datapipeline.execute()
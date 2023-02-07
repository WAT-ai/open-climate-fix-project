import json
import zipfile
from os import getcwd
import matplotlib.pyplot as plt
import xarray as xr
from huggingface_hub import hf_hub_download
from datetime import date, datetime, timedelta

# For each chunk
# 1. Download from HuggingFace
# 2. Unzip data
# 3. Preprocess
# 4. Upload to GCP


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
        with zipfile.ZipFile(source, 'r') as zip_ref:
            zip_ref.extractall(dest)

    def gcp_upload(self, source: str, dest: str) -> None:
        """
        Upload data from the source dir to a GCP bucket at dest

        Args:
            source: source file path
            dest: GCP bucket file path
        
        Returns:
            None
        """
        pass

    def teardown(self, filepath: str) -> None:
        """
        Deletes the directory at filepath and all of its contents

        Args:
            filepath: location of directory to delete
        """
        pass




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
                repo_id='openclimatefix/eumetsat-rss',
                filename=filepath,
                repo_type=self.config['hf_repo_type'],
                token=self.config['hf_token'],
                cache_dir=getcwd() + '/cache'
            )
            return download_path
        except Exception as error:
            log_file = open(self.config['error_log_path'] + 'error_logs.txt', 'a')
            log_file.write(str(error))
            print(error)

    def preprocess(self, filepath: str) -> None:
        """
        Preprocesses the zarr file at filepath according to configuration parameters

        Args:
            filepath: location of zarr path
        """
        pass

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

            # unzip file
            unzipped_path = 'unzipped/' + download_path[-30:-4]
            self.unzip(download_path, unzipped_path)
            self.teardown(download_path)

            # preprocess data
                # read in the data using xr, apply crops etc etc and save to disk 
                # in place of the original data and delete the xarray object in memory
            self.preprocess(unzipped_path)
            
            # upload to GCP
            self.gcp_upload(unzipped_path, self.config['gcp_bucket'] + '/' + unzipped_path[-30:-4])
            self.teardown(unzipped_path)

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
        self.cleanup()


if __name__ == '__main__':
    config_path = 'nwp_config.json'
    datapipeline = NWPPipeline(config=config_path)
    datapipeline.execute()

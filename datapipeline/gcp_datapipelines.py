import json
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
# 5. Clean up


class GCPPipeline:
    def __init__(self, config: str) -> None:
        """
            Intialization function for pipeline. Expects a path to JSON configuration file.
        """
        self.config = json.load(open(config))

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

    def download(self, filename: str) -> str:
        """
            Downloads data from "file_name" location from HugginFace and 
            returns the location of the downloaded data. If a filename is not found,
            it logs the error in a log file

            Args:
                file_name: HuggingFace data location
            
            Returns:
                returns downloaded location of the data
        """
        print(f'\nDownloading: {filename}...')
        try:
            download_path = hf_hub_download(
                repo_id=self.config['hf_repo_id'],
                filename=filename,
                repo_type=self.config['hf_repo_type'],
                token=self.config['hf_token']
            )
            return download_path
        except Exception as error:
            log_file = open(self.config['error_log_path'] + 'error_logs.txt', 'a')
            log_file.write(str(error))
            print(error)

        
    def preprocess(self) -> None:
        """
            Preprocesses the data as desired
            This function should probably take instructions from the download_configurations JSON file about the crop range, date range and features to drop
        """
        pass

    def format_date(self, date_str) -> tuple[date, date]:
        """
            Takes a date in mm-dd-yyyy format and returns a date object
        """
        return datetime.strptime(date_str, '%m-%d-%Y').date()

    def execute(self: str) -> None:
        """
            Runs the NWP pipeline according to the configuration file

            Args:
                None
            
            Returns:
                None
        """
        assert self.config['data_type'] == 'nwp', 'Configuration Error: Expects "nwp" data_type in configuration'

        START_DATE, END_DATE = self.format_date(self.config['start_date']), self.format_date(self.config['end_date'])
        assert START_DATE <= END_DATE, 'Configuration Error: start date must <= end date'

        TEMPLATE_PATH = f"data/surfac/{'YEAR'}/{'MONTH'}/{'DATE'}.zarr.zip"
        
        cur_date = START_DATE
        while cur_date <= END_DATE:

            # get file path
            filename = TEMPLATE_PATH.replace('YEAR', str(cur_date.year)) \
                                    .replace('MONTH', str(cur_date.month).zfill(2)) \
                                    .replace('DATE', str(cur_date.strftime("%Y%m%d")))

            # download file
            self.download(filename)

            # unzip
            # preprocess
            # upload
            # clean up

            # increment date
            cur_date += timedelta(days=1)


if __name__ == '__main__':
    config_path = 'nwp_config.json'
    datapipeline = NWPPipeline(config=config_path)
    datapipeline.execute()


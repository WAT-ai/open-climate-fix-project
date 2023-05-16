import json
import logging
import os
from datetime import timedelta, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='')

class GCPUtils:

    def download_dir(self, remote_path: str, destination_path: str) -> None:
        """
        Downloads a given directory from GCP

        Args:
            remote_path: path of dir in GCP storage
            destination_path: file path where data will be downloaded
        """

        logging.info(f'Downloading: {remote_path}')
        logging.info(f'To {destination_path}\n')
        os.system(f'gsutil -m cp -r {remote_path} {destination_path}')
        print('\n')

    def gcp_download_nwp(self, config_path: str) -> None:
        """
        Downloads NWP data as outlined in the config
        If some date within the given range is not found in the GCP bucket,
        it will be skipped and a corresponding message will be logged

        Args:
            config_path: path to JSON config object. It must contain:
                - destination_path: path to where data will be downloaded
                - start_date: start_date in dd/mm/yyy
                - end_date: end_date in dd/mm/yyy
        """
        config: dict = json.load(open(config_path))
        if not os.path.isdir(Path(config['destination_path'])):
            os.mkdir(Path(config['destination_path']))
        
        start_date = datetime.strptime(config['start_date'], '%d/%m/%Y')
        end_date =  datetime.strptime(config['end_date'], '%d/%m/%Y')
        current_date = start_date
        while current_date <= end_date:
            remote_path = f"gs://ocf_base_data/nwp/surface/{current_date.year}/{current_date.month:02}/{current_date.year}{current_date.month:02}{current_date.day:02}.zarr"
            self.download_dir(remote_path, config['destination_path'])
            current_date += timedelta(1)


if __name__ == "__main__":
    config_path = './gcp_download_config.json'
    util = GCPUtils()
    util.gcp_download_nwp(config_path=config_path)

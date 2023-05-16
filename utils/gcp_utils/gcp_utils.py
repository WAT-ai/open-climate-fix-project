import os
import logging
from pathlib import Path
from datetime import date, timedelta

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

    def gcp_download_nwp(
        self, destination_path: str, start_date: date, end_date: date
    ) -> None:
        """
        Downloads NWP data from start_date to end_date.
        If some date within the given range is not found in the GCP bucket,
        it will be skipped and a corresponding message will be logged

        Args:
            destination_path: path to where data will be downloaded
            start_date: start_date
            end_date: end_date

        Returns:
            None

        """
        if not os.path.isdir(Path(destination_path)):
            os.mkdir(Path(destination_path))

        cur = start_date
        while cur <= end_date:
            remote_path = f"gs://ocf_base_data/nwp/surface/{cur.year}/{cur.month:02}/{cur.year}{cur.month:02}{cur.day:02}.zarr"
            self.download_dir(remote_path, destination_path)
            cur += timedelta(1)


if __name__ == "__main__":
    start = date(2022, 1, 21)
    end = date(2022, 1, 22)
    path = Path("C:/Users/areel/watai/watai/data/experiment")
    util = GCPUtils()
    util.gcp_download_nwp(path, start, end)

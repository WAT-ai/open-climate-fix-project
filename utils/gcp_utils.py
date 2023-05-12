import os
from pathlib import Path
from datetime import date, timedelta

from google.cloud import storage

"""# Initialise a client
storage_client = storage.Client("WATai")
# Create a bucket object for our bucket
bucket = storage_client.get_bucket("ocf_base_data")
# Create a blob object from the filepath
blob = bucket.blob("folder_one/foldertwo/filename.extension")
# Download the file to a destination
blob.download_to_filename(destination_file_name)"""

class GCPUtils:

    def __init__(self) -> None:
        pass

    def create_gcp_storage_client(self, project: str, bucket: str):
        """
        Creates GCP storage client

        Args:
            project: GCP project name
            bucket: bucket name
        """
        self.storage_client = storage.Client(project)
        self.bucket = self.storage_client.get_bucket(bucket)


    def download_dir(self, remote_dir_path: str, destination: str):
        """
        Downloads a given directory from GCP

        Args:
            remote_dir_path: path of dir in GCP storage (after bucket name)
            destination: file path where data will be downloaded
        """
        print('Downloading:')
        print(f'    {remote_dir_path}')
        print('To')
        print(f'    {destination}')
        blobs = self.bucket.list_blobs(remote_dir_path)
        for blob in blobs:
            filename = blob.name.replace('/', '_')
            print(filename)
            blob.download_to_filename(destination + filename)

    def gcp_download_nwp(self, destination: str, start_date: date, end_date: date):
        """
        Downloads NWP data from start_date to end_date.
        If some date within the given range is not found in the GCP bucket,
        it will be skipped and a corresponding message will be logged

        Args:
            destination: file path where data will be downloaded
            start_date: start_date
            end_date: end_date
        
        Returns:
            None

        """
        if not os.path.isdir(Path(destination)):
            os.mkdir(Path(destination))

        self.create_gcp_storage_client('WAT.ai', 'ocf_base_data')

        cur = start_date
        while cur <= end_date:
            dir_path = f'nwp/surface/{cur.year}/{cur.month:02}/{cur.year}{cur.month:02}{cur.day:02}.zarr/'
            self.download_dir(dir_path, destination)
            cur += timedelta(1)

if __name__ == "__main__":
    start = date(2023, 1, 1)
    end = date(2023, 1, 1)
    path = Path('C:/Users/areel/watai/watai/data/experiment')
    util = GCPUtils()
    util.gcp_download_nwp(path, start, end)
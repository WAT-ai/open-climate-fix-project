import os
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


def gcp_download_nwp(destination: str, start_date: date, end_date: date):
    """
        downloads NWP data from start_date to end_date
        if data within the given range is not found in the GCP bucket,
        it will be skipped and a corresponding message will be logged

        
    """
    while start_date <= end_date:
        start_date += timedelta(1)

if __name__ == "__main__":
    start = date(2023, 1, 1)
    end = date(2023, 1, 20)
    gcp_download_nwp(start, end)
import json
import logging
import os
import pandas as pd
import zipfile
import shutil
import xarray as xr

from google.cloud import storage
from huggingface_hub import hf_hub_download
from datetime import date, datetime, timedelta

logging.basicConfig(level=logging.INFO)

class GCPPipelineUtils:
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
        logging.info(f'\nUnzipping: {source}')
        logging.info(f'\nExtracting to: {dest}')
        with zipfile.ZipFile(source, 'r') as zip_ref:
            zip_ref.extractall(dest)

    def gcp_upload_file(self, source: str, bucket_name: str, blob_name: str) -> None:
        """
        Uploads the source file to the GCP bucket specified in the configuration
        Used by the gcp_upload_dir method

        Args:
            source: file path of source file
            bucket_name: name of GCP bucket
            blob_name: desired name of file in GCP
        
        Returns:
            None
        """
        logging.info(f'\nUploading {source} to {blob_name}.')
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(source)
        logging.info(f'\nFile {source} succesfully uploaded to {blob_name}.')
        return None

    def gcp_upload_dir(self, source: str, bucket_name: str, blob_name: str):
        """
        Uploads the source dir to the GCP bucket specified in the configuration

        Args:
            source: file path of source dir
            bucket_name: name of GCP bucket
            blob_name: desired name of file in GCP
        
        Returns:
            None
        """
        logging.info(f'\nUploading {source} to {blob_name}.')
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        for path, subdirs, files in os.walk(source):
            for name in files:
                path_local = os.path.join(path, name)
                blob_path = path_local.replace('\\','/')
                remote_path = f'{blob_name}/{"/".join(blob_path.split(os.sep)[1:])}{blob_path.replace(source, "")[1:]}'
                blob = bucket.blob(remote_path)
                blob.upload_from_filename(path_local)

    def teardown(self, filepath: str) -> None:
        """
        Deletes the directory at filepath and all of its contents

        Args:
            filepath: location of directory to delete
        """
        logging.info(f'\nDeleting: {filepath}')
        shutil.rmtree(filepath)


class NWPPipeline(GCPPipelineUtils):
    def __init__(self, config: str) -> None:
        super().__init__(config)

    def download(self, filepath: str) -> str:
        """
        Downloads data from "filepath" location from HuggingFace and 
        returns the location of the downloaded data. If the filepath 
        is not found, it logs the error in a log file.

        Args:
            filepath: HuggingFace data location
        
        Returns:
            returns filepath of downloaded data
        """
        logging.info(f'\nDownloading: {filepath}...')
        try:
            download_path = hf_hub_download(
                repo_id=self.config['hf_repo_id'],
                filename=filepath,
                repo_type=self.config['hf_repo_type'],
                token=self.config['hf_token'],
                cache_dir=os.getcwd() + '/cache/downloaded'
            )
            return download_path
        except Exception as error:
            log_file = open(self.config['error_log_path'] + 'error_logs.txt', 'a')
            error_log = str(error) + f'\nAttempted to download filepath: {filepath}'
            log_file.write(str(error_log))
            return error

    def preprocess(self, filepath: str, to_path: str) -> None:
        """
        Preprocesses the zarr file at filepath according to configuration parameters

        Selecting time slice (6 AM to 9 PM)
        Dropping features
        Selecting a region (longitudinal latitudinal coords)

        Args:
            filepath: location of zarr path
        """
        logging.info(f'\nPreprocessing {filepath} and saving to {to_path}.')
        dataset = xr.open_dataset(filepath, engine='zarr', chunks='auto')
        dataset = self.drop_dataset_features(
            dataset=dataset, features=self.config['preprocess']['features']
            )
        dataset = self.crop_dataset_region(
            dataset=dataset,
            lat_range=self.config['preprocess']['latitude'],
            lon_range=self.config['preprocess']['longitude']
        )
        dataset = self.crop_dataset_time(
            dataset=dataset,
            time_range=self.config['preprocess']['time_range']
        )

        dataset.to_zarr(to_path)
        del dataset

    def crop_dataset_region(
            self,
            dataset: xr.Dataset,
            lat_range: tuple(int, int),
            lon_range: tuple(int, int)
        ) -> xr.Dataset:
        """
        Takes an Xarray dataset and returns a new dataset cropped within the given region

        Args:
            dataset: an Xarray dataset
            lat_range: a tuple of min and max latitudinal
            lon_range: a tuple of min and max longitudinal

        Returns:
            An Xarray dataset with cropped region
        """
        max_lat, min_lat = lat_range
        min_lon, max_lon = lon_range
        return dataset.sel(
            latitude=slice(min_lat, max_lat),
            longitude=slice(min_lon, max_lon)
        )
   
    def crop_dataset_time(self, dataset: xr.Dataset, time_range: tuple(int, int)) -> xr.Dataset:
        """
        Takes an Xarray dataset and returns a new dataset cropped within the given time range

        Args:
            dataset: an Xarray dataset
            lon_range: a tuple of min and max times

        Returns:
            An Xarray dataset with cropped time
        """
        min_time, max_time = time_range
        return dataset.sel(
            time=slice(dataset['time'][int(min_time)], dataset['time'][int(max_time)])
        )
    
    def drop_dataset_features(self, dataset: xr.Dataset, features: list[str]) -> xr.Dataset:
        """
        Takes an Xarray dataset and returns a new dataset with only the given features

        Args:
            dataset: an Xarray dataset
            features: list of features (must be valid features of dataset)

        Returns:
            An Xarray dataset with features dropped
        """
        return dataset[features]
    
    def interpolate_nwp(self, dataset: xr.Dataset):
        """
        Takes an Xarray of NWP data and interpolates the data
            interpolate after joining with PV

        Args:
            dataset: an Xarray of NWP data
        """
        pass

    def join_nwp_pv(
            self,
            path_to_pv_timeseries: str,
            path_to_pv_metadata: str,
            nwp_dataset: xr.Dataset
        ):
        """
        Takes an nwp_dataset and a path to PV data and joins them together

        Args:
            path_to_pv: file path to PV dataset
            nwp_dataset: an Xarray NWP dataset
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
                .replace('DATE', str(cur_date.strftime('%Y%m%d')))
            download_path = self.download(huggingface_path)

            if not isinstance(download_path, str):
                cur_date += timedelta(days=1)
                continue

            # unzip file
            unzipped_path = './cache/unzipped/' + download_path[-25:-4]
            self.unzip(download_path, unzipped_path)

            # preprocess data
            processed_path = './cache/preprocessed/' + download_path[-25:-4]
            self.preprocess(unzipped_path, processed_path)

        
            # upload to GCP
            blob_file_name = huggingface_path[5:-4]
            self.gcp_upload_dir(
                source=processed_path,
                bucket_name=self.config['gcp_bucket'],
                blob_name=self.config['gcp_dest_blob'] + blob_file_name
            )
            self.teardown('cache')

            # increment date
            cur_date += timedelta(days=1)


class PVPipeline(GCPPipelineUtils):
    def __init__(self, config: str) -> None:
        super().__init__(config)

    def preprocess(self, key: str, df: pd.DataFrame) -> pd.DataFrame:

        if 'instantaneous_power_gen_W' in df.columns:
            df.dropna(subset=['instantaneous_power_gen_W'], inplace=True)
            df = df.resample('15T').mean()
            df.rename(columns={'instantaneous_power_gen_W': 'instantaneous_power_W'}, inplace=True)
            df.drop('cumulative_energy_gen_Wh', axis=1, inplace=True)

        # if there is no instantaneous power, use diff operator to convert cumulative to instantaneous
        elif 'cumulative_energy_gen_Wh' in df.columns:
            df.dropna(subset=['cumulative_energy_gen_Wh'], inplace=True)
            df = df.resample('15T').mean()
            df['cumulative_energy_gen_Wh'].diff(inplace=True)
            df[df['cumulative_energy_gen_Wh'] < 0]['cumulative_energy_gen_Wh'] = 0
            df.rename(columns={'cumulative_energy_gen_Wh': 'instantaneous_power_W'}, inplace=True)

        elif 'energy_gen_Wh' in df.columns:
            df.rename(columns={'energy_gen_Wh': 'instantaneous_power_W'}, inplace=True)
            df = df.resample('15T').mean()
            # CHECK IF THIS CASE NEEDS TO BE DIFFERENCED

        else:
            logging.warning(f'\n{key} does not contain the necessary columns for power generation')

        df['system_id'] = key.split('/')[-1]
        df['timestamp'] = df.index

        # importing some preprocessing parameters from config
        start_date = datetime.strptime(self.config['preprocess']['date_range'][0], '%m-%d-%Y')
        end_date = datetime.strptime(self.config['preprocess']['date_range'][1], '%m-%d-%Y')
        assert start_date <= end_date, 'Configuration Error: start date must <= end date'

        # drop invalid values and dates
        df.dropna(subset=['instantaneous_power_W'], inplace=True)
        df = df[(df.index < end_date) & (df.index > start_date)]

        return df

    def execute(self) -> None:
        assert self.config['data_type'] == 'pv', 'Configuration Error: Expects "pv" data_type in configuration'
        filepath = self.config['file_path']
        hdf_path = None

        if not os.path.exists(filepath):
            logging.critical('\nDirectory does not exist at specified filepath')

        for file in os.listdir(filepath):

            if file.endswith('.hdf'):
                hdf_path = filepath + '/' + file

            # metadata is uploaded directly to GCP
            elif file.endswith('.csv'):
                self.gcp_upload_file(
                    source=filepath + '/' + file, 
                    bucket_name=self.config['gcp_bucket'],
                    blob_name=self.config['gcp_dest_blob'] + file
                )

        if not hdf_path:
            logging.critical('\nHDF5 file does not exist within the specified directory')

        with pd.HDFStore(hdf_path) as hdf:
            keys = hdf.keys()

        # create temp directory
        tmpdir = filepath + '/tmp'
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)

        # first two keys are uploaded directly to GCP
        pd.read_hdf(hdf_path, keys[0]).to_csv(tmpdir + '/pv_stats.csv')
        pd.read_hdf(hdf_path, keys[1]).to_csv(tmpdir + '/pv_missing.csv')
        self.gcp_upload_file(
            source=tmpdir + '/pv_stats.csv',
            bucket_name=self.config['gcp_bucket'],
            blob_name=self.config['gcp_dest_blob'] + 'pv_stats.csv'
        )
        self.gcp_upload_file(
            source=tmpdir + '/pv_stats.csv',
            bucket_name=self.config['gcp_bucket'],
            blob_name=self.config['gcp_dest_blob'] + 'pv_missing.csv'
        )

        # preprocessing and aggregating site-level data
        sites = []
        for i, key in enumerate(keys[2:]):
            logging.info(f'\nProcessing Key #{i}: {key}')
            site_df = pd.read_hdf(hdf_path, key)
            site_df = self.preprocess(key, site_df)
            sites.append(site_df)

        # OR using list comprehension
        # sites = [self.preprocess(site_df) for site_df in [pd.read_hdf(hdf_path, key) for key in keys]]

        total_df = pd.concat(sites, axis=0, ignore_index=True)

        # create MultiIndex
        total_df.set_index(['system_id', 'timestamp'], inplace=True)
        total_df.sort_index(inplace=True)

        # upload to GCP
        total_df.to_csv(tmpdir + '/pv_time_series.csv')
        self.gcp_upload_file(
            source=tmpdir + '/pv_time_series.csv',
            bucket_name=self.config['gcp_bucket'],
            blob_name=self.config['gcp_dest_blob'] + 'pv_time_series.csv'
        )
        self.teardown(tmpdir)


if __name__ == '__main__':
    config_path = './nwp_config.json'
    datapipeline = NWPPipeline(config_path)
    datapipeline.execute()

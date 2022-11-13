import wget
import sys

SATELLITE_2021_FILES = ['01', '02', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '18', '19', '21', '23', '24', '25', '29', '30', '31', '32', '33', '34', '37', '38', '40', '41', '42', '43', '46', '47', '49', '52', '54', '55', '58', '59', '62', '63', '64', '66', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '81', '82', '83', '84', '85', '86', '87', '88']
SATELLITE_2022_FILES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '28', '29', '30', '31', '32', '33', '34', '36', '38', '39', '40', '41', '42', '43', '44', '45']
SATELLITE_2021_COUNT = 88
SATELLITE_2022_COUNT = 56
SATELLITE_DATA_LINK = 'https://huggingface.co/datasets/openclimatefix/eumetsat-rss/resolve/main/data'

def download_data(files, link, year, file_count, output_dir):
    total = len(files)
    for i, file in enumerate(files[-2:]):
        file_name = f'{year}_0000{file}-of-0000{file_count}.zarr.zip'
        url = f'{link}/{year}/hrv/{file_name}'
        print(f'\n({i + 1}/{total}) - Downloading {file_name}\n')
        wget.download(url, output_dir + file_name)
    print(f"Download for {year} Complete")

if __name__ == "__main__":
    output_dir = sys.argv[1]
    year = sys.argv[2]
    if year == '2022':
        download_data(SATELLITE_2022_FILES, SATELLITE_DATA_LINK, 2022, SATELLITE_2022_COUNT, output_dir)
    else:
        download_data(SATELLITE_2021_FILES, SATELLITE_DATA_LINK, 2021, SATELLITE_2021_COUNT, output_dir)
    print("All Downloads Complete")

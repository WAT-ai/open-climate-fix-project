import zipfile
import os

INPUT_DIR = '../data'
OUTPUT_DIR = '../data'

if __name__ == '__main__':
    dir_list = os.listdir(INPUT_DIR)
    for index, file_name in enumerate(dir_list):
        print(f'({index + 1}/{len(dir_list)}) Unzipping: {file_name}')
        output_dir_file = OUTPUT_DIR + '/' + file_name[:-4]
        os.mkdir(output_dir_file)
        with zipfile.ZipFile(INPUT_DIR + '/' + file_name, 'r') as zip_ref:
            zip_ref.extractall(output_dir_file)
    print('Done')

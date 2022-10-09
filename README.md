# open-climate-fix-project

This is the code base for the 2022/23 Open Climate Fix project.

## utils/
Scripts for miscellaneous tasks can be stored in the utils directory.
The `unzip.py` file can unzip all files within a directory and save them to any other directory. This script is useful since most of the data that OCF provides will be zipped.
To use the `unzip.py` script:
1. Change the `INPUT_DIR` variable in `unzip.py` to the **absolute path** of the directory containing all the zip files you want to extract
2. Change `OUTPUT_DIR` to the **absolute path** of the directory where you want to save the extracted files.
3. Open a command prompt or terminal
4. Navigate to the directory containing `unzip.py`.
5. Run the command: `python unzip.py`

## Note on version control for data
Make sure not to add changes to data files to your commits. GitHub will not allow you to push changes to a file that is larger than ~50 MB, we cannot version our data using GitHub. To make things easier, create a directory called `data/` in your working directory and store all your data there. GitHub will not track any files in the `data/` directory (since the `data/` directory is in the .gitignore file)

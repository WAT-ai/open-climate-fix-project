# open-climate-fix-project

This is the code base for the 2022/23 Open Climate Fix project.

# Set up
This set up guide assumes you have Python and Ananconda installed, and you know basic command prompt or terminal commands.

### Development Environment
It is recommended to create a Python environment specifically for this project to minimize clutter. You can create a conda Python environment using the following command (assumes you have conda installed):
`conda create -n watai`. This will create an environment in you local machine called `watai`. In this environment you can install python and other libraries specific to this project. Although technically you can use any environment name you want (by replacing `watai` with something else in the command earlier) it is recommended you use `watai` as your env name, so that all developers on the team can share commands and setup scripts. Once the environment is created, run `conda activate watai` to activate the environment.
You can create environments without conda, by running `python -m venv watai`, but it is strongly recommended to use conda, since there are many package we will use down the road which are too complicated to install without conda (like Cartopy, geopandas, and others).



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

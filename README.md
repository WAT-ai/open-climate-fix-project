
# open-climate-fix-project

This is the code base for the 2022/23 Open Climate Fix project.

# Set up
This set up guide assumes you have Python and Anaconda installed, and you know basic cmd/terminal commands.

The dependencies for this project are a little finicky. Results can differ depending on your machine and operating system, so be ready to troubleshoot installation errors.

**Please make sure you are not Anaconda version 4.14.X**. Although this is the latest version (at the time of writing this), its been known to cause issues while installing packages that require SSL verification. Any version of Anaconda besides this should be fine, but I (Areel) used Anaconda 4.12.
## Steps
Here are the steps to install the dependencies:
1- Run `conda env create -f environment.yml`. This will create an environment in you local called `watai` with python version 3.10.4. It will install dependencies into the `watai` env.
2- Run `pip install -r requirements.txt`. This will install remaining dependencies.

#### Pip and Conda?
In general it is a bad idea to install packages using both Pip and Conda. The packages may not respect each others dependencies and versions, causing issues down the road. However, there are special cases (like this one), where both Pip and Conda need to be used. There are some packages that are too complex to be installed using just Pip, a package manager like Conda is needed. On the other hand, there are some packages that are not support by Conda. Hence, we use both. 
# utils/
Scripts for miscellaneous tasks can be stored in the utils directory.
## utils/unzip.py
The `unzip.py` file can unzip all files within a directory and save them to any other directory. This script is useful since most of the data that OCF provides will be zipped.
To use the `unzip.py` script:
1. Change the `INPUT_DIR` variable in `unzip.py` to the **absolute path** of the directory containing all the zip files you want to extract
2. Change `OUTPUT_DIR` to the **absolute path** of the directory where you want to save the extracted files.
3. Open a command prompt or terminal
4. Navigate to the directory containing `unzip.py`.
5. Run the command: `python unzip.py`

## Note on version control for data
Make sure not to add changes to data files to your commits. GitHub will not allow you to push changes to a file that is larger than ~50 MB, we cannot version our data using GitHub. To make things easier, create a folder called `data/` in your working directory and store all your data there. GitHub will not track any files in the `data/` directory (since the `data/` directory is in the .gitignore file)

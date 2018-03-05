## Version 1
* run_all.py - script to run through the entire design. Call submodules from here

### Running
run_all.py - will run entire design

run_all.py # - will run starting from step # Ie: run_all.py 2 (starts from step 2)

### Setting up the environment
* You'll need Python3 and pip3
* The commands below are for Linux

__Virtualenv__:
* To create a virtualenv environment with Python3:
```
virtualenv -p python3 [env name]
```
* To activate the virtualenv:
```
source [env name]/bin/activate
```
* All the packages needed are listed in _requirements.txt_. To install using pip:
```
pip install -r requirements.txt
```
* To update this file (within the virtual environment):
```
pip freeze > requirements.txt
```
* To leave the virtualenv:
```
deactivate
```

__Anaconda__:
* The packages needed, with explicit download links, are listed in _spec-file.txt_
* To set up an Anaconda virtual environment with all the required dependencies installed:
```
conda install --name myenv --file spec-file.txt
```
* To update this file (within the virtual environment):
```
conda list --explicit > spec-file.txt
```

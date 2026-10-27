# AfriSAR-2-ML Report

This repository contains the python `afrisar2ml_report` package that accompanies the project report and documentation of the AfriSAR-2-ML dataset.


## Installation

- clone this repository
- activate or create a python environment (e.g., `conda activate YOUR_ENV_NAME` or `conda create -n afrisar2ml_report python=3.13`)
- preferrably install GDAL using conda: `conda install gdal==3.13.3`
- install this package by running `pip install -e .` in the repository root folder (where `pyproject.toml` is located)


## Running

- adjust paths to the project folder and datasets in `src/afrisar2ml_report/paths.py`
- check config in `src/afrisar2ml_report/config.py`
- optionally, adjust device (CPU / GPU) in `src/afrisar2ml_report/hardware.py`
- run the benchmarks and generate report figures with `python -m afrisar2ml_report`

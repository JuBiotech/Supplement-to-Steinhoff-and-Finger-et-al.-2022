[![DOI](https://zenodo.org/badge/462239542.svg)](https://zenodo.org/badge/latestdoi/462239542)
 
 This repository contains the data and analysis for ODE-modeling based $K_S$ determination.

# Contents
* `data` contains raw datasets
* `notebooks` contains notebooks and externalized Python code for the analysis
* `results` contains the intermediate and final results of the analysis
# Installation
A Python environment for running the notebooks can be created with the following command:

```
conda env create -f environment.yml
```

The new environment is named `murefi_env` and can be activated with `conda activate murefi_env`.

After that a Jupyter notebook server can be launched with `jupyter notebook`.

# Contributions
The git history of this project was forked from https://github.com/jubiotech/calibr8-paper, but most of the content changed since.

Simone Schito contributed to the creation of calibration and cultivation data.

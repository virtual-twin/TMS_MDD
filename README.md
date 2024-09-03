# Transcranial Magnetic Stimulation (TMS) in 'The Virtual Brain' (TVB)
#### Author: Dr. Timo Hofsähs

This repository contains the code and data necessary to reproduce the results presented in the paper:


### The Virtual Brain links transcranial magnetic stimulation evoked potentials and neurotransmitter changes in major depressive disorder
#### Timo Hofsähs, Marius Pille, Jil Meier, Petra Ritter  
(in prep)

## Table of Contents
1. [Project Structure](#1-project-structure)
2. [Data](#2-data)
3. [Prerequisites](#3-Prerequisites)
4. [Usage](#4-usage)
5. [License](#5-license)
6. [Contact Information](#6-contact-information)
7. [Acknowledgements](#7-acknowledgements)


## 1. Project Structure

- `data/`: Contains files necessary to run the code
  - `leadfield/`: Leadfield matrix to project from source- to sensor-level (EEG)
  - `Schaefer2018_200Parcels_7Networks_count.csv`: Unfitted structural connectivity weights matrix
  - `Schaefer2018_200Parcels_7Networks_distance.csv`: Structural connectivity tract length matrix
  - `stimulus_weights.npy`: Spatial information of TMS stimulus
  - `TEPs.mat`: Empirical TMS-EEG timeseries for 20 healthy subjects, needs to be downloaded separately
- `analysis.py`: Script 3 for generating plots and analyzing simulation results
- `create_parameters.py`: Creates `parameters.txt`
- `environment.yml`: Generates environment with necessary software
- `fitting.py`: Script 1 for fitting EEG timeseries to empirical data
- `functions.py`: Contains shared functions used across other scripts
- `parameters.txt`: Specifies parameters for both `fitting.py` and `simulation.py`for parallel computing
- `requirements.txt`: Installs further necessary software
- `simulation.py`: Script 2 for running simulations based on fitted data


## 2. Data

All data necessary to run this code and reproduce the results, except the empirical TMS-EEG data, is stored in the `data/` directory. To run the fitting on the empirical TMS-EEG data, download the dataset under the link below and place `TEPs.mat` in `data/`. Please observe the data protection regulations of your respective country. All other data stored in `data/` originates from the publication listed below.

**Empirical TMS-EEG Data**<br>
Biabani M, Fornito A, Mutanen TP, Morrow J, Rogasch NC, Characterizing and minimizing the contribution of sensory inputs to TMS-evoked potentials. Brain Stimulation, 12(6), 1537-1552, 2019, DOI: https://doi.org/https://doi.org/10.1016/j.brs.2019.07.009<br>
Download: https://bridges.monash.edu/articles/dataset/TEPs-_SEPs/7440713?file=13772894<br>
Files: `TEPs.mat`

**Structural Connectivity & further Simulation Data**<br>
Momi D, Wang Z, Griffiths D (2023) TMS-evoked responses are driven by recurrent large-scale network dynamics eLife 12:e83232, DOI: https://doi.org/10.7554/eLife.83232,<br>
Download: https://github.com/GriffithsLab/PyTepFit<br>
Files: `Schaefer2018_200Parcels_7Networks_count.csv`, `Schaefer2018_200Parcels_7Networks_distance.csv`, `leadfield`, `stimulus_weights.npy`


## 3. Prerequisites

All requirements are stored in environment.yml and requirements.txt. To create a conda environment named `tms` please execute the following commands:

```bash
conda env create -f environment.yml
conda activate tms
pip install -r requirements.txt
```


## 4. Usage

The recreation of the publication results requires three steps, which must be performed in order:
1. Fitting: The script `fitting.py` fits EEG timeseries of whole-brain simulations to empirical TMS-evoked potential EEG timeseries. A gradient-descent algorithm is applied to fit the structural connectivity weights individually. The fitting is repeated for 20 healthy subjects and 100 different initial conditions each, which generates 2,000 individual fittings.
2. Simulation: In the script `simulation.py`, the fittings from step 1 are used to generate whole-brain simulations. The simulations are repeated with one of two parameters (Jansen & Rit neural mass model parameter b or C4) altered in a range of -50% to +50% from default in 2% steps, resulting in 50 simulations per parameter, adding up to one default simulation, generating 101 simulations per fitting. Results for all simulations are stored. 
3. Analysis: The script `analysis.py` takes the source- (RAW) and sensor-level (EEG) timeseries to generate plots.

To create the results, the scripts were executed on a high-performance computing cluster utilizing parallelization techniques with the argparse package. Both `fitting.py` and `simulation.py` were run with the parameters specified in each line of parameters.txt, leveraging multiple cores for computation.


## 5. License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC-BY 4.0, https://creativecommons.org/licenses/by/4.0/). You are free to share, copy, distribute, and transmit the work, as well as to adapt the work, provided that appropriate credit is given to the authors, a link to the license is provided, and you indicate if changes were made.


## 6. Contact Information

For questions or comments, please contact:
- Timo Hofsähs: timo.hofsaehs@charite.de
- Jil Meier: jil.meier@bih-charite.de
- Petra Ritter: petra.ritter@bih-charite.de


## 7. Acknowledgments

The fitting method and all functions in this script 'functions.py' except from 'gmfa' and  'gmfa_timepoint' are based on the code provided with the following publication:

Momi D, Wang Z, Griffiths JD. 2023. TMS-EEG evoked responses are driven by recurrent large-scale network dynamics. eLife2023;12:e83232 DOI: https://doi.org/10.7554/eLife.83232 
Licensed under a Creative Commons Attributionlicense (CC-BY). The original code can be found at: https://github.com/GriffithsLab/PyTepFit/blob/main/tepfit/fit.py
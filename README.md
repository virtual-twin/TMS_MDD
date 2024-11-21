# Transcranial Magnetic Stimulation (TMS) in 'The Virtual Brain' (TVB)
#### Author: Dr. Timo Hofsähs

This repository contains the code and data necessary to reproduce the results presented in the paper:


### The Virtual Brain links transcranial magnetic stimulation evoked potentials and inhibitory neurotransmitter changes in major depressive disorder
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

- `1_optimization.py`: Script 1 for optimizing the SC per subject
- `2_simulation.py`: Script 2 for running simulations utilizing optimized SCs
- `3_analysis.py`: Script 3 for analyzing simulation results and generating plots
- `4_bids_conversion.py`: Script 4 for converting the generated results into BIDS standard
- `data/`: Create this directory manually and place `TEPs.mat` in it before running `1_optimization.py`, contains all required files to run scripts 
  - `leadfield/`: Leadfield matrix to project from source-level to sensor-level (EEG)
  - `Schaefer2018_200Parcels_7Networks_count.csv`: Empirical structural connectivity weights matrix before optimization
  - `Schaefer2018_200Parcels_7Networks_distance.csv`: Structural connectivity tract length matrix
  - `stim_weights.npy`: Spatial information of TMS stimulus
  - `TEPs.mat`: Empirical TMS-EEG timeseries for 20 healthy subjects, needs to be downloaded separately and placed manually in data
- `create_parameters.py`: Creates `parameters.txt`
- `environment.yml`: Generates environment with necessary software
- `functions.py`: Contains shared functions used across other scripts
- `parameters.txt`: Specifies parameters for both `1_optimization.py` and `2_simulation.py`for parallel computing
- `requirements.txt`: Installs further necessary software


## 2. Data

Data necessary to run the code and reproduce the results need to be downloaded and stored in the `data/` directory. To run the optimization on the empirical TMS-EEG data, manually create the directory `data/`, download the dataset under the link below and place the file `TEPs.mat` in `data/`. Please observe the data protection regulations of your respective country. All other data will be downloaded automatically when running `1_optimization.py` from the repository provided.

**Empirical TMS-EEG Data**<br>
Biabani M, Fornito A, Mutanen TP, Morrow J, Rogasch NC, Characterizing and minimizing the contribution of sensory inputs to TMS-evoked potentials. Brain Stimulation, 12(6), 1537-1552, 2019, DOI: https://doi.org/https://doi.org/10.1016/j.brs.2019.07.009<br>
Download: https://bridges.monash.edu/articles/dataset/TEPs-_SEPs/7440713?file=13772894<br>
Files: `TEPs.mat`

**Structural Connectivity & further Simulation Data**<br>
Momi D, Wang Z, Griffiths D (2023) TMS-evoked responses are driven by recurrent large-scale network dynamics eLife 12:e83232, DOI: https://doi.org/10.7554/eLife.83232,<br>
Download: https://github.com/GriffithsLab/PyTepFit<br>
Files: `Schaefer2018_200Parcels_7Networks_count.csv`, `Schaefer2018_200Parcels_7Networks_distance.csv`, `leadfield`, `stim_weights.npy`


## 3. Prerequisites

All requirements are stored in environment.yml and requirements.txt. To create a conda environment named `tms` please execute the following commands:

```bash
conda env create -f environment.yml
conda activate tms
pip install -r requirements.txt
```


## 4. Usage

The recreation of the publication results requires three steps, which must be performed in order:
1. Optimization: The script `1_optimization.py` optimizes the structural connectivity individually per subject, to generate EEG timeseries of whole-brain simulations with a high correlation to empirical TMS-evoked potential EEG timeseries. The optimization is repeated for 20 healthy subjects and 100 different initial conditions each, which generates 2,000 individual optimizations.
2. Simulation: In the script `2_simulation.py`, the optimized SCs from step 1 are used to generate whole-brain simulations. The simulations are repeated with one of two parameters (Jansen & Rit neural mass model parameter b or C4) altered in a range of -50% to +50% from default in 2% steps, resulting in 50 simulations per parameter, adding up to one default simulation, generating 101 simulations per optimization. Results for all simulations are stored. 
3. Analysis: The script `3_analysis.py` takes the source- (RAW) and sensor-level (EEG) timeseries to generate plots.
4. BIDS Conversion: The script `4_bids_conversion.py` takes the complete directory structure consisting of downloaded data files and results from `1_optimization.py` and `2_simulation.py` and creates a complete new directory into Brain Imaging Data Standard (BIDS). The structure follows the BIDS Extension Proposal 034 (BEP034) for Computational Model Specifications (https://zenodo.org/records/7962032). Please note, that the script requires additional software packages (MNE, gdown, requests) and performs further file downloads.

To create the results, the scripts were executed on a high-performance computing cluster utilizing parallelization techniques with the argparse package. Both `1_optimization.py` and `2_simulation.py` were run with the parameters specified in each line of parameters.txt, leveraging multiple cores for computation.


## 5. License

This project is licensed under the **European Union Public License (EUPL), version 1.2**.  
You may use, modify, and distribute this software under the terms of the EUPL v1.2.  

For more information, see the [EUPL v1.2 license text](https://joinup.ec.europa.eu/collection/eupl/eupl-text-11-12) or the accompanying `LICENSE` file in this repository.



## 6. Contact Information

For questions or comments, please contact:
- Timo Hofsähs: timo.hofsaehs@charite.de
- Jil Meier: jil-mona.meier@bih-charite.de
- Petra Ritter: petra.ritter@bih-charite.de


## 7. Acknowledgements

The optimization method and all functions in `functions.py` except 'gmfa' and  'gmfa_timepoint' are based on the code provided with the following publication:
Momi D, Wang Z, Griffiths JD. 2023. TMS-EEG evoked responses are driven by recurrent large-scale network dynamics, eLife2023;12:e83232, DOI: https://doi.org/10.7554/eLife.83232. 
Licensed under a Creative Commons Attributionlicense (CC-BY).
Code: https://github.com/GriffithsLab/PyTepFit/blob/main/tepfit/fit.py
Licensed under MIT License: 'Copyright (c) [2023] [Davide Momi]. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.'

Brain Imaging Data Structure (BIDS)
Gorgolewski, K., Auer, T., Calhoun, V. et al. The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. Sci Data 3, 160044 (2016). https://doi.org/10.1038/sdata.2016.44

BIDS Extension for Computational Models:
Schirner M., Ritter P., BIDS Extension Proposal 034 (BEP034): BIDS Computational Model Specification, Version 1.0, https://zenodo.org/records/7962032
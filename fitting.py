# ------------------------------------------------------------------------------
# fitting.py
# Author: Dr. Timo Hofsähs
#
# Description: 
# This Python script is part of the code accompanying the scientific publication:
# The Virtual Brain links transcranial magnetic stimulation evoked potentials and 
# neurotransmitter changes in major depressive disorder
# Dr. Timo Hofsähs, Marius Pille, Dr. Jil Meier, Prof. Petra Ritter
# (in prep)
# 
# This code performs fitting of simulated TMS-evoked potentials to empirical data.
# Fitting results are stored. 
#
# The fitting method is based on the publication:
# Momi D, Wang Z, Griffiths JD. 2023. TMS-EEG evoked responses are driven by recurrent 
# large-scale network dynamics. eLife2023;12:e83232 DOI: https://doi.org/10.7554/eLife.83232
# Licensed under a Creative Commons Attribution license (CC-BY)
# The original code can be found at:
# https://github.com/GriffithsLab/PyTepFit/tree/main
#
# Copyright (c) 2024 Dr. Timo Hofsähs. All rights reserved.
#
# License: This code is licensed under the Creative Commons Attribution 4.0 International 
# License (CC-BY 4.0), which allows for redistribution, adaptation, and use in source 
# and binary forms, with or without modification, provided proper credit is given to 
# the original authors. You can view the full terms of this license at:
# https://creativecommons.org/licenses/by/4.0/
# ------------------------------------------------------------------------------

import argparse
import numpy as np
import os
import pandas as pd
import pickle
import scipy
import time
import zipfile

from functions import ParamsJR, Model_fitting, RNNJANSEN, Costs, OutputNM


def get_config():
    '''
    defines all values that can be changed from outside the script
    '''
    print('get_config')
    
    parser = argparse.ArgumentParser()
    cmd_parameters = list()

    jr_std = 0.0
    jr_init = 0.0
    
    cmd_parameters.append(["sub", 0, int])
    cmd_parameters.append(["run", 0, int])
    cmd_parameters.append(["seed", 0, int])
    cmd_parameters.append(["num_epochs", 2, int])
    cmd_parameters.append(["filename", "test", str])
    cmd_parameters.append(["jr_A_default", 3.25, float])
    cmd_parameters.append(["jr_A_std", 0, float])
    cmd_parameters.append(["jr_A_init", 0, float])
    cmd_parameters.append(["jr_B_default", 22.0, float])
    cmd_parameters.append(["jr_B_std", 0, float])
    cmd_parameters.append(["jr_B_init", 0, float])
    cmd_parameters.append(["jr_a_default", 100, float])
    cmd_parameters.append(["jr_a_std", jr_std, float])
    cmd_parameters.append(["jr_a_init", jr_init, float])
    cmd_parameters.append(["jr_b_default", 50, float])
    cmd_parameters.append(["jr_b_std", jr_std, float])
    cmd_parameters.append(["jr_b_init", jr_init, float])
    cmd_parameters.append(["jr_c1_default", 135, float])
    cmd_parameters.append(["jr_c1_std", jr_std, float])
    cmd_parameters.append(["jr_c1_init", jr_init, float])
    cmd_parameters.append(["jr_c2_default", (135*0.8), float])
    cmd_parameters.append(["jr_c2_std", jr_std, float])
    cmd_parameters.append(["jr_c2_init", jr_init, float])
    cmd_parameters.append(["jr_c3_default", (135*0.25), float])
    cmd_parameters.append(["jr_c3_std", jr_std, float])
    cmd_parameters.append(["jr_c3_init", jr_init, float])
    cmd_parameters.append(["jr_c4_default", (135*0.25), float])
    cmd_parameters.append(["jr_c4_std", jr_std, float])
    cmd_parameters.append(["jr_c4_init", jr_init, float])
    cmd_parameters.append(["jr_vmax_default", 5, float])
    cmd_parameters.append(["jr_vmax_std", 0, float])
    cmd_parameters.append(["jr_vmax_init", 0, float])
    cmd_parameters.append(["jr_v0_default", 6, float])
    cmd_parameters.append(["jr_v0_std", 0, float])
    cmd_parameters.append(["jr_v0_init", 0, float])
    cmd_parameters.append(["jr_r_default", 0.56, float])
    cmd_parameters.append(["jr_r_std", 0, float])
    cmd_parameters.append(["jr_r_init", 0, float])
    cmd_parameters.append(["jr_mu_default", 1e-09, float])
    cmd_parameters.append(["jr_mu_std", 0, float])
    cmd_parameters.append(["jr_mu_init", 0, float])
    cmd_parameters.append(["sc_weights_std", 1000, float])
    cmd_parameters.append(["sc_weights_init", 0, float])
    cmd_parameters.append(["g_default", 1000, float])
    cmd_parameters.append(["g_std", 0, float])
    cmd_parameters.append(["g_init", 0, float])
    cmd_parameters.append(["speed_default", 2.5, float])
    cmd_parameters.append(["speed_std", 0, float])
    cmd_parameters.append(["std_in_default", 250, float])
    cmd_parameters.append(["std_in_std", 0, float])
    cmd_parameters.append(["k_default", 7.5, float])
    cmd_parameters.append(["k_std", 0, float])

    for (parname, default, partype) in cmd_parameters:
        parser.add_argument(f"-{parname}", default=default, type=partype)
    config = parser.parse_args()
    return config


def run_fitting(config):
     '''Set up and run fitting'''
     
     start_time = time.time()
     print(f"\nConfig: {config}\n")
     sub = config.sub
     print(f'Subject: {sub}')
     run = config.run
     print(f'Run no.: {run}')
     num_epoches = config.num_epochs
     print(f'Number of epochs: {num_epoches}\n')
     filename = config.filename
     print(f'Filename: {filename}\n')

     file_sc_weights = './data/Schaefer2018_200Parcels_7Networks_count.csv'
     file_sc_distances = './data/Schaefer2018_200Parcels_7Networks_distance.csv'
     file_empirical_timeseries = './data/TEPs.mat'
     file_leadfield_matrix = './data/leadfield'
     file_stimulus = './data/stimulus_weights.npy'

     sc_weights_original = pd.read_csv(file_sc_weights, header=None, sep=' ').values
     sc_weights_norm = np.log1p(sc_weights_original)/np.linalg.norm(np.log1p(sc_weights_original))
     sc_distances = pd.read_csv(file_sc_distances, header=None, sep=' ').values
     empirical_timeseries = scipy.io.loadmat(file_empirical_timeseries)['meanTrials'][0][2][0]
     leadfield_matrix = np.load(file_leadfield_matrix, allow_pickle=True)
     stimulus_spatial = np.load(file_stimulus)

     dir_save = './results/results_fitting/'
     if not os.path.exists(dir_save):
         os.makedirs(dir_save)

     step_size = 0.00025
     tr = 0.001
     batch_size = 50
     input_size = 3
     state_size = 6
     base_batch_num = 20
     node_size = sc_weights_norm.shape[0] # (200)
     output_size = leadfield_matrix.shape[0] # (62)
     stimulus_temporal = np.zeros((node_size, int(tr/step_size), 400))
     stimulus_temporal[:, :, 100:110] = 1000
     print(f"batch_size: {batch_size}\nstep_size: {step_size}\ninput_size: {input_size}\ntr: {tr}\nstate_size: {state_size}\nbase_batch_num: {base_batch_num}\nnode_size: {node_size}\noutput_size: {output_size}\n")

     parameter = ParamsJR('JR',
         # Jansen Rit parameters
         A = [config.jr_A_default, config.jr_A_std, config.jr_A_init],
         B = [config.jr_B_default, config.jr_B_std, config.jr_B_init],
         a = [config.jr_a_default, config.jr_a_std, config.jr_a_init],
         b = [config.jr_b_default, config.jr_b_std, config.jr_b_init],
         c1 = [config.jr_c1_default, config.jr_c1_std, config.jr_c1_init],
         c2 = [config.jr_c2_default, config.jr_c2_std, config.jr_c2_init],
         c3 = [config.jr_c3_default, config.jr_c3_std, config.jr_c3_init],
         c4 = [config.jr_c4_default, config.jr_c4_std, config.jr_c4_init],
         vmax = [config.jr_vmax_default, config.jr_vmax_std, config.jr_vmax_init],
         v0 = [config.jr_v0_default, config.jr_v0_std, config.jr_v0_init],
         r = [config.jr_r_default, config.jr_r_std, config.jr_r_init],
         mu = [config.jr_mu_default, config.jr_mu_std, config.jr_mu_init],
         
         # sc
         w_bb = [sc_weights_norm, config.sc_weights_std, config.sc_weights_init],

         # global
         g = [config.g_default, config.g_std, config.g_init],
         speed = [config.speed_default, config.speed_std],
         std_in = [config.std_in_default, config.std_in_std],
         
         # leadfield matrix
         lm = [leadfield_matrix, np.zeros((output_size, node_size))],

         # stimulus
         k = [config.k_default, config.k_std], # stimulus scaling
         ki = [stimulus_spatial[:, np.newaxis], 0] # stimulus weights
         )
     print({k: v for k, v in parameter.__dict__.items() if k not in ['w_bb', 'lm', 'ki']}) # print dictionary except sc, lfm & stimulus
     print("\n")

     data_mean = np.array([empirical_timeseries[sub]] * num_epoches) # shape (120, 62, 2000)
     model = RNNJANSEN(input_size, node_size, batch_size, step_size, output_size, tr, sc_weights_norm, leadfield_matrix, sc_distances, parameter, config.seed)
     F = Model_fitting(model, data_mean[:, :, 900:1300], num_epoches, 0) # -1: Cost method
     output_train = F.train(u=stimulus_temporal)
     print(f'\nComplete train function')
     X0 = np.random.uniform(0, 5, (node_size, state_size))
     hE0 = np.random.uniform(0, 5, (node_size, 500))
     output_test = F.test(X0, hE0, base_batch_num, u=stimulus_temporal)
     print(f'Complete test function')

     dict_out = {
             'A': parameter.A[0],
             'B': parameter.B[0],
             'a': parameter.a[0],
             'b': parameter.b[0],
             'c1': parameter.c1[0],
             'c2': parameter.c2[0],
             'c3': parameter.c3[0],
             'c4': parameter.c4[0],
             'vmax': parameter.vmax[0],
             'v0': parameter.v0[0],
             'r': parameter.r[0],
             'mu': parameter.mu[0],
     
             'g': parameter.g[0],
             'speed': parameter.speed[0],
             'std_in': parameter.std_in[0],
     
             'k': parameter.k[0],

             'course_parameter_values': F.output_sim.course_parameter_values,
             
             'fitted_sc': F.output_sim.course_sc_values[-1],
             
             'ts_raw_fitted': (F.output_sim.E_test-F.output_sim.I_test).T.astype(np.float32),

             'course_loss': F.output_sim.course_loss,
             'course_cos_sim': F.output_sim.course_cos_sim,
             'course_pcc': F.output_sim.course_pcc,

             'config': config
         }

     filename_pkl = f"{filename}.pkl"
     filename_zip = f"{dir_save}{filename}.zip"
     with zipfile.ZipFile(filename_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
         with zipf.open(filename_pkl, 'w') as f:
             pickle.dump(dict_out, f)

     print(f"Files exported\n")
     print(f'Running time: {int(time.time() - start_time)} seconds')

if (__name__ == "__main__"):
     config = get_config()
     run_fitting(config)

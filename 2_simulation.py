# ------------------------------------------------------------------------------
# 2_simulation.py
# Author: Dr. Timo Hofsähs
#
# Copyright © 2024 Charité Universitätsmedizin Berlin. 
# This software is licensed under the terms of the European Union Public License 
# (EUPL) version 1.2 or later.
# https://joinup.ec.europa.eu/collection/eupl/eupl-text-11-12
# 
# Description: 
# This Python script is part of the code accompanying the scientific publication:
# The Virtual Brain links transcranial magnetic stimulation evoked potentials and 
# inhibitory neurotransmitter changes in major depressive disorder
# Timo Hofsähs, Marius Pille, Lukas Kern, Anuja Negi, Jil Meier, Petra Ritter
# (in prep)
# 
# This code performs simulations of TMS-evoked responses in EEG using 'The Virtual 
# Brain' (TVB) simulator. The script loads optimized SCs created with '1_optimization.py'
# repeats the simulation with one of two parameters modified and stores the results.
# ------------------------------------------------------------------------------

import argparse
import math
import numpy as np
import os
import pandas as pd
import pickle
import zipfile
from numba import guvectorize, float64
from tvb.simulator.lab import *
from functions import gmfa

def get_config():
    '''
    Defines all values that can be changed from outside the script
    '''
    print('get_config')
    
    parser = argparse.ArgumentParser()
    cmd_parameters = list()

    cmd_parameters.append(["sub", "0", str])
    cmd_parameters.append(["run", "0", str])
    cmd_parameters.append(["seed", 0, int])
    cmd_parameters.append(["sim_duration", 1400, float])
    cmd_parameters.append(["jr_A", 3.25, float])
    cmd_parameters.append(["jr_B", 22.0, float])
    cmd_parameters.append(["jr_a", 100, float])
    cmd_parameters.append(["jr_b", 50, float])
    cmd_parameters.append(["jr_c1", 135, float])
    cmd_parameters.append(["jr_c2", 108, float])
    cmd_parameters.append(["jr_c3", 33.75, float])
    cmd_parameters.append(["jr_c4", 33.75, float])
    cmd_parameters.append(["jr_v0", 6.0, float])
    cmd_parameters.append(["jr_r", 0.56, float])
    cmd_parameters.append(["jr_mu", 1e-9, float])
    cmd_parameters.append(["jr_e0", 5.0, float])
    cmd_parameters.append(["jr_stvar", 4, int])
    cmd_parameters.append(["g", 1000.0, float])
    cmd_parameters.append(["speed", 2.5, float])
    cmd_parameters.append(["dtx", 0.25, float])
    cmd_parameters.append(["raw_p", 1.0, float])
    cmd_parameters.append(["stim_pattern", "SinglePulse", str])
    cmd_parameters.append(["stim_onset", 1000.0, float])
    cmd_parameters.append(["stim_length", 10.0, float])
    cmd_parameters.append(["stim_frequency", 1000.0, float])
    cmd_parameters.append(["stim_scaling", 7.5, float])
    cmd_parameters.append(["filename", "test", str])
    for (parname, default, partype) in cmd_parameters:
        parser.add_argument(f"-{parname}", default=default, type=partype)
    config = parser.parse_args()
    return config

def run_simulation(config):

     print(f"\nConfig: {config}")

     # DEFINE SUBJECT
     sub = config.sub
     print(f"Sub: {sub}")

     # DEFINE OPTIMIZATION RUN
     run = config.run
     print(f"Run: {run}")

     # DEFINE SEED
     np.random.seed(config.seed)

     # LOAD OPTIMIZED PARAMETERS
     with zipfile.ZipFile(f"./results/results_optimization/{config.filename}.zip", 'r') as zipf:
         with zipf.open(f"{config.filename}.pkl") as f:
             dict_sub = pickle.load(f)
     print(f"Load optimized SCs\n")

     # LOAD FACTORS
     factors_explore = ['jr_b', 'jr_c4']
     range_low = 0.5
     range_high = 1.501
     range_step_size = 0.02
     factors_values = np.arange(range_low, range_high, range_step_size)
     
     # SIMULATION LENGTH - in ms
     sim_duration = config.sim_duration
     print(f"- Sim duration: {sim_duration}\n")
     
     # CONNECTIVITY - averaged HCP
     sc = dict_sub['fitted_sc']
     diagonal = -np.diag(np.sum(sc, axis=1))
     sc_edit = sc + diagonal
     file_sc_distances = './data/Schaefer2018_200Parcels_7Networks_distance.csv'
     sc_distances = pd.read_csv(file_sc_distances, header=None, sep=' ').values
     ce = np.array([0])
     white_matter = connectivity.Connectivity(tract_lengths = sc_distances, weights = sc_edit, region_labels=np.zeros(200,), centres=ce)
     white_matter.speed = np.array(dict_sub['speed'])
     white_matter.configure()
     print(f"- White matter: {white_matter}\n")
     print(f"- White matter speed: {white_matter.speed}\n")

     # NEURAL MASS MODEL - Jansen Rit extended
     from tvb.basic.neotraits.api import NArray, List, Range
     class ExtendedJansenRit(models.JansenRit):
         variables_of_interest = List(of=str, label="Variables watched by Monitors", choices=(['y1-y2', 'y0','y1','y2','y3','y4','y5']), default=(['y1-y2']))
 
         state_variables = tuple('y0 y1 y2 y3 y4 y5'.split())
         _nvar = 6
         cvar = np.array([1, 2], dtype=np.int32)
 
         def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
             y0, y1, y2, y3, y4, y5 = state_variables
             lrc = coupling[0, :] - coupling[1, :]
             short_range_coupling = local_coupling*(y1 - y2)

             exp = np.exp
 
             sigm_y1_y2 = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - (y1 - y2))))
             sigm_y0_1  = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - (self.a_1 * self.J * y0))))
             sigm_y0_3  = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - (self.a_3 * self.J * y0))))
 
             return np.array([
                 y3,
                 y4,
                 y5,
                 self.A * self.a * sigm_y1_y2 - 2.0 * self.a * y3 - self.a ** 2 * y0,
                 self.A * self.a * (self.mu + self.a_2 * self.J * sigm_y0_1 + lrc + short_range_coupling)
                     - 2.0 * self.a * y4 - self.a ** 2 * y1,
                 self.B * self.b * (self.a_4 * self.J * sigm_y0_3) - 2.0 * self.b * y5 - self.b ** 2 * y2,
             ])
    
         def dfun(self, y, c, local_coupling=0.0):
             r"""
             """
             src =  local_coupling*(y[1] - y[2])[:, 0]
             y_ = y.reshape(y.shape[:-1]).T
             c_ = c.reshape(c.shape[:-1]).T
             deriv = _numba_dfun_jr(y_, c_, src,
                                 self.nu_max, self.r, self.v0, self.a, self.a_1, self.a_2, self.a_3, self.a_4,
                                 self.A, self.b, self.B, self.J, self.mu
                                 )
             return deriv.T[..., np.newaxis]
         
     @guvectorize([(float64[:],) * 17], '(n),(m)' + ',()'*14 + '->(n)', nopython=True)
     def _numba_dfun_jr(y, c,
                     src,
                     nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu,
                     dx):
     
         sigm_y1_y2 = 2.0 * nu_max[0] / (1.0 + math.exp(r[0] * (v0[0] - (y[1] - y[2]))))
         sigm_y0_1 = 2.0 * nu_max[0] / (1.0 + math.exp(r[0] * (v0[0] - (a_1[0] * J[0] * y[0]))))
         sigm_y0_3 = 2.0 * nu_max[0] / (1.0 + math.exp(r[0] * (v0[0] - (a_3[0] * J[0] * y[0]))))
         dx[0] = y[3]
         dx[1] = y[4]
         dx[2] = y[5]
         dx[3] = A[0] * a[0] * sigm_y1_y2 - 2.0 * a[0] * y[3] - a[0] ** 2 * y[0]
         dx[4] = A[0] * a[0] * (mu[0] + a_2[0] * J[0] * sigm_y0_1 + (c[0] - c[1]) + src[0]) - 2.0 * a[0] * y[4] - a[0] ** 2 * y[1]
         dx[5] = B[0] * b[0] * (a_4[0] * J[0] * sigm_y0_3) - 2.0 * b[0] * y[5] - b[0] ** 2 * y[2]

     jr = {
                'jr_A':        dict_sub['A'],
                'jr_B':        dict_sub['B'],
                'jr_a':        dict_sub['a'] / 1000,
                'jr_b':        dict_sub['b'] / 1000,
                'jr_c1':       dict_sub['c1'] / 135,
                'jr_c2':       dict_sub['c2'] / 135,
                'jr_c3':       dict_sub['c3'] / 135,
                'jr_c4':       dict_sub['c4'] / 135,
                'jr_v0':       dict_sub['v0'],
                'jr_r':        dict_sub['r'],
                'jr_nu_max':   dict_sub['vmax'] / 2000, # = 0.0025
                'jr_mu':       dict_sub['mu'],
                'jr_stvar':    4  # stvar = state variable the stimulus is applied to
        }
     print(f"- JR: {jr}")
     
     # COUPLING
     limit = 0.5
     white_matter_coupling = coupling.Sigmoidal(
         a = np.array([4 * float(config.g) / 1000]),
         cmin = np.array([-limit]),
         cmax = np.array([limit]),
         sigma = np.array([1.0])
             )
     white_matter_coupling.configure()
     print(f"- GC: {white_matter_coupling}\n")
     
     # INTEGRATOR
     dtx = config.dtx
     integ = integrators.EulerStochastic(dt=dtx)
     integ.noise.nsig = np.full((6, 200, 1), 0)
     integ.noise.nsig[4] = 0.0003
     integ.configure()  
     
     # MONITORS
     mon_raw = monitors.Raw(period = config.raw_p)
     mon_raw.configure()
     
     # INITIAL CONDITIONS
     init = np.random.rand(config.sim_duration, 6, 200, 1)

     # STIMULUS 
     stim_scaling = config.stim_scaling * dict_sub['A'] * dict_sub['a'] / 1000
     stimulus_weighting_file = np.load('./data/stim_weights.npy')
     stimulus_temporal = equations.PulseTrain()
     stimulus_temporal.parameters["onset"] = config.stim_onset
     stimulus_temporal.parameters["T"] = config.stim_frequency
     stimulus_temporal.parameters["tau"] = config.stim_length
     stimulus_temporal.parameters.update()
     print(f"- Stimulus temporal equations: {stimulus_temporal}\n")
     stimulus_weighting = stimulus_weighting_file * stim_scaling
     stim = patterns.StimuliRegion(temporal=stimulus_temporal, connectivity=white_matter, weight=stimulus_weighting)
     stim.configure()
     print(f"- Stim: {stim}\n")

     # FORWARD MODEL
     file_leadfield_matrix = './data/leadfield'
     leadfield_matrix = np.load(file_leadfield_matrix, allow_pickle=True)

     return_raw = np.zeros((len(factors_explore), len(factors_values), int(400/dtx), 200))
     return_eeg = np.zeros((len(factors_explore), len(factors_values), int(400/dtx), 62))
     return_gmfa = np.zeros((len(factors_explore), len(factors_values), int(300/dtx)))
     for ifex, fex in enumerate(factors_explore):
         for ifx, fx in enumerate(factors_values):
             jr_copy = jr.copy()
             jr_copy[fex] = float(jr_copy[fex]) * float(fx)

             # NEURAL MASS MODEL
             neuron_model = ExtendedJansenRit(
                     A = np.array(jr_copy['jr_A']),
                     B = np.array(jr_copy['jr_B']),
                     a = np.array(jr_copy['jr_a']),
                     b = np.array(jr_copy['jr_b']),
                     a_1 = np.array(jr_copy['jr_c1']),
                     a_2 = np.array(jr_copy['jr_c2']),
                     a_3 = np.array(jr_copy['jr_c3']),
                     a_4 = np.array(jr_copy['jr_c4']),
                     v0 = np.array(jr_copy['jr_v0']),
                     r =  np.array(jr_copy['jr_r']),
                     mu = np.array(jr_copy['jr_mu']),
                     nu_max = np.array(jr_copy['jr_nu_max'])
                     )
             neuron_model.stvar = np.array(jr_copy['jr_stvar'])
             neuron_model.configure()
             print(f"- Factor explored {fex}: {np.round(jr[fex], 4)} * {np.round(fx, 3)} = {np.round(jr[fex]*fx, 4)}")

             # SIMULATOR
             sim = simulator.Simulator(
                     model = neuron_model,                          
                     connectivity = white_matter,
                     coupling = white_matter_coupling,
                     integrator = integ,
                     monitors = [mon_raw],
                     initial_conditions = init,
                     stimulus = stim
                 )
             sim.configure()

             # RUN SIMULATION
             (raw_time, raw_data), = sim.run(simulation_length = sim_duration)

             # EXPORT
             raw = np.squeeze(raw_data)
             eeg = 0.0005 * leadfield_matrix.dot(raw.T).T
             return_raw[ifex][ifx] = raw[int((config.stim_onset-100)/dtx):int((config.stim_onset+300)/dtx), :]
             return_eeg[ifex][ifx] = eeg[int((config.stim_onset-100)/dtx):int((config.stim_onset+300)/dtx), :]
             return_gmfa[ifex][ifx] = gmfa(eeg, int(config.stim_onset/dtx), int((config.stim_onset+300)/dtx))

     print(f"- Ran sim: {sim}\n")

     # EXPORT
     for directory in ['raw', 'eeg', 'gmfa', 'config']:
         if not os.path.exists(f'./results/results_simulation/{directory}/'):
             os.makedirs(f'./results/results_simulation/{directory}/')


     with zipfile.ZipFile(f"./results/results_simulation/raw/{config.filename}_raw.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
         with zipf.open(f"{config.filename}_raw.npy", 'w') as f:
             np.save(f, return_raw.astype(np.float32))
     with zipfile.ZipFile(f"./results/results_simulation/eeg/{config.filename}_eeg.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
         with zipf.open(f"{config.filename}_eeg.npy", 'w') as f:
             np.save(f, return_eeg.astype(np.float32))
     with zipfile.ZipFile(f"./results/results_simulation/gmfa/{config.filename}_gmfa.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
         with zipf.open(f"{config.filename}_gmfa.npy", 'w') as f:
             np.save(f, return_gmfa.astype(np.float32))
     with zipfile.ZipFile(f"./results/results_simulation/config/{config.filename}_config.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
         with zipf.open(f"{config.filename}_config.pkl", 'w') as f:
             pickle.dump(dict(config=config), f)
         
     print(f"- Files exported\n")

if (__name__ == "__main__"):
     config = get_config()
     run_simulation(config)
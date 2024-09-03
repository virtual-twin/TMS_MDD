# ------------------------------------------------------------------------------
# functions.py
# Author: Dr. Timo Hofsähs
#
# Description: 
# This Python script is part of the code accompanying the scientific publication:
# The Virtual Brain links transcranial magnetic stimulation evoked potentials and 
# neurotransmitter changes in major depressive disorder
# Dr. Timo Hofsähs, Marius Pille, Dr. Jil Meier, Prof. Petra Ritter
# (in prep)
# 
# This code provides all functions necessary to perform the fitting in 'fitting.py'.
# The fitting method and all functions in this script except from 'gmfa' and 
# 'gmfa_timepoint' are based on the code provided with the following publication:
# Momi D, Wang Z, Griffiths JD. 2023. TMS-EEG evoked responses are driven by recurrent 
# large-scale network dynamics. eLife2023;12:e83232 DOI: https://doi.org/10.7554/eLife.83232
# Licensed under a Creative Commons Attribution license (CC-BY)
# The original code can be found at:
# https://github.com/GriffithsLab/PyTepFit/blob/main/tepfit/fit.py
#
# Copyright (c) 2024 Dr. Timo Hofsähs. All rights reserved.
#
# License: This code is licensed under the Creative Commons Attribution 4.0 International 
# License (CC-BY 4.0), which allows for redistribution, adaptation, and use in source 
# and binary forms, with or without modification, provided proper credit is given to 
# the original authors. You can view the full terms of this license at:
# https://creativecommons.org/licenses/by/4.0/
# ------------------------------------------------------------------------------

import numpy as np
import scipy.stats
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class OutputNM():
    mode_all = ['train', 'test']
    stat_vars_all = ['m', 'v']

    def __init__(self, model_name, node_size, param, fit_weights=False):
        self.loss = np.array([])
        if model_name == "JR":
            state_names = ['E', 'Ev', 'I', 'Iv', 'P', 'Pv']
            self.output_name = "eeg"
        for name in state_names + [self.output_name]:
            for m in self.mode_all:
                setattr(self, name + '_' + m, [])

        vars = [a for a in dir(param) if not a.startswith('__') and not callable(getattr(param, a))]
        for var in vars:
            if np.any(getattr(param, var)[1] > 0):
                if var != 'std_in':
                    setattr(self, var, np.array([]))
                    for stat_var in self.stat_vars_all:
                        setattr(self, var + '_' + stat_var, [])
                else:
                    setattr(self, var, [])

        self.weights = []

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class ParamsJR():
    '''Jansen & Rit neural mass model'''
    def __init__(self, model_name, **kwargs):
        if model_name == "JR":
            param = {"A ": [3.25, 0], "a": [100, 0.], "B": [22, 0], "b": [50, 0], "g": [100, 0],"c1": [135, 0.], "c2": [135 * 0.8, 0.],"c3": [135 * 0.25, 0.],"c4": [135 * 0.25, 0.],"std_in": [100, 0], "vmax": [5, 0], "v0": [6, 0], "r": [0.56, 0], "mu": [.5, 0],"speed": [2.5, 0],"k": [5, 0],"ki": [1, 0]}
        for var in param:
            setattr(self, var, param[var])
        for var in kwargs:
            setattr(self, var, kwargs[var])

def sys2nd(A, a, u, x, v):
    '''
    Jansen & Rit neural mass model equation
    A = maximum amplitude of PSP (A=EPSP, B=IPSP)
    a = maximum firing rate of populations (a = excitatory = PC, EIN; b = inhibitory = IIN)
    u = input activity
    x,v = state variable activity
    '''
    return A*a*u -2*a*v-a**2*x

def sigmoid(x, vmax, v0, r):
    '''
    Jansen & Rit neural mass model sigmoid function (potential to rate operator)
    x = input membrane potential
    vmax = maximum firing rate of all neuronal populations (default = 5 s^-1)
    v0 = PSP for which half of the maximum firing rate of the neuronal population is achieved; can be interpreted as excitability of all neuronal populations (default = 6mV)
    r = steepness at the firing threshold; represents the variance of firing thresholds within the NMM (default = 0.56 mV^-1)
    '''
    return vmax/(1+torch.exp(r*(v0-x)))


class RNNJANSEN(torch.nn.Module):
    '''
    Embedding of whole-brain simulation parameters in fitting algorithm.
    Defines parameters of the whole-brain simulation and if and to what extent they are included in the fitting.
    '''
    state_names = ['E', 'Ev', 'I', 'Iv', 'P', 'Pv']
    model_name = "JR"

    def __init__(self, 
                 input_size: int, 
                 node_size: int,
                 batch_size: int, 
                 step_size: float, 
                 output_size: int, 
                 tr: float, 
                 sc: float, 
                 lm: float, 
                 dist: float,
                 param: ParamsJR,
                 seed_int: int) -> None:
            
        super(RNNJANSEN, self).__init__()
        self.state_size = 6  # number of state variables
        self.input_size = input_size  # 1 or 2 or 3
        self.tr = tr  # tr ms (integration step 0.1 ms)
        self.step_size = torch.tensor(step_size, dtype=torch.float32)  # integration step 0.1 ms
        self.hidden_size = int(tr / step_size)
        self.batch_size = batch_size  # size of batch in ms applied at each fittin iteration
        self.node_size = node_size  # number of regions of the whole-brain simulation
        self.output_size = output_size  # number of EEG channels
        self.sc = sc  # structural connectivity weights matrix (shaped node_size^2)
        self.dist = torch.tensor(dist, dtype=torch.float32) # structural connectivity tract length matrix (shaped node_size^2)
        self.param = param # parameters of the whole-brain simulation
        self.seed_int = seed_int # number of the seed
         
        np.random.seed(self.seed_int) # define seed
        self.lm = torch.tensor(lm, dtype=torch.float32)  # leadfield matrix from sourced data to eeg

        # define if a parameter is fitted or not and add variance of default parameter if defined
        vars = [a for a in dir(param) if not a.startswith('__') and not callable(getattr(param, a))]
        for var in vars:
            if np.any(getattr(param, var)[1] > 0):
                setattr(self, var, Parameter(torch.tensor(getattr(param, var)[0] + getattr(param, var)[2] * np.random.randn(1,)[0], dtype=torch.float32)))
                dict_nv = {}
                dict_nv['m'] = getattr(param, var)[0]
                dict_nv['v'] = getattr(param, var)[1]

                dict_np = {}
                dict_np['m'] = var + '_m'
                dict_np['v'] = var + '_v'

                for key in dict_nv:
                    setattr(self, dict_np[key], Parameter(torch.tensor(dict_nv[key], dtype=torch.float32)))
            else:
                setattr(self, var, torch.tensor(getattr(param, var)[0], dtype=torch.float32))

    def forward(self, input, noise_in, noise_out, hx, hE):
        '''
        Forward function of the concurrent simulation and fitting.
        Calculates the state variables of the neural mass model for every time step.
        '''
        bound_coupling = 500  # upper and lower limit of the coupling function
        dt = self.step_size
        self.delays = (self.dist / self.speed).type(torch.int64) # delays of the coupling per region

        w = torch.exp(self.w_bb) * torch.tensor(self.sc, dtype=torch.float32) # multiplication of SC weights matrix with fitting transformation factor w_bb
        w_n = torch.log1p(0.5 * (w + w.T)) / torch.linalg.norm(torch.log1p(0.5 * (w + w.T))) # normalization, logarithmization, symmetrication of altered SC weights matrix
        self.sc_m = w_n
        dg = -torch.diag(torch.sum(w_n, axis=1)) # negative of the sum of all SC weights per region

        M = hx[:, 0:1]  # current of main population
        E = hx[:, 1:2]  # current of excitory population
        I = hx[:, 2:3]  # current of inhibitory population
        Mv = hx[:, 3:4]  # voltage of main population
        Ev = hx[:, 4:5]  # voltage of exictory population
        Iv = hx[:, 5:6]  # voltage of inhibitory population

        current_state = torch.zeros_like(hx)
        next_state = {}
        eeg_batch = []
        E_batch = []
        I_batch = []
        M_batch = []
        Ev_batch = []
        Iv_batch = []
        Mv_batch = []


        for i_batch in range(self.batch_size):
            # noiseEEG = noise_out[:, i_batch:i_batch + 1]
            for i_hidden in range(self.hidden_size):
                # Define SC matrix & coupling
                hE_new = hE.clone() # hE = history object, history of excitatory state variable
                Ed = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32) # delays of E
                Ed = hE_new.gather(1, self.delays) # delay of E as input through structural connectivity
                LEd = torch.reshape(torch.sum(w_n * torch.transpose(Ed, 0, 1), 1), (self.node_size, 1)) # weighted delayed input from E

                # Define noise
                noiseE = noise_in[:, i_hidden, i_batch, 0:1] * self.std_in
                # noiseI = noise_in[:, i_hidden, i_batch, 1:2] * self.std_in
                # noiseM = noise_in[:, i_hidden, i_batch, 2:3] * self.std_in

                # Define stimulus
                u = input[:, i_hidden:i_hidden + 1, i_batch]
                stimulus = self.k * self.ki * u

                # Define activity input for Jansen & Rit equations
                rM = sigmoid(E - I, self.vmax, self.v0, self.r)
                rE = self.c2 * sigmoid(self.c1 * M , self.vmax, self.v0, self.r)
                rI = self.c4 * sigmoid(self.c3 * M, self.vmax, self.v0, self.r)

                # Define coupling
                coupled_activity = self.g * (LEd + torch.matmul(dg, E-I))
                coupling = bound_coupling * torch.tanh(coupled_activity / bound_coupling)
                
                # Jansen & Rit neural mass model equations
                ddM = M + dt * Mv
                ddE = E + dt * Ev
                ddI = I + dt * Iv
                ddMv = Mv + dt * sys2nd(self.A, self.a, rM, M, Mv)
                ddEv = Ev + dt * sys2nd(self.A, self.a, (rE + coupling + stimulus + noiseE + self.mu), E, Ev)
                ddIv = Iv + dt * sys2nd(self.B, self.b, rI, I, Iv)

                E = ddE
                I = ddI
                M = ddM
                Ev = ddEv
                Iv = ddIv
                Mv = ddMv

                hE[:, 0] = E[:, 0] - I[:, 0] # create new time step for history of E object

            # Put M E I Mv Ev and Iv at every tr to the placeholders for checking them visually.
            M_batch.append(M)
            I_batch.append(I)
            E_batch.append(E)
            Mv_batch.append(Mv)
            Iv_batch.append(Iv)
            Ev_batch.append(Ev)

            hE = torch.cat([(E - I), hE[:, :-1]], axis=1)  # update placeholders for E buffer

            # Put the EEG signal each tr to the placeholder being used in the cost calculation.
            eeg_currently = 0.0005 * torch.matmul(self.lm, E - I)
            eeg_batch.append(eeg_currently)

        # Update the current state.
        current_state = torch.cat([M, E, I, Mv, Ev, Iv], axis=1)
        next_state['current_state'] = current_state
        next_state['eeg_batch'] = torch.cat(eeg_batch, axis=1)
        next_state['E_batch'] = torch.cat(E_batch, axis=1)
        next_state['I_batch'] = torch.cat(I_batch, axis=1)
        next_state['P_batch'] = torch.cat(M_batch, axis=1)
        next_state['Ev_batch'] = torch.cat(Ev_batch, axis=1)
        next_state['Iv_batch'] = torch.cat(Iv_batch, axis=1)
        next_state['Pv_batch'] = torch.cat(Mv_batch, axis=1)

        return next_state, hE

class Costs:
    '''
    Class that defines equations for the cost function of the fittin algorithm
    '''
    def __init__(self, method):
        self.method = method

    def cost_dist(self, sim, emp):
        losses = torch.sqrt(torch.mean((sim - emp) ** 2))
        return losses

    def cost_beamform(self, model, emp):
        corr = torch.matmul(emp, emp.T)
        corr_inv = torch.inverse(corr)
        corr_inv_s = torch.inverse(torch.matmul(model.lm.T, torch.matmul(corr_inv, model.lm)))
        W = torch.matmul(corr_inv_s, torch.matmul(model.lm.T, corr_inv))
        return torch.trace(torch.matmul(W, torch.matmul(corr, W.T)))

    def cost_r(self, logits_series_tf, labels_series_tf):
        node_size = logits_series_tf.shape[0]
        labels_series_tf_n = labels_series_tf - torch.reshape(torch.mean(labels_series_tf, 1), [node_size, 1]) # remove mean across time
        logits_series_tf_n = logits_series_tf - torch.reshape(torch.mean(logits_series_tf, 1), [node_size, 1])
        cov_sim = torch.matmul(logits_series_tf_n, torch.transpose(logits_series_tf_n, 0, 1)) # correlation
        cov_def = torch.matmul(labels_series_tf_n, torch.transpose(labels_series_tf_n, 0, 1))
        FC_sim_T = torch.matmul(torch.matmul(torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_sim)))), cov_sim), torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_sim))))) # fc for sim and empirical BOLDs
        FC_T = torch.matmul(torch.matmul(torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_def)))), cov_def), torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_def)))))
        ones_tri = torch.tril(torch.ones_like(FC_T), -1) # mask for lower triangle without diagonal
        zeros = torch.zeros_like(FC_T) # create a tensor all ones
        mask = torch.greater(ones_tri, zeros)  # boolean tensor, mask[i] = True iff x[i] > 1
        FC_tri_v = torch.masked_select(FC_T, mask) # mask out fc to vector with elements of the lower triangle
        FC_sim_tri_v = torch.masked_select(FC_sim_T, mask)
        FC_v = FC_tri_v - torch.mean(FC_tri_v) # remove the mean across the elements
        FC_sim_v = FC_sim_tri_v - torch.mean(FC_sim_tri_v)
        corr_FC = torch.sum(torch.multiply(FC_v, FC_sim_v)) * torch.reciprocal(torch.sqrt(torch.sum(torch.multiply(FC_v, FC_v)))) * torch.reciprocal(torch.sqrt(torch.sum(torch.multiply(FC_sim_v, FC_sim_v)))) # corr_coef
        losses_corr = -torch.log(0.5000 + 0.5 * corr_FC) # use surprise: corr to calculate probability and -log
        return losses_corr

    def cost_eff(self, model, sim, emp):
        if self.method == 0: # in the current version this methd is applied
            return self.cost_dist(sim, emp)
        elif self.method == 1:
            return self.cost_beamform(model, emp) + self.cost_dist(sim, emp)
        else:
            return self.cost_r(sim, emp)


class Model_fitting:
    '''Fitting algorithm equations'''
    def __init__(self, model, ts, num_epoches, cost):
        self.model = model
        self.num_epoches = num_epoches
        self.ts = ts
        self.cost = Costs(cost)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def train(self, u=0):
        '''Train function of the fitting algorithm'''
        # define some constants
        lb = 0.001
        delays_max = 500
        state_ub = 2 # lower bound for initial conditions
        state_lb = 0.5 # upper bound for initial conditions
        w_cost = 1.0 # factor to scale cost of similarity against cost of parameter change
        epoch_min = 200 # minimum amount of epochs to run, stop criterium
        r_lb = 0.85 # minimum pearson correlation value, stop criterium
        self.u = u # stimulus

        self.output_sim = OutputNM(self.model.model_name, # placeholder for output(EEG and histoty of model parameters and loss)
                                   self.model.node_size, 
                                   self.model.param)
        
        # define an optimizor(ADAM)
        optimizer = optim.Adam(self.model.parameters(), lr=0.05, eps=1e-7)

        X = torch.tensor(np.random.uniform(state_lb, state_ub, (self.model.node_size, self.model.state_size)), dtype=torch.float32) # initial conditions
        hE = torch.tensor(np.random.uniform(state_lb, state_ub, (self.model.node_size, delays_max)), dtype=torch.float32) # history object

        # define masks for geting lower triangle matrix
        mask = np.tril_indices(self.model.node_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)

        # placeholders for the history of model parameters
        fit_param = {}
        fit_sc = [self.model.sc[mask].copy()] # sc weights history
        for key, value in self.model.state_dict().items():
            fit_param[key] = [value.detach().numpy().ravel().copy()]

        loss_his = []
        num_batches = int(self.ts.shape[2] / self.model.batch_size) # define num_batches

        # empty dicts and arrays for course of parameter values, gradients, similarity metrics and loss
        parameters_fitted_initial = {}
        for key, value in self.model.state_dict().items():
            if '_' not in key:
                parameters_fitted_initial[key] = value.detach().numpy().astype(np.float32)
        parameters_fitted_export = {}
        parameters_fitted_export[-1] = parameters_fitted_initial
        # parameters_gradients_export = {}
        course_loss = np.zeros((3, self.num_epoches))
        course_cos_sim = np.zeros((self.num_epoches,))
        course_pcc = np.zeros((self.num_epoches,))
        course_sc_values = np.zeros((self.num_epoches+1, self.model.node_size, self.model.node_size))
        course_sc_values[0] = self.model.sc

        for i_epoch in range(self.num_epoches):
            eeg = self.ts[i_epoch % self.ts.shape[0]]
            # Create placeholders for the simulated EEG E I M Ev Iv and Mv of entire time series.
            for name in self.model.state_names + [self.output_sim.output_name]:
                setattr(self.output_sim, name + '_train', [])

            external = torch.tensor(np.zeros([self.model.node_size, self.model.hidden_size, self.model.batch_size]), dtype=torch.float32)

            # Perform the training in batches
            for i_batch in range(num_batches):

                optimizer.zero_grad() # Reset the gradient to zeros after update model parameters.

                # Get the input and output noises for the module.
                noise_in = torch.tensor(np.random.randn(self.model.node_size, self.model.hidden_size, self.model.batch_size, self.model.input_size), dtype=torch.float32)
                noise_out = torch.tensor(np.random.randn(self.model.node_size, self.model.batch_size), dtype=torch.float32)

                if not isinstance(self.u, int):
                    external = torch.tensor((self.u[:, :, i_batch * self.model.batch_size:(i_batch + 1) * self.model.batch_size]), dtype=torch.float32)

                # Use the model.forward() function to update next state and get simulated EEG in this batch.
                next_batch, hE_new = self.model(external, noise_in, noise_out, X, hE)

                # Get the batch of emprical EEG signal.
                ts_batch = torch.tensor((eeg.T[i_batch * self.model.batch_size:(i_batch + 1) * self.model.batch_size, :]).T, dtype=torch.float32)

                loss_prior = []
                m = torch.nn.ReLU() # define the relu function
                variables_p = [a for a in dir(self.model.param) if
                               not a.startswith('__') and not callable(getattr(self.model.param, a))]
                
                # get penalty on each fitted model parameter based on the derivation from default
                for var in variables_p:
                    if np.any(getattr(self.model.param, var)[1] > 0):
                        dict_np = {}
                        dict_np['m'] = var + '_m'
                        dict_np['v'] = var + '_v'
                        loss_prior.append(torch.sum((lb + (1 / m(self.model.get_parameter(dict_np['v'])))) * (m(self.model.get_parameter(var)) - m(self.model.get_parameter(dict_np['m']))) ** 2))
                # calculate total loss
                loss_similarity = w_cost * self.cost.cost_eff(self.model, next_batch['eeg_batch'], ts_batch) # loss from difference between simulated and empirical timeseries
                loss_complexity = sum(loss_prior) # loss from derivation of fitted parameters from default
                loss = loss_similarity + loss_complexity

                # Put the batch of the simulated EEG, E I M Ev Iv Mv in to placeholders for entire time-series.
                for name in self.model.state_names + [self.output_sim.output_name]:
                    name_next = name + '_batch'
                    tmp_ls = getattr(self.output_sim, name + '_train')
                    tmp_ls.append(next_batch[name_next].detach().numpy())
                    setattr(self.output_sim, name + '_train', tmp_ls)

                loss_his.append(loss.detach().numpy())
                loss.backward(retain_graph=True) # Calculate gradient using backward (backpropagation) method of the loss function.
                optimizer.step() # Optimize the model based on the gradient method in updating the model parameters.
                for key, value in self.model.state_dict().items(): # Put the updated model parameters into the history placeholders.
                    fit_param[key].append(value.detach().numpy().ravel().copy())
                fit_sc.append(self.model.sc_m.detach().numpy()[mask].copy()) # add newly fitted SC to list
                X = torch.tensor(next_batch['current_state'].detach().numpy(), dtype=torch.float32)
                hE = torch.tensor(hE_new.detach().numpy(), dtype=torch.float32) # update history object

            fc = np.corrcoef(self.ts.mean(0))
            tmp_ls = getattr(self.output_sim, self.output_sim.output_name + '_train')
            ts_sim = np.concatenate(tmp_ls, axis=1)
            fc_sim = np.corrcoef(ts_sim[:, 10:])

            for name in self.model.state_names + [self.output_sim.output_name]:
                tmp_ls = getattr(self.output_sim, name + '_train')
                setattr(self.output_sim, name + '_train', np.concatenate(tmp_ls, axis=1))
            self.output_sim.loss = np.array(loss_his)

            # store current SC and current SC gradients
            current_sc = np.zeros((200,200))
            current_sc_mask = np.tril_indices(200,-1)
            current_sc[current_sc_mask] = np.array(fit_sc)[-10:,:].mean(0)
            current_sc = current_sc+current_sc.T
            course_sc_values[i_epoch+1] = current_sc
            # course_sc_gradients[i_epoch] = self.model.w_bb.grad.detach().numpy()

            # store current parameter values and gradients for export
            parameters_fitted_epoch = {}
            # parameters_gradients_epoch = {}
            for key, value in self.model.state_dict().items():
                if '_' not in key:
                    parameters_fitted_epoch[key] = value.detach().numpy().astype(np.float32)
                    # parameters_gradients_epoch[key] = np.array(value.grad, dtype=np.float32)
            parameters_fitted_export[i_epoch] = parameters_fitted_epoch
            # parameters_gradients_export[i_epoch] = parameters_gradients_epoch

            # store current loss values
            course_loss[0] = loss_similarity.detach().numpy()
            course_loss[1] = loss_complexity.detach().numpy()
            course_loss[2] = loss.detach().numpy()

            # store and print current cosine similarity value
            course_cos_sim[i_epoch] = np.round(np.diag(cosine_similarity(ts_sim, self.ts.mean(0))).mean(), 4)
            print(f'epoch: {i_epoch}  -  cosine similarity: {np.round(course_cos_sim[i_epoch], 4)}')

            pcc_value = np.zeros((62,))
            for j in range(62):
                channel_emp = self.ts.mean(0).T[:,j]
                channel_sim = ts_sim.T[:,j]
                r, p = scipy.stats.pearsonr(channel_emp, channel_sim)
                pcc_value[j] = r
            course_pcc[i_epoch] = np.round(np.mean(pcc_value), 4)
            # print(f'epoch: {i_epoch}  -  pcc: {course_pcc[i_epoch]}')

            # FITTING CHECK
            # if i_epoch > 0 and i_epoch % 10 == 0:
            #     # PLOT FITTED HETEROGENEOUS PARAMETERS
            #     fig, axs = plt.subplots(len(parameters_fitted_epoch), 1, figsize=(5, 7), dpi=150)
            #     for i, key in enumerate(parameters_fitted_epoch):
            #         axs[i].plot(np.array(parameters_fitted_epoch[key]))
            #         axs[i].set_ylabel(key)
            #         yticks = axs[i].get_yticks()
            #         axs[i].set_yticklabels(['{:,.2f}'.format(y) for y in yticks])
            #     plt.tight_layout()
            #     plt.show()

            #     # PLOT FITTED SC
            #     plt.figure(figsize=(5, 5), dpi=200)
            #     plt.imshow(current_sc)
            #     plt.colorbar()
            #     plt.title('SC')
            #     plt.show()

            #     # PLOT TIMESERIES
            #     fig, axs = plt.subplots(1, 2, figsize=(8, 2), dpi=70)
            #     for j in range(200):
            #         axs[0].plot(np.arange(-100, 300), (E_train[j] - I_train[j]).T, alpha=0.2, c='red' if self.model.ki[j] != 0 else 'black')
            #     axs[0].set_title('fitted RAW')
            #     axs[0].set_xlim(-100, 300)
            #     axs[1].plot(np.arange(-100, 300), ts_sim.T)
            #     axs[1].set_title('fitted EEG')
            #     axs[1].set_xlim(-100, 300)
            #     plt.tight_layout()
            #     plt.show()

            if i_epoch > epoch_min and np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1] > r_lb:
                break

        # store outputs
        self.output_sim.weights = np.array(fit_sc)
        self.output_sim.course_parameter_values = parameters_fitted_export
        # self.output_sim.course_parameter_gradients = parameters_gradients_export
        self.output_sim.course_sc_values = course_sc_values
        # self.output_sim.course_sc_gradients = course_sc_gradients
        self.output_sim.course_loss = course_loss
        self.output_sim.course_cos_sim = course_cos_sim
        self.output_sim.course_pcc = course_pcc

        for key, value in fit_param.items():
            setattr(self.output_sim, key, np.array(value))

    def test(self, x0, he0, base_batch_num, u=0):
        '''Test function of the fitting algorithm'''
        transient_num = 10
        self.u = u

        # initial state
        X = torch.tensor(x0, dtype=torch.float32)
        hE = torch.tensor(he0, dtype=torch.float32)

        num_batches = int(self.ts.shape[2] / self.model.batch_size) + base_batch_num
        # Create placeholders for the simulated BOLD E I x f and q of entire time series.
        for name in self.model.state_names + [self.output_sim.output_name]:
            setattr(self.output_sim, name + '_test', [])

        u_hat = np.zeros((self.model.node_size, self.model.hidden_size, base_batch_num * self.model.batch_size + self.ts.shape[2]))
        u_hat[:, :, base_batch_num * self.model.batch_size:] = self.u

        # Perform the training in batches.
        for i_batch in range(num_batches):
            # Get the input and output noises for the module.
            noise_in = torch.tensor(np.random.randn(self.model.node_size, self.model.hidden_size, self.model.batch_size, self.model.input_size), dtype=torch.float32)
            noise_out = torch.tensor(np.random.randn(self.model.node_size, self.model.batch_size), dtype=torch.float32)
            external = torch.tensor((u_hat[:, :, i_batch * self.model.batch_size:(i_batch + 1) * self.model.batch_size]), dtype=torch.float32)

            # Use the model.forward() function to update next state and get simulated EEG in this batch.
            next_batch, hE_new = self.model(external, noise_in, noise_out, X, hE)

            if i_batch > base_batch_num - 1:
                for name in self.model.state_names + [self.output_sim.output_name]:
                    name_next = name + '_batch'
                    tmp_ls = getattr(self.output_sim, name + '_test')
                    tmp_ls.append(next_batch[name_next].detach().numpy())
                    setattr(self.output_sim, name + '_test', tmp_ls)

            X = torch.tensor(next_batch['current_state'].detach().numpy(), dtype=torch.float32)
            hE = torch.tensor(hE_new.detach().numpy(), dtype=torch.float32)

        fc = np.corrcoef(self.ts.mean(0))
        tmp_ls = getattr(self.output_sim, self.output_sim.output_name + '_test')
        ts_sim = np.concatenate(tmp_ls, axis=1)

        fc_sim = np.corrcoef(ts_sim[:, transient_num:])

        for name in self.model.state_names + [self.output_sim.output_name]:
            tmp_ls = getattr(self.output_sim, name + '_test')
            setattr(self.output_sim, name + '_test', np.concatenate(tmp_ls, axis=1))


# GMFA
def gmfa(data, start_time, end_time):
    '''
    Calculates the Global Mean Field Amplitude of an EEG time series for a defined time interval
    data: EEG data (n_timepoints, n_channels)
    start_time: Start index in the time series
    end_time: End index in the time series
    all: array shaped (n_timepoints, ), contains the gmfp for every time point
    '''
    interval_data = data[start_time:end_time, :]
    mean_values = np.mean(interval_data, axis=1)
    deviations = interval_data - mean_values[:, np.newaxis]
    squared_deviations = deviations ** 2
    sum_squared_deviations = np.sum(squared_deviations, axis=1)
    all = np.sqrt(sum_squared_deviations / interval_data[0].shape[0])
    return all

def gmfa_timepoint(data, timepoint):
    '''
    Calculates the Global Mean Field Amplitude of an EEG time series for a defined time interval
    data: EEG data (n_timepoints, n_channels)
    start_time: Start index in the time series
    end_time: End index in the time series

    Returns 2 arrays
    all: array shaped (n_timepoints, ), contains the gmfp for every time point
    avg: average value of all time points
    '''
    data_timepoint = data[timepoint, :]
    mean_value = np.mean(data_timepoint)
    deviations = data_timepoint - mean_value
    squared_deviations = deviations ** 2
    sum_squared_deviations = np.sum(squared_deviations)
    division = np.sqrt(sum_squared_deviations / data.shape[1])

    return division
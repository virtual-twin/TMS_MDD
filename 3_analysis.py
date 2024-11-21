# ------------------------------------------------------------------------------
# 3_analysis.py
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
# Dr. Timo Hofsähs, Marius Pille, Dr. Jil Meier, Prof. Petra Ritter
# (in prep)
# 
# This code is used to analyze the results generated with '2_simulation.py'. 
# It reads data and results, performs the statistical analysis and plots the results.
# ------------------------------------------------------------------------------

import matplotlib.cm as cm
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import pandas as pd
import pickle
import scipy
import seaborn as sns
import zipfile

from functions import gmfa, gmfa_timepoint
from matplotlib.font_manager import FontProperties
from scipy import stats


# DEFINE VARIABLES
dir_exp = f"./analysis/"
if not os.path.exists(dir_exp):
    os.makedirs(dir_exp)

subs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
runs = np.arange(0,100)
n_epochs = 150
sim_duration = 1400
dtx = 0.25
ts = np.arange(0, sim_duration / dtx)

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

factors_explore = ['jr_b', 'jr_c4']
range_low = 0.5
range_high = 1.501
range_step_size = 0.02
factors_values = np.round(np.arange(range_low, range_high, range_step_size),5)
parameter_description = ['b (1/s)', 'C4']
factors_values_b_c4 = np.vstack((factors_values * 50, factors_values * 33.75))


# SETTINGS PLOTTING
resolution=500
fig_width=15

# font
fontsize_title = 28
fontsize_title_axes = 25
fontsize_axes = 20
fontsize_label = 17
fontsize_legend = 15
font_style = 'Helvetica'
plt.rcParams["font.family"] = font_style
plt.rcParams['font.size'] = fontsize_label
plt.rcParams['figure.titlesize'] = fontsize_title
plt.rcParams['axes.titlesize'] = fontsize_title_axes
plt.rcParams['axes.labelsize'] = fontsize_axes
plt.rcParams['xtick.labelsize'] = fontsize_label
plt.rcParams['ytick.labelsize'] = fontsize_label
plt.rcParams['legend.fontsize'] = fontsize_legend

# color
cmap_custom_mult_colors = [(0, "black"),(0.1, (0.0, 0.325, 0.51)),(0.25, (0.0, 0.718, 1.0)),(0.5, (0.824, 0.0, 0.467)),(0.7, (0.969, 0.38, 0.69)),(0.9, (0.994, 0.852, 0.991)),(1, "white")]
cmap_custom_mult = mcolors.LinearSegmentedColormap.from_list('cmap_custom_mult', cmap_custom_mult_colors)
cmap_custom_mult.set_bad(color='black')
cmap_custom_binary2_colors = [(0, cmap_custom_mult(0.25)), (0.5, ((178+70)/256, (138+70)/256, 1.0)), (1, cmap_custom_mult(0.7))]
cmap_custom_binary2 = mcolors.LinearSegmentedColormap.from_list('cmap_custom_binary2', cmap_custom_binary2_colors)
cmap_custom_binary2.set_bad(color='black')
cmap_custom_stim_colors = [(0, (0.0, 0.325, 0.51)),(0.01, (0.0, 0.718, 1.0)),(0.3, (0.994, 0.852, 0.991)),(0.6, (0.969, 0.38, 0.69)),(1.0, (0.824, 0.0, 0.467)),]
cmap_custom_stim = mcolors.LinearSegmentedColormap.from_list('cmap_custom_stim', cmap_custom_stim_colors)
cmap_custom_stim.set_bad(color='black')
cmap_custom_blue_colors = [(0.0, ((52-50)/256, (165-50)/256, (248-50)/256)), (0.5, (52/256, 165/256, 248/256)), (1, ((52+25)/256, (165+25)/256, (248+25)/256))]
cmap_custom_blue = mcolors.LinearSegmentedColormap.from_list('cmap_custom_blue', cmap_custom_blue_colors)
cmap_custom_purple_colors = [(0.0, ((178-50)/256, (138-50)/256, (213-50)/256)), (0.5, (178/256, 138/256, 213/256)), (1, ((178+25)/256, (138+25)/256, (213+25)/256))]
cmap_custom_purple = mcolors.LinearSegmentedColormap.from_list('cmap_custom_purple', cmap_custom_purple_colors)
cmap_custom_magenta_colors = [(0.0, ((206-50)/256, (94-50)/256, (159-50)/256)), (0.5, (206/256, 94/256, 159/256)), (1, ((206+25)/256, (94+25)/256, (159+25)/256))]
cmap_custom_magenta = mcolors.LinearSegmentedColormap.from_list('cmap_custom_magenta', cmap_custom_magenta_colors)


# UPLOAD

# optimization results
files_missing = []
dict_fitting = {}
for isub, sub in enumerate(subs):
    dict_sub = {}
    for irun, run in enumerate(runs):
        filename = f"sub_{sub}_run_{run}"
        filepath = f"./results/results_optimization/{filename}.zip"
        if os.path.exists(filepath):
            with zipfile.ZipFile(filepath, 'r') as zipf:
                with zipf.open(f"{filename}.pkl") as f:
                    dict_run = pickle.load(f)
            dict_sub[run] = dict_run
        else:
            files_missing.append(filename)
    dict_fitting[sub] = dict_sub
print(f'Number of files missing: {len(files_missing)}')
ts_raw_fit = np.zeros((len(subs), len(runs), 400, 200), dtype=np.float32)
ts_eeg_fit = np.zeros((len(subs), len(runs), 400, 62), dtype=np.float32)
fits_pcc_all = np.zeros((len(subs), len(runs)), dtype=np.float32)
fits_cos_sim_course = np.zeros((len(subs), len(runs), n_epochs), dtype=np.float32)
for isub, sub in enumerate(subs):
    for irun, run in enumerate(runs):
        raw = dict_fitting[sub][run]['ts_raw_fitted']
        ts_raw_fit[isub, irun] = raw
        ts_eeg_fit[isub, irun] = 0.0005 * np.matmul(leadfield_matrix, raw.T).T
        fits_pcc_all[isub, irun] = dict_fitting[sub][run]['course_pcc'][-1]
        fits_cos_sim_course[isub, irun] = dict_fitting[sub][run]['course_cos_sim']


# simulation results
sim_res_gmfa = np.zeros((len(subs), len(runs), len(factors_explore), len(factors_values), int(300/dtx)), dtype=np.float32) # (20,100,2,51,1200)
for isub, sub in enumerate(subs):
    dict_sub = {}
    for irun, run in enumerate(runs):
        filename = f"sub_{sub}_run_{run}"
        filepath = f"./results/results_simulation/gmfa/{filename}_gmfa.zip"
        with zipfile.ZipFile(filepath, 'r') as zipf:
            with zipf.open(f"{filename}_gmfa.npy", 'r') as f:
                sim_res_gmfa[isub, irun] = np.load(f)
gmfa_tep_relative = np.zeros((len(factors_explore), len(runs), len(subs), len(factors_values)), dtype=np.float32) # (2, 100, 20, 6, 21) = (factors, runs, subjects, x+y values, parameter values explored)
index_default = 25
range_gmfa = [55, 275]
for ifx, fx in enumerate(factors_explore):
    for irun, run in enumerate(runs):
        for isub, sub in enumerate(subs):
            gmfa_sub_default = np.sum(sim_res_gmfa[isub, irun, ifx, index_default, int(range_gmfa[0]/dtx):int(range_gmfa[1]/dtx)])  # GMFA of TEP of the subject at default
            gmfa_sub = np.sum(sim_res_gmfa[isub, irun, ifx, :, int(range_gmfa[0]/dtx):int(range_gmfa[1]/dtx)], axis=1)  # GMFA of TEP of the subject at all values of parameter
            gmfa_tep_relative[ifx, irun, isub] = (gmfa_sub / gmfa_sub_default)*100



### METHOD PLOTS

ticks = [50, 100, 150, 200]
sub_choice = 14
sc_fitted_example = dict_fitting[subs[sub_choice]][0]['fitted_sc']
raw_fitted_example = ts_raw_fit[sub_choice, 0]
eeg_fitted_example = ts_eeg_fit[sub_choice, 0]

# empirical TEP
ts_emp_choice = empirical_timeseries[sub_choice][:,950:1300].T
cmap_empirical = cm.gray.reversed()
color = iter(cmap_empirical(np.linspace(0, 1, 120)))
plt.figure(figsize=(6,3.5), dpi=resolution)
for j in range(50):
    col = next(color)
for j in range(62):
    col = next(color)
    plt.plot(np.arange(-50,300), ts_emp_choice[:,j], color=col, alpha=0.8)
plt.axvline(x=0, color='red', linestyle='--', label='Stimulus')
plt.xlim(-50,300)
plt.legend(loc='lower right')
plt.ylabel('Amplitude (µV)')
plt.xlabel('Time (ms)')
plt.tight_layout()
plt.savefig(f"{dir_exp}methodplot_TEP_empirical_sub{sub_choice}.png", transparent=True)
plt.close()

# input SC
plt.figure(figsize=(7,4), dpi=resolution)
im = plt.imshow(sc_weights_norm, cmap=cmap_custom_mult)
plt.xlabel('Region')
plt.ylabel('Region')
cbar = plt.colorbar(im, label="Connection Weights")
cbar.set_label("Connection Weights")
plt.xticks(ticks)
plt.yticks(ticks)
plt.tight_layout()
plt.savefig(f"{dir_exp}methodplot_sc_weights.png", transparent=True)
plt.close()

# fitted SC logarithmized and 0 replaced
from matplotlib.colors import LogNorm
small_value = np.min(sc_fitted_example[np.where(sc_fitted_example != 0)])
sc_fitted_replaced = np.where(sc_fitted_example == 0, small_value, sc_fitted_example)
plt.figure(figsize=(7, 4), dpi=resolution)
im = plt.imshow(sc_fitted_replaced, cmap=cmap_custom_mult, norm=LogNorm())
plt.xlabel('Region')
plt.ylabel('Region')
cbar = plt.colorbar(im, label="Connection Weights")
cbar.set_label("Connection Weights")
cbar.ax.text(1.9, 0, '0', ha='left', va='center', transform=cbar.ax.transAxes)
plt.xticks(ticks)
plt.yticks(ticks)
plt.tight_layout()
plt.savefig(f"{dir_exp}methodplot_sc_fitted_logarithmized_0_replaced.png", transparent=True)
plt.close()

# big timeseries plot
ex_low = eeg_fitted_example[50:] * 0.75
ex_def = eeg_fitted_example[50:]
ex_high = eeg_fitted_example[50:] * 1.25
gmfa_low = [gmfa(ex_low, i, i+1) for i in range(350)]
gmfa_def = [gmfa(ex_def, i, i+1) for i in range(350)]
gmfa_high = [gmfa(ex_high, i, i+1) for i in range(350)]

fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(fig_width+7,3), dpi=resolution, gridspec_kw={'width_ratios': [1, 1, 1, 0.001, 1]})
color = iter(cmap_custom_blue.reversed()(np.linspace(0, 1, 62)))
for j in range(62):
    col = next(color)
    axs[0].plot(np.arange(-50,300), ex_high[:,j], c=col, alpha=0.8)
color = iter(cmap_custom_purple.reversed()(np.linspace(0, 1, 62)))
for j in range(62):
    col = next(color)
    axs[1].plot(np.arange(-50,300), ex_def[:,j], c=col, alpha=0.8)
color = iter(cmap_custom_magenta.reversed()(np.linspace(0, 1, 62)))
for j in range(62):
    col = next(color)
    axs[2].plot(np.arange(-50,300), ex_low[:,j], c=col, alpha=0.8)
for i in range(3):
    axs[i].set_ylim(-6.8,10.1)
axs[4].plot(np.arange(-50,300), gmfa_high, color=cmap_custom_blue(0.5), label='Low b', lw=4)
axs[4].plot(np.arange(-50,300), gmfa_def, color=cmap_custom_purple(0.5), label='Default b', lw=4)
axs[4].plot(np.arange(-50,300), gmfa_low, color=cmap_custom_magenta(0.5), label='High b', lw=4)

for i, ax in enumerate(axs):
    if i != 3:
        ax.set_ylabel('Amplitude (µV)' if i < 3 else 'GMFA')
        ax.set_xlabel('Time (ms)')
        ax.set_xlim(-50, 300)
        ax.axvline(x=0, color='black', linestyle='--', label='Stimulus')
        ax.legend(loc='upper right')
axs[3].axis('off')
plt.tight_layout()
plt.savefig(f"{dir_exp}methodplot_TEP_GMFA_sim.png", transparent=True)
plt.show()



### RESULTS PLOTS

parameter_description2 = ['b', f'$C_4$']

# main plot
fig, ax = plt.subplots(1, 2, figsize=(15,5), dpi=resolution)#, gridspec_kw={'hspace': 0.05})
for ifx, fx in enumerate(factors_explore):
    axs=ax[ifx]
    mean_per_subject = np.nanmean(gmfa_tep_relative[ifx], axis=0).T
    mean = np.nanmean(mean_per_subject, axis=1)
    std = np.nanstd(mean_per_subject, axis=1)

    axs.plot(factors_values_b_c4[ifx], mean, color="black", lw=2, label="GMFA mean")
    polygon = axs.fill_between(factors_values_b_c4[ifx], mean-std, mean+std, alpha=0.5, color='none', label="GMFA standard deviation")
    verts = np.vstack([p.vertices for p in polygon.get_paths()])
    gradient = axs.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=cmap_custom_binary2, aspect='auto', alpha=0.9,
                        extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    gradient.set_clip_path(polygon.get_paths()[0], transform=axs.transData)

    # linear regression
    x = factors_values_b_c4[ifx]
    y = mean.copy()
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * x + intercept
    p_label = np.round(p_value,3) if p_value >= 0.001 else "<0.001"
    axs.plot(factors_values_b_c4[ifx], line, linestyle='--', color='black', lw=4, label=f"Linear Regression\nr={r_value:.2f}, p={p_label}")

    axs.axhline(y=100, color='dimgray', linestyle='dotted', lw=3, label="Default GMFA (100%)")#, label='default value')    
    axs.set_xlabel(parameter_description2[ifx])
    axs.set_xlim(factors_values_b_c4[ifx][0], factors_values_b_c4[ifx][-1])
    axs.legend(loc="upper center")

ax[0].set_ylabel(f"GMFA (% of default)")
ax[1].set_ylabel(f"GMFA (% of default)")

plt.tight_layout()
plt.subplots_adjust(wspace=0.23)
plt.savefig(f'{dir_exp}results_main.png', transparent=True)
plt.show()

# b plots
sub_b_choice = 8
idx_b_low = 0
idx_b_high = -1
filename = f"sub_{sub_b_choice}_run_0"
filepath = f"./results/results_simulation/eeg/{filename}_eeg.zip"
with zipfile.ZipFile(filepath, 'r') as zipf:
    with zipf.open(f"{filename}_eeg.npy", 'r') as f:
        example_b = np.load(f)
eeg_b_low = example_b[0,idx_b_low][::4,:]
eeg_b_high = example_b[0,idx_b_high][::4,:]

fig, ax = plt.subplots(1, 4, figsize=(18,3.), dpi=300, gridspec_kw={'width_ratios': [1, 1, 0.000001, 1]})
for j, col in enumerate(cmap_custom_blue.reversed()(np.linspace(0, 1, 62))):
    ax[0].plot(np.arange(-50, 300), eeg_b_low[50:, j], c=col, alpha=0.8)
ax[0].set_title(f"EEG sub-0{sub_b_choice}, b={int(factors_values_b_c4[0,idx_b_low])}s$^-$$^1$")
for j, col in enumerate(cmap_custom_magenta.reversed()(np.linspace(0, 1, 62))):
    ax[1].plot(np.arange(-50, 300), eeg_b_high[50:, j], c=col, alpha=0.6)
ax[1].set_title(f"EEG sub-0{sub_b_choice}, b={int(factors_values_b_c4[0,idx_b_high])}s$^-$$^1$")
ax[1].sharey(ax[0])
ax[2].axis('off')
ax[3].plot(sim_res_gmfa[sub_b_choice,0,0,idx_b_low,:][::4], label=f"b={int(factors_values_b_c4[0, idx_b_low])}s$^-$$^1$", color=cmap_custom_binary2([0.1]), lw=4)
ax[3].plot(sim_res_gmfa[sub_b_choice,0,0,idx_b_high,:][::4], label=f"b={int(factors_values_b_c4[0, idx_b_high])}s$^-$$^1$", color=cmap_custom_binary2([0.9]), lw=4)
ax[3].legend(loc="upper right", fontsize=11)
ax[3].set_ylabel('GMFA')
ax[3].set_title(f"GMFA Comparison")
ax[3].set_xlim(0,300)
for i in [0, 1]:
    ax[i].axvline(x=0, color='black', linestyle='--')
    ax[i].set_xlim(-50,300)
    ax[i].set_ylabel('Amplitude (mV)')
for i in [0, 1, 3]:
    ax[i].set_xlabel("Time (ms)")
fig.tight_layout()
plt.savefig(f'{dir_exp}resultsplot_timeseries_gmfa_b_sub_{sub_b_choice}.png', transparent=True)
plt.show()

# C4 Plots
sub_c4_choice = 7
idx_c4_low = 11
idx_c4_high = -1
filename = f"sub_{sub_c4_choice}_run_0"
filepath = f"./results/results_simulation/eeg/{filename}_eeg.zip"
with zipfile.ZipFile(filepath, 'r') as zipf:
    with zipf.open(f"{filename}_eeg.npy", 'r') as f:
        example_c4 = np.load(f)
eeg_c4_low = example_c4[1,idx_c4_low][::4,:]
eeg_c4_high = example_c4[1,idx_c4_high][::4,:]

fig, ax = plt.subplots(1, 4, figsize=(18,3.), dpi=300, gridspec_kw={'width_ratios': [1, 0.000001, 1, 1]})
ax[0].plot(sim_res_gmfa[sub_c4_choice,0,1,idx_c4_low,:][::4], label=f"C$_4$={np.round(factors_values_b_c4[1, idx_c4_low],3)}", color=cmap_custom_binary2([0.1]), lw=4)
ax[0].plot(sim_res_gmfa[sub_c4_choice,0,1,idx_c4_high,:][::4], label=f"C$_4$={np.round(factors_values_b_c4[1, idx_c4_high],3)}", color=cmap_custom_binary2([0.9]), lw=4)
ax[0].legend(loc="upper right", fontsize=11)
ax[0].set_xlim(0,300)
ax[0].set_ylabel('GMFA')
ax[0].set_title(f"GMFA Comparison")
ax[1].axis('off')
for j, col in enumerate(cmap_custom_blue.reversed()(np.linspace(0, 1, 62))):
    ax[2].plot(np.arange(-50, 300), eeg_c4_low[50:, j], c=col, alpha=0.8)
ax[2].set_title(f"EEG sub-0{sub_c4_choice}, C$_4$={np.round(factors_values_b_c4[1,idx_c4_low],3)}")
for j, col in enumerate(cmap_custom_magenta.reversed()(np.linspace(0, 1, 62))):
    ax[3].plot(np.arange(-50, 300), eeg_c4_high[50:, j], c=col, alpha=0.6)
ax[3].set_title(f"EEG sub-0{sub_c4_choice}, C$_4$={np.round(factors_values_b_c4[1,idx_c4_high],3)}")
ax[3].sharey(ax[2])
for i in [2, 3]:
    ax[i].axvline(x=0, color='black', linestyle='--')
    ax[i].set_xlim(-50,300)
    ax[i].set_ylabel('Amplitude (mV)')
for i in [0, 2, 3]:
    axs = ax[i]
    axs.set_xlabel("Time (ms)")
fig.tight_layout()
plt.savefig(f'{dir_exp}resultsplot_timeseries_gmfa_c4_sub_{sub_c4_choice}.png', transparent=True)
plt.show()
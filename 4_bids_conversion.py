# ------------------------------------------------------------------------------
# 4_bids_conversion.py
# Author: Dr. Timo Hofsähs
# Based on code provided by Dr. Jil Meier
#
# Description: 
# This Python script is part of the code accompanying the scientific publication:
# The Virtual Brain links transcranial magnetic stimulation evoked potentials and 
# inhibitory neurotransmitter changes in major depressive disorder
# Dr. Timo Hofsähs, Marius Pille, Dr. Jil Meier, Prof. Petra Ritter
# (in prep)
# 
# This code converts the optimization & simulation results and the required data to the 
# Brain Imaging Data Structure (BIDS) format.
#
# BIDS:
# Gorgolewski, K., Auer, T., Calhoun, V. et al. The brain imaging data structure, 
# a format for organizing and describing outputs of neuroimaging experiments. 
# Sci Data 3, 160044 (2016). https://doi.org/10.1038/sdata.2016.44
#
# BIDS Extension for Computational Models:
# Schirner M., Ritter P., BIDS Extension Proposal 034 (BEP034): BIDS Computational 
# Model Specification, Version 1.0, https://zenodo.org/records/7962032
#
# The code to generate the .xml file for the equations was provided by Leon Martin,
# Brain Simulation Section, Charité - Universitätsmedizin Berlin.
#
# License: This code is licensed under the Creative Commons Attribution 4.0 International 
# License (CC-BY 4.0), which allows for redistribution, adaptation, and use in source 
# and binary forms, with or without modification, provided proper credit is given to 
# the original authors (cite as indicated above). You can view the full terms of this 
# license at: https://creativecommons.org/licenses/by/4.0/
# ------------------------------------------------------------------------------

import gdown
import json
import mne
import numpy as np
import os
import pandas as pd
import pickle
import requests
import scipy.io as sio
import time
import zipfile

dir_data = "./data/"
dir_results_optimization = "./results/results_optimization/"
dir_results_simulation = "./results/results_simulation/"
dir_bids = './bids/'
dirs_bids = ['coord/', 'net/', 'param/', 'eeg/', 'eq/', 'spatial/']
if not os.path.exists(dir_bids):
    os.makedirs(dir_bids)
for d in dirs_bids:
    if not os.path.exists(dir_bids + d):
        os.makedirs(dir_bids + d)
dir_coord = f"{dir_bids}coord/"
dir_net = f"{dir_bids}net/"
dir_param = f"{dir_bids}param/"
dir_eeg = f"{dir_bids}eeg/"
dir_eq = f"{dir_bids}eq/"
dir_spatial = f"{dir_bids}spatial/"
bids_desc = "desc-tmsmdd"

# simulation parameters
subs = []
for i in range(20):
    subs.append(f"{i+1:02}")
runs = np.arange(0,100)
factors_explore = ['jr_b', 'jr_c4']
range_low = 0.5
range_high = 1.501
range_step_size = 0.02
factors_values = np.arange(range_low, range_high, range_step_size)
parameter_description = ['b (1/s)', 'C4']
factors_values_b_c4 = np.vstack((factors_values * 0.05, factors_values * 0.25))
sim_duration = 1400
dtx = 0.25

# json metadata
SourceCode = ["https://github.com/the-virtual-brain/tvb-framework/tree/1.5", "/tvb-framework-1.5"]
SourceCodeVersion = "1.5"
SoftwareVersion = "1.5"
SoftwareName = "TVB"
SoftwareRepository = ["https://github.com/the-virtual-brain/tvb-framework/tree/1.5", "/tvb-framework-1.5"]
BIDSDataFolder = dir_bids
datasetIdentifier = bids_desc
NumberOfRegions = 200
data ={}
data["SoftwareName"] = SoftwareName
data["SoftwareRepository"] = SoftwareRepository
data["SoftwareVersion"] = SoftwareVersion
data["SourceCode"] = SourceCode
data["SourceCodeVersion"] = SourceCodeVersion

# create sub directories
for sub in subs:
    dir_sub = dir_bids+"sub-"+sub
    if not os.path.isdir(dir_sub):
        os.makedirs(dir_sub)
        os.makedirs(dir_sub+"/net")
        os.makedirs(dir_sub+"/ts")

# dataset description
dataset_description = {
    "Name": "tms_mdd",
    "BIDSVersion": "1.6.1-dev",
    "DatasetType": "comp",
    "Authors": ["Timo Hofsähs", "Marius Pille", "Jil Meier", "Petra Ritter"],
    "License": "We explicitly grant re-use of the overall dataset under the Creative Commons License Attribution-ShareAlike 4.0 International. The license explicitly grants reuse of all non-personal aspects of the dataset. Importantly, the personal data contained within this data set is governed by the EU General Data Protection Regulation. As a consequence, the license only applies to non-personal aspects of the dataset, for example, relating to the structure and organization of the dataset or the way the dataset was produced, but not to the personal information contained within the dataset.",
    "HowToAcknowledge": "This dataset was obtained from the EBRAINS knowlege graph and originally published in Hofsähs et al., 2024 (in prep).",
    "Funding": ["PR acknowledges support by the Virtual Research Environment at the Charite Berlin – a node of EBRAINS Health Data Cloud, by EU Horizon Europe program Horizon EBRAINS2.0 (101147319), Virtual Brain Twin (101137289), EBRAINS-PREP 101079717, AISN – 101057655, EBRAIN-Health 101058516, Digital Europe TEF-Health 101100700, EU H2020 Virtual Brain Cloud 826421, Human Brain Project SGA2 785907; Human Brain Project SGA3 945539, German Research Foundation SFB 1436 (project ID 425899996); SFB 1315 (project ID 327654276); SFB 936 (project ID 178316478; SFB-TRR 295 (project ID 424778381); SPP Computational Connectomics RI 2073/6-1, RI 2073/10-2, RI 2073/9-1; DFG Clinical Research Group BECAUSE-Y 504745852, PHRASE Horizon EIC grant 101058240; Berlin Institute of Health and Foundation Charite."],
    "ReferencesAndLinks": ["https://www.thevirtualbrain.org"],
}
with open(f"{dir_bids}dataset_description.json","w") as f:
    json.dump(dataset_description, f, indent=4)


# COORD
# _labels: brain region labels tsv
response = requests.get("https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_200Parcels_7Networks_order_info.txt")
if response.status_code == 200:
    with open(dir_data+"Schaefer2018_200Parcels_7Networks_order_info.txt", "wb") as file:
        file.write(response.content)
with open("./data/Schaefer2018_200Parcels_7Networks_order_info.txt", 'r') as file:
    region_labels_lines = file.readlines()
region_labels = region_labels_lines[::2]
region_labels = [line.strip() for line in region_labels]
file_labels_tsv = f"{dir_coord}{bids_desc}_labels.tsv"
with open(file_labels_tsv, 'w') as file:
    for label in region_labels:
        file.write(label + '\n')
# .json
data_labels ={}
data_labels["NumberOfRows"] = NumberOfRegions
data_labels["NumberOfColumns"] = NumberOfRegions
data_labels["Units"] = ''
data_labels["Description"] = "These are the region labels in Schaefer parcellation with 200 regions and 7 networks. They are the same for all subjects."
with open(BIDSDataFolder +"/coord/desc-" + datasetIdentifier + "_labels.json","w") as f:
    json.dump(data_labels, f, indent=4)

# _times: time steps tsv
time_steps = np.round(np.arange(0.25,1400.01, 0.25), 2)
file_times_tsv = f"{dir_coord}{bids_desc}_times.tsv"
with open(file_times_tsv, 'w') as file:
    for step in time_steps:
        file.write(str(step) + '\n')
# .json
data_times ={}
data_times["NumberOfRows"] = 5600
data_times["NumberOfColumns"] = 1
data_times["Units"] = 'ms'
data_times["SamplingPeriod"] = 0.25
data_times["Description"] = "RAW and EEG time steps of the simulated time series."
with open(BIDSDataFolder + "/coord/desc-" + datasetIdentifier + "_times.json","w") as f:
    json.dump(data_times, f, indent=4)

# _conv: leadfield matrix
file_leadfield_matrix = dir_data+"leadfield"
leadfield_matrix = np.load(file_leadfield_matrix, allow_pickle=True).T
file_conv_tsv = f"{dir_coord}{bids_desc}_conv.tsv"
np.savetxt(file_conv_tsv, leadfield_matrix, delimiter='\t', fmt='%s')
# .json
data_lfm ={}
data_lfm["NumberOfRows"] = NumberOfRegions
data_lfm["NumberOfColumns"] = 62
data_lfm["Description"] = f"The projection matrix (leadfield matrix) that is utilized as forward solution to get EEG from RAW timeseries data. It was originally published here: Momi D, Wang Z, Griffiths D (2023) TMS-evoked responses are driven by recurrent large-scale network dynamics eLife 12:e83232, DOI: https://doi.org/10.7554/eLife.83232, Download: https://github.com/GriffithsLab/PyTepFit"
with open(BIDSDataFolder + "/coord/desc-" + datasetIdentifier + "_conv.json","w") as f:
    json.dump(data_lfm, f, indent=4)


# EEG
# _electrodes: download and store EEG channel information
data_url = 'https://drive.google.com/drive/folders/1iwsxrmu_rnDCvKNYDwTskkCNt709MPuF'
data_files = {'all_avg.mat_avg_high_epoched': '17Eb_hMUKpbMb16CmnvaJR8b7Nxx8wQMs'}
for filename, file_id in data_files.items():
    if not os.path.exists(f'./data/{filename}'):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', f'./data/{filename}', quiet=True)
mat_file = f"{dir_data}all_avg.mat_avg_high_epoched"
epochs = mne.read_epochs(mat_file, verbose=False)
info_ch = epochs.info['chs']
electrodes = {}
for i, ch in enumerate(info_ch):
    key = ch['scanno']
    name = ch['ch_name']
    loc = ch['loc']
    electrodes[key] = name, loc
file_electrodes_tsv = f"{dir_eeg}{bids_desc}_electrodes.tsv"
electrodes_names = []
electrodes_x = []
electrodes_y = []
electrodes_z = []
for i in range(1,63):
    electrodes_names.append(electrodes[i][0])
    electrodes_x.append(electrodes[i][1][0])
    electrodes_y.append(electrodes[i][1][1])
    electrodes_z.append(electrodes[i][1][2])
dict_electrodes = {"name": electrodes_names, "x": electrodes_x, "y": electrodes_y, "z": electrodes_z}
df = pd.DataFrame(dict_electrodes)
df.to_csv(file_electrodes_tsv, sep='\t', index=False)
# .json
data_electrodes ={}
data_electrodes["NumberOfRows"] = 62
data_electrodes["NumberOfColumns"] = 4
data_electrodes["EEGChannelCount"] = 62
data_electrodes["Description"] = "The coordinates of the EEG electrodes in the MNI space."
with open(f"{dir_eeg}{bids_desc}_electrodes.json","w") as f:
    json.dump(data_electrodes, f, indent=4)


# EQ
# .xml
eq_xml_content = '''
<?xml version="1.0" ?>
<Lems xmlns="http://www.neuroml.org/lems/0.7.6" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.neuroml.org/lems/0.7.6 https://raw.githubusercontent.com/LEMS/LEMS/development/Schemas/LEMS/LEMS_v0.7.6.xsd">
  <Dimension name="voltage" m="1" l="2" t="-3" i="-1"/>
  <Dimension name="time" t="1"/>
  <Dimension name="current" i="1"/>
  <Unit symbol="mV" dimension="voltage" power="-3" scale="1.0"/>
  <Unit symbol="ms" dimension="time" power="-3" scale="1.0"/>
  <Unit symbol="mA" dimension="current" power="-12" scale="1.0"/>
  <ComponentType name="derivatives" description="Time derivates of the Neural Mass model">
    <Constant name="A" symbol="A" value="3.25" dimension="lo=2.6, hi=9.75, step=0.05" description="Maximum amplitude of EPSP mV Also called average synaptic gain "/>
    <Constant name="B" symbol="B" value="22" dimension="lo=17.6, hi=110.0, step=0.2" description="Maximum amplitude of IPSP mV Also called average synaptic gain "/>
    <Constant name="J" symbol="J" value="135" dimension="lo=65.0, hi=1350.0, step=1.0" description="Average number of synapses between populations "/>
    <Constant name="a_1" symbol="\\alpha_1\" value="1" dimension="lo=0.5, hi=1.5, step=0.1" description="Average probability constant of the number of synapses made by the pyramidal cells to the dendrites of the excitatory interneurons Jansen et al 1993 Jansen Rit 1995 It characterizes the connectivity between the PCs and EINs "/>
    <Constant name="a_2" symbol="\\alpha_2\" value="0.8" dimension="lo=0.4, hi=1.2, step=0.1" description="Average probability constant of the number of synapses made by the EINs to the dendrites of the PCs Jansen et al 1993 Jansen Rit 1995 It characterizes the excitatory connectivity between the EINs and PCs "/>
    <Constant name="a_3" symbol="\\alpha_3\" value="0.25" dimension="lo=0.125, hi=0.375, step=0.005" description="Average probability constant of the number of synapses made by the PCs to the dendrites of the IINs Jansen et al 1993 Jansen Rit 1995 It characterizes the connectivity between the PCs and inhibitory IINs "/>
    <Constant name="a_4" symbol="\\alpha_4\" value="0.25" dimension="lo=0.125, hi=0.375, step=0.005" description="Average probability constant of the number of synapses made by the IINs to the dendrites of the PCs Jansen et al 1993 Jansen Rit 1995 It characterizes the connectivity between the IINs and PCs "/>
    <Constant name="a" symbol="a" value="0.1" dimension="lo=0.05, hi=0.15, step=0.01" description="Rate constant of the excitatory post synaptic potential EPSP or average synaptic gain Jansen et al 1993 Jansen Rit 1995 van Rotterdam et al 1982 It is interpreted as being the lumped representation of the sum of the reciprocal of the time constant of passive membrane and all other spatially distributed delays including temporal dispersion in the afferent tract synaptic diffusion and resistive capacitive delay in the dendritic network Freeman 1975 Jansen et al 1993 Note In TVB the parameter is converted in ms 1 "/>
    <Constant name="b" symbol="b" value="0.05" dimension="lo=0.025, hi=0.075, step=0.005" description="Rate constant of the inhibitory post synaptic potential IPSP or average synaptic gain Jansen et al 1993 Jansen Rit 1995 van Rotterdam et al 1982 It is interpreted as being the lumped representation of the sum of the reciprocal of the time constant of passive membrane and all other spatially distributed delays including temporal dispersion in the afferent tract synaptic diffusion and resistive capacitive delay in the dendritic network Freeman 1975 Note In TVB the parameter is converted in ms 1 "/>
    <Constant name="mu" symbol="\\mu" value="0.22" dimension="lo=0.0, hi=0.22, step=0.01" description="Mean excitatory external input to the derivative of the state variable y4 JR PCs represented by a pulse density that consists of activity originating from adjacent and more distant cortical columns as well as from subcortical structures e g thalamus "/>
    <Constant name="nu_max" symbol="\\nu_{max}" value="0.0025" dimension="lo=0.00125, hi=0.00375, step=0.00001" description="Asymptotic of the sigmoid function Sigm JR corresponds to the maximum firing rate of the neural populations "/>
    <Constant name="r" symbol="r" value="0.56" dimension="lo=0.28, hi=0.84, step=0.01" description="Steepness or gain parameter of the sigmoid function Sigm JR Jansen et al 1993 Jansen Rit 1995 "/>
    <Constant name="v0" symbol="v_0" value="5.52" dimension="lo=3.12, hi=6.0, step=0.02" description="Average firing threshold PSP for which half of the firing rate is achieved Note The usual value for this parameter is 6 0 Jansen et al 1993 Jansen Rit 1995 "/>
    <Exposure name="y0" dimension="None" description="First state variable of the first Jansen Rit population PCs Jansen et al 1993 Jansen Rit 1995 It represents the averaged excitatory post synaptic membrane potentials from the PCs "/>
    <Exposure name="y1" dimension="None" description="First state variable of the second Jansen Rit population EINs Jansen et al 1993 Jansen Rit 1995 It represents the averaged excitatory post synaptic membrane potentials from the EINs "/>
    <Exposure name="y1 - y2" dimension="None"/>
    <Exposure name="y2" dimension="None" description="First state variable of the third Jansen Rit population IINs Jansen et al 1993 Jansen Rit 1995 It represents the averaged inhibitory post synaptic membrane potentials from the IINs "/>
    <Exposure name="y3" dimension="None" description="Second state variable of the first Jansen Rit population excitatory PCs Jansen et al 1993 Jansen Rit 1995 "/>
    <Dynamics>
      <StateVariable name="y0" dimension="-1.0, 1.0"/>
      <StateVariable name="y1" dimension="-500.0, 500.0"/>
      <StateVariable name="y2" dimension="-50.0, 50.0"/>
      <StateVariable name="y3" dimension="-6.0, 6.0"/>
      <StateVariable name="y4" dimension="-20.0, 20.0"/>
      <StateVariable name="y5" dimension="-500.0, 500.0"/>
      <DerivedVariable name="sigm_y1_y2" exposure="sigm_y1_y2" value="2.0*nu_max/(exp(r*(v0 - (y1 - y2))) + 1.0)"/>
      <DerivedVariable name="sigm_y0_3" exposure="sigm_y0_3" value="2.0*nu_max/(exp(r*(-J*a_3*y0 + v0)) + 1.0)"/>
      <DerivedVariable name="sigm_y0_1" exposure="sigm_y0_1" value="2.0*nu_max/(exp(r*(-J*a_1*y0 + v0)) + 1.0)"/>
      <DerivedVariable name="short_range_coupling" exposure="short_range_coupling" value="local_coupling*(y1 - y2)"/>
      <TimeDerivative variable="y0dot" value="y3"/>
      <TimeDerivative variable="y1dot" value="y4"/>
      <TimeDerivative variable="y2dot" value="y5"/>
      <TimeDerivative variable="y3dot" value="A*a*sigm_y1_y2 - a^2*y0 - 2.0*a*y3"/>
      <TimeDerivative variable="y4dot" value="A*a*(J*a_2*sigm_y0_1 + c_pop0 + mu + short_range_coupling) - a^2*y1 - 2.0*a*y4"/>
      <TimeDerivative variable="y5dot" value="B*J*a_4*b*sigm_y0_3 - b^2*y2 - 2.0*b*y5"/>
    </Dynamics>
  </ComponentType>
  <ComponentType name="coupling_function" description="Sigmoidal Coupling Function">
    <Parameter name="g_ij" dimension="0" description="connectivity weights matrix"/>
    <Parameter name="x_i" dimension="0" description="current state"/>
    <Parameter name="x_j" dimension="0" description="delayed state"/>
    <DerivedParameter name="c_pop0" description="" value="global_coupling"/>
    <Constant name="cmin" value="-1.0" dimension="lo=-1000.0, hi=1000.0, step=10.0" description="Minimum of the Sigmoid function"/>
    <Constant name="cmax" value="1.0" dimension="lo=-1000.0, hi=1000.0, step=10.0" description="Maximum of the Sigmoid function"/>
    <Constant name="midpoint" value="0.0" dimension="lo=-1000.0, hi=1000.0, step=10.0" description="Midpoint of the linear portion of the sigmoid"/>
    <Constant name="a" value="1.0" dimension="lo=0.01, hi=1000.0, step=10.0" description="Scaling of sigmoidal"/>
    <Constant name="sigma" value="230.0" dimension="lo=0.01, hi=1000.0, step=10.0" description="Standard deviation of sigmoidal"/>
    <Dynamics>
      <DerivedVariable name="sum" value="(g_ij * x_j).sum(axis=2)"/>
      <DerivedVariable name="post" value="cmin + ((cmax - cmin) / (1.0 + exp(-a *((sum - midpoint) / sigma))))"/>
    </Dynamics>
  </ComponentType>
</Lems>
'''
with open(f"{dir_eq}{bids_desc}_eq.xml", "w") as f:
    f.write(eq_xml_content)
# .json
data["Description"] = "These are the equations to simulate the time series with the Jansen & Rit neural mass model."
with open(BIDSDataFolder + "/eq/desc-" + datasetIdentifier + "_eq.json","w") as f:
    json.dump(data, f, indent=4)



# NET
# _distances: tract lengths
file_tract_lengths = dir_data+"Schaefer2018_200Parcels_7Networks_distance.csv"
tract_lengths = pd.read_csv(file_tract_lengths, header=None, sep=' ').values
file_distances_tsv = f"{dir_net}{bids_desc}_distances.tsv"
np.savetxt(file_distances_tsv, tract_lengths, delimiter='\t', fmt='%s')
# .json
data_sc ={}
data_sc["NumberOfRows"] = NumberOfRegions
data_sc["NumberOfColumns"] = NumberOfRegions
data_sc["CoordsRows"] = ["../../coord/desc-" + datasetIdentifier + "_labels.json", "../../coord/desc-" + datasetIdentifier + "_nodes.json"]
data_sc["CoordsColumns"] = ["../../coord/desc-" + datasetIdentifier + "_labels.json", "../../coord/desc-" + datasetIdentifier + "_nodes.json"]
data_sc["Description"] = "These tract lengths were derived from DTI data in mm, averaged over 400 individuals from the Human Connectome Project. It was originally published here: Momi D, Wang Z, Griffiths D (2023) TMS-evoked responses are driven by recurrent large-scale network dynamics eLife 12:e83232, DOI: https://doi.org/10.7554/eLife.83232, Download: https://github.com/GriffithsLab/PyTepFit"
with open(f"{dir_net}{bids_desc}_distances.json","w") as f:
    json.dump(data_sc, f, indent=4)
# _weights: tract weights
file_tract_weights = dir_data+"Schaefer2018_200Parcels_7Networks_count.csv"
tract_weights = pd.read_csv(file_tract_weights, header=None, sep=' ').values
file_weights_tsv = f"{dir_net}{bids_desc}_weights.tsv"
np.savetxt(file_weights_tsv, tract_weights, delimiter='\t', fmt='%s')
# .json
data_sc["Description"] = "This SC was derived from streamlines in DTI data, averaged over 400 individuals from the Human Connectome Project. It was originally published here: Momi D, Wang Z, Griffiths D (2023) TMS-evoked responses are driven by recurrent large-scale network dynamics eLife 12:e83232, DOI: https://doi.org/10.7554/eLife.83232, Download: https://github.com/GriffithsLab/PyTepFit"
with open(f"{dir_net}{bids_desc}_weights.json","w") as f:
    json.dump(data_sc, f, indent=4)


# PARAM
import math
from numba import guvectorize, float64
from tvb.simulator.lab import *
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
    # dx[4] = A[0] * a[0] * (mu[0] + a_2[0] * J[0] * sigm_y0_1 + c[0] + src[0]) - 2.0 * a[0] * y[4] - a[0] ** 2 * y[1]
    dx[4] = A[0] * a[0] * (mu[0] + a_2[0] * J[0] * sigm_y0_1 + (c[0] - c[1]) + src[0]) - 2.0 * a[0] * y[4] - a[0] ** 2 * y[1]
    dx[5] = B[0] * b[0] * (a_4[0] * J[0] * sigm_y0_3) - 2.0 * b[0] * y[5] - b[0] ** 2 * y[2]
neuron_model = ExtendedJansenRit(b = np.array(0.025))
neuron_model.configure()
# _param: parameter xml files
# b
for ib, b in enumerate(factors_values_b_c4[0]):
    file_param_xml = f"{dir_param}{bids_desc}_b-{np.round(b,3)}-a_4-0.25_param.xml"
    with open(file_param_xml,"w") as f: ##
            f.write((
                  "<?xml version=\"1.0\" ?>\n<Lems xmlns=\"http://www.neuroml.org/lems/0.7.6\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://www.neuroml.org/lems/0.7.6 https://raw.githubusercontent.com/LEMS/LEMS/development/Schemas/LEMS/LEMS_v0.7.6.xsd\">\n    <ComponentType name=\"global_parameters\">\n    "
                  "\t<Constant name=\"A\" symbol=\"A\" value=\"3.25\" dimension=\"lo=2.6, hi=9.75, step=0.05\" description=\"Maximum amplitude of EPSP mV Also called average synaptic gain \"/>\n    "
                  "\t<Constant name=\"B\" symbol=\"B\" value=\"22\" dimension=\"lo=17.6, hi=110.0, step=0.2\" description=\"Maximum amplitude of IPSP mV Also called average synaptic gain \"/>\n    "
                  "\t<Constant name=\"J\" symbol=\"J\" value=\"135\" dimension=\"lo=65.0, hi=1350.0, step=1.0\" description=\"Average number of synapses between populations \"/>\n    "
                  "\t<Constant name=\"a_1\" symbol=\"\\alpha_1\" value=\"1\" dimension=\"lo=0.5, hi=1.5, step=0.1\" description=\"Average probability constant of the number of synapses made by the pyramidal cells to the dendrites of the excitatory interneurons Jansen et al 1993 Jansen Rit 1995 It characterizes the connectivity between the PCs and EINs \"/>\n    "
                  "\t<Constant name=\"a_2\" symbol=\"\\alpha_2\" value=\"0.8\" dimension=\"lo=0.4, hi=1.2, step=0.1\" description=\"Average probability constant of the number of synapses made by the EINs to the dendrites of the PCs Jansen et al 1993 Jansen Rit 1995 It characterizes the excitatory connectivity between the EINs and PCs \"/>\n    "
                  "\t<Constant name=\"a_3\" symbol=\"\\alpha_3\" value=\"0.25\" dimension=\"lo=0.125, hi=0.375, step=0.005\" description=\"Average probability constant of the number of synapses made by the PCs to the dendrites of the IINs Jansen et al 1993 Jansen Rit 1995 It characterizes the connectivity between the PCs and inhibitory IINs \"/>\n    "
                  "\t<Constant name=\"a_4\" symbol=\"\\alpha_4\" value=\"0.25\" dimension=\"lo=0.125, hi=0.375, step=0.005\" description=\"Average probability constant of the number of synapses made by the IINs to the dendrites of the PCs Jansen et al 1993 Jansen Rit 1995 It characterizes the connectivity between the IINs and PCs \"/>\n    "
                  "\t<Constant name=\"a\" symbol=\"a\" value=\"0.1\" dimension=\"lo=0.05, hi=0.15, step=0.01\" description=\"Rate constant of the excitatory post synaptic potential EPSP or average synaptic gain Jansen et al 1993 Jansen Rit 1995 van Rotterdam et al 1982 It is interpreted as being the lumped representation of the sum of the reciprocal of the time constant of passive membrane and all other spatially distributed delays including temporal dispersion in the afferent tract synaptic diffusion and resistive capacitive delay in the dendritic network Freeman 1975 Jansen et al 1993 Note In TVB the parameter is converted in ms 1 \"/>\n    "
                  f"\t<Constant name=\"b\" symbol=\"b\" value=\"{np.round(b,3)}\" dimension=\"lo=0.025, hi=0.075, step=0.005\" description=\"Rate constant of the inhibitory post synaptic potential IPSP or average synaptic gain Jansen et al 1993 Jansen Rit 1995 van Rotterdam et al 1982 It is interpreted as being the lumped representation of the sum of the reciprocal of the time constant of passive membrane and all other spatially distributed delays including temporal dispersion in the afferent tract synaptic diffusion and resistive capacitive delay in the dendritic network Freeman 1975 Note In TVB the parameter is converted in ms 1 \"/>\n    "
                  "\t<Constant name=\"mu\" symbol=\"\mu\" value=\"0.22\" dimension=\"lo=0.0, hi=0.22, step=0.01\" description=\"Mean excitatory external input to the derivative of the state variable y4 JR PCs represented by a pulse density that consists of activity originating from adjacent and more distant cortical columns as well as from subcortical structures e g thalamus \"/>\n    "
                  "\t<Constant name=\"nu_max\" symbol=\"\\nu_{max}\" value=\"0.0025\" dimension=\"lo=0.00125, hi=0.00375, step=0.00001\" description=\"Asymptotic of the sigmoid function Sigm JR corresponds to the maximum firing rate of the neural populations \"/>\n    "
                  "\t<Constant name=\"r\" symbol=\"r\" value=\"0.56\" dimension=\"lo=0.28, hi=0.84, step=0.01\" description=\"Steepness or gain parameter of the sigmoid function Sigm JR Jansen et al 1993 Jansen Rit 1995 \"/>\n    "
                  "\t<Constant name=\"v0\" symbol=\"v_0\" value=\"5.52\" dimension=\"lo=3.12, hi=6.0, step=0.02\" description=\"Average firing threshold PSP for which half of the firing rate is achieved Note The usual value for this parameter is 6 0 Jansen et al 1993 Jansen Rit 1995 \"/>\n    "
                  "\t<Constant name=\"cmin\" value=\"-0.5\" dimension=\"lo=-1000.0, hi=1000.0, step=10.0\" description=\"Minimum of the Sigmoid function\"/>\n    "
                  "\t<Constant name=\"cmax\" value=\"0.5\" dimension=\"lo=-1000.0, hi=1000.0, step=10.0\" description=\"Maximum of the Sigmoid function\"/>\n    "
                  "\t<Constant name=\"midpoint\" value=\"0.0\" dimension=\"lo=-1000.0, hi=1000.0, step=10.0\" description=\"Midpoint of the linear portion of the sigmoid\"/>\n    "
                  "\t<Constant name=\"a\" value=\"4.0\" dimension=\"lo=0.01, hi=1000.0, step=10.0\" description=\"Scaling of sigmoidal\"/>\n    "
                  "\t<Constant name=\"sigma\" value=\"230.0\" dimension=\"lo=0.01, hi=1000.0, step=10.0\" description=\"Standard deviation of sigmoidal\"/>\n    "
                  "\t<Constant name=\"speed\" value=\"2.5\" dimension=\"\"/>\n    "
                  "</ComponentType>\n</Lems>"))
# c4
for ic4, c4 in enumerate(factors_values_b_c4[1]):
    file_param_xml = f"{dir_param}{bids_desc}_b-0.05-a_4-{np.round(c4,3)}_param.xml"
    with open(file_param_xml,"w") as f: ##
            f.write((
                  "<?xml version=\"1.0\" ?>\n<Lems xmlns=\"http://www.neuroml.org/lems/0.7.6\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://www.neuroml.org/lems/0.7.6 https://raw.githubusercontent.com/LEMS/LEMS/development/Schemas/LEMS/LEMS_v0.7.6.xsd\">\n    <ComponentType name=\"global_parameters\">\n    "
                  "\t<Constant name=\"A\" symbol=\"A\" value=\"3.25\" dimension=\"lo=2.6, hi=9.75, step=0.05\" description=\"Maximum amplitude of EPSP mV Also called average synaptic gain \"/>\n    "
                  "\t<Constant name=\"B\" symbol=\"B\" value=\"22\" dimension=\"lo=17.6, hi=110.0, step=0.2\" description=\"Maximum amplitude of IPSP mV Also called average synaptic gain \"/>\n    "
                  "\t<Constant name=\"J\" symbol=\"J\" value=\"135\" dimension=\"lo=65.0, hi=1350.0, step=1.0\" description=\"Average number of synapses between populations \"/>\n    "
                  "\t<Constant name=\"a_1\" symbol=\"\\alpha_1\" value=\"1\" dimension=\"lo=0.5, hi=1.5, step=0.1\" description=\"Average probability constant of the number of synapses made by the pyramidal cells to the dendrites of the excitatory interneurons Jansen et al 1993 Jansen Rit 1995 It characterizes the connectivity between the PCs and EINs \"/>\n    "
                  "\t<Constant name=\"a_2\" symbol=\"\\alpha_2\" value=\"0.8\" dimension=\"lo=0.4, hi=1.2, step=0.1\" description=\"Average probability constant of the number of synapses made by the EINs to the dendrites of the PCs Jansen et al 1993 Jansen Rit 1995 It characterizes the excitatory connectivity between the EINs and PCs \"/>\n    "
                  "\t<Constant name=\"a_3\" symbol=\"\\alpha_3\" value=\"0.25\" dimension=\"lo=0.125, hi=0.375, step=0.005\" description=\"Average probability constant of the number of synapses made by the PCs to the dendrites of the IINs Jansen et al 1993 Jansen Rit 1995 It characterizes the connectivity between the PCs and inhibitory IINs \"/>\n    "
                  f"\t<Constant name=\"a_4\" symbol=\"\\alpha_4\" value=\"{np.round(c4,3)}\" dimension=\"lo=0.125, hi=0.375, step=0.005\" description=\"Average probability constant of the number of synapses made by the IINs to the dendrites of the PCs Jansen et al 1993 Jansen Rit 1995 It characterizes the connectivity between the IINs and PCs \"/>\n    "
                  "\t<Constant name=\"a\" symbol=\"a\" value=\"0.1\" dimension=\"lo=0.05, hi=0.15, step=0.01\" description=\"Rate constant of the excitatory post synaptic potential EPSP or average synaptic gain Jansen et al 1993 Jansen Rit 1995 van Rotterdam et al 1982 It is interpreted as being the lumped representation of the sum of the reciprocal of the time constant of passive membrane and all other spatially distributed delays including temporal dispersion in the afferent tract synaptic diffusion and resistive capacitive delay in the dendritic network Freeman 1975 Jansen et al 1993 Note In TVB the parameter is converted in ms 1 \"/>\n    "
                  "\t<Constant name=\"b\" symbol=\"b\" value=\"0.05\" dimension=\"lo=0.025, hi=0.075, step=0.005\" description=\"Rate constant of the inhibitory post synaptic potential IPSP or average synaptic gain Jansen et al 1993 Jansen Rit 1995 van Rotterdam et al 1982 It is interpreted as being the lumped representation of the sum of the reciprocal of the time constant of passive membrane and all other spatially distributed delays including temporal dispersion in the afferent tract synaptic diffusion and resistive capacitive delay in the dendritic network Freeman 1975 Note In TVB the parameter is converted in ms 1 \"/>\n    "
                  "\t<Constant name=\"mu\" symbol=\"\mu\" value=\"0.22\" dimension=\"lo=0.0, hi=0.22, step=0.01\" description=\"Mean excitatory external input to the derivative of the state variable y4 JR PCs represented by a pulse density that consists of activity originating from adjacent and more distant cortical columns as well as from subcortical structures e g thalamus \"/>\n    "
                  "\t<Constant name=\"nu_max\" symbol=\"\\nu_{max}\" value=\"0.0025\" dimension=\"lo=0.00125, hi=0.00375, step=0.00001\" description=\"Asymptotic of the sigmoid function Sigm JR corresponds to the maximum firing rate of the neural populations \"/>\n    "
                  "\t<Constant name=\"r\" symbol=\"r\" value=\"0.56\" dimension=\"lo=0.28, hi=0.84, step=0.01\" description=\"Steepness or gain parameter of the sigmoid function Sigm JR Jansen et al 1993 Jansen Rit 1995 \"/>\n    "
                  "\t<Constant name=\"v0\" symbol=\"v_0\" value=\"5.52\" dimension=\"lo=3.12, hi=6.0, step=0.02\" description=\"Average firing threshold PSP for which half of the firing rate is achieved Note The usual value for this parameter is 6 0 Jansen et al 1993 Jansen Rit 1995 \"/>\n    "
                  "\t<Constant name=\"cmin\" value=\"-0.5\" dimension=\"lo=-1000.0, hi=1000.0, step=10.0\" description=\"Minimum of the Sigmoid function\"/>\n    "
                  "\t<Constant name=\"cmax\" value=\"0.5\" dimension=\"lo=-1000.0, hi=1000.0, step=10.0\" description=\"Maximum of the Sigmoid function\"/>\n    "
                  "\t<Constant name=\"midpoint\" value=\"0.0\" dimension=\"lo=-1000.0, hi=1000.0, step=10.0\" description=\"Midpoint of the linear portion of the sigmoid\"/>\n    "
                  "\t<Constant name=\"a\" value=\"4.0\" dimension=\"lo=0.01, hi=1000.0, step=10.0\" description=\"Scaling of sigmoidal\"/>\n    "
                  "\t<Constant name=\"sigma\" value=\"230.0\" dimension=\"lo=0.01, hi=1000.0, step=10.0\" description=\"Standard deviation of sigmoidal\"/>\n    "
                  "\t<Constant name=\"speed\" value=\"2.5\" dimension=\"\"/>\n    "
                  "</ComponentType>\n</Lems>"))    
# .json
for ib, b in enumerate(factors_values_b_c4[0]):
    file_param_json= f"{dir_param}{bids_desc}_b-{np.round(b,3)}-a_4-0.25_param.json"
    data["Description"] = f"These are the global parameter values for the Jansen & Rit neural mass model model. All values are default, except b (={np.round(b,3)})."
    with open(file_param_json,"w") as f:
        json.dump(data, f, indent=4)
for ic4, c4 in enumerate(factors_values_b_c4[1]):
    file_param_json = f"{dir_param}{bids_desc}_b-0.05-a_4-{np.round(c4,3)}_param.json"
    data["Description"] = f"These are the global parameter values for the Jansen & Rit neural mass model model. All values are default, except c4 (={np.round(c4,3)})."
    with open(file_param_json,"w") as f:
        json.dump(data, f, indent=4)


# SPATIAL
# _stim-node: store spatial stimulus distribution
stim_weighting = np.load(dir_data+"stim_weights.npy")
file_stim_node_tsv = f"{dir_spatial}{bids_desc}_stim-node.tsv"
with open(file_stim_node_tsv, 'w') as file:
    for node in stim_weighting:
        file.write(str(node) + '\n')
# .json
data_stimnode ={}
data_stimnode["NumberOfRows"] = NumberOfRegions
data_stimnode["NumberOfColumns"] = 1
data_stimnode["RegionLabels"] = f"{dir_coord}{bids_desc}_labels.tsv"
data_stimnode["Description"] = "This array contains the TMS stimulus strength per region, derived from the SimNIBS software. It was originally published here: Momi D, Wang Z, Griffiths D (2023) TMS-evoked responses are driven by recurrent large-scale network dynamics eLife 12:e83232, DOI: https://doi.org/10.7554/eLife.83232, Download: https://github.com/GriffithsLab/PyTepFit"
with open(f"{dir_spatial}{bids_desc}_stim-node.json","w") as f:
    json.dump(data_stimnode, f, indent=4)


# SUB-XX
# net/_weights: store subject-individual SC
for isub, sub in enumerate(subs):
    dir_target_sub = f"{dir_bids}sub-{sub}/"
    for run in runs:
        dict_optimized_file = f"sub_{str(isub)}_run_{run}"
        with zipfile.ZipFile(f'{dir_results_optimization}{dict_optimized_file}.zip', 'r') as zipf:
            with zipf.open(f"{dict_optimized_file}.pkl") as f:
                dict_run = pickle.load(f)
                sub_sc = dict_run["fitted_sc"]
        filename_tsv = f"sub-{sub}_{bids_desc}_run-{str(run)}_weights.tsv"
        file_weights_tsv = f"{dir_target_sub}net/{filename_tsv}"
        np.savetxt(file_weights_tsv, sub_sc, delimiter='\t', fmt='%s')
        with zipfile.ZipFile(f"{file_weights_tsv}.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(file_weights_tsv, os.path.basename(file_weights_tsv))
        os.remove(file_weights_tsv)
        # .json
        data_sc ={}
        data_sc["NumberOfRows"] = NumberOfRegions
        data_sc["NumberOfColumns"] = NumberOfRegions
        data_sc["CoordsRows"] = ["../../coord/desc-" + datasetIdentifier + "_labels.json", "../../coord/desc-" + datasetIdentifier + "_nodes.json"]
        data_sc["CoordsColumns"] = ["../../coord/desc-" + datasetIdentifier + "_labels.json", "../../coord/desc-" + datasetIdentifier + "_nodes.json"]
        data_sc["Description"] = f"These are the structural connectivity weights of subject {sub} after optimization repetition number {run}."
        with open(f"{dir_target_sub}net/sub-{sub}_{bids_desc}_run-{str(run)}_weights.json","w") as f:
            json.dump(data_sc, f, indent=4)

# ts/-raw_stimuli: store raw timeseries
time_start = time.time()
data_ts = data.copy()
data_ts["NumberOfRows"] = int(400 / dtx)
data_ts["CoordsRows"] = f"../../coord/{bids_desc}_times.tsv"
data_ts["ModelEq"] = f"../../eq/{bids_desc}_eq.xml"
data_ts["SamplingPeriod"] = 400/1000
data_ts["SamplingFrequency"] = int(1000 / dtx)
for isub, sub in enumerate(subs):
    dir_target_sub = f"{dir_bids}sub-{sub}/"
    for run in runs:
        # raw
        with zipfile.ZipFile(f"{dir_results_simulation}raw/sub_{isub}_run_{run}_raw.zip", 'r') as zipf:
            with zipf.open(f"sub_{isub}_run_{run}_raw.npy", 'r') as f:
                sim_raw_ts = np.load(f)
        # eeg
        with zipfile.ZipFile(f"{dir_results_simulation}eeg/sub_{isub}_run_{run}_eeg.zip", 'r') as zipf:
            with zipf.open(f"sub_{isub}_run_{run}_eeg.npy", 'r') as f:
                sim_eeg_ts = np.load(f)
        data_ts["Network"] = [f"../net/sub-{sub}_{bids_desc}_run-{run}_weights.json", f"../../net/{bids_desc}_distances.json"]
        for ival, val in enumerate(factors_values_b_c4[0]):
            b = str(np.round(val, 3))
            c4 = str(np.round(factors_values_b_c4[1][ival], 3))
            # b - RAW
            file_raw_stimuli_tsv = f"{dir_target_sub}ts/sub-{sub}_{bids_desc}_run-{str(run)}_b-{b}-a_4-0.25-raw_stimuli.tsv"
            np.savetxt(file_raw_stimuli_tsv, sim_raw_ts[0,ival], delimiter='\t', fmt='%s')
            with zipfile.ZipFile(f"{file_raw_stimuli_tsv}.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(file_raw_stimuli_tsv, os.path.basename(file_raw_stimuli_tsv))
            os.remove(file_raw_stimuli_tsv)
            # .json
            data_ts["ModelParam"] = f"../../param/{bids_desc}_b-{b}-a_4-0.25_param.xml"
            data_ts["Description"] = f"This is the time series for subject {sub} with RAW monitor from TVB with Jansen & Rit parameter b={b}"
            data_ts["NumberOfColumns"] = 200
            data_ts["CoordsColumns"] = [f"../../coord/{bids_desc}_labels.tsv"]
            data_ts = dict(sorted(data_ts.items()))
            with open(f"{dir_target_sub}ts/sub-{sub}_{bids_desc}_run-{str(run)}_b-{b}-a_4-0.25-raw_stimuli.json","w") as f:
                json.dump(data_ts, f, indent=4)
            # b - EEG
            file_eeg_stimuli_tsv = f"{dir_target_sub}ts/sub-{sub}_{bids_desc}_run-{str(run)}_b-{b}-a_4-0.25-eeg_stimuli.tsv"
            np.savetxt(file_eeg_stimuli_tsv, sim_eeg_ts[0,ival], delimiter='\t', fmt='%s')
            with zipfile.ZipFile(f"{file_eeg_stimuli_tsv}.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(file_eeg_stimuli_tsv, os.path.basename(file_eeg_stimuli_tsv))
            os.remove(file_eeg_stimuli_tsv)
            # .json
            data_ts["Description"] = f"This is the time series for subject {sub} with EEG monitor from TVB with Jansen & Rit parameter b={b}"
            data_ts["NumberOfColumns"] = 62
            data_ts["CoordsColumns"] = [f"../../eeg/{bids_desc}_electrodes.tsv"]
            data_ts = dict(sorted(data_ts.items()))
            with open(f"{dir_target_sub}ts/sub-{sub}_{bids_desc}_run-{str(run)}_b-{b}-a_4-0.25-eeg_stimuli.json","w") as f:
                json.dump(data_ts, f, indent=4)
            # c4 - RAW
            file_raw_stimuli_tsv = f"{dir_target_sub}ts/sub-{sub}_{bids_desc}_run-{str(run)}_b-0.05-a_4-{c4}-raw_stimuli.tsv"
            np.savetxt(file_raw_stimuli_tsv, sim_raw_ts[1,ival], delimiter='\t', fmt='%s')
            with zipfile.ZipFile(f"{file_raw_stimuli_tsv}.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(file_raw_stimuli_tsv, os.path.basename(file_raw_stimuli_tsv))
            os.remove(file_raw_stimuli_tsv)
            # .json
            data_ts["ModelParam"] = f"../../param/{bids_desc}_b-0.05-a_4-{c4}_param.xml"
            data_ts["Description"] = f"This is the time series for subject {sub} with RAW monitor from TVB with Jansen & Rit parameter C4={c4}"
            data_ts["NumberOfColumns"] = 200
            data_ts["CoordsColumns"] = [f"../../coord/{bids_desc}_labels.tsv"]
            data_ts = dict(sorted(data_ts.items()))
            with open(f"{dir_target_sub}ts/sub-{sub}_{bids_desc}_run-{str(run)}_b-0.05-a_4-{c4}-raw_stimuli.json","w") as f:
                json.dump(data_ts, f, indent=4)
            # c4 - EEG
            file_eeg_stimuli_tsv = f"{dir_target_sub}ts/sub-{sub}_{bids_desc}_run-{str(run)}_b-0.05-a_4-{c4}-eeg_stimuli.tsv"
            np.savetxt(file_eeg_stimuli_tsv, sim_eeg_ts[1,ival], delimiter='\t', fmt='%s')
            with zipfile.ZipFile(f"{file_eeg_stimuli_tsv}.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(file_eeg_stimuli_tsv, os.path.basename(file_eeg_stimuli_tsv))
            os.remove(file_eeg_stimuli_tsv)
            # .json
            data_ts["Description"] = f"This is the time series for subject {sub} with EEG monitor from TVB with Jansen & Rit parameter C4={c4}"
            data_ts["NumberOfColumns"] = 62
            data_ts["CoordsColumns"] = [f"../../eeg/{bids_desc}_electrodes.tsv"]
            data_ts = dict(sorted(data_ts.items()))
            with open(f"{dir_target_sub}ts/sub-{sub}_{bids_desc}_run-{str(run)}_b-0.05-a_4-{c4}-eeg_stimuli.json","w") as f:
                json.dump(data_ts, f, indent=4)
        if run % 25 == 0:
            elapsed_time = time.time() - time_start
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"Sub {sub}, Run {run}, Time {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
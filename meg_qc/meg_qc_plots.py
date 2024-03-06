import os
import ancpbids
import time
import json

import sys

from meg_qc.source.universal_plots import boxplot_all_time_csv, boxplot_epoched_xaxis_channels_csv, boxplot_epoched_xaxis_epochs_csv, Plot_psd_csv, make_head_pos_plot_csv, make_head_pos_plot_mne, plot_muscle_csv
from meg_qc.source.plots_ecg_eog import plot_artif_per_ch_correlated_lobes_csv, plot_correlation_csv


# Needed to import the modules without specifying the full path, for command line and jupyter notebook
sys.path.append('./')
sys.path.append('./meg_qc/source/')

# relative path for `make html` (docs)
sys.path.append('../meg_qc/source/')

# relative path for `make html` (docs) run from https://readthedocs.org/
# every time rst file is nested insd of another, need to add one more path level here:
sys.path.append('../../meg_qc/source/')
sys.path.append('../../../meg_qc/source/')
sys.path.append('../../../../meg_qc/source/')


# What we want: 
# save in the right folders the csvs - to do in pipeline as qc derivative
# get it from the folders fro plotting - over ancp bids again??
# plot only whst is requested by user: separate config + if condition like in pipeline?
# do we save them as derivatives to write over abcp bids as report as before?

def make_plots_meg_qc(config_file_path, internal_config_file_path):

    derivs_path  = '/Volumes/M2_DATA/'
    verbose_plots = True

    # STD
    f_path = derivs_path+'STDs_by_lobe.csv'
    derivs_std = []
    fig_std_epoch0 = []
    fig_std_epoch1 = []
    m_or_g_chosen = ['mag']


    for m_or_g in m_or_g_chosen:

        derivs_std += [boxplot_all_time_csv(f_path, ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)]

        # fig_std_epoch0 += [boxplot_epoched_xaxis_channels(chs_by_lobe_copy[m_or_g], df_std, ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)]
        fig_std_epoch0 += [boxplot_epoched_xaxis_channels_csv(f_path, ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)]

        fig_std_epoch1 += [boxplot_epoched_xaxis_epochs_csv(f_path, ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)]

    derivs_std += fig_std_epoch0+fig_std_epoch1 


    # PtP
    f_path = derivs_path+'PtPs_by_lobe.csv'



    # PSD
    # TODO: also add pie psd plot

    derivs_psd = []

    for m_or_g in m_or_g_chosen:

        method = 'welch' #is also hard coded in PSD_meg_qc() for now
        f_path = derivs_path+'PSDs_by_lobe.csv'

        psd_plot_derivative=Plot_psd_csv(m_or_g, f_path, method, verbose_plots)

        derivs_psd += [psd_plot_derivative]


    # ECG
        
    ecg_derivs = []
        
    f_path = derivs_path+'ECGs_by_lobe.csv'
        
    for m_or_g in m_or_g_chosen:
        affected_derivs = plot_artif_per_ch_correlated_lobes_csv(f_path, m_or_g, 'ECG', flip_data=False, verbose_plots=verbose_plots)
        correlation_derivs = plot_correlation_csv(f_path, 'ECG', m_or_g, verbose_plots=verbose_plots)

    ecg_derivs += affected_derivs + correlation_derivs

    # EOG

    eog_derivs = []
    f_path = derivs_path+'EOGs_by_lobe.csv'
        
    for m_or_g in m_or_g_chosen:
        affected_derivs = plot_artif_per_ch_correlated_lobes_csv(f_path, m_or_g, 'EOG', flip_data=False, verbose_plots=verbose_plots)
        correlation_derivs = plot_correlation_csv(f_path, 'EOG', m_or_g, verbose_plots=verbose_plots)

    eog_derivs += affected_derivs + correlation_derivs 

    # Muscle

    f_path = derivs_path+'muscle.csv'

    if 'mag' in m_or_g_chosen:
            m_or_g_decided=['mag']
    elif 'grad' in m_or_g_chosen and 'mag' not in m_or_g_chosen:
            m_or_g_decided=['grad']
    else:
            print('___MEG QC___: ', 'No magnetometers or gradiometers found in data. Artifact detection skipped.')


    muscle_derivs =  plot_muscle_csv(f_path, m_or_g_decided[0], verbose_plots = True)

    # Head

    f_path = derivs_path+'Head.csv'
        
    head_pos_derivs, head_pos_baselined = make_head_pos_plot_csv(f_path, verbose_plots=verbose_plots)
    # head_pos_derivs2 = make_head_pos_plot_mne(raw, head_pos, verbose_plots=verbose_plots)
    # head_pos_derivs += head_pos_derivs2

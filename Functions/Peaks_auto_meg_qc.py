# Calculating peak-to-peak amplitudes using mne annotations.

import numpy as np
import pandas as pd
import mne
from universal_plots import boxplot_std_hovering_plotly, boxplot_channel_epoch_hovering_plotly

def get_amplitude_annots_per_channel(raw: mne.io.Raw, peak: float, flat: float, channels: list, bad_percent:  int, min_duration: float) -> tuple[pd.DataFrame, list]:
    """Function creates amplitude (peak-to-peak annotations) for every channel separately"""
    
    amplit_annot_with_ch_names=mne.Annotations(onset=[], duration=[], description=[], orig_time=raw.annotations.orig_time) #initialize 
    bad_channels=[]

    for channel in channels:
        #get annotation object:
        amplit_annot=mne.preprocessing.annotate_amplitude(raw, peak=peak, flat=flat , bad_percent=bad_percent, min_duration=min_duration, picks=[channel], verbose=False)
        
        bad_channels.append(amplit_annot[1]) #Can later add these into annotation as well.

        if len(amplit_annot[0])>0:

            #create new annot obj and add there all data + channel name:
            amplit_annot_with_ch_names.append(onset=amplit_annot[0][0]['onset'], duration=amplit_annot[0][0]['duration'], description=amplit_annot[0][0]['description'], ch_names=[[channel[0]]])

    df_ptp_amlitude_annot=amplit_annot_with_ch_names.to_data_frame()
    return df_ptp_amlitude_annot, bad_channels, amplit_annot_with_ch_names


def PP_auto_meg_qc(sid:str, config, channels:list, data: mne.io.Raw, m_or_g_chosen):

    ptp_mne_section = config['PTP_mne']

    peak_m = ptp_mne_section['peak_m']
    flat_m = ptp_mne_section['flat_m']
    peak_g = ptp_mne_section['peak_g']
    flat_g = ptp_mne_section['flat_g']
    bad_percent = ptp_mne_section.getint('bad_percent')
    min_duration = ptp_mne_section.getfloat('min_duration')

    peaks = {}
    flats = {}
    dfs_ptp_amlitude_annot = {}
    bad_channels = {}
    amplit_annot_with_ch_names = {}

    if 'mags' in m_or_g_chosen:
        if  not peak_m or not flat_m:
            print('Magnetometers were chosen for analysis, but no peak or flat values given for auto peak-to-peak detection. Please add values in config file.')
            return
        else:
            peaks['mags'] = float(peak_m)
            flats['mags'] = float(flat_m)

    if 'grads' in m_or_g_chosen:
        if not peak_g or not flat_g:
            print('Gradiometers were chosen for analysis, but no peak or flat values given for auto peak-to-peak detection. Please add values in config file.')
            return
        else:
            peaks['grads'] = float(peak_g)
            flats['grads'] = float(flat_g)

    for  m_or_g in m_or_g_chosen:
        dfs_ptp_amlitude_annot[m_or_g], bad_channels[m_or_g], amplit_annot_with_ch_names[m_or_g] = get_amplitude_annots_per_channel(data, peaks[m_or_g], flats[m_or_g], channels[m_or_g], bad_percent, min_duration)

    return dfs_ptp_amlitude_annot, bad_channels, amplit_annot_with_ch_names

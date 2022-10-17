# This version of main allows to only choose mags, grads or both for 
# the entire pipeline in the beginning. Not for separate QC measures

import pandas as pd
import mne
import configparser
from PSD_meg_qc import PSD_QC as PSD_QC

config = configparser.ConfigParser()
config.read('settings.ini')
# sids = config['DEFAULT']['sid']
# sid_list = list(sids.split(","))


#def initial_stuff(duration: int or None, config: dict):
def initial_stuff(sid):

    '''Here all the initial actions need to work with MEG data are done: 
    - load fif file and convert into raw,
    - create folders in BIDS compliant format,
    - crop the data if needed,
    - filter and downsample the data,
    - epoch the data.

    Args:
    duration (int): how long the cropped data should be, in seconds

    Returns: 
    n_events (int): number of events(=number of epochs)
    df_epochs_mags (pd. Dataframe): data frame containing data for all epochs for mags 
    df_epochs_grads (pd. Dataframe): data frame containing data for all epochs for grads 
    epochs_mags (mne. Epochs): epochs as mne data structure for magnetometers
    epochs_grads (mne. Epochs): epochs as mne data structure for gradiometers 
    mags (list of tuples): magnetometer channel name + its index
    grads (list of tuples): gradiometer channel name + its index
    raw_bandpass(mne.raw): data only filtered, cropped (*)
    raw_bandpass_resamp(mne.raw): data filtered and resampled, cropped (*)
    raw_cropped(mne.io.Raw): data in raw format, cropped, not filtered, not resampled (*)
    raw(mne.io.Raw): original data in raw format, not cropped, not filtered, not resampled.
    (*): if duration was set to None - the data will not be cropped and these outputs 
    will return what is stated, but in origibal duration.

    Yes, these are a lot  of data output option, we can reduce them later when we know what will not be used.
    '''

    config = configparser.ConfigParser()
    config.read('settings.ini')

    default_section = config['DEFAULT']
    data_file = default_section['data_file']

    from data_load_and_folders import load_meg_data, make_folders_meg, Epoch_meg

    raw, mags, grads=load_meg_data(data_file)

    #Create folders:
    make_folders_meg(sid)

    #crop the data to calculate faster:
    tmin = default_section['data_crop_tmin']
    tmax = default_section['data_crop_tmax']

    if not tmin: 
        tmin = 0
    else:
        tmin=float(tmin)
    if not tmax: 
        tmax = raw.times[-1] 
    else:
        tmax=float(tmax)

    raw_cropped = raw.copy()
    raw_cropped.crop(tmin, tmax)

    #Data filtering:
    filtering_section = config['Filter_and_resample']
    if filtering_section['apply_filtering'] is True:
        l_freq = filtering_section.getfloat('l_freq') 
        h_freq = filtering_section.getfloat('h_freq') 
        method = filtering_section['method']
    
        raw_cropped.load_data(verbose=True) #Data has to be loaded into mememory before filetering:
        raw_bandpass = raw_cropped.copy()
        raw_bandpass.filter(l_freq=l_freq, h_freq=h_freq, picks='meg', method=method, iir_params=None)

        #And downsample:
        raw_bandpass_resamp=raw_bandpass.copy()
        raw_bandpass_resamp.resample(sfreq=h_freq*5)
        #frequency to resample is 5 times higher than the maximum chosen frequency of the function

    else:
        raw_bandpass = raw_cropped.copy()
        raw_bandpass_resamp=raw_bandpass.copy()
        #OR maybe we dont need these 2 copies of data at all? Think how to get rid of them, 
        # because they are used later. Referencing might mess up things, check that.


    #Apply epoching: USE NON RESAMPLED DATA. Or should we resample after epoching? 
    # Since sampling freq is 1kHz and resampling is 500Hz, it s not that much of a win...

    epoching_section = config['Epoching']
    event_dur = epoching_section.getfloat('event_dur') 
    epoch_tmin = epoching_section.getfloat('epoch_tmin') 
    epoch_tmax = epoching_section.getfloat('epoch_tmax') 
    stim_channel = default_section['stim_channel'] 

    if len(stim_channel) == 0:
        picks_stim = mne.pick_types(raw.info, stim=True)
        stim_channel = []
        for ch in picks_stim:
            stim_channel.append(raw.info['chs'][ch]['ch_name'])
    
    n_events, df_epochs_mags, df_epochs_grads, epochs_mags, epochs_grads=Epoch_meg(data=raw_bandpass, 
        stim_channel=stim_channel, event_dur=event_dur, epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax)

    channels = {'mags': mags, 'grads': grads}

    return n_events, df_epochs_mags, df_epochs_grads, epochs_mags, epochs_grads, channels, raw_bandpass, raw_bandpass_resamp, raw_cropped, raw


def selected_m_or_g(section: configparser.SectionProxy):
    """get do_for selection for given config: is the calculation of this particilatr quality measure done for mags, grads, both or none."""

    do_for = section['do_for']

    if do_for == 'none':
        return
    elif do_for == 'mags':
        return ['mags']
    elif do_for == 'grads':
        return ['grads']
    elif do_for == 'both':
        return ['mags', 'grads']

#%%
def MEG_QC_rmse(sid: str, config, channels: dict, m_or_g_title: dict, df_epochs:pd.DataFrame, filtered_d_resamp, n_events: int, m_or_g_chosen):

    from universal_plots import boxplot_channel_epoch_hovering_plotly
    from universal_html_report import make_RMSE_html_report

    rmse_section = config['RMSE']

    # import RMSE_meg_qc as rmse #or smth like this - when it's extracted to .py
    std_lvl = rmse_section.getint('std_lvl')

    list_of_figure_paths = []
    list_of_figure_paths_std_epoch = []
    big_std_with_value = {}
    small_std_with_value = {}
    fig = {}
    fig_path = {}
    df_std = {}
    fig_std_epoch = {}
    fig_path_std_epoch = {}
    rmse = {}

    # will run for both if mags and grads both chosen,otherwise just for one of them:
    for m_or_g in m_or_g_chosen:
        big_std_with_value[m_or_g], small_std_with_value[m_or_g], rmse[m_or_g] = RMSE_meg_all(data=filtered_d_resamp, channels=channels[m_or_g], std_lvl=1)

        fig[m_or_g], fig_path[m_or_g] = boxplot_std_hovering_plotly(std_data=rmse[m_or_g], tit=channels[m_or_g], channels=channels[m_or_g], sid=sid)
        
        df_std[m_or_g] = RMSE_meg_epoch(ch_type=m_or_g, channels=channels[m_or_g], std_lvl=std_lvl, n_events=n_events, df_epochs=df_epochs[m_or_g], sid=sid) 

        fig_std_epoch[m_or_g], fig_path_std_epoch[m_or_g] = boxplot_channel_epoch_hovering_plotly(df_mg=df_std[m_or_g], ch_type=m_or_g, sid=sid, what_data='stds')
        
        list_of_figure_paths.append(fig_path[m_or_g])
        list_of_figure_paths_std_epoch.append(fig_path_std_epoch[m_or_g])
    
    list_of_figure_paths += list_of_figure_paths_std_epoch

    make_RMSE_html_report(sid=sid, what_data='stds', list_of_figure_paths=list_of_figure_paths)



#%%
def MEG_peaks_manual(sid:str, config, channels:list, filtered_d_resamp: mne.io.Raw, m_or_g_chosen):
    #UNFINISHED

    PTP_manual_section = config['PTP_manual']

    # from Peaks_meg_qc import peak_amplitude_per_epoch as pp_epoch 
    ptp_manual_section = config['PTP_manual']
    pair_dist_sec = ptp_manual_section.getint('pair_dist_sec') 
    thresh_lvl = ptp_manual_section.getint('thresh_lvl')

    sfreq = filtered_d_resamp.info['sfreq']
    df_pp_ampl_mags=peak_amplitude_per_epoch(mg_names=mags, df_epoch_mg=df_epochs_mags, sfreq=sfreq, n_events=n_events, thresh_lvl=thresh_lvl, pair_dist_sec=pair_dist_sec)
    df_pp_ampl_grads=peak_amplitude_per_epoch(mg_names=grads, df_epoch_mg=df_epochs_grads, sfreq=sfreq, n_events=n_events, thresh_lvl=thresh_lvl, pair_dist_sec=pair_dist_sec)

    from universal_plots import boxplot_channel_epoch_hovering_plotly
    _, fig_path_m_pp_ampl_epoch=boxplot_channel_epoch_hovering_plotly(df_mg=df_pp_ampl_mags, ch_type='Magnetometers', sid='1', what_data='peaks')
    _, fig_path_g_pp_ampl_epoch=boxplot_channel_epoch_hovering_plotly(df_mg=df_pp_ampl_grads, ch_type='Gradiometers', sid='1', what_data='peaks')

    from universal_html_report import make_peak_html_report
    list_of_figure_paths=[fig_path_m_pp_ampl_epoch, fig_path_g_pp_ampl_epoch]
    make_peak_html_report(sid=sid, what_data='peaks', list_of_figure_paths=list_of_figure_paths)


#%%
def MEG_peaks_auto(sid:str, config, channels:list, filtered_d_resamp: mne.io.Raw, m_or_g_chosen):
    #UNFINISHED

    PTP_mne_section = config['PTP_mne']

    # import peaks_mne #or smth like this - when it's extracted to .py

    ptp_mne_section = config['PTP_mne']
    peak = ptp_mne_section.getint('peak_m') 
    flat = ptp_mne_section.getint('flat_m') 
    df_ptp_amlitude_annot_mags, bad_channels_mags, amplit_annot_with_ch_names_mags=get_amplitude_annots_per_channel(raw_cropped, peak, flat, ch_type_names=mags)


#%%
#  ADD 4 MORE SECIONS HERE




#%%
def MEG_QC_measures(sid):

    """This function will call all the QC functions.
    Here goes several sections which will in the future be called over main, but are not yet, since they are in the notebooks"""

    n_events, df_epochs_mags, df_epochs_grads, epochs_channels_mags, epochs_channels_grads, channels, filtered_d, filtered_d_resamp, raw_cropped, raw = initial_stuff(sid)

    m_or_g_title = {
        'grads': 'Gradiometers',
        'mags': 'Magnetometers'
    }
    df_epochs = {
        'grads': df_epochs_grads,
        'mags': df_epochs_mags
    }
    epochs_channels = {
        'grads': epochs_channels_grads,
        'mags': epochs_channels_mags
    }

    config = configparser.ConfigParser()
    config.read('settings.ini')
    default_section = config['DEFAULT']
    m_or_g_chosen = selected_m_or_g(default_section)

    if m_or_g_chosen != ['mags'] and m_or_g_chosen != ['grads'] and m_or_g_chosen != ['mags', 'grads']:
        raise ValueError('Type of channels to analise has to be chose in setting.ini. Use "mags", "grads" or "both" as parameter of do_for. Otherwise the analysis can not be done.')


    # MEG_QC_rmse(sid, config, channels, df_epochs, m_or_g_title, filtered_d_resamp, n_events)

    psd_section = config['PSD']
    PSD_QC(sid, channels, filtered_d_resamp, m_or_g_chosen, psd_section)

    # MEG_peaks_manual()

    # MEG_peaks_auto()

    # MEG_EOG()

    # MEG_ECG()

    # MEG_head_movements()

    # MEG_muscle()


#%%
#Run the pipleine over subjects
#  UNCOMMENT THIS PART ONLY WHEN ALL MEASUREMENTS ARE SAVED INTO PY FILES. OTHERWISE IT WILL TRY TO RUN IT AND FAIL EVERY TIME THIS FILE IS CALLED IN ANY WAY
# for sid in sid_list:
#     n_events, df_epochs_mags, df_epochs_grads, epochs_mags, epochs_grads, mags, grads, raw_bandpass, raw_bandpass_resamp, raw_cropped, raw = initial_stuff(sid)
#     MEG_QC_measures(sid)

#%%

MEG_QC_measures(sid='1')
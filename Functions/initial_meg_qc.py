import os
import mne
import configparser
import plotly.graph_objects as go
import numpy as np
from universal_plots import QC_derivative


def get_all_config_params(config_file_name: str):
    '''Parse all the parameters from config and put into a python dictionary 
    divided by sections. Parsing approach can be changed here, which 
    will not affect working of other fucntions.
    
    Return:
    all_qc_params: dictionary of dictionaries, where each one refers to 
    a QC pipeline section and contains corresponding parameters.
    '''
    
    all_qc_params = {}

    config = configparser.ConfigParser()
    config.read(config_file_name)

    default_section = config['DEFAULT']

    m_or_g_chosen = default_section['do_for'] 
    m_or_g_chosen = m_or_g_chosen.replace(" ", "")
    m_or_g_chosen = m_or_g_chosen.split(",")

    if 'mag' not in m_or_g_chosen and 'grad' not in m_or_g_chosen:
        print('___MEG QC___: ', 'No channels to analyze. Check parameter do_for in config file.')
        return None

    try:
        dataset_path = default_section['data_directory']
        tmin = default_section['data_crop_tmin']
        tmax = default_section['data_crop_tmax']

        if not tmin: 
            tmin = 0
        else:
            tmin=float(tmin)
        if not tmax: 
            tmax = None
        else:
            tmax=float(tmax)

        default_params = dict({
            'm_or_g_chosen': m_or_g_chosen, 
            'dataset_path': dataset_path,
            'crop_tmin': tmin,
            'crop_tmax': tmax})
        all_qc_params['default'] = default_params

        filtering_section = config['Filtering']
        if filtering_section.getboolean('apply_filtering') is True:
            filtering_params = dict({
                'l_freq': filtering_section.getfloat('l_freq'),
                'h_freq': filtering_section.getfloat('h_freq'),
                'method': filtering_section['method']})
            all_qc_params['Filtering'] = filtering_params
        else: 
            all_qc_params['Filtering'] = 'Not apply'


        epoching_section = config['Epoching']
        stim_channel = epoching_section['stim_channel'] 
        stim_channel = stim_channel.replace(" ", "")
        stim_channel = stim_channel.split(",")
        if stim_channel==['']:
            stim_channel=None

        epoching_params = dict({
        'event_dur': epoching_section.getfloat('event_dur'),
        'epoch_tmin': epoching_section.getfloat('epoch_tmin'),
        'epoch_tmax': epoching_section.getfloat('epoch_tmax'),
        'stim_channel': stim_channel})
        all_qc_params['Epoching'] = epoching_params

        rmse_section = config['RMSE']
        all_qc_params['RMSE'] = dict({
            'std_lvl':  rmse_section.getint('std_lvl'), 
            'allow_percent_noisy': rmse_section.getfloat('allow_percent_noisy_epochs'),
            'allow_percent_flat': rmse_section.getfloat('allow_percent_flat_epochs')})
        

        psd_section = config['PSD']
        freq_min = psd_section['freq_min']
        freq_max = psd_section['freq_max']
        if not freq_min: 
            freq_min = 0
        else:
            freq_min=float(freq_min)
        if not freq_max: 
            freq_max = np.inf
        else:
            freq_max=float(freq_max)

        all_qc_params['PSD'] = dict({
        'freq_min': freq_min,
        'freq_max': freq_max,
        'mean_power_per_band_needed': psd_section.getboolean('mean_power_per_band_needed'),
        'n_fft': psd_section.getint('n_fft'),
        'n_per_seg': psd_section.getint('n_per_seg')})


        ptp_manual_section = config['PTP_manual']
        all_qc_params['PTP_manual'] = dict({
        'max_pair_dist_sec': ptp_manual_section.getfloat('max_pair_dist_sec'),
        'thresh_lvl': ptp_manual_section.getfloat('ptp_thresh_lvl'),
        'allow_percent_noisy': ptp_manual_section.getfloat('allow_percent_noisy_epochs'),
        'allow_percent_flat': ptp_manual_section.getfloat('allow_percent_flat_epochs'),
        'ptp_top_limit': ptp_manual_section.getfloat('ptp_top_limit'),
        'ptp_bottom_limit': ptp_manual_section.getfloat('ptp_bottom_limit')})


        ptp_mne_section = config['PTP_auto']
        all_qc_params['PTP_auto'] = dict({
        'peak_m': ptp_mne_section.getfloat('peak_m'),
        'flat_m': ptp_mne_section.getfloat('flat_m'),
        'peak_g': ptp_mne_section.getfloat('peak_g'),
        'flat_g': ptp_mne_section.getfloat('flat_g'),
        'bad_percent': ptp_mne_section.getint('bad_percent'),
        'min_duration': ptp_mne_section.getfloat('min_duration')})


        ecg_section = config['ECG']
        all_qc_params['ECG'] = dict({
        'ecg_epoch_tmin': ecg_section.getfloat('ecg_epoch_tmin'),
        'ecg_epoch_tmax': ecg_section.getfloat('ecg_epoch_tmax'),
        'norm_lvl': ecg_section.getfloat('norm_lvl'),
        'use_abs_of_all_data': ecg_section['use_abs_of_all_data']})

        eog_section = config['EOG']
        all_qc_params['EOG'] = dict({
        'eog_epoch_tmin': eog_section.getfloat('eog_epoch_tmin'),
        'eog_epoch_tmax': eog_section.getfloat('eog_epoch_tmax'),
        'norm_lvl': eog_section.getfloat('norm_lvl'),
        'use_abs_of_all_data': eog_section['use_abs_of_all_data']})

        head_section = config['Head_movement']
        all_qc_params['Head'] = dict({})

        muscle_section = config['Muscle']
        list_thresholds = muscle_section['threshold_muscle']
        #separate values in list_thresholds based on coma, remove spaces and convert them to floats:
        list_thresholds = [float(i) for i in list_thresholds.split(',')]

        all_qc_params['Muscle'] = dict({
        'threshold_muscle': list_thresholds,
        'min_distance_between_different_muscle_events': muscle_section.getfloat('min_distance_between_different_muscle_events')})

    except:
        print('___MEG QC___: ', 'Invalid setting in config file! Please check instructions for each setting. \nGeneral directions: \nDon`t write any parameter as None. Don`t use quotes.\nLeaving blank is only allowed for parameters: \n- stim_channel, \n- data_crop_tmin, data_crop_tmax, \n -freq_min and freq_max in Filtering section, \n- all parameters of Filtering section if apply_filtering is set to False.')
        return None

    return all_qc_params


def Epoch_meg(epoching_params, data: mne.io.Raw):

    '''Gives epoched data in 2 separated data frames: mag and grad + as epoch objects.
    
    Args:
    config
    data (mne.io.Raw): data in raw format
    
    Returns: 
    n_events (int): number of events(=number of epochs)
    df_epochs_mag (pd. Dataframe): data frame containing data for all epochs for mag 
    df_epochs_grad (pd. Dataframe): data frame containing data for all epochs for grad 
    epochs_mag (mne. Epochs): epochs as mne data structure for magnetometers
    epochs_grad (mne. Epochs): epochs as mne data structure for gradiometers '''

    event_dur = epoching_params['event_dur']
    epoch_tmin = epoching_params['epoch_tmin']
    epoch_tmax = epoching_params['epoch_tmax']
    stim_channel = epoching_params['stim_channel']

    if stim_channel is None:
        picks_stim = mne.pick_types(data.info, stim=True)
        stim_channel = []
        for ch in picks_stim:
            stim_channel.append(data.info['chs'][ch]['ch_name'])
    print('___MEG QC___: ', 'Stimulus channels detected:', stim_channel)

    picks_magn = data.copy().pick_types(meg='mag').ch_names if 'mag' in data else None
    picks_grad = data.copy().pick_types(meg='grad').ch_names if 'grad' in data else None

    events = mne.find_events(data, stim_channel=stim_channel, min_duration=event_dur)
    n_events=len(events)

    if n_events == 0:
        print('___MEG QC___: ', 'No events with set minimum duration were found using all stimulus channels. No epoching can be done. Try different event duration in config file.')
        epochs_grad, epochs_mag = None, None
    else:
        epochs_mag = mne.Epochs(data, events, picks=picks_magn, tmin=epoch_tmin, tmax=epoch_tmax, preload=True, baseline = None)
        epochs_grad = mne.Epochs(data, events, picks=picks_grad, tmin=epoch_tmin, tmax=epoch_tmax, preload=True, baseline = None)

    dict_epochs_mg = {
    'mag': epochs_mag,
    'grad': epochs_grad}

    return dict_epochs_mg


def initial_processing(default_settings: dict, filtering_settings: dict, epoching_params:dict, data_file: str):

    '''Here all the initial actions need to work with MEG data are done: 
    - load fif file and convert into raw,
    - create folders in BIDS compliant format,
    - crop the data if needed,
    - filter and downsample the data,
    - epoch the data.

    Args:
    config: config file like settings.ini
    data_file (str): path to the data file

    Returns: 
    dict_of_dfs_epoch (dict with 2 pd. Dataframe): 2 data frames containing data for all epochs for mag and grad
    epochs_mg (dict with 2 mne. Epochs): 2 epoch objects for mag and  grad
    channels (dict): mag and grad channels names
    raw_bandpass(mne.raw): data only filtered, cropped (*)
    raw_bandpass_resamp(mne.raw): data filtered and resampled, cropped (*)
    raw_cropped(mne.io.Raw): data in raw format, cropped, not filtered, not resampled (*)
    raw(mne.io.Raw): original data in raw format, not cropped, not filtered, not resampled.
    (*): if duration was set to None - the data will not be cropped and these outputs 
    will return what is stated, but in origibal duration.
    '''

    active_shielding_used = False
    try:
        raw = mne.io.read_raw_fif(data_file, on_split_missing='ignore')
    except: 
        raw = mne.io.read_raw_fif(data_file, allow_maxshield=True, on_split_missing='ignore')
        active_shielding_used = True

    mag_ch_names = raw.copy().pick_types(meg='mag').ch_names if 'mag' in raw else None
    grad_ch_names = raw.copy().pick_types(meg='grad').ch_names if 'grad' in raw else None
    channels = {'mag': mag_ch_names, 'grad': grad_ch_names}

    #crop the data to calculate faster:
    tmax=default_settings['crop_tmax']
    if tmax is None: 
        tmax = raw.times[-1] 
    raw_cropped = raw.copy().crop(tmin=default_settings['crop_tmin'], tmax=tmax)

    #Data filtering:
    raw_cropped_filtered = raw_cropped.copy()
    if filtering_settings != 'Not apply':
        raw_cropped.load_data(verbose=True) #Data has to be loaded into mememory before filetering:
        raw_cropped_filtered = raw_cropped.copy()
        raw_cropped_filtered.filter(l_freq=filtering_settings['l_freq'], h_freq=filtering_settings['h_freq'], picks='meg', method=filtering_settings['method'], iir_params=None)
        
        #And downsample:
        raw_cropped_filtered_resampled = raw_cropped_filtered.copy().resample(sfreq=filtering_settings['h_freq']*5)
        #frequency to resample is 5 times higher than the maximum chosen frequency of the function
    else:
        raw_cropped_filtered_resampled = raw_cropped_filtered.copy()
        #OR maybe we dont need these 2 copies of data at all? Think how to get rid of them, 
        # because they are used later. Referencing might mess up things, check that.
    

    #Apply epoching: USE NON RESAMPLED DATA. Or should we resample after epoching? 
    # Since sampling freq is 1kHz and resampling is 500Hz, it s not that much of a win...

    dict_epochs_mg = Epoch_meg(epoching_params, data=raw_cropped_filtered)

    return dict_epochs_mg, channels, raw_cropped_filtered, raw_cropped_filtered_resampled, raw_cropped, raw, active_shielding_used


def sanity_check(m_or_g_chosen, channels):
    '''Check if the channels which the user gave in config file to analize actually present in the data set'''

    if 'mag' not in m_or_g_chosen and 'grad' not in m_or_g_chosen:
        m_or_g_chosen = []
    if channels['mag'] is None and 'mag' in m_or_g_chosen:
        print('___MEG QC___: ', 'There are no magnetometers in this data set: check parameter do_for in config file. Analysis will be done only for gradiometers.')
        m_or_g_chosen.remove('mag')
    elif channels['grad'] is None and 'grad' in m_or_g_chosen:
        print('___MEG QC___: ', 'There are no gradiometers in this data set: check parameter do_for in config file. Analysis will be done only for magnetometers.')
        m_or_g_chosen.remove('grad')
    elif channels['mag'] is None and channels['grad'] is None:
        print ('There are no magnetometers or gradiometers in this data set. Analysis will not be done.')
        m_or_g_chosen = []
    return m_or_g_chosen


def detect_extra_channels(raw):
    picks_ECG = mne.pick_types(raw.info, ecg=True)
    if picks_ECG.size == 0:
        print('___MEG QC___: ', 'No ECG channels found is this data set. Attempting to reconstruct ECG data from magnetometers...')
        ECG_channel_name = None
    else:
        ECG_channel_name=[]
        for i in range(0,len(picks_ECG)):
            ECG_channel_name.append(raw.info['chs'][picks_ECG[i]]['ch_name'])

    picks_EOG = mne.pick_types(raw.info, eog=True)
    if picks_EOG.size == 0:
        print('___MEG QC___: ', 'No EOG channels found is this data set - EOG artifacts can not be detected.')
        EOG_channel_name = None
    else:
        EOG_channel_name=[]
        for i in range(0,len(picks_EOG)):
            EOG_channel_name.append(raw.info['chs'][picks_EOG[i]]['ch_name'])
    
    return ECG_channel_name, EOG_channel_name #, picks_HPI, picks_stim


def detect_noisy_ecg_eog(raw_cropped, picked_channels_ecg_or_eog:list[str],  thresh_lvl:float, plotflag:bool):

    bad_ecg_eog = False
    if picked_channels_ecg_or_eog is None:
        return [], None

    sfreq=raw_cropped.info['sfreq']
    #threshold for peak detection. to whatlevel allowed the noisy peaks to be in comparison with most of other peaks
    duration_crop = len(raw_cropped)/raw_cropped.info['sfreq']/60  #duration in minutes


    if 'ecg' or 'ECG' in picked_channels_ecg_or_eog[0]:
            max_peak_dist=35 #allow the lowest pulse to be 35/min. this is the maximal possible distance between 2 pulses.

    elif 'eog' or 'EOG' in picked_channels_ecg_or_eog[0]:
            max_peak_dist=5 #normal spontaneous blink rate is between 12 and 15/min, allow 5 blinks a min minimum. However do we really need to limit here?

    
    for picked in picked_channels_ecg_or_eog:
        ch_data=raw_cropped.get_data(picks=picked)[0] 
        # get_data creates list inside of a list becausee expects to create a list for each channel. 
        # but interation takes 1 ch at a time anyways. this is why [0]
        thresh=(max(ch_data) - min(ch_data)) / thresh_lvl 

        pos_peak_locs, pos_peak_magnitudes = mne.preprocessing.peak_finder(ch_data, extrema=1, thresh=thresh, verbose=False) #positive peaks
        neg_peak_locs, neg_peak_magnitudes = mne.preprocessing.peak_finder(ch_data, extrema=-1, thresh=thresh, verbose=False) #negative peaks

        #find where places of recording without peaks at all:
        normal_pos_peak_locs, _ = mne.preprocessing.peak_finder(ch_data, extrema=1, verbose=False) #all positive peaks of the data
        ind_break_start = np.where(np.diff(normal_pos_peak_locs)/sfreq/60>max_peak_dist)#find where the distance between positive peaks is too long

        #_, amplitudes=neighbour_peak_amplitude(max_pair_dist_sec,sfreq, pos_peak_locs, neg_peak_locs, pos_peak_magnitudes, neg_peak_magnitudes)
        # if amplitudes is not None and len(amplitudes)>3*duration_crop/60: #allow 3 non-standard peaks per minute. Or 0? DISCUSS
        #     bad_ecg_eog=True
        #     print('___MEG QC___: ', picked, ' channel is too noisy. Number of unusual amplitudes detected over the set limit: '+str(len (amplitudes)))

        all_peaks=np.concatenate((pos_peak_locs,neg_peak_locs),axis=None)
        if len(all_peaks)/duration_crop>3:
        # allow 3 non-standard peaks per minute. Or 0? DISCUSS. implies that noiseness has to be repeated regularly.  
        # if there is only 1 little piece of time with noise and the rest is good, will not show that one. 
        # include some time limitation of noisy times?
            print('___MEG QC___: ', 'ECG channel might be corrupted. Atypical peaks in ECG amplitudes detected: '+str(len (all_peaks))+'. Peaks per minute: '+str(round(len(all_peaks)/duration_crop)))
            
        if len(ind_break_start[0])/duration_crop>3: #allow 3 breaks per minute. Or 0? DISCUSS
            #ind_break_start[0] - here[0] because np.where created array of arrays above
            bad_ecg_eog=True
            print('___MEG QC___: ', picked, ' channel has breaks in recording. Number of breaks detected: '+str(len(ind_break_start[0]))+'. Breaks per minute: '+str(round(len(ind_break_start[0])/duration_crop)))

        noisy_ch_derivs=[]
        if plotflag:
            t=np.arange(0, duration_crop, 1/60/sfreq) 
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=ch_data, name=picked+' data'));
            fig.add_trace(go.Scatter(x=t[pos_peak_locs], y=pos_peak_magnitudes, mode='markers', name='+peak'));
            fig.add_trace(go.Scatter(x=t[neg_peak_locs], y=neg_peak_magnitudes, mode='markers', name='-peak'));

            for n in ind_break_start[0]:
                fig.add_vline(x=t[normal_pos_peak_locs][n],
                annotation_text='break', annotation_position="bottom right",line_width=0.6,annotation=dict(font_size=8))

            fig.update_layout(
                title={
                'text': picked+": atypical peaks and breaks",
                'y':0.85,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
                xaxis_title="Time in minutes",
                yaxis = dict(
                    showexponent = 'all',
                    exponentformat = 'e'))
                
            fig.show()
            noisy_ch_derivs += [QC_derivative(fig, 'Noisy_ECG_channel', None, 'plotly')]
        

    return noisy_ch_derivs, bad_ecg_eog
import mne
import configparser
import numpy as np


def get_all_config_params(config_file_name: str):
    '''Parse all the parameters from config and put into a python dictionary 
    divided by sections. Parsing approach can be changed here, which 
    will not affect working of other fucntions.
    

    Parameters:
    ----------
    config_file_name: str
        The name of the config file.

    Returns:
    -------
    all_qc_params: dict
        A dictionary with all the parameters from the config file.

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
            'allow_percent_noisy_flat_epochs': rmse_section.getfloat('allow_percent_noisy_flat_epochs'),
            'noisy_multiplier': rmse_section.getfloat('noisy_multiplier'),
            'flat_multiplier': rmse_section.getfloat('flat_multiplier'),})
        

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
        'psd_step_size': psd_section.getfloat('psd_step_size')})

        # 'n_fft': psd_section.getint('n_fft'),
        # 'n_per_seg': psd_section.getint('n_per_seg'),


        ptp_manual_section = config['PTP_manual']
        all_qc_params['PTP_manual'] = dict({
        'max_pair_dist_sec': ptp_manual_section.getfloat('max_pair_dist_sec'),
        'ptp_thresh_lvl': ptp_manual_section.getfloat('ptp_thresh_lvl'),
        'allow_percent_noisy_flat_epochs': ptp_manual_section.getfloat('allow_percent_noisy_flat_epochs'),
        'ptp_top_limit': ptp_manual_section.getfloat('ptp_top_limit'),
        'ptp_bottom_limit': ptp_manual_section.getfloat('ptp_bottom_limit'),
        'std_lvl': ptp_manual_section.getfloat('std_lvl'),
        'noisy_multiplier': ptp_manual_section.getfloat('noisy_multiplier'),
        'flat_multiplier': ptp_manual_section.getfloat('flat_multiplier')})


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

    '''Epochs MEG data based on the parameters provided in the config file.
    
    Parameters
    ----------
    epoching_params : dict
        Dictionary with parameters for epoching.
    data : mne.io.Raw
        MEG data to be epoch.
        
    Returns
    -------
    dict_epochs_mg : dict
        Dictionary with epochs for each channel type: mag, grad.

    '''

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
    - read fif file,
    - crop the data if needed,
    - filter and downsample the data,
    - epoch the data.

    Parameters
    ----------
    default_settings : dict
        Dictionary with default settings for MEG QC.
    filtering_settings : dict
        Dictionary with parameters for filtering.
    epoching_params : dict
        Dictionary with parameters for epoching.
    data_file : str
        Path to the fif file with MEG data.

    Returns
    -------
    dict_epochs_mg : dict
        Dictionary with epochs for each channel type: mag, grad.
    channels : dict
        Dictionary with channel names for each channel type: mag, grad.
    raw_crop_filtered : mne.io.Raw
        Filtered and cropped MEG data.
    raw_crop_filtered_resampled : mne.io.Raw
        Filtered, cropped and resampled MEG data.
    raw_cropped : mne.io.Raw
        Cropped MEG data.
    raw : mne.io.Raw
        MEG data.
    active_shielding_used : bool
        True if active shielding was used during recording.
    
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
    
    '''Check if the channels which the user gave in config file to analize actually present in the data set.
    
    Parameters
    ----------
    m_or_g_chosen : list
        List with channel types to analize: mag, grad. These are theones the user chose.
    channels : dict
        Dictionary with channel names for each channel type: mag, grad. These are the ones present in the data set.
    
    Returns
    -------
    m_or_g_chosen : list
        List with channel types to analize: mag, grad.
        
    '''

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



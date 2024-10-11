import os
import mne
import configparser
import numpy as np
import pandas as pd
from meg_qc.plotting.universal_plots import QC_derivative, assign_channels_properties, sort_channels_by_lobe


def get_all_config_params(config_file_name: str):

    """
    Parse all the parameters from config and put into a python dictionary 
    divided by sections. Parsing approach can be changed here, which 
    will not affect working of other fucntions.
    

    Parameters
    ----------
    config_file_name: str
        The name of the config file.

    Returns
    -------
    all_qc_params: dict
        A dictionary with all the parameters from the config file.

    """
    
    all_qc_params = {}

    config = configparser.ConfigParser()
    config.read(config_file_name)

    default_section = config['DEFAULT']

    m_or_g_chosen = default_section['ch_types'] 
    m_or_g_chosen = [chosen.strip() for chosen in m_or_g_chosen.split(",")]
    if 'mag' not in m_or_g_chosen and 'grad' not in m_or_g_chosen:
        print('___MEGqc___: ', 'No channels to analyze. Check parameter ch_types in config file.')
        return None

    #TODO: save list of mags and grads here and use later everywhere? because for CTF types are messed up.

    subjects = default_section['subjects']
    subjects = [sub.strip() for sub in subjects.split(",")]

    run_STD = default_section.getboolean('STD')
    run_PSD = default_section.getboolean('PSD')
    run_PTP_manual = default_section.getboolean('PTP_manual')
    run_PTP_auto_mne = default_section.getboolean('PTP_auto_mne')
    run_ECG = default_section.getboolean('ECG')
    run_EOG = default_section.getboolean('EOG')
    run_Head = default_section.getboolean('Head')
    run_Muscle = default_section.getboolean('Muscle')

    ds_paths = default_section['data_directory']
    ds_paths = [path.strip() for path in ds_paths.split(",")]
    if len(ds_paths) < 1:
        print('___MEGqc___: ', 'No datasets to analyze. Check parameter data_directory in config file. Data path can not contain spaces! You can replace them with underscores or remove completely.')
        return None

    tmin = default_section['data_crop_tmin']
    tmax = default_section['data_crop_tmax']
    try:
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
            'subjects': subjects,
            'run_STD': run_STD,
            'run_PSD': run_PSD,
            'run_PTP_manual': run_PTP_manual,
            'run_PTP_auto_mne': run_PTP_auto_mne,
            'run_ECG': run_ECG,
            'run_EOG': run_EOG,
            'run_Head': run_Head,
            'run_Muscle': run_Muscle,
            'dataset_path': ds_paths,
            'plot_mne_butterfly': default_section.getboolean('plot_mne_butterfly'),
            'plot_interactive_time_series': default_section.getboolean('plot_interactive_time_series'),
            'plot_interactive_time_series_average': default_section.getboolean('plot_interactive_time_series_average'),
            'crop_tmin': tmin,
            'crop_tmax': tmax})
        all_qc_params['default'] = default_params

        filtering_section = config['Filtering']
        try:
            lfreq = filtering_section.getfloat('l_freq')
        except:
            lfreq = None

        try:
            hfreq = filtering_section.getfloat('h_freq')
        except:
            hfreq = None
        
        all_qc_params['Filtering'] = dict({
            'apply_filtering': filtering_section.getboolean('apply_filtering'),
            'l_freq': lfreq,
            'h_freq': hfreq,
            'method': filtering_section['method'],
            'downsample_to_hz': filtering_section.getint('downsample_to_hz')})


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
        'stim_channel': stim_channel,
        'event_repeated': epoching_section['event_repeated']})
        all_qc_params['Epoching'] = epoching_params

        std_section = config['STD']
        all_qc_params['STD'] = dict({
            'std_lvl':  std_section.getint('std_lvl'), 
            'allow_percent_noisy_flat_epochs': std_section.getfloat('allow_percent_noisy_flat_epochs'),
            'noisy_channel_multiplier': std_section.getfloat('noisy_channel_multiplier'),
            'flat_multiplier': std_section.getfloat('flat_multiplier'),})
        

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
        'psd_step_size': psd_section.getfloat('psd_step_size')})


        ptp_manual_section = config['PTP_manual']
        all_qc_params['PTP_manual'] = dict({
        'max_pair_dist_sec': ptp_manual_section.getfloat('max_pair_dist_sec'),
        'ptp_thresh_lvl': ptp_manual_section.getfloat('ptp_thresh_lvl'),
        'allow_percent_noisy_flat_epochs': ptp_manual_section.getfloat('allow_percent_noisy_flat_epochs'),
        'ptp_top_limit': ptp_manual_section.getfloat('ptp_top_limit'),
        'ptp_bottom_limit': ptp_manual_section.getfloat('ptp_bottom_limit'),
        'std_lvl': ptp_manual_section.getfloat('std_lvl'),
        'noisy_channel_multiplier': ptp_manual_section.getfloat('noisy_channel_multiplier'),
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
        'drop_bad_ch': ecg_section.getboolean('drop_bad_ch'),
        'n_breaks_bursts_allowed_per_10min': ecg_section.getint('n_breaks_bursts_allowed_per_10min'),
        'allowed_range_of_peaks_stds': ecg_section.getfloat('allowed_range_of_peaks_stds'),
        'norm_lvl': ecg_section.getfloat('norm_lvl'),
        'gaussian_sigma': ecg_section.getint('gaussian_sigma'),
        'thresh_lvl_peakfinder': ecg_section.getfloat('thresh_lvl_peakfinder'),
        'height_multiplier': ecg_section.getfloat('height_multiplier')})

        eog_section = config['EOG']
        all_qc_params['EOG'] = dict({
        'n_breaks_bursts_allowed_per_10min': eog_section.getint('n_breaks_bursts_allowed_per_10min'),
        'allowed_range_of_peaks_stds': eog_section.getfloat('allowed_range_of_peaks_stds'),
        'norm_lvl': eog_section.getfloat('norm_lvl'),
        'gaussian_sigma': ecg_section.getint('gaussian_sigma'),
        'thresh_lvl_peakfinder': eog_section.getfloat('thresh_lvl_peakfinder'),})

        head_section = config['Head_movement']
        all_qc_params['Head'] = dict({})

        muscle_section = config['Muscle']
        list_thresholds = muscle_section['threshold_muscle']
        #separate values in list_thresholds based on coma, remove spaces and convert them to floats:
        list_thresholds = [float(i) for i in list_thresholds.split(',')]
        muscle_freqs = [float(i) for i in muscle_section['muscle_freqs'].split(',')]

        all_qc_params['Muscle'] = dict({
        'threshold_muscle': list_thresholds,
        'min_distance_between_different_muscle_events': muscle_section.getfloat('min_distance_between_different_muscle_events'),
        'muscle_freqs': muscle_freqs,
        'min_length_good': muscle_section.getfloat('min_length_good')})

    except:
        print('___MEGqc___: ', 'Invalid setting in config file! Please check instructions for each setting. \nGeneral directions: \nDon`t write any parameter as None. Don`t use quotes.\nLeaving blank is only allowed for parameters: \n- stim_channel, \n- data_crop_tmin, data_crop_tmax, \n- freq_min and freq_max in Filtering section, \n- all parameters of Filtering section if apply_filtering is set to False.')
        return None

    return all_qc_params

def get_internal_config_params(config_file_name: str):
    
    """
    Parse all the parameters from config and put into a python dictionary 
    divided by sections. Parsing approach can be changed here, which 
    will not affect working of other fucntions.
    These are interanl parameters, NOT to be changed by the user.
    

    Parameters
    ----------
    config_file_name: str
        The name of the config file.

    Returns
    -------
    internal_qc_params: dict
        A dictionary with all the parameters.

    """
    
    internal_qc_params = {}

    config = configparser.ConfigParser()
    config.read(config_file_name)

    ecg_section = config['ECG']
    internal_qc_params['ECG'] = dict({
        'max_n_peaks_allowed_for_ch': ecg_section.getint('max_n_peaks_allowed_for_ch'),
        'max_n_peaks_allowed_for_avg': ecg_section.getint('max_n_peaks_allowed_for_avg'),
        'ecg_epoch_tmin': ecg_section.getfloat('ecg_epoch_tmin'),
        'ecg_epoch_tmax': ecg_section.getfloat('ecg_epoch_tmax'),
        'before_t0': ecg_section.getfloat('before_t0'),
        'after_t0': ecg_section.getfloat('after_t0'),
        'window_size_for_mean_threshold_method': ecg_section.getfloat('window_size_for_mean_threshold_method')})
    
    eog_section = config['EOG']
    internal_qc_params['EOG'] = dict({
        'max_n_peaks_allowed_for_ch': eog_section.getint('max_n_peaks_allowed_for_ch'),
        'max_n_peaks_allowed_for_avg': eog_section.getint('max_n_peaks_allowed_for_avg'),
        'eog_epoch_tmin': eog_section.getfloat('eog_epoch_tmin'),
        'eog_epoch_tmax': eog_section.getfloat('eog_epoch_tmax'),
        'before_t0': eog_section.getfloat('before_t0'),
        'after_t0': eog_section.getfloat('after_t0'),
        'window_size_for_mean_threshold_method': eog_section.getfloat('window_size_for_mean_threshold_method')})
    
    psd_section = config['PSD']
    internal_qc_params['PSD'] = dict({
        'method': psd_section.get('method'),
        'prominence_lvl_pos_avg': psd_section.getint('prominence_lvl_pos_avg'),
        'prominence_lvl_pos_channels': psd_section.getint('prominence_lvl_pos_channels')})
    
    return internal_qc_params


def Epoch_meg(epoching_params, data: mne.io.Raw):

    """
    Epoch MEG data based on the parameters provided in the config file.
    
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

    """

    event_dur = epoching_params['event_dur']
    epoch_tmin = epoching_params['epoch_tmin']
    epoch_tmax = epoching_params['epoch_tmax']
    stim_channel = epoching_params['stim_channel']

    if stim_channel is None:
        picks_stim = mne.pick_types(data.info, stim=True)
        stim_channel = []
        for ch in picks_stim:
            stim_channel.append(data.info['chs'][ch]['ch_name'])
    print('___MEGqc___: ', 'Stimulus channels detected:', stim_channel)

    picks_magn = data.copy().pick('mag').ch_names if 'mag' in data else None
    picks_grad = data.copy().pick('grad').ch_names if 'grad' in data else None

    if not stim_channel:
        print('___MEGqc___: ', 'No stimulus channel detected. Setting stimulus channel to None to allow mne to detect events autamtically.')
        stim_channel = None
        #here for info on how None is handled by mne: https://mne.tools/stable/generated/mne.find_events.html
        #even if stim is None, mne will check once more when creating events.


    epochs_grad, epochs_mag = None, None

    try:
        events = mne.find_events(data, stim_channel=stim_channel, min_duration=event_dur)

        if len(events) < 1:
            print('___MEGqc___: ', 'No events with set minimum duration were found using all stimulus channels. No epoching can be done. Try different event duration in config file.')
        else:
            print('___MEGqc___: ', 'Events found:', len(events))
            epochs_mag = mne.Epochs(data, events, picks=picks_magn, tmin=epoch_tmin, tmax=epoch_tmax, preload=True, baseline = None, event_repeated=epoching_params['event_repeated'])
            epochs_grad = mne.Epochs(data, events, picks=picks_grad, tmin=epoch_tmin, tmax=epoch_tmax, preload=True, baseline = None, event_repeated=epoching_params['event_repeated'])

    except: #case when we use stim_channel=None, mne checks once more,  finds no other stim ch and no events and throws error:
        print('___MEGqc___: ', 'No stim channels detected, no events found.')
        pass #go to returning empty dict
        
    
    dict_epochs_mg = {
    'mag': epochs_mag,
    'grad': epochs_grad}

    return dict_epochs_mg

def get_units(raw):

    """
    UNFINISHED!! TODO
    For CTF especially so far not clear what are grads, what mags.

    Get what kind of channels present: mags, grads or both.
    Get units for each channel type.
    """

    picked_channels = mne.pick_types(raw.info, meg='mag')
    for ch in picked_channels:
        ch_name = raw.info['ch_names'][ch]
        print(f"Channel: {ch_name}")
        ch_unit_code = raw.info['chs'][ch]['unit']
        print(str(ch_unit_code))
        ch_unit_str = str(ch_unit_code)
        #find str after 'UNIT_' in ch_unit_code:
        match = re.search(r'UNIT_(\w)', ch_unit_str)
        if match:
            unit = match.group(1)
        else:
            unit = 'unknown'
        print(f"Unit: {unit}")


def check_chosen_ch_types(m_or_g_chosen, channels_objs):
    
    """
    Check if the channels which the user gave in config file to analize actually present in the data set.
    
    Parameters
    ----------
    m_or_g_chosen : list
        List with channel types to analize: mag, grad. These are theones the user chose.
    channels_objs : dict
        Dictionary with channel names for each channel type: mag, grad. These are the ones present in the data set.
    
    Returns
    -------
    m_or_g_chosen : list
        List with channel types to analize: mag, grad.
    m_or_g_skipped_str : str
        String with information about which channel types were skipped.
        
    """

    if not any(ch in m_or_g_chosen for ch in ['mag', 'grad']):
        skipped_str = "No channels to analyze. Check parameter ch_types in config file."
        raise ValueError(skipped_str)

    skipped_msgs = {
        'mag': "There are no magnetometers in this data set: check parameter ch_types in config file. Analysis will be done only for gradiometers.",
        'grad': "There are no gradiometers in this data set: check parameter ch_types in config file. Analysis will be done only for magnetometers."
    }

    for ch in ['mag', 'grad']:
        if len(channels_objs[ch]) == 0 and ch in m_or_g_chosen:
            skipped_str = skipped_msgs[ch]
            print(f'___MEGqc___: {skipped_str}')
            m_or_g_chosen.remove(ch)

    if not any(channels_objs[ch] for ch in ['mag', 'grad']):
        skipped_str = "There are no magnetometers nor gradiometers in this data set. Analysis will not be done."
        raise ValueError(skipped_str)

    # Now m_or_g_chosen contain only those channel types which are present in the data set and were chosen by the user.
        
    return m_or_g_chosen, skipped_str


def load_data(file_path):

    """
    Load MEG data from a file. It can be a CTF data or a FIF file.

    Parameters
    ----------
    file_path : str
        Path to the fif file with MEG data.

    Returns
    -------
    raw : mne.io.Raw
        MEG data.
    shielding_str : str
        String with information about active shielding.

    """

    shielding_str = ''

    meg_system = None

    if os.path.isdir(file_path) and file_path.endswith('.ds'):
        # It's a CTF data directory
        print("___MEGqc___: ", "Loading CTF data...")
        raw = mne.io.read_raw_ctf(file_path, preload=True)
        meg_system = 'CTF'

    elif os.path.isfile(file_path) and file_path.endswith('.fif'):
        # It's a FIF file
        meg_system = 'Triux'

        print("___MEGqc___: ", "Loading FIF data...")
        try:
            raw = mne.io.read_raw_fif(file_path, on_split_missing='ignore')
        except: 
            raw = mne.io.read_raw_fif(file_path, allow_maxshield=True, on_split_missing='ignore')
            shielding_str=''' <p>This fif file contains Internal Active Shielding data. Quality measurements calculated on this data should not be compared to the measuremnts calculated on the data without active shileding, since in the current case invironmental noise reduction was already partially performed by shileding, which normally should not be done before assesing the quality.</p>'''

    else:
        raise ValueError("Unsupported file format or file does not exist. The pipeline works with CTF data directories and FIF files.")
    
    return raw, shielding_str, meg_system


def initial_processing(default_settings: dict, filtering_settings: dict, epoching_params:dict, file_path: str):

    """
    Here all the initial actions needed to analyse MEG data are done: 

    - read fif file,
    - separate mags and grads names into 2 lists,
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
    chs_by_lobe : dict
        Dictionary with channel objects for each channel type: mag, grad. And by lobe. Each obj hold info about the channel name, 
        lobe area and color code, locations and (in the future) pther info, like: if it has noise of any sort.
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
    info_derivs : list
        List with QC_derivative objects with MNE info object.
    shielding_str : str
        String with information about active shielding.
    epoching_str : str
        String with information about epoching.
    sensors_derivs : list
        List with data frames with sensors info.
    time_series_derivs : list
        List with data frames with time series info.
    time_series_str : str
        String with information about time series plotting for report.
    m_or_g_chosen : list
        List with channel types to analize: mag, grad.
    m_or_g_skipped_str : str
        String with information about which channel types were skipped.
    lobes_color_coding_str : str
        String with information about color coding for lobes.
    plot_legend_use_str : str
        String with information about using the plot legend, where to click to hide/show channels.
    resample_str : str
        String with information about resampling.
    
    """


    print('___MEGqc___: ', 'Reading data from file:', file_path)

    raw, shielding_str, meg_system = load_data(file_path)

    info = raw.info
    info_derivs = [QC_derivative(content = info, name = 'RawInfo', content_type = 'info', fig_order=-1)]


    # from IPython.display import display
    # display(raw)

    #crop the data to calculate faster:
    tmax_possible = raw.times[-1] 
    tmax=default_settings['crop_tmax']
    if tmax is None or tmax > tmax_possible: 
        tmax = tmax_possible 
    raw_cropped = raw.copy().crop(tmin=default_settings['crop_tmin'], tmax=tmax)
    #When resampling for plotting, cropping or anything else you don't need permanent in raw inside any functions - always do raw_new=raw.copy() not just raw_new=raw. The last command doesn't create a new object, the whole raw will be changed and this will also be passed to other functions even if you don't return the raw.


    #Data filtering:
    raw_cropped_filtered = raw_cropped.copy()
    if filtering_settings['apply_filtering'] is True:
        raw_cropped.load_data() #Data has to be loaded into mememory before filetering:
        raw_cropped_filtered = raw_cropped.copy()

        #if filtering_settings['h_freq'] is higher than the Nyquist frequency, set it to Nyquist frequency:
        if filtering_settings['h_freq'] > raw_cropped_filtered.info['sfreq']/2 - 1:
            filtering_settings['h_freq'] = raw_cropped_filtered.info['sfreq']/2 - 1
            print('___MEGqc___: ', 'High frequency for filtering is higher than Nyquist frequency. High frequency was set to Nyquist frequency:', filtering_settings['h_freq'])
        raw_cropped_filtered.filter(l_freq=filtering_settings['l_freq'], h_freq=filtering_settings['h_freq'], picks='meg', method=filtering_settings['method'], iir_params=None)
        print('___MEGqc___: ', 'Data filtered from', filtering_settings['l_freq'], 'to', filtering_settings['h_freq'], 'Hz.')
        
        if filtering_settings['downsample_to_hz'] is False:
            raw_cropped_filtered_resampled = raw_cropped_filtered.copy()
            resample_str = 'Data not resampled. '
            print('___MEGqc___: ', resample_str)
        elif filtering_settings['downsample_to_hz'] >= filtering_settings['h_freq']*5:
            raw_cropped_filtered_resampled = raw_cropped_filtered.copy().resample(sfreq=filtering_settings['downsample_to_hz'])
            resample_str = 'Data resampled to ' + str(filtering_settings['downsample_to_hz']) + ' Hz. '
            print('___MEGqc___: ', resample_str)
        else:
            raw_cropped_filtered_resampled = raw_cropped_filtered.copy().resample(sfreq=filtering_settings['h_freq']*5)
            #frequency to resample is 5 times higher than the maximum chosen frequency of the function
            resample_str = 'Chosen "downsample_to_hz" value set was too low, it must be at least 5 time higher than the highest filer frequency. Data resampled to ' + str(filtering_settings['h_freq']*5) + ' Hz. '
            print('___MEGqc___: ', resample_str)

            
    else:
        print('___MEGqc___: ', 'Data not filtered.')
        #And downsample:
        if filtering_settings['downsample_to_hz'] is not False:
            raw_cropped_filtered_resampled = raw_cropped_filtered.copy().resample(sfreq=filtering_settings['downsample_to_hz'])
            if filtering_settings['downsample_to_hz'] < 500:
                resample_str = 'Data resampled to ' + str(filtering_settings['downsample_to_hz']) + ' Hz. Keep in mind: resampling to less than 500Hz is not recommended, since it might result in high frequency data loss (for example of the CHPI coils signal. '
                print('___MEGqc___: ', resample_str)
            else:
                resample_str = 'Data resampled to ' + str(filtering_settings['downsample_to_hz']) + ' Hz. '
                print('___MEGqc___: ', resample_str)
        else:
            raw_cropped_filtered_resampled = raw_cropped_filtered.copy()
            resample_str = 'Data not resampled. '
            print('___MEGqc___: ', resample_str)

        
    #Apply epoching: USE NON RESAMPLED DATA. Or should we resample after epoching? 
    # Since sampling freq is 1kHz and resampling is 500Hz, it s not that much of a win...

    dict_epochs_mg = Epoch_meg(epoching_params, data=raw_cropped_filtered)
    epoching_str = ''
    if dict_epochs_mg['mag'] is None and dict_epochs_mg['grad'] is None:
        epoching_str = ''' <p>No epoching could be done in this data set: no events found. Quality measurement were only performed on the entire time series. If this was not expected, try: 1) checking the presence of stimulus channel in the data set, 2) setting stimulus channel explicitly in config file, 3) setting different event duration in config file.</p><br></br>'''


    #Get channels and their properties. Currently not used in pipeline. But this might be a useful dictionary form if later want do add more information about each channels.
    #In this dict channels are separated by mag/grads. not by lobes.
    channels_objs, lobes_color_coding_str = assign_channels_properties(raw, meg_system)

    #Check if there are channels to analyze according to info in config file:
    m_or_g_chosen, m_or_g_skipped_str = check_chosen_ch_types(m_or_g_chosen=default_settings['m_or_g_chosen'], channels_objs=channels_objs)

    #Sort channels by lobe - this will be used often for plotting
    chs_by_lobe = sort_channels_by_lobe(channels_objs)
    print('___MEGqc___: ', 'Channels sorted by lobe.')

    #Get channels names - these will be used all over the pipeline. Holds only names of channels that are to be analyzed:
    channels={'mag': [ch.name for ch in channels_objs['mag']], 'grad': [ch.name for ch in channels_objs['grad']]}


    #Plot time series:
    #TODO: we still plot time series here? Decide if we dont need it at all or if we need it in some other form.

    time_series_derivs = []
    
    # for m_or_g in m_or_g_chosen:
    #     if default_settings['plot_interactive_time_series'] is True:
    #         time_series_derivs += plot_time_series(raw_cropped_filtered, m_or_g, chs_by_lobe[m_or_g])
    #     if default_settings['plot_interactive_time_series_average'] is True:
    #         time_series_derivs += plot_time_series_avg(raw_cropped, m_or_g)

    if time_series_derivs:
        time_series_str="For this visialisation the data is resampled to 100Hz but not filtered. If cropping was chosen in settings the cropped raw is presented here, otherwise - entire duratio."
    else:
        time_series_str = 'No time series plot was generated. To generate it, set plot_interactive_time_series or(and) plot_interactive_time_series_average to True in settings.'

    plot_legend_use_str = "<p></p><p>On each interactive plot: <br> - click twice on the legend to hide/show a group of channels;<br> - click one to hide/show individual channels;<br> - hover over the dot/line to see information about channel an metric value.</li></ul></p>"

    resample_str = '<p>' + resample_str + '</p>'

    #Extract chs_by_lobe into a data frame
    sensors_derivs = chs_dict_to_csv(chs_by_lobe,  file_name_prefix = 'Sensors')

    return meg_system, dict_epochs_mg, chs_by_lobe, channels, raw_cropped_filtered, raw_cropped_filtered_resampled, raw_cropped, raw, info_derivs, shielding_str, epoching_str, sensors_derivs, time_series_derivs, time_series_str, m_or_g_chosen, m_or_g_skipped_str, lobes_color_coding_str, plot_legend_use_str, resample_str


def chs_dict_to_csv(chs_by_lobe: dict, file_name_prefix: str):

    """
    Convert dictionary with channels objects to a data frame and save it as a csv file.

    Parameters
    ----------
    chs_by_lobe : dict
        Dictionary with channel objects for each channel type: mag, grad. And by lobe. Each obj hold info about the channel name, 
        lobe area and color code, locations and (in the future) pther info, like: if it has noise of any sort.
    file_name_prefix : str
        Prefix for the file name. Example: 'Sensors' will result in file name 'Sensors.csv'.

    Returns
    -------
    df_deriv : list
        List with data frames with sensors info.

    """

    #Extract chs_by_lobe into a data frame
    chs_by_lobe_df = {k1: {k2: pd.concat([channel.to_df() for channel in v2]) for k2, v2 in v1.items()} for k1, v1 in chs_by_lobe.items()}

    its = []
    for ch_type, content in chs_by_lobe_df.items():
        for lobe, items in content.items():
            its.append(items)

    df_fin = pd.concat(its)

    # if df already contains columns like 'STD epoch_' with numbers, 'STD epoch' needs to be removed from the data frame:
    if 'STD epoch' in df_fin and any(col.startswith('STD epoch_') and col[10:].isdigit() for col in df_fin.columns):
        # If there are, drop the 'STD epoch' column
        df_fin = df_fin.drop(columns='STD epoch')
    if 'PtP epoch' in df_fin and any(col.startswith('PtP epoch_') and col[10:].isdigit() for col in df_fin.columns):
        # If there are, drop the 'PtP epoch' column
        df_fin = df_fin.drop(columns='PtP epoch')
    if 'PSD' in df_fin and any(col.startswith('PSD_') and col[4:].isdigit() for col in df_fin.columns):
        # If there are, drop the 'STD epoch' column
        df_fin = df_fin.drop(columns='PSD')
    if 'ECG' in df_fin and any(col.startswith('ECG_') and col[4:].isdigit() for col in df_fin.columns):
        # If there are, drop the 'STD epoch' column
        df_fin = df_fin.drop(columns='ECG')
    if 'EOG' in df_fin and any(col.startswith('EOG_') and col[4:].isdigit() for col in df_fin.columns):
        # If there are, drop the 'STD epoch' column
        df_fin = df_fin.drop(columns='EOG')


    df_deriv = [QC_derivative(content = df_fin, name = file_name_prefix, content_type = 'df')]

    return df_deriv
import mne
import configparser
import numpy as np

from IPython.display import display
from universal_plots import plot_sensors_3d, plot_time_series


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

    m_or_g_chosen = default_section['do_for'] 
    m_or_g_chosen = m_or_g_chosen.replace(" ", "")
    m_or_g_chosen = m_or_g_chosen.split(",")
    if 'mag' not in m_or_g_chosen and 'grad' not in m_or_g_chosen:
        print('___MEG QC___: ', 'No channels to analyze. Check parameter do_for in config file.')
        return None

    ds_paths = default_section['data_directory']
    ds_paths = ds_paths.replace(" ", "")
    ds_paths = ds_paths.split(",")
    if len(ds_paths) < 1:
        print('___MEG QC___: ', 'No datasets to analyze. Check parameter data_directory in config file. Data path can not contain spaces! You can replace them with underscores or remove completely.')
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
            'dataset_path': ds_paths,
            'plot_interactive_time_series': default_section.getboolean('plot_interactive_time_series'),
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
            all_qc_params['Filtering'] = False


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
        'ecg_epoch_tmin': ecg_section.getfloat('ecg_epoch_tmin'),
        'ecg_epoch_tmax': ecg_section.getfloat('ecg_epoch_tmax'),
        'norm_lvl': ecg_section.getfloat('norm_lvl'),
        'flip_data': ecg_section.getboolean('flip_data')})

        eog_section = config['EOG']
        all_qc_params['EOG'] = dict({
        'n_breaks_bursts_allowed_per_10min': eog_section.getint('n_breaks_bursts_allowed_per_10min'),
        'allowed_range_of_peaks_stds': eog_section.getfloat('allowed_range_of_peaks_stds'),
        'eog_epoch_tmin': eog_section.getfloat('eog_epoch_tmin'),
        'eog_epoch_tmax': eog_section.getfloat('eog_epoch_tmax'),
        'norm_lvl': eog_section.getfloat('norm_lvl'),
        'flip_data': eog_section.getboolean('flip_data')})

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
        print('___MEG QC___: ', 'Invalid setting in config file! Please check instructions for each setting. \nGeneral directions: \nDon`t write any parameter as None. Don`t use quotes.\nLeaving blank is only allowed for parameters: \n- stim_channel, \n- data_crop_tmin, data_crop_tmax, \n- freq_min and freq_max in Filtering section, \n- all parameters of Filtering section if apply_filtering is set to False.')
        return None

    return all_qc_params


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


def sanity_check(m_or_g_chosen, channels):
    
    """
    Check if the channels which the user gave in config file to analize actually present in the data set.
    
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
        
    """

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


class MEG_channels:

    """ 
    Channel with info about it such as name, type, lobe area and color code for plotting.

    Attributes
    ----------
    name : str
        The name of the channel.
    type : str
        The type of the channel: 'mag', 'grad'
    lobe_area : str
        The lobe area of the channel: 'left frontal', 'right frontal', 'left temporal', 'right temporal', 'left parietal', 'right parietal', 'left occipital', 'right occipital', 'central', 'subcortical', 'unknown'.
    color_code : str
        The color code for plotting with plotly according to the lobe area of the channel.


    """

    def __init__(self, name, type, lobe_area, color_code):

        """
        Constructor method
        
        Parameters
        ----------
        name : str
            The name of the channel.
        type : str
            The type of the channel: 'mag', 'grad'
        lobe_area : str
            The lobe area of the channel: 'left frontal', 'right frontal', 'left temporal', 'right temporal', 'left parietal', 'right parietal', 'left occipital', 'right occipital', 'central', 'subcortical', 'unknown'.
        color_code : str
            The color code for plotting with plotly according to the lobe area of the channel.

        """

        self.name = name
        self.type = type
        self.lobe_area = lobe_area
        self.color_code = color_code

    def __repr__(self):

        """
        Returns the string representation of the object.
        
        """

        return self.name, f': type: {self.type}, lobe area: {self.lobe_area}, color code: {self.color_code}'



def assign_channels_to_areas(channels):

    if len(channels['mag']) == 102 and len(channels['grad']) == 204: #for 306 channel data in Elekta/Neuromag Treux system
        #loop over all values in the dictionary:
        for key, ch in channels.items():
            pass
    
    lobes = {'Left frontal': ['MEG0121', 'MEG0122', 'MEG0123', 'MEG0341', 'MEG0342', 'MEG0343', 'MEG0321', 'MEG0322', 'MEG0323', 'MEG0331',  'MEG0332', 'MEG0333', 'MEG0643', 'MEG0642', 'MEG0641', 'MEG0611', 'MEG0612'. 'MEG0613', 'MEG0541', 'MEG0542', 'MEG0543', 'MEG0311', 'MEG0312', 'MEG0313', 'MEG0511', 'MEG0512', 'MEG0513', 'MEG0521', 'MEG0522', 'MEG0523', 'MEG0531', 'MEG0532', 'MEG0533'],
            'Right frontal': ['MEG0811', 'MEG0812', 'MEG0813', 'MEG0911', 'MEG0912', 'MEG0913', 'MEG0921', 'MEG0922', 'MEG0923', 'MEG0931', 'MEG0932', 'MEG0933', 'MEG0941', 'MEG0942', 'MEG0943', 'MEG1011', 'MEG1012', 'MEG1013', 'MEG1021', 'MEG1022', 'MEG1023', 'MEG1031', 'MEG1032', 'MEG1033', 'MEG1211', 'MEG1212', 'MEG1213', 'MEG1221', 'MEG1222', 'MEG1223', 'MEG1231', 'MEG1232', 'MEG1233', 'MEG1241', 'MEG1242', 'MEG1243', 'MEG1411', 'MEG1412', 'MEG1413'],
            'Left temporal': ['MEG0111', 'MEG0112', 'MEG0113', 'MEG0131', 'MEG0132', 'MEG0133', 'MEG0141', 'MEG0142', 'MEG0143', 'MEG0211', 'MEG0212', 'MEG0213', 'MEG0221', 'MEG0222', 'MEG0223', 'MEG0231', 'MEG0232', 'MEG0233', 'MEG0241', 'MEG0242', 'MEG0243', 'MEG1511', 'MEG1512', 'MEG1513', 'MEG1521', 'MEG1522', 'MEG1523', 'MEG1531', 'MEG1532', 'MEG1533', 'MEG1541', 'MEG1542', 'MEG1543', 'MEG1611', 'MEG1612', 'MEG1613', 'MEG1621', 'MEG1622', 'MEG1623'],
            'Right temporal': ['MEG1311', 'MEG1312', 'MEG1313', 'MEG1321', 'MEG1322', 'MEG1333', 'MEG1421', 'MEG1422', 'MEG1423', 'MEG1431', 'MEG1432', 'MEG1433', 'MEG1441', 'MEG1442', 'MEG1443', 'MEG1341', 'MEG1342', 'MEG1343', 'MEG1331', 'MEG1332', 'MEG1333', 'MEG2611', 'MEG2612', 'MEG2613', 'MEG2621', 'MEG2622', 'MEG2623', 'MEG2631', 'MEG2632', 'MEG2633', 'MEG2641', 'MEG2642', 'MEG2643', 'MEG2411', 'MEG2412', 'MEG2413', 'MEG2421', 'MEG2422', 'MEG2423'],
            'Left Parietal': ['MEG0411', 'MEG0412', 'MEG0413', 'MEG0421', 'MEG0422', 'MEG0423', 'MEG0431', 'MEG0432', 'MEG0433', 'MEG0441', 'MEG0442', 'MEG0443', 'MEG0711', 'MEG0712', 'MEG0713', 'MEG0741', 'MEG0742', 'MEG0743', 'MEG1811', 'MEG1812', 'MEG1813', 'MEG1821', 'MEG1822', 'MEG1823', 'MEG1831', 'MEG1832', 'MEG1833', 'MEG1841', 'MEG1842', 'MEG1843', 'MEG061', 'MEG0632', 'MEG0633', 'MEG1631', 'MEG1632', 'MEG1633', 'MEG2011', 'MEG2012', 'MEG2013'],
            'Right Parietal': ['MEG1041', 'MEG1042', 'MEG1043', ],'}
             
            


def initial_processing(default_settings: dict, filtering_settings: dict, epoching_params:dict, data_file: str):

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
    epoching_str : str
        String with information about epoching.
    
    """

    print('___MEG QC___: ', 'Reading data from file:', data_file)

    try:
        raw = mne.io.read_raw_fif(data_file, on_split_missing='ignore')
        shielding_str = ''
    except: 
        raw = mne.io.read_raw_fif(data_file, allow_maxshield=True, on_split_missing='ignore')
        shielding_str=''' <p>This file contains Internal Active Shielding data. Quality measurements calculated on this data should not be compared to the measuremnts calculated on the data without active shileding, since in the current case invironmental noise reduction was already partially performed by shileding, which normally should not be done before assesing the quality.</p>'''

    display(raw)

    #crop the data to calculate faster:
    tmax=default_settings['crop_tmax']
    if tmax is None: 
        tmax = raw.times[-1] 
    raw_cropped = raw.copy().crop(tmin=default_settings['crop_tmin'], tmax=tmax)


    #Data filtering:
    raw_cropped_filtered = raw_cropped.copy()
    if filtering_settings is not False:
        raw_cropped.load_data() #Data has to be loaded into mememory before filetering:
        raw_cropped_filtered = raw_cropped.copy()

        #if filtering_settings['h_freq'] is higher than the Nyquist frequency, set it to Nyquist frequency:
        if filtering_settings['h_freq'] > raw_cropped_filtered.info['sfreq']/2 - 1:
            filtering_settings['h_freq'] = raw_cropped_filtered.info['sfreq']/2 - 1
            print('___MEG QC___: ', 'High frequency for filtering is higher than Nyquist frequency. High frequency was set to Nyquist frequency:', filtering_settings['h_freq'])
        raw_cropped_filtered.filter(l_freq=filtering_settings['l_freq'], h_freq=filtering_settings['h_freq'], picks='meg', method=filtering_settings['method'], iir_params=None)
        print('___MEG QC___: ', 'Data filtered from', filtering_settings['l_freq'], 'to', filtering_settings['h_freq'], 'Hz.')
        
        #And downsample:
        raw_cropped_filtered_resampled = raw_cropped_filtered.copy().resample(sfreq=filtering_settings['h_freq']*5)
        #frequency to resample is 5 times higher than the maximum chosen frequency of the function
        print('___MEG QC___: ', 'Data resampled to', filtering_settings['h_freq']*5, 'Hz.')
    else:
        raw_cropped_filtered_resampled = raw_cropped_filtered.copy()
        #OR maybe we dont need these 2 copies of data at all? Think how to get rid of them, 
        # because they are used later. Referencing might mess up things, check that.
        print('___MEG QC___: ', 'Data not filtered.')
    

    #Apply epoching: USE NON RESAMPLED DATA. Or should we resample after epoching? 
    # Since sampling freq is 1kHz and resampling is 500Hz, it s not that much of a win...

    dict_epochs_mg = Epoch_meg(epoching_params, data=raw_cropped_filtered)
    epoching_str = ''
    if dict_epochs_mg['mag'] is None and dict_epochs_mg['grad'] is None:
        epoching_str = ''' <p>No epoching could be done in this data set: no events found. Quality measurement were only performed on the entire time series. If this was not expected, try: 1) checking the presence of stimulus channel in the data set, 2) setting stimulus channel explicitly in config file, 3) setting different event duration in config file.</p><br></br>'''
            

    mag_ch_names = raw.copy().pick_types(meg='mag').ch_names if 'mag' in raw else None
    grad_ch_names = raw.copy().pick_types(meg='grad').ch_names if 'grad' in raw else None

    channels = {'mag': [MEG_channels(m, 'mag', None, None) for m in mag_ch_names] if mag_ch_names is not None else None,
                'grad': [MEG_channels(g, 'grad', None, None) for g in grad_ch_names] if grad_ch_names is not None else None}

    #Check if there are channels to analyze:
    m_or_g_chosen = sanity_check(m_or_g_chosen=default_settings['m_or_g_chosen'], channels=channels)
    m_or_g_skipped_str = ''
    if len(m_or_g_chosen) == 0: 
        m_or_g_skipped_str = '''<p>No channels to analyze. Check presence of mag and grad in your data set and parameter do_for in settings.</p>'''
        raise ValueError(m_or_g_skipped_str)
    if 'mag' not in m_or_g_chosen:
        m_or_g_skipped_str = ''' <p>This data set contains no magnetometers or they were not chosen for analysis. Quality measurements were performed only on gradiometers.</p>'''
    if 'grad' not in m_or_g_chosen:
        m_or_g_skipped_str = ''' <p>This data set contains no gradiometers or they were not chosen for analysis. Quality measurements were performed only on magnetometers.</p>'''





    #Plot sensors:
    sensors_derivs = plot_sensors_3d(raw)

    #Plot time series:
    time_series_derivs = []
    if default_settings['plot_interactive_time_series'] is True:
        time_series_str="For this visialisation the data is resampled to 100Hz but not filtered. If cropping was chosen in settings the cropped raw is presented here, otherwise - entire duratio."
        for ch_type in m_or_g_chosen:
            time_series_derivs += plot_time_series(raw_cropped, ch_type)
    else:
        time_series_str = 'No time series plot was generated. To generate it, set plot_interactive_time_series to True in settings.'
        time_series_derivs = []
        
    return dict_epochs_mg, channels, raw_cropped_filtered, raw_cropped_filtered_resampled, raw_cropped, raw, shielding_str, epoching_str, sensors_derivs, time_series_derivs, time_series_str, m_or_g_chosen, m_or_g_skipped_str

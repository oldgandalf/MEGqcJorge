import mne
import numpy as np
from universal_html_report import simple_metric_basic
from universal_plots import QC_derivative, get_tit_and_unit
import plotly.graph_objects as go
from scipy.signal import find_peaks


def check_3_conditions(picked: str, ch_data: list or np.ndarray, fs: int, ecg_or_eog: str, n_breaks_allowed_per_10min: int = 3, allowed_range_of_peaks_stds: float = 0.05):

    """
    Check if the ECG/EOG channel is not corrupted using 3 conditions:
    - peaks have similar amplitude
    - intervals between ECG/EOG events are in the normal healthy human range (extended the range in case of a special experiment)
    - recording has no or only a few breaks

    Parameters
    ----------
    picked: str
        Named of picked ECG/EOG channel
    ch_data: list or np.ndarray
        ECG/EOG channel data
    fs: int
        Sampling frequency
    ecg_or_eog: str
        'ECG' or 'EOG'
    n_breaks_allowed_per_10min: int
        Number of breaks allowed per 10 minutes of recording
    allowed_range_of_peaks_stds : float, optional
        Allowed range of peaks standard deviations. The default is 0.05.
        
        - The channel data will be scaled from 0 to 1, so the setting is universal for all data sets.
        - The peaks will be detected on the scaled data
        - The average std of all peaks has to be within this allowed range, If it is higher - the channel has too high deviation in peaks height and is counted as noisy

    
    
    Returns
    -------
    similar_ampl : bool
        True if peaks have similar amplitude
    mean_rr_interval_ok: bool
        True if intervals between ECG/EOG events are in the normal healthy human range
    no_breaks : bool
        True if recording has no or only a few breaks 
    fig : plotly.graph_objects.Figure
        Figure with ECG/EOG channel data and peaks marked

    
    """

    # 1. Check if R peaks (or EOG peaks)  have similar amplitude. If not - data is too noisy:
    # Find R peaks (or peaks of EOG wave) using find_peaks
    height = np.mean(ch_data) + 1 * np.std(ch_data)
    peaks, _ = find_peaks(ch_data, height=height, distance=round(0.5 * fs)) #assume there are no peaks within 0.5 seconds from each other.


    # scale ecg data between 0 and 1: here we dont care about the absolute values. important is the pattern: 
    # are the peak magnitudes the same on average or not? Since absolute values and hence mean and std 
    # can be different for different data sets, we can just scale everything between 0 and 1 and then
    # compare the peak magnitudes
    ch_data_scaled = (ch_data - np.min(ch_data))/(np.max(ch_data) - np.min(ch_data))
    peak_amplitudes = ch_data_scaled[peaks]

    amplitude_std = np.std(peak_amplitudes)

    if amplitude_std < allowed_range_of_peaks_stds: 
        similar_ampl = True
        print("___MEG QC___: Peaks have similar amplitudes, amplitude std: ", amplitude_std)
    else:
        similar_ampl = False
        print("___MEG QC___: Peaks do not have similar amplitudes, amplitude std: ", amplitude_std)


    # 2. Calculate RR intervals (time differences between consecutive R peaks)
    rr_intervals = np.diff(peaks) / fs
    mean_RR_dist = np.mean(rr_intervals)

    if ecg_or_eog == 'ECG':
        rr_dist_allowed = [0.6, 1.6] #take possible pulse rate of 100-40 bpm (hense distance between peaks is 0.6-1.6 seconds)
    elif ecg_or_eog == 'EOG':
        rr_dist_allowed = [1, 10] #take possible blink rate of 60-5 per minute (hense distance between peaks is 1-10 seconds). Yes, 60 is a very high rate, but I see this in some data sets often.

    if mean_RR_dist < rr_dist_allowed[0] or mean_RR_dist > rr_dist_allowed[1]: 
        print("___MEG QC___: Mean peak distance is not between " + str(rr_dist_allowed) + " sec. Mean dist: %s sec" % mean_RR_dist)
        mean_rr_interval_ok = False
    else:
        mean_rr_interval_ok = True
        print("___MEG QC___: Mean peak distance is between " + str(rr_dist_allowed) + " sec. Mean dist: %s sec" % mean_RR_dist)

    # 3. Check for breaks in recording:
    # Calculate average time difference and standard deviation
    avg_time_diff = np.mean(rr_intervals)
    std_time_diff = np.std(rr_intervals)

    # Check for large deviations from average time difference
    deviation_threshold = 5  # Set threshold for deviation from average time difference
    deviations = np.abs(rr_intervals - avg_time_diff)

    n_breaks = np.sum(deviations > deviation_threshold * std_time_diff)

    #allow x breaks per 10 minutes:
    if n_breaks > len(rr_intervals)/60*10/n_breaks_allowed_per_10min:
        no_breaks = False
        print("___MEG QC___: There are parts in the data without regular peaks, number of breaks: ", np.sum(deviations > deviation_threshold * std_time_diff))  
    elif 0 < n_breaks < len(rr_intervals)/60*10/n_breaks_allowed_per_10min:
        no_breaks = True
        print("___MEG QC___: There are parts in the data without regular peaks, but within allowed number of breaks, number of breaks total: ", n_breaks, ", average number of breaks per 10 minutes: ", n_breaks/(len(rr_intervals)/60*10))
    else:
        no_breaks = True
        print("___MEG QC___: All parts of the data have regular peaks")


    # Plot the signal using plotly:
    fig = plot_channel(ch_data, peaks, ch_name = picked, fs = fs)

    return (similar_ampl, mean_rr_interval_ok, no_breaks), fig


def plot_channel(ch_data: np.ndarray or list, peaks: np.ndarray or list, ch_name: str, fs: float):

    time = np.arange(len(ch_data))/fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=ch_data, mode='lines', name=ch_name + ' data'))
    fig.add_trace(go.Scatter(x=time[peaks], y=ch_data[peaks], mode='markers', name='peaks'))
    fig.update_layout(xaxis_title='time, s', 
                yaxis = dict(
                showexponent = 'all',
                exponentformat = 'e'),
                yaxis_title='Amplitude',
                title={
                'text': ch_name,
                'y':0.85,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
    fig.show()

    return fig

def detect_noisy_ecg_eog(raw: mne.io.Raw, picked_channels_ecg_or_eog: list[str],  ecg_or_eog: str, n_breaks_allowed_per_10min: int =3, allowed_range_of_peaks_stds: float = 0.05):
    """
    Detects noisy ecg or eog channels.

    The channel is noisy when:

    1. The distance between the peaks of ECG/EOG signal is too large (events are not frequent enoigh for a human) or too small (events are too frequent for a human).
    2. There are too many breaks in the data (indicating lack of heartbeats or blinks for a too long period) -corrupted channel or dustructed recording
    3. Peaks are of significantly different amplitudes (indicating that the channel is noisy).

    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    picked_channels_ecg_or_eog : list[str]
        List of ECH or EOG channel names to be checked.
    ecg_or_eog : str
        'ECG' or 'EOG'
    n_breaks_allowed_per_10min : int, optional
        Number of breaks allowed per 10 minutes of recording. The default is 3.
    allowed_range_of_peaks_stds : float, optional
        Allowed range of peaks standard deviations. The default is 0.05.

        - The channel data will be scaled from 0 to 1, so the setting is universal for all data sets.
        - The peaks will be detected on the scaled data
        - The average std of all peaks has to be within this allowed range, If it is higher - the channel has too high deviation in peaks height and is counted as noisy


    Returns
    -------
    noisy_ch_derivs : list[QC_derivative]
        List of figures (requested channels plots)  as QC_derivative instances.
    bad_ecg_eog : dict
        Dictionary with channel names as keys and 'good' or 'bad' as values.

        
    """

    sfreq=raw.info['sfreq']

    bad_ecg_eog = {}
    noisy_ch_derivs=[]
    for picked in picked_channels_ecg_or_eog:

        ch_data=raw.get_data(picks=picked)[0] 
        # get_data creates list inside of a list becausee expects to create a list for each channel. 
        # but iteration takes 1 ch at a time. this is why [0]

        ecg_eval, fig = check_3_conditions(picked, ch_data, sfreq, ecg_or_eog, n_breaks_allowed_per_10min, allowed_range_of_peaks_stds)
        print(f'___MEG QC___: {picked} satisfied conditions for a good channel: ', ecg_eval)

        if all(ecg_eval):
            print(f'___MEG QC___: Overall good {ecg_or_eog} channel: {picked}')
            bad_ecg_eog[picked] = 'good'
        else:
            print(f'___MEG QC___: Overall bad {ecg_or_eog} channel: {picked}')
            bad_ecg_eog[picked] = 'bad'

        noisy_ch_derivs += [QC_derivative(fig, bad_ecg_eog[picked]+' '+picked, 'plotly', description_for_user = picked+' is '+ bad_ecg_eog[picked]+ ': 1) peaks have similar amplitude: '+str(ecg_eval[0])+', 2) mean interval between peaks as expected: '+str(ecg_eval[1])+', 3) no or few breaks in the data: '+str(ecg_eval[2]))]
        
    return noisy_ch_derivs, bad_ecg_eog


class Avg_artif:
    
    """ 
    Instance of this class:

    - contains average ECG/EOG epoch for a particular channel,
    - calculates its main peak (location and magnitude),
    - evaluates if this epoch is concidered as artifact or not based on the main peak amplitude.
    

    Attributes
    ----------
    name : str
        name of the channel
    mean_artifact_epoch : list
        list of floats, average ecg epoch for a particular channel
    peak_loc : int
        locations of peaks inside the artifact epoch
    peak_magnitude : float
        magnitudes of peaks inside the artifact epoch
    wave_shape : bool
        True if the average epoch has typical wave shape, False otherwise. R wave shape  - for ECG or just a wave shape for EOG.
    artif_over_threshold : bool
        True if the main peak is concidered as artifact, False otherwise. True if artifact sas magnitude over the threshold
    main_peak_loc : int
        location of the main peak inside the artifact epoch
    main_peak_magnitude : float
        magnitude of the main peak inside the artifact epoch

        
    Methods
    -------
    __init__(self, name: str, mean_artifact_epoch:list, peak_loc=None, peak_magnitude=None, wave_shape:bool=None, artif_over_threshold:bool=None, main_peak_loc: int=None, main_peak_magnitude: float=None)
        Constructor
    __repr__(self)
        Returns a string representation of the object

        
    """

    def __init__(self, name: str, mean_artifact_epoch:list, peak_loc=None, peak_magnitude=None, wave_shape:bool=None, artif_over_threshold:bool=None, main_peak_loc: int=None, main_peak_magnitude: float=None):
        """Constructor"""
        
        self.name =  name
        self.mean_artifact_epoch = mean_artifact_epoch
        self.peak_loc = peak_loc
        self.peak_magnitude = peak_magnitude
        self.wave_shape =  wave_shape
        self.artif_over_threshold = artif_over_threshold
        self.main_peak_loc = main_peak_loc
        self.main_peak_magnitude = main_peak_magnitude

    def __repr__(self):
        """
        Returns a string representation of the object
        
        """

        return 'Mean artifact peak on: ' + str(self.name) + '\n - peak location inside artifact epoch: ' + str(self.peak_loc) + '\n - peak magnitude: ' + str(self.peak_magnitude) +'\n - main_peak_loc: '+ str(self.main_peak_loc) +'\n - main_peak_magnitude: '+str(self.main_peak_magnitude)+'\n wave_shape: '+ str(self.wave_shape) + '\n - artifact magnitude over threshold: ' + str(self.artif_over_threshold)+ '\n'
    
    def get_peaks_wave(self, max_n_peaks_allowed, thresh_lvl_peakfinder=None):

        """
        Find peaks in the average artifact epoch and decide if the epoch has wave shape: 
        few peaks (different number allowed for ECG and EOG) - wave shape, many or no peaks - not.
        Many peaks would mean that the epoch a mean over artifact-free data, looking noisy due to the lack of the pattern.
        
        Parameters
        ----------
        max_n_peaks_allowed : int
            maximum number of peaks allowed in the average artifact epoch
        thresh_lvl_peakfinder : float
            threshold for peakfinder function
            
        Returns
        -------
        peak_loc : list
            locations of peaks inside the artifact epoch
        peak_magnitudes : list
            magnitudes of peaks inside the artifact epoch
        peak_locs_pos : list
            locations of positive peaks inside the artifact epoch
        peak_locs_neg : list
            locations of negative peaks inside the artifact epoch
        peak_magnitudes_pos : list
            magnitudes of positive peaks inside the artifact epoch
        peak_magnitudes_neg : list
            magnitudes of negative peaks inside the artifact epoch
        
            
        """
        
        peak_locs_pos, peak_locs_neg, peak_magnitudes_pos, peak_magnitudes_neg = find_epoch_peaks(ch_data=self.mean_artifact_epoch, thresh_lvl_peakfinder=thresh_lvl_peakfinder)
        
        self.peak_loc=np.concatenate((peak_locs_pos, peak_locs_neg), axis=None)

        if np.size(self.peak_loc)==0: #no peaks found - set peaks as just max of the whole epoch
            self.peak_loc=np.array([np.argmax(np.abs(self.mean_artifact_epoch))])
            self.wave_shape=False
        elif 1<=len(self.peak_loc)<=max_n_peaks_allowed:
            self.wave_shape=True
        elif len(self.peak_loc)>max_n_peaks_allowed:
            self.wave_shape=False
        else:
            self.wave_shape=False
            print('___MEG QC___: ', self.name + ': no expected artifact wave shape, check the reason!')

        self.peak_magnitude=np.array(self.mean_artifact_epoch[self.peak_loc])

        peak_locs=np.concatenate((peak_locs_pos, peak_locs_neg), axis=None)
        peak_magnitudes=np.concatenate((peak_magnitudes_pos, peak_magnitudes_neg), axis=None)

        return peak_locs, peak_magnitudes, peak_locs_pos, peak_locs_neg, peak_magnitudes_pos, peak_magnitudes_neg


    def plot_epoch_and_peak(self, fig, t, fig_tit, ch_type):

        """
        Plot the average artifact epoch and the peak inside it.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            figure to plot the epoch and the peak
        t : list
            time vector
        fig_tit: str
            title of the figure
        ch_type: str
            type of the channel ('mag, 'grad')

        Returns
        -------
        fig : plotly.graph_objects.Figure
            figure with the epoch and the peak
        
        
        """

        fig_ch_tit, unit = get_tit_and_unit(ch_type)

        fig.add_trace(go.Scatter(x=np.array(t), y=np.array(self.mean_artifact_epoch), name=self.name))
        fig.add_trace(go.Scatter(x=np.array(t[self.peak_loc]), y=self.peak_magnitude, mode='markers', name='peak: '+self.name));

        fig.update_layout(
            xaxis_title='Time in seconds',
            yaxis = dict(
                showexponent = 'all',
                exponentformat = 'e'),
            yaxis_title='Mean artifact magnitude in '+unit,
            title={
                'text': fig_tit+fig_ch_tit,
                'y':0.85,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        return fig

    def get_highest_peak(self, t: np.ndarray, timelimit_min: float, timelimit_max: float):

        """
        Find the highest peak of the artifact epoch inside the give time window. 
        Time window is centered around the t0 of the ecg/eog event and limited by timelimit_min and timelimit_max.
        

        Parameters
        ----------
        t : list
            time vector
        timelimit_min : float
            minimum time limit for the peak
        timelimit_max : float
            maximum time limit for the peak
            
        Returns
        -------
        main_peak_magnitude : float
            magnitude of the main peak
        main_peak_loc : int
            location of the main peak
        
        
        """

        if self.peak_loc is None:
            self.main_peak_magnitude=None
            self.main_peak_loc=None
            return None, None

        self.main_peak_magnitude = -1000
        for peak_loc in self.peak_loc:
            if timelimit_min<t[peak_loc]<timelimit_max: #if peak is inside the timelimit_min and timelimit_max was found:
                if self.mean_artifact_epoch[peak_loc] > self.main_peak_magnitude: #if this peak is higher than the previous one:
                    self.main_peak_magnitude=self.mean_artifact_epoch[peak_loc]
                    self.main_peak_loc=peak_loc 
  
        if self.main_peak_magnitude == -1000: #if no peak was found inside the timelimit_min and timelimit_max:
            self.main_peak_magnitude=None
            self.main_peak_loc=None

        return self.main_peak_loc, self.main_peak_magnitude


def detect_channels_above_norm(norm_lvl: float, list_mean_ecg_epochs: list, mean_ecg_magnitude_peak: float, t: np.ndarray, t0_actual: float, ecg_or_eog: str):

    """
    Find the channels which got average artifact amplitude higher than the average over all channels*norm_lvl.
    
    Parameters
    ----------
    norm_lvl : float
        The norm level is the scaling factor for the threshold. The mean artifact amplitude over all channels is multiplied by the norm_lvl to get the threshold.
    list_mean_ecg_epochs : list
        List of MeanArtifactEpoch objects, each hold the information about mean artifact for one channel.
    mean_ecg_magnitude_peak : float
        The magnitude the mean artifact amplitude over all channels.
    t : np.ndarray
        Time vector.
    t0_actual : float
        The time of the ecg/eog event.
    ecg_or_eog : str
        Either 'ECG' or 'EOG'.

    Returns
    -------
    affected_channels : list
        List of channels which got average artifact amplitude higher than the average over all channels*norm_lvl. -> affected by ECG/EOG artifact
    not_affected_channels : list
        List of channels which got average artifact amplitude lower than the average over all channels*norm_lvl. -> not affected by ECG/EOG artifact
    artifact_lvl : float
        The threshold for the artifact amplitude: average over all channels*norm_lvl.
    
    
    """


    if ecg_or_eog=='ECG':
        window_size=0.02
    elif ecg_or_eog=='EOG':
        window_size=0.1
    else:
        print('___MEG QC___: ', 'ecg_or_eog should be either ECG or EOG')

    timelimit_min=-window_size+t0_actual
    timelimit_max=window_size+t0_actual


    #Find the channels which got peaks over this mean:
    affected_channels=[]
    not_affected_channels=[]
    artifact_lvl=mean_ecg_magnitude_peak/norm_lvl #data over this level will be counted as artifact contaminated
    for potentially_affected_channel in list_mean_ecg_epochs:
        #if np.max(np.abs(potentially_affected_channel.peak_magnitude))>abs(artifact_lvl) and potentially_affected_channel.wave_shape is True:


        #find the highest peak inside the timelimit_min and timelimit_max:
        main_peak_loc, main_peak_magnitude = potentially_affected_channel.get_highest_peak(t, timelimit_min, timelimit_max)

        print('___MEG QC___: ', potentially_affected_channel.name, ' Main Peak magn: ', potentially_affected_channel.main_peak_magnitude, ', Main peak loc ', potentially_affected_channel.main_peak_loc, ' Wave shape: ', potentially_affected_channel.wave_shape)
        
        if main_peak_magnitude is not None: #if there is a peak in time window of artifact - check if it s high enough and has right shape
            if main_peak_magnitude>abs(artifact_lvl) and potentially_affected_channel.wave_shape is True:
                potentially_affected_channel.artif_over_threshold=True
                affected_channels.append(potentially_affected_channel)
            else:
                not_affected_channels.append(potentially_affected_channel)
                print('___MEG QC___: ', potentially_affected_channel.name, ' Peak magn over th: ', potentially_affected_channel.main_peak_magnitude>abs(artifact_lvl), ', in the time window: ', potentially_affected_channel.main_peak_loc, ' Wave shape: ', potentially_affected_channel.wave_shape)
        else:
            not_affected_channels.append(potentially_affected_channel)
            print('___MEG QC___: ', potentially_affected_channel.name, ' Peak magn over th: NO PEAK in time window')

    return affected_channels, not_affected_channels, artifact_lvl


def plot_affected_channels(artif_affected_channels: list, artifact_lvl: float, t: np.ndarray, ch_type: str, fig_tit: str, flip_data: bool or str = 'flip'):

    """
    Plot the mean artifact amplitude for all affected (not affected) channels in 1 plot together with the artifact_lvl.
    
    Parameters
    ----------
    artif_affected_channels : list
        List of ECG/EOG artifact affected channels.
    artifact_lvl : float
        The threshold for the artifact amplitude: average over all channels*norm_lvl.
    t : np.ndarray
        Time vector.
    ch_type : str
        Either 'mag' or 'grad'.
    fig_tit: str
        The title of the figure.
    flip_data : bool
        If True, the absolute value of the data will be used for the calculation of the mean artifact amplitude. Default to 'flip'. 
        'flip' means that the data will be flipped if the peak of the artifact is negative. 
        This is donr to get the same sign of the artifact for all channels, then to get the mean artifact amplitude over all channels and the threshold for the artifact amplitude onbase of this mean
        And also for the reasons of visualization: the artifact amplitude is always positive.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure with the mean artifact amplitude for all affected (not affected) channels in 1 plot together with the artifact_lvl.

        
    """

    fig_ch_tit, unit = get_tit_and_unit(ch_type)

    fig=go.Figure()

    for ch in artif_affected_channels:
        fig=ch.plot_epoch_and_peak(fig, t, 'Channels affected by ECG artifact: ', ch_type)

    fig.add_trace(go.Scatter(x=t, y=[(artifact_lvl)]*len(t), name='Thres=mean_peak/norm_lvl'))

    if flip_data == 'False':
        fig.add_trace(go.Scatter(x=t, y=[(-artifact_lvl)]*len(t), name='-Thres=mean_peak/norm_lvl'))

    fig.update_layout(
        xaxis_title='Time in seconds',
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'),
        yaxis_title='Mean artifact magnitude in '+unit,
        title={
            'text': fig_tit+str(len(artif_affected_channels))+' '+fig_ch_tit,
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})


    fig.show()

    return fig


def find_epoch_peaks(ch_data: np.ndarray, thresh_lvl_peakfinder: float):
    
    """
    Find the peaks in the epoch data using the peakfinder algorithm.

    Parameters
    ----------
    ch_data : np.ndarray
        The data of the channel.
    thresh_lvl_peakfinder : float
        The threshold for the peakfinder algorithm.

    Returns
    -------
    peak_locs_pos : np.ndarray
        The locations of the positive peaks.
    peak_locs_neg : np.ndarray
        The locations of the negative peaks.
    peak_magnitudes_pos : np.ndarray
        The magnitudes of the positive peaks.
    peak_magnitudes_neg : np.ndarray
        The magnitudes of the negative peaks.

        
    """

    thresh_mean=(max(ch_data) - min(ch_data)) / thresh_lvl_peakfinder
    peak_locs_pos, _ = find_peaks(ch_data, prominence=thresh_mean)
    peak_locs_neg, _ = find_peaks(-ch_data, prominence=thresh_mean)

    try:
        peak_magnitudes_pos=ch_data[peak_locs_pos]
    except:
        peak_magnitudes_pos=np.empty(0)

    try:
        peak_magnitudes_neg=ch_data[peak_locs_neg]
    except:
        peak_magnitudes_neg=np.empty(0)

    return peak_locs_pos, peak_locs_neg, peak_magnitudes_pos, peak_magnitudes_neg



def flip_channels(avg_ecg_epoch_data_nonflipped: np.ndarray, channels: list, max_n_peaks_allowed: int, thresh_lvl_peakfinder: float, t0_estimated_ind_start: int, t0_estimated_ind_end: int, t0_estimated_ind: int):

    """
    Flip the channels if the peak of the artifact is negative and located close to the estimated t0.

    Parameters
    ----------
    avg_ecg_epoch_data_nonflipped : np.ndarray
        The data of the channels.
    channels : list
        The list of the channels.
    max_n_peaks_allowed : int
        The maximum number of peaks allowed in the epoch.
    thresh_lvl_peakfinder : float
        The threshold for the peakfinder algorithm.
    t0_estimated_ind_start : int
        The start index of the time window for the estimated t0.
    t0_estimated_ind_end : int
        The end index of the time window for the estimated t0.
    t0_estimated_ind : int
        The index of the estimated t0.


    Returns
    -------
    ecg_epoch_per_ch : list
        The list of the ecg epochs.
    avg_ecg_epoch_per_ch_only_data : np.ndarray
        The data of the channels after flipping.


    """

    ecg_epoch_per_ch_only_data=np.empty_like(avg_ecg_epoch_data_nonflipped)
    ecg_epoch_per_ch=[]

    for i, ch_data in enumerate(avg_ecg_epoch_data_nonflipped): 
        ecg_epoch_nonflipped = Avg_artif(name=channels[i], mean_artifact_epoch=ch_data)
        peak_locs, peak_magnitudes, _, _, _, _ = ecg_epoch_nonflipped.get_peaks_wave(max_n_peaks_allowed, thresh_lvl_peakfinder)
        #print('___MEG QC___: ', channels[i], ' peak_locs:', peak_locs)

        #find peak_locs which is located the closest to t0_estimated_ind:
        if peak_locs.size>0:
            peak_loc_closest_to_t0=peak_locs[np.argmin(np.abs(peak_locs-t0_estimated_ind))]

            #if peak_loc_closest_t0 is negative and is located in the estimated time window of the wave - flip the data:
            if (ch_data[peak_loc_closest_to_t0]<0) & (peak_loc_closest_to_t0>t0_estimated_ind_start) & (peak_loc_closest_to_t0<t0_estimated_ind_end):
                ecg_epoch_per_ch_only_data[i]=-ch_data
                peak_magnitudes=-peak_magnitudes
                #print('___MEG QC___: ', channels[i]+' was flipped: peak_loc_near_t0: ', peak_loc_closest_to_t0, t[peak_loc_closest_to_t0], ', peak_magn:', ch_data[peak_loc_closest_to_t0], ', t0_estimated_ind_start: ', t0_estimated_ind_start, t[t0_estimated_ind_start], 't0_estimated_ind_end: ', t0_estimated_ind_end, t[t0_estimated_ind_end])
            else:
                ecg_epoch_per_ch_only_data[i]=ch_data
                #print('___MEG QC___: ', channels[i]+' was NOT flipped: peak_loc_near_t0: ', peak_loc_closest_to_t0, t[peak_loc_closest_to_t0], ', peak_magn:', ch_data[peak_loc_closest_to_t0], ', t0_estimated_ind_start: ', t0_estimated_ind_start, t[t0_estimated_ind_start], 't0_estimated_ind_end: ', t0_estimated_ind_end, t[t0_estimated_ind_end])
        else:
            ecg_epoch_per_ch_only_data[i]=ch_data
            #print('___MEG QC___: ', channels[i]+' was NOT flipped: peak_loc_near_t0: ', peak_loc_closest_to_t0, t[peak_loc_closest_to_t0], ', peak_magn:', ch_data[peak_loc_closest_to_t0], ', t0_estimated_ind_start: ', t0_estimated_ind_start, t[t0_estimated_ind_start], 't0_estimated_ind_end: ', t0_estimated_ind_end, t[t0_estimated_ind_end])
            
        ecg_epoch_per_ch.append(Avg_artif(name=channels[i], mean_artifact_epoch=ecg_epoch_per_ch_only_data[i], peak_loc=peak_locs, peak_magnitude=peak_magnitudes, wave_shape=ecg_epoch_nonflipped.wave_shape))

    return ecg_epoch_per_ch, ecg_epoch_per_ch_only_data


def estimate_t0(ecg_or_eog: str, avg_ecg_epoch_data_nonflipped: list, t: np.ndarray):
    
    """ 
    Estimate t0 for the artifact. MNE has it s own estimation of t0, but it is often not accurate.
    Steps:

    1. find peaks on all channels in time frame around -0.02<t[peak_loc]<0.012 
        (here R wave is typically detected by mne - for ecg, for eog it is -0.1<t[peak_loc]<0.2)
    2. take 5 channels with most prominent peak 
    3. find estimated average t0 for all 5 channels, because t0 of event which mne estimated is often not accurate.
    

    Parameters
    ----------
    ecg_or_eog : str
        The type of the artifact: 'ECG' or 'EOG'.
    avg_ecg_epoch_data_nonflipped : np.ndarray
        The data of the channels.
    t : np.ndarray
        The time vector.
        
    Returns
    -------
    t0_estimated_ind : int
        The index of the estimated t0.
    t0_estimated : float
        The estimated t0.
    t0_estimated_ind_start : int
        The start index of the time window for the estimated t0.
    t0_estimated_ind_end : int
        The end index of the time window for the estimated t0.
    
        
    """
    

    if ecg_or_eog=='ECG':
        timelimit_min=-0.02
        timelimit_max=0.012
        window_size=0.02
    elif ecg_or_eog=='EOG':
        timelimit_min=-0.1
        timelimit_max=0.2
        window_size=0.1

        #these define different windows: 
        # - timelimit is where the peak of the wave is normally located counted from event time defined by mne. 
        #       It is a larger window, it is used to estimate t0, more accurately than mne does (based on 5 most promiment channels).
        # - window_size - where the peak of the wave must be located, counted from already estimated t0. It is a smaller window.
    else:
        print('___MEG QC___: ', 'Choose ecg_or_eog input correctly!')


    #find indexes of t where t is between timelimit_min and timelimit_max (limits where R wave typically is detected by mne):
    t_event_ind=np.argwhere((t>timelimit_min) & (t<timelimit_max))

    # cut the data of each channel to the time interval where R wave is expected to be:
    avg_ecg_epoch_data_nonflipped_limited_to_event=avg_ecg_epoch_data_nonflipped[:,t_event_ind[0][0]:t_event_ind[-1][0]]

    #find 5 channels with max values in the time interval where R wave is expected to be:
    max_values=np.max(np.abs(avg_ecg_epoch_data_nonflipped_limited_to_event), axis=1)
    max_values_ind=np.argsort(max_values)[::-1]
    max_values_ind=max_values_ind[:5]

    # find the index of max value for each of these 5 channels:
    max_values_ind_in_avg_ecg_epoch_data_nonflipped=np.argmax(np.abs(avg_ecg_epoch_data_nonflipped_limited_to_event[max_values_ind]), axis=1)
    
    #find average index of max value for these 5 channels th then derive t0_estimated:
    t0_estimated_average=int(np.round(np.mean(max_values_ind_in_avg_ecg_epoch_data_nonflipped)))
    #limited to event means that the index is limited to the time interval where R wave is expected to be.
    #Now need to get back to actual time interval of the whole epoch:

    #find t0_estimated to use as the point where peak of each ch data should be:
    t0_estimated_ind=t_event_ind[0][0]+t0_estimated_average #sum because time window was cut from the beginning of the epoch previously
    t0_estimated=t[t0_estimated_ind]

    # window of 0.015 or 0.05s around t0_estimated where the peak on different channels should be detected:
    t0_estimated_ind_start=np.argwhere(t==round(t0_estimated-window_size, 3))[0][0] 
    t0_estimated_ind_end=np.argwhere(t==round(t0_estimated+window_size, 3))[0][0]
    #yes you have to round it here because the numbers stored in in memery like 0.010000003 even when it looks like 0.01, hence np.where cant find the target float in t vector


    #another way without round would be to find the closest index of t to t0_estimated-0.015:
    #t0_estimated_ind_start=np.argwhere(t==np.min(t[t<t0_estimated-window_size]))[0][0]
    # find the closest index of t to t0_estimated+0.015:
    #t0_estimated_ind_end=np.argwhere(t==np.min(t[t>t0_estimated+window_size]))[0][0]

    print('___MEG QC___: ', t0_estimated_ind, '-t0_estimated_ind, ', t0_estimated, '-t0_estimated,     ', t0_estimated_ind_start, '-t0_estimated_ind_start, ', t0_estimated_ind_end, '-t0_estimated_ind_end')

    
    return t0_estimated, t0_estimated_ind, t0_estimated_ind_start, t0_estimated_ind_end




def find_affected_channels(ecg_epochs: mne.Epochs, channels: list, m_or_g: str, norm_lvl: float, ecg_or_eog: str, thresh_lvl_peakfinder: float, sfreq:float, tmin: float, tmax: float, plotflag=True, flip_data='flip'):

    """
    Find channels that are affected by ECG or EOG events.
    The function calculates average ECG epoch for each channel and then finds the peak of the wave on each channel.
    Then it compares the peak amplitudes across channels to decide which channels are affected the most.
    The function returns a list of channels that are affected by ECG or EOG events.

    0. For each separate channel get the average ECG epoch. If needed, flip this average epoch to make it's main peak positive.
    Flip approach: 

    - define  a window around the ecg/eog event deteceted by mne. This is not the real t0, but  an approximation. 
        The size of the window defines by how large on average the error of mne is when mne algorythm estimates even time. 
        So for example if mne is off by 0.05s on average, then the window should be -0.05 to 0.05s. 
    - take 5 channels with the largest peak in this window - assume these peaks are the actual artifact.
    - find the average of these 5 peaks - this is the new estimated_t0 (but still not the real t0)
    - create a new window around this new t0 - in this time window all the artifact wave shapes should be located on all channels.
    - flip the channels, if they have a peak inside of this new window, but the peak is negative and it is the closest peak to estimated t0. 
        if the peak is positive - do not flip.
    - collect all final flipped+unflipped eppochs of these channels 
        
    1. Then, for each chennel make a check if the epoch has a typical wave shape. This is the first step to detect affected channels. 
    If no wave shape - it s automatically a not affected channel. If it has - check further.
    It could make sense to do this step after the next one, but actually check for wave shape is done together with peak detection in step 0. Hence this order)

    2. Calculate average ECG epoch on the collected epochs from all channels. Check if average has a wave shape. 
    If no wave shape - no need to check for affected channels further.
    If it has - check further

    3. Set a threshold which defines a high amplitude of ECG event. (All above this threshold counted as potential ECG peak.)
    Threshold is the magnitude of the peak of the average ECG/EOG epoch multiplued by norm_lvl. 
    norl_lvl is chosen by user in config file
    
    4. Find all peaks above this threshold.
    Finding approach:

    - again, set t0 actual as the time point of the peak of an average artifact (over all channels)
    - again, set a window around t0_actual. this new window is defined by how long the wave of the artifact normally is. 
        The window is centered around t0 and for ECG it will be -0.-02 to 0.02s, for EOG it will be -0.1 to 0.1s.
    - find one main peak of the epoch for each channel which would be inside this window and closest to t0.
    - if this peaks magnitude is over the threshold - this channels is considered to be affected by ECG or EOG. Otherwise - not affected.
        (The epoch has to have a wave shape).

    5. Affected and non affected channels will be plotted and added to the dictionary for final report and json file.


    Parameters
    ----------
    ecg_epochs : mne.Epochs
        ECG epochs.
    channels : list
        List of channels to use.
    m_or_g : str
        'mag' or 'grad'.
    norm_lvl : float
        Normalization level.
    ecg_or_eog : str
        'ECG' or 'EOG'.
    thresh_lvl_peakfinder : float
        Threshold level for peakfinder.
    sfreq : float
        Sampling frequency.
    tmin : float
        Start time.
    tmax : float
        End time.
    plotflag : bool, optional
        Plot flag. The default is True.
    flip_data : bool, optional    
        Use absolute value of all data. The default is 'flip'.

        
    Returns 
    -------
    artif_affected_channels : list
        List of instances of Mean_artif_peak_on_channel. The list of channels affected by ecg.eog artifact.
        Each instance contains info about the average ecg/eog artifact on this channel and the peak amplitude of the artifact.
    ecg_not_affected_channels : list
        List of instances of Mean_artif_peak_on_channel. The list of channels not affected by ecg.eog artifact.
        Each instance contains info about the average ecg/eog artifact on this channel and the peak amplitude of the artifact.
    affected_derivs : list
        List of instances of QC_deriv with figures for average ecg/eog and affected + not affected channels. Last 2 are optional: oly if average is good.
    bad_avg: bool
        True if the average ecg/eog artifact is bad: too noisy. In case of a noisy average ecg/eog artifact, no affected channels should be further detected.

        
    """

    if  ecg_or_eog=='ECG':
        max_n_peaks_allowed_for_ch_per_ms=8 #this is for an individual ch, it can be more noisy, therefore more peaks are allowed. It also depends on the length of chosen window
        max_n_peaks_allowed_for_avg_per_epoch = 3 #this is for the whole averaged over all channels ecg epoch, it should be much smoother - therefore less peaks are allowed.
    elif ecg_or_eog=='EOG':
        max_n_peaks_allowed_for_ch_per_ms=5
        max_n_peaks_allowed_for_avg_per_epoch = 3
    else:
        print('___MEG QC___: ', 'Choose ecg_or_eog input correctly!')

    max_n_peaks_allowed=round(((abs(tmin)+abs(tmax))/0.1)*max_n_peaks_allowed_for_ch_per_ms)
    print('___MEG QC___: ', 'max_n_peaks_allowed: '+str(max_n_peaks_allowed))

    t = np.round(np.arange(tmin, tmax+1/sfreq, 1/sfreq), 3) #yes, you need to round


    #1.:
    #averaging the ECG epochs together:
    avg_ecg_epochs = ecg_epochs.average(picks=channels)#.apply_baseline((-0.5, -0.2))
    #avg_ecg_epochs is evoked:Evoked objects typically store EEG or MEG signals that have been averaged over multiple epochs.
    #The data in an Evoked object are stored in an array of shape (n_channels, n_times)

    ecg_epoch_per_ch=[]

    if flip_data is False:
        ecg_epoch_per_ch_only_data=avg_ecg_epochs.data
        for i, ch_data in enumerate(ecg_epoch_per_ch_only_data):
            ecg_epoch_per_ch.append(Avg_artif(name=channels[i], mean_artifact_epoch=ch_data))
            ecg_epoch_per_ch[i].get_peaks_wave(max_n_peaks_allowed, thresh_lvl_peakfinder)

    elif flip_data is True:

        # New ecg flip approach:

        # 1. find peaks on all channels it time frame around -0.02<t[peak_loc]<0.012 (here R wave is typica;ly dettected by mne - for ecg, for eog it is -0.1<t[peak_loc]<0.2)
        # 2. take 5 channels with most prominent peak 
        # 3. find estimated average t0 for all 5 channels, because t0 of event which mne estimated is often not accurate
        # 4. flip all channels with negative peak around estimated t0 


        avg_ecg_epoch_data_nonflipped=avg_ecg_epochs.data

        _, t0_estimated_ind, t0_estimated_ind_start, t0_estimated_ind_end = estimate_t0(ecg_or_eog, avg_ecg_epoch_data_nonflipped, t)
        ecg_epoch_per_ch, ecg_epoch_per_ch_only_data = flip_channels(avg_ecg_epoch_data_nonflipped, channels, max_n_peaks_allowed, thresh_lvl_peakfinder, t0_estimated_ind_start, t0_estimated_ind_end, t0_estimated_ind)

      
    else:
        print('___MEG QC___: ', 'Wrong set variable: flip_data=', flip_data)


    # Find affected channels after flipping:
    # 5. calculate average of all channels
    # 6. find peak on average channel and set it as actual t0
    # 7. affected channels will be the ones which have peak amplitude over average in limits of -0.05:0.05s from actual t0 """


    avg_ecg_overall=np.mean(ecg_epoch_per_ch_only_data, axis=0) 
    # will show if there is ecg artifact present  on average. should have ecg shape if yes. 
    # otherwise - it was not picked up/reconstructed correctly

    avg_ecg_overall_obj=Avg_artif(name='Mean_'+ecg_or_eog+'_overall',mean_artifact_epoch=avg_ecg_overall)
    avg_ecg_overall_obj.get_peaks_wave(max_n_peaks_allowed_for_avg_per_epoch, thresh_lvl_peakfinder)

    mean_ecg_magnitude_peak=np.max(avg_ecg_overall_obj.peak_magnitude)
    mean_ecg_loc_peak = avg_ecg_overall_obj.peak_loc[np.argmax(avg_ecg_overall_obj.peak_magnitude)]
    
    #set t0_actual as the time of the peak of the average ecg artifact:
    t0_actual=t[mean_ecg_loc_peak]

    if avg_ecg_overall_obj.wave_shape is True:
        desc = "GOOD " +ecg_or_eog+ " average. Detected " + str(len(avg_ecg_overall_obj.peak_magnitude)) + " peak(s). Allowed max: " + str(max_n_peaks_allowed_for_avg_per_epoch) + " peaks (pos+neg)."
        print('___MEG QC___: ', desc)
        bad_avg=False
    else:
        desc = "BAD " +ecg_or_eog+ " average. Detected " + str(len(avg_ecg_overall_obj.peak_magnitude)) + " peak(s). Allowed max: " + str(max_n_peaks_allowed_for_avg_per_epoch) + " peaks (pos+neg)."
        print('___MEG QC___: ', desc)
        bad_avg=True

    affected_derivs=[]
    if plotflag is True:
        fig_avg = go.Figure()
        avg_ecg_overall_obj.plot_epoch_and_peak(fig_avg, t, 'Mean '+ecg_or_eog+' artifact over all data: ', m_or_g)
        fig_avg.show()
        affected_derivs += [QC_derivative(fig_avg, 'overall_average_ECG_epoch_'+m_or_g, 'plotly', description_for_user = desc)]

    #2. and 3.:
    if bad_avg is False:
        artif_affected_channels, ecg_not_affected_channels, artifact_lvl = detect_channels_above_norm(norm_lvl=norm_lvl, list_mean_ecg_epochs=ecg_epoch_per_ch, mean_ecg_magnitude_peak=mean_ecg_magnitude_peak, t=t, t0_actual=t0_actual, ecg_or_eog=ecg_or_eog)

        if plotflag is True:
            fig_affected = plot_affected_channels(artif_affected_channels, artifact_lvl, t, ch_type=m_or_g, fig_tit=ecg_or_eog+' affected channels: ', flip_data=flip_data)
            fig_not_affected = plot_affected_channels(ecg_not_affected_channels, artifact_lvl, t, ch_type=m_or_g, fig_tit=ecg_or_eog+' not affected channels: ', flip_data=flip_data)
            affected_derivs += [QC_derivative(fig_affected, ecg_or_eog+'_affected_channels_'+m_or_g, 'plotly')]
            affected_derivs += [QC_derivative(fig_not_affected, ecg_or_eog+'_not_affected_channels_'+m_or_g, 'plotly')]

        if avg_ecg_overall_obj.wave_shape is False and (not ecg_not_affected_channels or len(ecg_not_affected_channels)/len(channels)<0.2):
            print('___MEG QC___: ', 'Something went wrong! The overall average ' +ecg_or_eog+ ' is  bad, but all  channels are affected by ' +ecg_or_eog+ ' artifact.')
    else:
        artif_affected_channels = []

    return artif_affected_channels, affected_derivs, bad_avg



#%%
def make_dict_global_ECG_EOG(all_affected_channels: list, channels: list):
    """
    Make a dictionary for the global part of simple metrics for ECG/EOG artifacts.
    For ECG/EOG no local metrics are calculated, so global is the only one.
    
    Parameters
    ----------
    all_affected_channels : list
        List of all affected channels.
    channels : list
        List of all channels.
        
    Returns
    -------
    dict_global_ECG_EOG : dict
        Dictionary with simple metrics for ECG/EOG artifacts.

        
    """

    if not all_affected_channels:
        number_of_affected_ch = 0
        percent_of_affected_ch = 0
        affected_chs = None
        #top_10_magnitudes = None
    else:
        number_of_affected_ch = len(all_affected_channels)
        percent_of_affected_ch = round(len(all_affected_channels)/len(channels)*100, 1)

        # sort all_affected_channels by main_peak_magnitude:
        all_affected_channels_sorted = sorted(all_affected_channels, key=lambda ch: ch.main_peak_magnitude, reverse=True)
        affected_chs = {ch.name: ch.main_peak_magnitude for ch in all_affected_channels_sorted}

    metric_global_content = {
        'number_of_affected_ch': number_of_affected_ch,
        'percent_of_affected_ch': percent_of_affected_ch, 
        'details':  affected_chs}

    return metric_global_content


def make_simple_metric_ECG_EOG(all_affected_channels: dict, m_or_g_chosen: list, ecg_or_eog: str, channels: dict, bad_avg: dict):
    """
    Make simple metric for ECG/EOG artifacts as a dictionary, which will further be converted into json file.
    
    Parameters
    ----------
    all_affected_channels : dict
        Dictionary with listds of affected channels for each channel type.
    m_or_g_chosen : list
        List of channel types chosen for the analysis. 
    ecg_or_eog : str
        String 'ecg' or 'eog' depending on the artifact type.
    channels : dict
        Dictionary with listds of channels for each channel type.
    bad_avg : dict
        Dictionary with boolean values for mag and grad, indicating if the average artifact is bad or not. 
        
    Returns
    -------
    simple_metric : dict
        Dictionary with simple metrics for ECG/EOG artifacts.
        

    """

    metric_global_name = 'all_'+ecg_or_eog+'_affected_channels'
    metric_global_content={'mag': None, 'grad': None}
    metric_global_description = 'Affected channels are the channels with average (over '+ecg_or_eog+' epochs of this channel) ' +ecg_or_eog+ ' artifact above the threshold. Channels are listed here in order from the highest to lowest artifact amplitude. Non affected channels are not listed. Threshld is defined as average '+ecg_or_eog+' artifact peak magnitude over al channels * norm_lvl. norm_lvl is defined in the config file. Metrci also provides a list of 10 most strongly affected channels + their artfact peaks magnitdes.'

    for m_or_g in m_or_g_chosen:
        if bad_avg[m_or_g] is False:
            metric_global_content[m_or_g]= make_dict_global_ECG_EOG(all_affected_channels[m_or_g], channels[m_or_g])

    simple_metric = simple_metric_basic(metric_global_name, metric_global_description, metric_global_content['mag'], metric_global_content['grad'], display_only_global=True)

    return simple_metric


def plot_ecg_eog_mne(ecg_epochs: mne.Epochs, m_or_g: str, tmin: float, tmax: float):

    """
    Plot ECG/EOG artifact with topomap and average over epochs (MNE plots based on matplotlib)

    Parameters
    ----------
    ecg_epochs : mne.Epochs
        ECG/EOG epochs.
    m_or_g : str
        String 'mag' or 'grad' depending on the channel type.
    tmin : float
        Start time of the epoch.
    tmax : float
        End time of the epoch.
    
    Returns
    -------
    ecg_derivs : list
        List of QC_derivative objects with plots.
    
    
    """

    mne_ecg_derivs = []
    fig_ecg = ecg_epochs.plot_image(combine='mean', picks = m_or_g)[0] #plot averageg over ecg epochs artifact
    # [0] is to plot only 1 figure. the function by default is trying to plot both mag and grad, but here we want 
    # to do them saparetely depending on what was chosen for analysis
    mne_ecg_derivs += [QC_derivative(fig_ecg, 'mean_ECG_epoch_'+m_or_g, 'matplotlib')]
    fig_ecg.show()

    #averaging the ECG epochs together:
    avg_ecg_epochs = ecg_epochs.average() #.apply_baseline((-0.5, -0.2))
    # about baseline see here: https://mne.tools/stable/auto_tutorials/preprocessing/10_preprocessing_overview.html#sphx-glr-auto-tutorials-preprocessing-10-preprocessing-overview-py

    #plot average artifact with topomap
    fig_ecg_sensors = avg_ecg_epochs.plot_joint(times=[tmin-tmin/100, tmin/2, 0, tmax/2, tmax-tmax/100], picks = m_or_g)
    # tmin+tmin/10 and tmax-tmax/10 is done because mne sometimes has a plotting issue, probably connected tosamplig rate: 
    # for example tmin is  set to -0.05 to 0.02, but it  can only plot between -0.0496 and 0.02.

    mne_ecg_derivs += [QC_derivative(fig_ecg_sensors, 'ECG_field_pattern_sensors_'+m_or_g, 'matplotlib')]
    fig_ecg_sensors.show()

    return mne_ecg_derivs

#%%
def ECG_meg_qc(ecg_params: dict, raw: mne.io.Raw, channels: list, m_or_g_chosen: list):
    
    """
    Main ECG function. Calculates average ECG artifact and finds affected channels.
    
    Parameters
    ----------
    ecg_params : dict
        Dictionary with ECG parameters originating from config file.
    raw : mne.io.Raw
        Raw data.
    channels : dict
        Dictionary with listds of channels for each channel type (typer mag and grad).
    m_or_g_chosen : list
        List of channel types chosen for the analysis.
        
    Returns
    -------
    ecg_derivs : list
        List of all derivatives (plotly figures) as QC_derivative instances
    simple_metric_ECG : dict
        Dictionary with simple metrics for ECG artifacts to be exported into json file.
    no_ecg_str : str
        String with information about ECG channel used in the final report.
        

    """

    picks_ECG = mne.pick_types(raw.info, ecg=True)
    ecg_ch_name = [raw.info['chs'][name]['ch_name'] for name in picks_ECG]

    ecg_derivs = []

    noisy_ch_derivs, bad_ecg_eog = detect_noisy_ecg_eog(raw, ecg_ch_name,  ecg_or_eog = 'ECG', n_breaks_allowed_per_10min = ecg_params['n_breaks_allowed_per_10min'], allowed_range_of_peaks_stds = ecg_params['allowed_range_of_peaks_stds'])

    ecg_derivs += noisy_ch_derivs
    if ecg_ch_name:
        for ch in ecg_ch_name:
            if bad_ecg_eog[ch] == 'bad': #ecg channel present but noisy - drop it and  try to reconstruct
                if ecg_params['drop_bad_ch'] is True:
                    no_ecg_str = 'ECG channel data is too noisy, cardio artifacts were reconstructed. ECG channel was dropped from the analysis. Consider checking the quality of ECG channel on your recording device.'
                    print('___MEG QC___: ', no_ecg_str)
                    raw.drop_channels(ch)
                elif ecg_params['drop_bad_ch'] is False:
                    no_ecg_str = 'ECG channel data is too noisy, still attempt to calculate artifats using this channel. Consider checking the quality of ECG channel on your recording device.'
                    print('___MEG QC___: ', no_ecg_str)
                else:
                    raise ValueError('drop_bad_ch should be either True or False')
            elif bad_ecg_eog[ch] == 'good': #ecg channel present and good - use it
                no_ecg_str = ch+' is good and is used to identify hearbeats: '
                print('___MEG QC___: ', no_ecg_str)
    else:
        no_ecg_str = 'No ECG channel found. The signal is reconstructed based on magnetometers data.'
        print('___MEG QC___: ', no_ecg_str)
    

    #ecg_events_times  = (ecg_events[:, 0] - raw.first_samp) / raw.info['sfreq']
    
    sfreq=raw.info['sfreq']
    tmin=ecg_params['ecg_epoch_tmin']
    tmax=ecg_params['ecg_epoch_tmax']
    norm_lvl=ecg_params['norm_lvl']
    flip_data=ecg_params['flip_data']
    
    ecg_affected_channels={}
    bad_avg = {}

    for m_or_g  in m_or_g_chosen:

        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, picks=channels[m_or_g], tmin=tmin, tmax=tmax)

        ecg_derivs += plot_ecg_eog_mne(ecg_epochs, m_or_g, tmin, tmax)

        ecg_affected_channels[m_or_g], affected_derivs, bad_avg[m_or_g]=find_affected_channels(ecg_epochs, channels[m_or_g], m_or_g, norm_lvl, ecg_or_eog='ECG', thresh_lvl_peakfinder=6, tmin=tmin, tmax=tmax, plotflag=True, sfreq=sfreq, flip_data=flip_data)
        ecg_derivs += affected_derivs


        if bad_avg[m_or_g] is True:
            tit, _ = get_tit_and_unit(m_or_g)
            no_ecg_str += '<br>'+tit+': ECG signal detection/reconstruction did not produce reliable results. Hearbeat artifacts and affected channels can not be estimated. <br>'
        else:
            no_ecg_str += ''

    simple_metric_ECG = make_simple_metric_ECG_EOG(ecg_affected_channels, m_or_g_chosen, 'ECG', channels, bad_avg)

    return ecg_derivs, simple_metric_ECG, no_ecg_str


#%%
def EOG_meg_qc(eog_params: dict, raw: mne.io.Raw, channels: dict, m_or_g_chosen: list):
    
    """
    Main EOG function. Calculates average EOG artifact and finds affected channels.
    
    Parameters
    ----------
    eog_params : dict
        Dictionary with EOG parameters originating from the config file.
    raw : mne.io.Raw
        Raw MEG data.
    channels : dict
        Dictionary with listds of channels for each channel type (typer mag and grad).
    m_or_g_chosen : list
        List of channel types chosen for the analysis.
        
    Returns
    -------
    eog_derivs : list
        List of all derivatives (plotly figures) as QC_derivative instances
    simple_metric_EOG : dict
        Dictionary with simple metrics for ECG artifacts to be exported into json file.
    no_eog_str : str
        String with information about EOG channel used in the final report.
    
    """
    eog_derivs = []
    simple_metric_EOG = {'description': 'EOG artifacts could not be calculated'}

    picks_EOG = mne.pick_types(raw.info, eog=True)
    eog_ch_name = [raw.info['chs'][name]['ch_name'] for name in picks_EOG]
    if picks_EOG.size == 0:
        no_eog_str = 'No EOG channels found is this data set - EOG artifacts can not be detected.'
        print('___MEG QC___: ', no_eog_str)
        return eog_derivs, simple_metric_EOG, no_eog_str
    else:
        no_eog_str = 'Only blinks can be calculated using MNE, not saccades.'
        print('___MEG QC___: ', 'EOG channels found: ', eog_ch_name)



    # Notify id EOG channel is bad (dont drop it in any case, no analysis possible without it):
    # COMMENTED OUT because it is not working properly. Approch same as for ECG, parameters different, but still detects good channels where they are bad.
    # Might work on in later, left out for now, because this step doesnt influence the final results, just a warning.
    # noisy_ch_derivs, bad_ecg_eog = detect_noisy_ecg_eog(raw, eog_ch_name,  ecg_or_eog = 'EOG', n_breaks_allowed_per_10min = eog_params['n_breaks_allowed_per_10min'], allowed_range_of_peaks_stds = eog_params['allowed_range_of_peaks_stds'])
    # eog_derivs += noisy_ch_derivs

    # for ch_eog in eog_ch_name:
    #     if bad_ecg_eog[ch_eog] == 'bad': #ecg channel present but noisy give warning, otherwise just continue. 
    #         #BTW we dont relly care if the bad escg channel is the one for saccades, becase we only use blinks. Identify this in the warning?
    #         #IDK how because I dont know which channel is the one for saccades
    #         print('___MEG QC___:  '+ch_eog+' channel data is noisy. EOG data will be estimated, but might not be accurate. Cosider checking the quality of ECG channel on your recording device.')

    #plot EOG channels
    for ch_eog in eog_ch_name:
        ch_data = raw.get_data(picks=ch_eog)[0]
        fig_ch = plot_channel(ch_data, peaks = [], ch_name = ch_eog, fs = raw.info['sfreq'])
        eog_derivs += [QC_derivative(fig_ch, ch_eog, 'plotly')]

    #eog_events=mne.preprocessing.find_eog_events(raw, thresh=None, ch_name=None)
    # ch_name: This doesn’t have to be a channel of eog type; it could, for example, also be an ordinary 
    # EEG channel that was placed close to the eyes, like Fp1 or Fp2.
    # or just use none as channel, so the eog will be found automatically

    #eog_events_times  = (eog_events[:, 0] - raw.first_samp) / raw.info['sfreq']

    sfreq=raw.info['sfreq']
    tmin=eog_params['eog_epoch_tmin']
    tmax=eog_params['eog_epoch_tmax']
    norm_lvl=eog_params['norm_lvl']
    flip_data=eog_params['flip_data']

    eog_affected_channels={}
    bad_avg = {}
    no_eog_str = ''
    for m_or_g  in m_or_g_chosen:

        eog_epochs = mne.preprocessing.create_eog_epochs(raw, picks=channels[m_or_g], tmin=tmin, tmax=tmax)

        eog_derivs += plot_ecg_eog_mne(eog_epochs, m_or_g, tmin, tmax)

        eog_affected_channels[m_or_g], affected_derivs, bad_avg[m_or_g] = find_affected_channels(eog_epochs, channels[m_or_g], m_or_g, norm_lvl, ecg_or_eog='EOG', thresh_lvl_peakfinder=2, tmin=tmin, tmax=tmax, plotflag=True, sfreq=sfreq, flip_data=flip_data)
        eog_derivs += affected_derivs

        if bad_avg[m_or_g] is True:
            tit, _ = get_tit_and_unit(m_or_g)
            no_eog_str += '<br>'+tit+': EOG signal detection did not produce reliable results. Eyeblink artifacts and affected channels can not be estimated. <br>'
        else:
            no_eog_str += ''

    simple_metric_EOG=make_simple_metric_ECG_EOG(eog_affected_channels, m_or_g_chosen, 'EOG', channels, bad_avg)

    return eog_derivs, simple_metric_EOG, no_eog_str

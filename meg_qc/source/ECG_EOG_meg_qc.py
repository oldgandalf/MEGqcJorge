import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks
import matplotlib #this is in case we will need to suppress mne matplotlib plots
from copy import deepcopy
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from meg_qc.source.universal_html_report import simple_metric_basic
from meg_qc.source.universal_plots import QC_derivative, get_tit_and_unit, plot_df_of_channels_data_as_lines_by_lobe
from IPython.display import display


def check_3_conditions(ch_data: list or np.ndarray, fs: int, ecg_or_eog: str, n_breaks_bursts_allowed_per_10min: int, allowed_range_of_peaks_stds: float):

    """
    Check if the ECG/EOG channel is not corrupted using 3 conditions:
    - peaks have similar amplitude
    - no breaks longer than normal max distance between peaks of hear beats
    - no bursts: too short intervals between peaks

    Parameters
    ----------
    ch_data : list or np.ndarray
        Data of the channel to check
    fs : int
        Sampling frequency of the data
    ecg_or_eog : str
        'ECG' or 'EOG'
    n_breaks_bursts_allowed_per_10min : int, optional
        Number of breaks allowed per 10 minutes of recording, by default 3. Can also set to 0, but then it can falsely detect a break/burst if the peak detection was not perfect.
    allowed_range_of_peaks_stds : float, optional
        Allowed range of standard deviations of peak amplitudes, by default 0.05. Works for ECG channel, but not good for EOG channel.
    
    Returns
    -------
    similar_ampl : bool
        True if peaks have similar amplitude
    no_breaks : bool
        True if there are up to allowed number of breaks in the data
    no_bursts : bool
        True if there are up to allowed number of bursts in the data
    fig : plotly.graph_objects.Figure
        Plot of the channel data and detected peaks

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

    if amplitude_std <= allowed_range_of_peaks_stds: 
        similar_ampl = True
        print("___MEG QC___: Peaks have similar amplitudes, amplitude std: ", amplitude_std)
    else:
        similar_ampl = False
        print("___MEG QC___: Peaks do not have similar amplitudes, amplitude std: ", amplitude_std)


    # 2. Calculate RR intervals (time differences between consecutive R peaks)
    rr_intervals = np.diff(peaks) / fs

    if ecg_or_eog == 'ECG':
        rr_dist_allowed = [0.6, 1.6] #take possible pulse rate of 100-40 bpm (hense distance between peaks is 0.6-1.6 seconds)
    elif ecg_or_eog == 'EOG':
        rr_dist_allowed = [1, 10] #take possible blink rate of 60-5 per minute (hense distance between peaks is 1-10 seconds). Yes, 60 is a very high rate, but I see this in some data sets often.


    #Count how many segment there are in rr_intervals with breaks or bursts:
    n_breaks = 0
    n_bursts = 0
    for i in range(len(rr_intervals)):
        if rr_intervals[i] > rr_dist_allowed[1]:
            n_breaks += 1
        if rr_intervals[i] < rr_dist_allowed[0]:
            n_bursts += 1

    no_breaks, no_bursts = True, True
    #Check if there are too many breaks:
    if n_breaks > len(rr_intervals)/60*10/n_breaks_bursts_allowed_per_10min:
        print("___MEG QC___: There are more than 2 breaks in the data, number: ", n_breaks)
        no_breaks = False
    if n_bursts > len(rr_intervals)/60*10/n_breaks_bursts_allowed_per_10min:
        print("___MEG QC___: There are more than 2 bursts in the data, number: ", n_bursts)
        no_bursts = False


    return (similar_ampl, no_breaks, no_bursts), peaks



def plot_ECG_EOG_channel(ch_data: np.ndarray or list, peaks: np.ndarray or list, ch_name: str, fs: float, verbose_plots: bool):

    """
    Plot the ECG channel data and detected peaks
    
    Parameters
    ----------
    ch_data : list or np.ndarray
        Data of the channel
    peaks : list or np.ndarray
        Indices of the peaks in the data
    ch_name : str
        Name of the channel
    fs : int
        Sampling frequency of the data
    verbose_plots : bool
        If True, show the figure in the notebook
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plot of the channel data and detected peaks
        
    """

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
    
    if verbose_plots is True:
        fig.show()

    return fig

def detect_noisy_ecg(raw: mne.io.Raw, ecg_ch: str,  ecg_or_eog: str, n_breaks_bursts_allowed_per_10min: int, allowed_range_of_peaks_stds: float):
    
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
    ecg_ch : str
        ECG channel names to be checked.
    ecg_or_eog : str
        'ECG' or 'EOG'
    n_breaks_bursts_allowed_per_10min : int, optional
        Number of breaks allowed per 10 minutes of recording. The default is 3.
    allowed_range_of_peaks_stds : float, optional
        Allowed range of peaks standard deviations. The default is 0.05.

        - The channel data will be scaled from 0 to 1, so the setting is universal for all data sets.
        - The peaks will be detected on the scaled data
        - The average std of all peaks has to be within this allowed range, If it is higher - the channel has too high deviation in peaks height and is counted as noisy


    Returns
    -------
    bad_ecg_eog : dict
        Dictionary with channel names as keys and 'good' or 'bad' as values.
    all_ch_data[0] : list
        data of the ECG channel recorded.
    ecg_eval : tuple
        Tuple of 3 booleans, indicating if the channel is good or bad according to 3 conditions.


        
    """

    sfreq=raw.info['sfreq']

    bad_ecg_eog = {}
    peaks = []

    ch_data = raw.get_data(picks=ecg_ch)[0] #here ch_data will be the RAW DATA
    # get_data creates list inside of a list becausee expects to create a list for each channel. 
    # but iteration takes 1 ch at a time. this is why [0]

    ecg_eval, peaks = check_3_conditions(ch_data, sfreq, ecg_or_eog, n_breaks_bursts_allowed_per_10min, allowed_range_of_peaks_stds)
    print(f'___MEG QC___: {ecg_ch} satisfied conditions for a good channel: ', ecg_eval)

    if all(ecg_eval):
        print(f'___MEG QC___: Overall good {ecg_or_eog} channel: {ecg_ch}')
        bad_ecg_eog[ecg_ch] = 'good'
    else:
        print(f'___MEG QC___: Overall bad {ecg_or_eog} channel: {ecg_ch}')
        bad_ecg_eog[ecg_ch] = 'bad'

    return bad_ecg_eog, ch_data, peaks, ecg_eval


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


class Avg_artif:
    
    """ 
    Instance of this class:

    - contains average ECG/EOG epoch for a particular channel,
    - calculates its main peak (location and magnitude), possibe on both smoothed and non smoothed data.
    - evaluates if this epoch is concidered as artifact or not based on the main peak amplitude.
    

    Attributes
    ----------
    name : str
        name of the channel
    artif_data : list
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
    artif_data_smoothed : list
        list of floats, average ecg epoch for a particular channel, smoothed usig Gaussian filter
    peak_loc_smoothed : int
        locations of peaks inside the artifact epoch calculated on smoothed data
    peak_magnitude_smoothed : float
        magnitudes of peaks inside the artifact epoch calculated on smoothed data
    wave_shape_smoothed : bool
        True if the average epoch has typical wave shape, False otherwise. R wave shape  - for ECG or just a wave shape for EOG. Calculated on smoothed data
    artif_over_threshold_smoothed : bool
        True if the main peak is concidered as artifact, False otherwise. True if artifact sas magnitude over the threshold. Calculated on smoothed data
    main_peak_loc_smoothed : int
        location of the main peak inside the artifact epoch. Calculated on smoothed data
    main_peak_magnitude_smoothed : float
        magnitude of the main peak inside the artifact epoch. Calculated on smoothed data
    corr_coef : float
        correlation coefficient between the ECG/EOG channels data and average data of this mag/grad channel
    p_value : float
        p-value of the correlation coefficient between the ECG/EOG channels data and average data of this mag/grad channel
    lobe: str
        which lobe his channel belongs to
    color: str
        color code for this channel according to the lobe it belongs to
    


        
    Methods
    -------
    __init__(self, name: str, artif_data:list, peak_loc=None, peak_magnitude=None, wave_shape:bool=None, artif_over_threshold:bool=None, main_peak_loc: int=None, main_peak_magnitude: float=None)
        Constructor
    __repr__(self)
        Returns a string representation of the object

        
    """

    def __init__(self, name: str, artif_data:list, peak_loc=None, peak_magnitude=None, wave_shape:bool=None, artif_over_threshold:bool=None, main_peak_loc: int=None, main_peak_magnitude: float=None, artif_data_smoothed: list or None = None, peak_loc_smoothed=None, peak_magnitude_smoothed=None, wave_shape_smoothed:bool=None, artif_over_threshold_smoothed:bool=None, main_peak_loc_smoothed: int=None, main_peak_magnitude_smoothed: float=None, corr_coef: float = None, p_value: float = None, lobe: str = None, color: str = None):
        """Constructor"""
        
        self.name =  name
        self.artif_data = artif_data
        self.peak_loc = peak_loc
        self.peak_magnitude = peak_magnitude
        self.wave_shape =  wave_shape
        self.artif_over_threshold = artif_over_threshold
        self.main_peak_loc = main_peak_loc
        self.main_peak_magnitude = main_peak_magnitude
        self.artif_data_smoothed = artif_data_smoothed
        self.peak_loc_smoothed = peak_loc_smoothed
        self.peak_magnitude_smoothed = peak_magnitude_smoothed
        self.wave_shape_smoothed =  wave_shape_smoothed
        self.artif_over_threshold_smoothed = artif_over_threshold_smoothed
        self.main_peak_loc_smoothed = main_peak_loc_smoothed
        self.main_peak_magnitude_smoothed = main_peak_magnitude_smoothed
        self.corr_coef = corr_coef
        self.p_value = p_value
        self.lobe = lobe
        self.color = color


    def __repr__(self):
        """
        Returns a string representation of the object
        
        """

        return 'Mean artifact for: ' + str(self.name) + '\n - peak location inside artifact epoch: ' + str(self.peak_loc) + '\n - peak magnitude: ' + str(self.peak_magnitude) +'\n - main_peak_loc: '+ str(self.main_peak_loc) +'\n - main_peak_magnitude: '+str(self.main_peak_magnitude)+'\n - wave_shape: '+ str(self.wave_shape) + '\n - artifact magnitude over threshold: ' + str(self.artif_over_threshold)+ '\n'
    


    def get_peaks_wave(self, max_n_peaks_allowed: int, thresh_lvl_peakfinder: float):

        """
        Find peaks in the average artifact epoch and decide if the epoch has wave shape: 
        few peaks (different number allowed for ECG and EOG) - wave shape, many or no peaks - not.
        On non smoothed data!
        
        Parameters
        ----------
        max_n_peaks_allowed : int
            maximum number of peaks allowed in the average artifact epoch
        thresh_lvl_peakfinder : float
            threshold for peakfinder function.
        
            
        """

        peak_locs_pos_orig, peak_locs_neg_orig, peak_magnitudes_pos_orig, peak_magnitudes_neg_orig = find_epoch_peaks(ch_data=self.artif_data, thresh_lvl_peakfinder=thresh_lvl_peakfinder)
        
        self.peak_loc=np.concatenate((peak_locs_pos_orig, peak_locs_neg_orig), axis=None)
        self.peak_magnitude=np.concatenate((peak_magnitudes_pos_orig, peak_magnitudes_neg_orig), axis=None)

        if np.size(self.peak_loc)==0: #no peaks found
            self.wave_shape=False
        elif 1<=len(self.peak_loc)<=max_n_peaks_allowed:
            self.wave_shape=True
        elif len(self.peak_loc)>max_n_peaks_allowed:
            self.wave_shape=False
        else:
            print('Something went wrong with peak detection')


    def get_peaks_wave_smoothed(self, gaussian_sigma: int, max_n_peaks_allowed: int, thresh_lvl_peakfinder: float):

        """
        Find peaks in the average artifact epoch and decide if the epoch has wave shape: 
        few peaks (different number allowed for ECG and EOG) - wave shape, many or no peaks - not.
        On smoothed data! If it was not smoothed yet - it will be smoothed inside this function
        
        Parameters
        ----------
        gaussian_sigma : int
            sigma for gaussian smoothing
        max_n_peaks_allowed : int
            maximum number of peaks allowed in the average artifact epoch
        thresh_lvl_peakfinder : float
            threshold for peakfinder function.

        
        """

        if self.artif_data_smoothed is None: #if no smoothed data available yet
            self.smooth_artif(gaussian_sigma) 

        peak_locs_pos_smoothed, peak_locs_neg_smoothed, peak_magnitudes_pos_smoothed, peak_magnitudes_neg_smoothed = find_epoch_peaks(ch_data=self.artif_data_smoothed, thresh_lvl_peakfinder=thresh_lvl_peakfinder)
        
        self.peak_loc_smoothed=np.concatenate((peak_locs_pos_smoothed, peak_locs_neg_smoothed), axis=None)
        self.peak_magnitude_smoothed=np.concatenate((peak_magnitudes_pos_smoothed, peak_magnitudes_neg_smoothed), axis=None)

        if np.size(self.peak_loc_smoothed)==0:
            self.wave_shape_smoothed=False
        elif 1<=len(self.peak_loc_smoothed)<=max_n_peaks_allowed:
            self.wave_shape_smoothed=True
        elif len(self.peak_loc_smoothed)>max_n_peaks_allowed:
            self.wave_shape_smoothed=False
        else:
            print('Something went wrong with peak detection')


    def plot_epoch_and_peak(self, t: np.ndarray, fig_tit: str, ch_type: str, fig: go.Figure = None, plot_original: bool = True, plot_smoothed: bool = True):

        """
        Plot the average artifact epoch and the peak inside it.
        Allowes to plot both originl and smoothed data or only 1 of them in same figure.

        Parameters
        ----------
        t : list
            time vector as numpy array. It can be created as: t = np.round(np.arange(tmin, tmax+1/sfreq, 1/sfreq), 3) #yes, you need to round
        fig_tit: str
            title of the figure not including ch type.
        ch_type: str
            type of the channel ('mag, 'grad'). Used only as title of the figure
        fig: plotly.graph_objects.Figure
            (Empty) plotly figure to be filled. If set to None - figure will be created inside this function. 
            Giving figure is useful if you want to plot more traces on top of the figure you already have using this function.
            But! If you plot into the same figure on the loop - create the figure first. 
            And then input same figure into this function on every iteration. If you ll input None, figure will be overwritten on every itertion.
        plot_original : bool
            if True - plot originl artifact data (non smoothed)
        plot_smoothed : bool
            if True - plot smoothed artifact data 

        Returns
        -------
        fig : plotly.graph_objects.Figure
            figure with the epoch and the peak
        
        
        """
        if fig is None: #if no figure is provided - create figure. Otherwise it will only create the data for figure, and figure would need to be made before.
            fig=go.Figure()

        fig_ch_tit, unit = get_tit_and_unit(ch_type)

        if plot_original is True and self.artif_data is not None:
            fig.add_trace(go.Scatter(x=np.array(t), y=np.array(self.artif_data), name=self.name, legendgroup='Original data', legendgrouptitle=dict(text='Original data')))
            fig.add_trace(go.Scatter(x=np.array(t[self.peak_loc]), y=self.peak_magnitude, mode='markers', name='peak: '+self.name, legendgroup='Original data', legendgrouptitle=dict(text='Original data')))
        elif plot_original is True and self.artif_data is None:
            print("Artifact contains no original data!")
        else:
            pass
        
        if plot_smoothed is True and self.artif_data_smoothed is not None:
            fig.add_trace(go.Scatter(x=np.array(t), y=np.array(self.artif_data_smoothed), name=self.name, legendgroup='Smoothed data', legendgrouptitle=dict(text='Smoothed data')))
            fig.add_trace(go.Scatter(x=np.array(t[self.peak_loc_smoothed]), y=self.peak_magnitude_smoothed, mode='markers', name='peak: '+self.name, legendgroup='Smoothed data', legendgrouptitle=dict(text='Smoothed data')))
        elif plot_smoothed is True and self.artif_data_smoothed is None:
            print("Plot of smoothed data was requested, but smoothing was not performed yet.")
        else:
            pass


        fig.update_layout(
            xaxis_title='Time in seconds',
            yaxis = dict(
                showexponent = 'all',
                exponentformat = 'e'),
            yaxis_title='Artifact magnitude in '+unit,
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
        main_peak_loc : int
            location of the main peak
        main_peak_magnitude : float
            magnitude of the main peak
        
        """

        if self.peak_loc is None: #if no peaks were found on original data:
            self.main_peak_magnitude=None
            self.main_peak_loc=None
        elif self.peak_loc is not None: #if peaks were found on original data:
            self.main_peak_magnitude = -1000
            for peak_loc in self.peak_loc:
                if timelimit_min<t[peak_loc]<timelimit_max: #if peak is inside the timelimit_min and timelimit_max was found:
                    if self.artif_data[peak_loc] > self.main_peak_magnitude: #if this peak is higher than the previous one:
                        self.main_peak_magnitude=self.artif_data[peak_loc]
                        self.main_peak_loc=peak_loc 
    
            if self.main_peak_magnitude == -1000: #if no peak was found inside the timelimit_min and timelimit_max:
                self.main_peak_magnitude=None
                self.main_peak_loc=None
        else:
            self.main_peak_loc, self.main_peak_magnitude = None, None


        return self.main_peak_loc, self.main_peak_magnitude
    
    def get_highest_peak_smoothed(self, t: np.ndarray, timelimit_min: float, timelimit_max: float):

        """
        Find the highest peak of the artifact epoch inside the give time window on SMOOTHED data.
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
        main_peak_magnitude_smoothed : float
            magnitude of the main peak on smoothed data
        main_peak_loc_smoothed : int
            location of the main peak on smoothed data
        
        
        """

        if self.peak_loc_smoothed is None:
            self.main_peak_magnitude_smoothed=None
            self.main_peak_loc_smoothed=None
        elif self.peak_loc_smoothed is not None:
            self.main_peak_magnitude_smoothed = -1000
            for peak_loc in self.peak_loc_smoothed:
                if timelimit_min<t[peak_loc]<timelimit_max:
                    if self.artif_data_smoothed[peak_loc] > self.main_peak_magnitude_smoothed:
                        self.main_peak_magnitude_smoothed=self.artif_data_smoothed[peak_loc]
                        self.main_peak_loc_smoothed=peak_loc 
    
            if self.main_peak_magnitude_smoothed == -1000:
                self.main_peak_magnitude_smoothed=None
                self.main_peak_loc_smoothed=None

        else:    
            self.main_peak_loc_smoothed, self.main_peak_magnitude_smoothed = None, None


        return self.main_peak_loc_smoothed, self.main_peak_magnitude_smoothed
    
    
    def smooth_artif(self, gauss_sigma: int):

        """ 
        Smooth the artifact epoch using gaussian filter.
        This is done do detect the wave shape in presence of noise. 
        Usually EOG are more noisy than ECG which prevents from detecting a wave shape with same settings on these 2 types of artifacts.
        
        Parameters
        ----------
        gauss_sigma : int
            sigma of the gaussian filter
            
        Returns
        -------
        self
            Avg_artif object with smoothed artifact epoch in self.artif_data_smoothed
        
        """

        data_copy=deepcopy(self.artif_data)
        self.artif_data_smoothed = gaussian_filter(data_copy, gauss_sigma)

        return self
    

    def flip_artif(self):
            
        """
        Flip the artifact epoch upside down on original (non smoothed) data.
        This is only done if the need to flip was detected in flip_channels() function.
        
        Returns
        -------
        self
            Avg_artif object with flipped artifact epoch in self.artif_data and self.peak_magnitude
        
        """

        if self.artif_data is not None:
            self.artif_data = -self.artif_data
        if self.peak_magnitude is not None:
            self.peak_magnitude = -self.peak_magnitude

        return self
    

    def flip_artif_smoothed(self):
            
        """
        Flip the SMOOTHED artifact epoch upside down.
        This is only done if the need to flip was detected in flip_channels() function.
        
        Returns
        -------
        self
            Avg_artif object with flipped smoothed artifact epoch in self.artif_data_smoothed and self.peak_magnitude_smoothed
        
        """

        if self.artif_data_smoothed is not None:
            self.artif_data_smoothed = -self.artif_data_smoothed
        
        if self.peak_magnitude_smoothed is not None:
            self.peak_magnitude_smoothed = -self.peak_magnitude_smoothed

        return self

    def detect_artif_above_threshold(self, artif_threshold_lvl: float, t: np.ndarray, timelimit_min: float, timelimit_max: float):

        """
        Detect if the highest peak of the artifact epoch is above a given threshold.
        Time window is centered around the t0 of the ecg/eog event and limited by timelimit_min and timelimit_max.

        Parameters
        ----------
        artif_threshold_lvl : float
            threshold level
        t : list
            time vector
        timelimit_min : float
            minimum time limit for the peak
        timelimit_max : float
            maximum time limit for the peak

        Returns
        -------
        self.artif_over_threshold : bool
            True if the highest peak is above the threshold, False otherwise

        """

        if self.artif_data is not None:
            #find the highest peak inside the timelimit_min and timelimit_max:
            _, main_peak_magnitude_orig = self.get_highest_peak(t=t, timelimit_min=timelimit_min, timelimit_max=timelimit_max)
            if main_peak_magnitude_orig is not None:
                if main_peak_magnitude_orig>abs(artif_threshold_lvl) and self.wave_shape is True:
                    self.artif_over_threshold=True
                else:
                    self.artif_over_threshold=False
            else:
                self.artif_over_threshold=False
        
        return self.artif_over_threshold


    def detect_artif_above_threshold_smoothed(self, artif_threshold_lvl: float, t: np.ndarray, timelimit_min: float, timelimit_max: float):

        """
        Detect if the highest peak of the artifact epoch is above a given threshold for SMOOTHED data.
        Time window is centered around the t0 of the ecg/eog event and limited by timelimit_min and timelimit_max.

        Parameters
        ----------
        artif_threshold_lvl : float
            threshold level
        t : list
            time vector
        timelimit_min : float
            minimum time limit for the peak
        timelimit_max : float
            maximum time limit for the peak

        Returns
        -------
        self.artif_over_threshold : bool
            True if the highest peak is above the threshold, False otherwise

        """

        if self.artif_data_smoothed is not None:
            #find the highest peak inside the timelimit_min and timelimit_max:
            _, main_peak_magnitude_smoothed = self.get_highest_peak(t=t, timelimit_min=timelimit_min, timelimit_max=timelimit_max)
            if main_peak_magnitude_smoothed is not None:
                if main_peak_magnitude_smoothed>abs(artif_threshold_lvl) and self.wave_shape_smoothed is True:
                    self.artif_over_threshold_smoothed=True
                else:
                    self.artif_over_threshold_smoothed=False
            else:
                self.artif_over_threshold_smoothed=False

        return self.artif_over_threshold_smoothed


def detect_channels_above_norm(norm_lvl: float, list_mean_artif_epochs: list, mean_magnitude_peak: float, t: np.ndarray, t0_actual: float, window_size_for_mean_threshold_method: float, mean_magnitude_peak_smoothed: float = None, t0_actual_smoothed: float = None):


    """
    Find the channels which got average artifact amplitude higher than the average over all channels*norm_lvl.
    
    Parameters
    ----------
    norm_lvl : float
        The norm level is the scaling factor for the threshold. The mean artifact amplitude over all channels is multiplied by the norm_lvl to get the threshold.
    list_mean_artif_epochs : list
        List of MeanArtifactEpoch objects, each hold the information about mean artifact for one channel.
    mean_magnitude_peak : float
        The magnitude the mean artifact amplitude over all channels.
    t : np.ndarray
        Time vector.
    t0_actual : float
        The time of the ecg/eog event.
    window_size_for_mean_threshold_method: float
        this value will be taken before and after the t0_actual. It defines the time window in which the peak of artifact on the channel has to present 
        to be counted as artifact peak and compared t the threshold. Unit: seconds
    mean_magnitude_peak_smoothed : float, optional
        The magnitude the mean artifact amplitude over all channels for SMOOTHED data. The default is None.
    t0_actual_smoothed : float, optional
        The time of the ecg/eog event for SMOOTHED data. The default is None.

    Returns
    -------
    affected_orig : list
        List of channels which got average artifact amplitude higher than the average over all channels*norm_lvl.
    not_affected_orig : list
        List of channels which got average artifact amplitude lower than the average over all channels*norm_lvl.
    artif_threshold_lvl : float
        The threshold level for the artifact amplitude.
    affected_smoothed : list
        List of channels which got average artifact amplitude higher than the average over all channels*norm_lvl for SMOOTHED data.
    not_affected_smoothed : list 
        List of channels which got average artifact amplitude lower than the average over all channels*norm_lvl for SMOOTHED data.
    artif_threshold_lvl_smoothed : float
        The threshold level for the artifact amplitude for SMOOTHED data.
    
    """

    timelimit_min=-window_size_for_mean_threshold_method+t0_actual
    timelimit_max=window_size_for_mean_threshold_method+t0_actual


    #Find the channels which got peaks over this mean:
    affected_orig=[]
    not_affected_orig=[]
    affected_smoothed=[]
    not_affected_smoothed=[]

    artif_threshold_lvl=mean_magnitude_peak/norm_lvl #data over this level will be counted as artifact contaminated

    if mean_magnitude_peak_smoothed is None or t0_actual_smoothed is None:
        print('___MEG QC___: ', 'mean_magnitude_peak_smoothed and t0_actual_smoothed should be provided')
    else:
        artifact_lvl_smoothed=mean_magnitude_peak_smoothed/norm_lvl  #SO WHEN USING SMOOTHED CHANNELS - USE SMOOTHED AVERAGE TOO!
        timelimit_min_smoothed=-window_size_for_mean_threshold_method+t0_actual_smoothed
        timelimit_max_smoothed=window_size_for_mean_threshold_method+t0_actual_smoothed


    for potentially_affected in list_mean_artif_epochs:

        result = potentially_affected.detect_artif_above_threshold(artif_threshold_lvl, t, timelimit_min, timelimit_max)
        if result is True:
            affected_orig.append(potentially_affected)
        else:
            not_affected_orig.append(potentially_affected)
        
        result_smoothed = potentially_affected.detect_artif_above_threshold_smoothed(artifact_lvl_smoothed, t, timelimit_min_smoothed, timelimit_max_smoothed)
        if result_smoothed is True:
            affected_smoothed.append(potentially_affected)
        else:
            not_affected_smoothed.append(potentially_affected)

    return affected_orig, not_affected_orig, artif_threshold_lvl, affected_smoothed, not_affected_smoothed, artifact_lvl_smoothed


def plot_affected_channels(artif_affected_channels: list, artifact_lvl: float, t: np.ndarray, ch_type: str, fig_tit: str, chs_by_lobe: dict, flip_data: bool or str = 'flip', smoothed: bool = False, verbose_plots: bool = True):

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
    chs_by_lobe : dict
        dictionary with channel objects sorted by lobe
    flip_data : bool
        If True, the absolute value of the data will be used for the calculation of the mean artifact amplitude. Default to 'flip'. 
        'flip' means that the data will be flipped if the peak of the artifact is negative. 
        This is donr to get the same sign of the artifact for all channels, then to get the mean artifact amplitude over all channels and the threshold for the artifact amplitude onbase of this mean
        And also for the reasons of visualization: the artifact amplitude is always positive.
    smoothed: bool
        Plot smoothed data (true) or nonrmal (false)
    verbose_plots : bool
        True for showing plot in notebook.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure with the mean artifact amplitude for all affected (not affected) channels in 1 plot together with the artifact_lvl.

        
    """

    if artif_affected_channels: #if affected channels present:

        #plot channels separated by lobes:
        affected_names_list = []
        affected_data_list = []
        for ch in artif_affected_channels:
            affected_names_list.append(ch.name)
            if smoothed is True:
                affected_data_list.append(ch.artif_data_smoothed)
            else:
                affected_data_list.append(ch.artif_data)

        affected_data_arr = np.array(affected_data_list)

        df_affected=pd.DataFrame(affected_data_arr.T, columns=affected_names_list)

        fig = plot_df_of_channels_data_as_lines_by_lobe(chs_by_lobe, df_affected, t)

        #decorate the plot:
        ch_type_tit, unit = get_tit_and_unit(ch_type)
        fig.update_layout(
            xaxis_title='Time in seconds',
            yaxis = dict(
                showexponent = 'all',
                exponentformat = 'e'),
            yaxis_title='Mean artifact magnitude in '+unit,
            title={
                'text': fig_tit+str(len(artif_affected_channels))+' '+ch_type_tit,
                'y':0.85,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})


    else:
        fig=go.Figure()
        ch_type_tit, _ = get_tit_and_unit(ch_type)
        title=fig_tit+'0 ' +ch_type_tit
        fig.update_layout(
            title={
            'text': title,
            'x': 0.5,
            'y': 0.9,
            'xanchor': 'center',
            'yanchor': 'top'})
        
    #in any case - add the threshold on the plot
    fig.add_trace(go.Scatter(x=t, y=[(artifact_lvl)]*len(t), line=dict(color='red'), name='Thres=mean_peak/norm_lvl')) #add threshold level

    if flip_data is False and artifact_lvl is not None: 
        fig.add_trace(go.Scatter(x=t, y=[(-artifact_lvl)]*len(t), line=dict(color='black'), name='-Thres=mean_peak/norm_lvl'))

    if verbose_plots is True:
        fig.show()

    return fig




def flip_channels(artif_per_ch_nonflipped: list, tmin: float, tmax: float, sfreq: int, params_internal: dict):

    """
    Flip the channels if the peak of the artifact is negative and located close to the estimated t0.

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

    
    Parameters
    ----------
    avg_artif_nonflipped : list
        List of Avg_artif objects with not flipped data.
    tmin : float
        time in sec before the peak of the artifact (negative number).
    tmax : float
        time in sec after the peak of the artifact (positive number).
    sfreq : int
        Sampling frequency.
    params_internal : dict
        Dictionary with internal parameters.


    Returns
    -------
    artifacts_flipped : list
        The list of the ecg epochs.
    artif_time_vector : np.ndarray
        The time vector for the ecg epoch (for plotting further).


    """

    artif_time_vector = np.round(np.arange(tmin, tmax+1/sfreq, 1/sfreq), 3) #yes, you need to round

    _, t0_estimated_ind, t0_estimated_ind_start, t0_estimated_ind_end = estimate_t0(artif_per_ch_nonflipped, artif_time_vector, params_internal)

    artifacts_flipped=[]

    for ch_artif in artif_per_ch_nonflipped: #for each channel:

        if ch_artif.peak_loc.size>0: #if there are any peaks - find peak_locs which is located the closest to t0_estimated_ind:
            peak_loc_closest_to_t0=ch_artif.peak_loc[np.argmin(np.abs(ch_artif.peak_loc-t0_estimated_ind))]

            #if peak_loc_closest_t0 is negative and is located in the estimated time window of the wave - flip the data:
            if (ch_artif.artif_data[peak_loc_closest_to_t0]<0) & (peak_loc_closest_to_t0>t0_estimated_ind_start) & (peak_loc_closest_to_t0<t0_estimated_ind_end):
                ch_artif.flip_artif()
                if ch_artif.artif_data_smoothed is not None: #if there is also smoothed data present - flip it as well:
                    ch_artif.flip_artif_smoothed()
            else:
                pass
        else:
            pass

        artifacts_flipped.append(ch_artif)

    return artifacts_flipped, artif_time_vector


def estimate_t0(artif_per_ch_nonflipped: list, t: np.ndarray, params_internal: dict):
    
    """ 
    Estimate t0 for the artifact. MNE has it s own estimation of t0, but it is often not accurate.
    Steps:

    1. find maxima on all channels (absolute values) in time frame around -0.02<t[peak_loc]<0.012 
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
    params_internal : dict
        Dictionary with internal parameters.
        
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

    window_size_for_mean_threshold_method=params_internal['window_size_for_mean_threshold_method']
    timelimit_min = params_internal['timelimit_min']
    timelimit_max = params_internal['timelimit_max']

    #collect artif data for each channel into nd array:
    avg_ecg_epoch_data_nonflipped = np.array([ch.artif_data for ch in artif_per_ch_nonflipped]) 

    #find indexes of t where t is between timelimit_min and timelimit_max (limits where R wave typically is detected by mne):
    t_event_ind=np.argwhere((t>timelimit_min) & (t<timelimit_max))

    # cut the data of each channel to the time interval where wave is expected to be:
    avg_ecg_epoch_data_nonflipped_limited_to_event=avg_ecg_epoch_data_nonflipped[:,t_event_ind[0][0]:t_event_ind[-1][0]]

    #find 5 channels with max values in the time interval where wave is expected to be:
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
    t0_estimated_ind_start=np.argwhere(t==round(t0_estimated-window_size_for_mean_threshold_method, 3))[0][0] 
    t0_estimated_ind_end=np.argwhere(t==round(t0_estimated+window_size_for_mean_threshold_method, 3))[0][0]
    #yes you have to round it here because the numbers stored in in memery like 0.010000003 even when it looks like 0.01, hence np.where cant find the target float in t vector


    #another way without round would be to find the closest index of t to t0_estimated-0.015:
    #t0_estimated_ind_start=np.argwhere(t==np.min(t[t<t0_estimated-window_size]))[0][0]
    # find the closest index of t to t0_estimated+0.015:
    #t0_estimated_ind_end=np.argwhere(t==np.min(t[t>t0_estimated+window_size]))[0][0]
    
    return t0_estimated, t0_estimated_ind, t0_estimated_ind_start, t0_estimated_ind_end



def calculate_artifacts_on_channels(artif_epochs: mne.Epochs, channels: list, chs_by_lobe: dict, thresh_lvl_peakfinder: float, tmin: float, tmax: float, params_internal: dict, gaussian_sigma: int):

    """
    Find channels that are affected by ECG or EOG events.
    The function calculates average ECG epoch for each channel and then finds the peak of the wave on each channel.
   

    Parameters
    ----------
    artif_epochs : mne.Epochs
        ECG epochs.
    channels : list
        List of channels to use.
    chs_by_lobe : dict
        dictionary with channel objects sorted by lobe
    thresh_lvl_peakfinder : float
        Threshold level for peakfinder.
    tmin : float
        Start time.
    tmax : float
        End time.
    params_internal : dict
        Dictionary with internal parameters.
    gaussian_sigma : int, optional
        Sigma for gaussian filter. The default is 6. Usually for EOG need higher (6-7), t s more noisy, for ECG - lower (4-5).

        
    Returns 
    -------
    all_artifs_nonflipped : list
        List of channels with Avg_artif objects, data in these is not flipped yet.
        
    """

    max_n_peaks_allowed_for_ch = params_internal['max_n_peaks_allowed_for_ch']

    max_n_peaks_allowed=round(((abs(tmin)+abs(tmax))/0.1)*max_n_peaks_allowed_for_ch)
    print('___MEG QC___: ', 'max_n_peaks_allowed_for_ch: '+str(max_n_peaks_allowed))

    #1.:
    #averaging the ECG epochs together:
    avg_epochs = artif_epochs.average(picks=channels)#.apply_baseline((-0.5, -0.2))
    #avg_ecg_epochs is evoked:Evoked objects typically store EEG or MEG signals that have been averaged over multiple epochs.
    #The data in an Evoked object are stored in an array of shape (n_channels, n_times)

    # 1. find maxima on all channels (absolute values) in time frame around -0.02<t[peak_loc]<0.012 (here R wave is typicaly detected by mne - for ecg, for eog it is -0.1<t[peak_loc]<0.2)
    # 2. take 5 channels with most prominent peak 
    # 3. find estimated average t0 for all 5 channels, because t0 of event which mne estimated is often not accurate

    avg_artif_data_nonflipped=avg_epochs.data #shape (n_channels, n_times)

    # 4. detect peaks on channels 
    all_artifs_nonflipped = []
    for i, ch_data in enumerate(avg_artif_data_nonflipped):  # find peaks and estimate detect wave shape on all channels
        artif_nonflipped = Avg_artif(name=channels[i], artif_data=ch_data)
        artif_nonflipped.get_peaks_wave(max_n_peaks_allowed=max_n_peaks_allowed, thresh_lvl_peakfinder=thresh_lvl_peakfinder)
        artif_nonflipped.get_peaks_wave_smoothed(gaussian_sigma = gaussian_sigma, max_n_peaks_allowed=max_n_peaks_allowed, thresh_lvl_peakfinder=thresh_lvl_peakfinder)
        all_artifs_nonflipped.append(artif_nonflipped)

    # assign lobe to each channel right away (for plotting)
    all_artifs_nonflipped = assign_lobe_to_artifacts(all_artifs_nonflipped, chs_by_lobe)

    return all_artifs_nonflipped


def find_mean_rwave_blink(ch_data: np.ndarray or list, event_indexes: np.ndarray, tmin: float, tmax: float, sfreq: int):

    """
    Calculate mean R wave on the data of either original ECG channel or reconstructed ECG channel.
    In some cases (for reconstructed) there are no events, so mean Rwave cant be estimated.
    This usually does not happen for real ECG channel. Because real ECG channel passes the check even earlier in the code. (see check_3_conditions())

    Parameters
    ----------
    ch_data : np.ndarray
        Data of the channel (real or reconstructed).
    event_indexes : array
        Array of event indexes (R wave peaks).
    tmin : float
        Start time of ECG epoch (negative value).
    tmax : float
        End time of ECG epoch (positive value).
    sfreq : int
        Sampling frequency.

    Returns
    -------
    mean_rwave : np.ndarray
        Mean R wave (1 dimentional).
    
    """

    # Initialize an empty array to store the extracted epochs
    epochs = np.zeros((len(event_indexes), int((tmax-tmin)*sfreq)+1))

    # Loop through each ECG event and extract the corresponding epoch
    for i, event in enumerate(event_indexes):
        start = np.round(event + tmin*sfreq).astype(int)
        end = np.round(event + tmax*sfreq).astype(int)+1

        if start < 0:
            continue

        if end > len(ch_data):
            continue

        epochs[i, :] = ch_data[start:end]

    #average all epochs:
    mean_rwave=np.mean(epochs, axis=0)

    return mean_rwave


def assign_lobe_to_artifacts(artif_per_ch, chs_by_lobe):

    """ Loop over all channels in artif_per_ch and assign lobe and lobe color to each channel for plotting purposes.

    Parameters
    ----------
    artif_per_ch : list
        List of channels with Avg_artif objects.
    chs_by_lobe : dict
        Dictionary of channels grouped by lobe with color codes.

    Returns
    -------
    artif_per_ch : list
        List of channels with Avg_artif objects, now with assigned lobe and color for plotting. 

    """
    

    for lobe,  ch_list in chs_by_lobe.items(): #loop over dict of channels for plotting
        for ch_for_plot in ch_list: #same, level deeper
            for ch_artif in artif_per_ch: #loop over list of instances of Avg_artif class
                if ch_artif.name == ch_for_plot.name:
                    ch_artif.lobe = ch_for_plot.lobe
                    ch_artif.color = ch_for_plot.lobe_color
                    break

    #Check that all channels have been assigned a lobe:
    for ch_artif in artif_per_ch:
        if ch_artif.lobe is None or ch_artif.color is None:
            print('___MEG QC___: ', 'Channel ', ch_artif.name, ' has not been assigned a lobe or color for plotting. Check assign_lobe_to_artifacts().')

    return artif_per_ch


def find_affected_by_correlation(mean_rwave: np.ndarray, artif_per_ch: list):

    """"
    Calculate correlation coefficient and p-value between mean R wave and each channel in artif_per_ch.
    Higher correlation coefficient means that the channel is more likely to be affected by ECG artifact.

    Here we assume that both vectors have sme length! these are defined by tmin and tmax which are set in config and propageted in this script. 
    Keep in mind if changing anything with tmin and tmax
    
    Parameters
    ----------
    mean_rwave : np.ndarray
        Mean R wave (1 dimentional).
    artif_per_ch : list
        List of channels with Avg_artif objects.

    Returns
    -------
    artif_per_ch : list
        List of channels with Avg_artif objects, now with assigned correlation coefficient and p-value.
    
    """

    
    if len(mean_rwave) != len(artif_per_ch[0].artif_data):
        print('___MEG QC___: ', 'mean_rwave and artif_per_ch.artif_data have different length! Both are defined by tmin and tmax in config.py and are use to cut the data. Keep in mind if changing anything with tmin and tmax')
        print('len(mean_rwave): ', len(mean_rwave), 'len(artif_per_ch[0].artif_data): ', len(artif_per_ch[0].artif_data))
        return

    for ch in artif_per_ch:
        ch.corr_coef, ch.p_value = pearsonr(ch.artif_data_smoothed, mean_rwave)

    return artif_per_ch


def plot_correlation(artif_per_ch, ecg_or_eog, m_or_g, verbose_plots=False):

    """
    Plot correlation coefficient and p-value between mean R wave and each channel in artif_per_ch.

    Parameters
    ----------
    artif_per_ch : list
        List of channels with Avg_artif objects.
    ecg_or_eog : str
        Either 'ECG' or 'EOG'.
    m_or_g : str
        Either 'mag' or 'grad'.
    verbose_plots : bool
        If True, plot will be displayed in a notebook.

    Returns
    -------
    corr_derivs : list
        List with 1 QC_derivative instance: Figure with correlation coefficient and p-value between mean R wave and each channel in artif_per_ch.
    
    """

    _, _, _, corr_val_of_last_most_correlated, corr_val_of_last_middle_correlated, corr_val_of_last_least_correlated = split_correlated_artifacts_into_3_groups(artif_per_ch)

    print('least', corr_val_of_last_least_correlated)
    print('middle', corr_val_of_last_middle_correlated)
    print('most', corr_val_of_last_most_correlated)

    traces = []

    tit, _ = get_tit_and_unit(m_or_g)

    for ch in artif_per_ch:
        traces += [go.Scatter(x=[abs(ch.corr_coef)], y=[ch.p_value], mode='markers', marker=dict(size=5, color=ch.color), name=ch.name, legendgroup=ch.lobe, legendgrouptitle=dict(text=ch.lobe.upper()), hovertemplate='Corr coef: '+str(ch.corr_coef)+'<br>p-value: '+str(abs(ch.p_value)))]

    fig = go.Figure(data=traces)

    #add rectangles to the plot to separate most correlated (red), middle (yellow) and least correlated (green) channels:
    #separate rage -1 to 1 into 6 equal parts:
    # ranges=np.linspace(-1, 1, 7)
    # x_most=[ranges[0], ranges[1]]
    # x_most2=[ranges[-1], ranges[-2]]
    # x_middle=[ranges[1], ranges[2]]
    # x_middle2=[ranges[-2], ranges[-3]]
    # x_least=[ranges[2], ranges[4]]

    # fig.add_shape(type="rect", xref="x", yref="y", x0=x_most[0], y0=-0.1, x1=x_most[1], y1=1.1, line=dict(color="Red", width=2), fillcolor="Red", opacity=0.1)
    # fig.add_shape(type="rect", xref="x", yref="y", x0=x_most2[0], y0=-0.1, x1=x_most2[1], y1=1.1, line=dict(color="Red", width=2), fillcolor="Red", opacity=0.1)
    # fig.add_shape(type="rect", xref="x", yref="y", x0=x_middle[0], y0=-0.1, x1=x_middle[1], y1=1.1, line=dict(color="Yellow", width=2), fillcolor="Yellow", opacity=0.1)
    # fig.add_shape(type="rect", xref="x", yref="y", x0=x_middle2[0], y0=-0.1, x1=x_middle2[1], y1=1.1, line=dict(color="Yellow", width=2), fillcolor="Yellow", opacity=0.1)
    # fig.add_shape(type="rect", xref="x", yref="y", x0=x_least[0], y0=-0.1, x1=x_least[1], y1=1.1, line=dict(color="Green", width=2), fillcolor="Green", opacity=0.1)

    fig.add_shape(type="rect", xref="x", yref="y", x0=0, y0=-0.1, x1=corr_val_of_last_least_correlated, y1=1.1, line=dict(color="Green", width=2), fillcolor="Green", opacity=0.1)
    fig.add_shape(type="rect", xref="x", yref="y", x0=corr_val_of_last_least_correlated, y0=-0.1, x1=corr_val_of_last_middle_correlated, y1=1.1, line=dict(color="Yellow", width=2), fillcolor="Yellow", opacity=0.1)
    fig.add_shape(type="rect", xref="x", yref="y", x0=corr_val_of_last_middle_correlated, y0=-0.1, x1=1, y1=1.1, line=dict(color="Red", width=2), fillcolor="Red", opacity=0.1)

    #set axis titles:
    fig.update_xaxes(title_text='Correlation coefficient')
    fig.update_yaxes(title_text='P-value')

    #set title:
    fig.update_layout(title_text=tit+': Pearson correlation between average '+ecg_or_eog+' recorded and average artifact for each channel')

    if verbose_plots is True:
        fig.show()

    corr_derivs = [QC_derivative(fig, 'Corr_values_'+ecg_or_eog, 'plotly', description_for_user='Absolute value of the correlation coefficient is shown here. The sign would only represent the position of the channel towards magnetic field. <p>- Green: 33% of all channels that have the weakest correlation with mean ' +ecg_or_eog +'; </p> <p>- Yellow: 33% of all channels that have mild correlation with mean ' +ecg_or_eog +';</p> <p>- Red: 33% of all channels that have the stronges correlation with mean ' +ecg_or_eog +'. </p>')]

    return corr_derivs


def split_correlated_artifacts_into_3_groups(artif_per_ch):

    """
    Collect artif_per_ch into 3 lists - for plotting:
    - a third of all channels that are the most correlated with mean_rwave
    - a third of all channels that are the least correlated with mean_rwave
    - a third of all channels that are in the middle of the correlation with mean_rwave

    Parameters
    ----------
    artif_per_ch : list
        List of objects of class Avg_artif

    Returns
    -------
    artif_per_ch : list
        List of objects of class Avg_artif, ranked by correlation coefficient
    most_correlated : list
        List of objects of class Avg_artif that are the most correlated with mean_rwave
    least_correlated : list
        List of objects of class Avg_artif that are the least correlated with mean_rwave
    middle_correlated : list
        List of objects of class Avg_artif that are in the middle of the correlation with mean_rwave
    corr_val_of_last_least_correlated : float
        Correlation value of the last channel in the list of the least correlated channels
    corr_val_of_last_middle_correlated : float
        Correlation value of the last channel in the list of the middle correlated channels
    corr_val_of_last_most_correlated : float
        Correlation value of the last channel in the list of the most correlated channels


    """

    #sort by correlation coef. Take abs of the corr coeff, because the channels might be just flipped due to their location against magnetic field::
    artif_per_ch.sort(key=lambda x: abs(x.corr_coef), reverse=True)

    most_correlated = artif_per_ch[:int(len(artif_per_ch)/3)]
    least_correlated = artif_per_ch[-int(len(artif_per_ch)/3):]
    middle_correlated = artif_per_ch[int(len(artif_per_ch)/3):-int(len(artif_per_ch)/3)]

    #get correlation values of all most correlated channels:
    all_most_correlated = [abs(ch.corr_coef) for ch in most_correlated]
    all_middle_correlated = [abs(ch.corr_coef) for ch in middle_correlated]
    all_least_correlated = [abs(ch.corr_coef) for ch in least_correlated]

    #find the correlation value of the last channel in the list of the most correlated channels:
    # this is needed for plotting correlation values, to know where to put separation rectangles.
    corr_val_of_last_most_correlated = max(all_most_correlated)
    corr_val_of_last_middle_correlated = max(all_middle_correlated)
    corr_val_of_last_least_correlated = max(all_least_correlated)

    return most_correlated, middle_correlated, least_correlated, corr_val_of_last_most_correlated, corr_val_of_last_middle_correlated, corr_val_of_last_least_correlated



def plot_artif_per_ch_correlated_lobes(artif_per_ch: list, tmin: float, tmax: float, m_or_g: str, ecg_or_eog: str, chs_by_lobe: dict, flip_data: bool, verbose_plots: bool):

    """
    Plot average artifact for each channel, colored by lobe, 
    channels are split into 3 separate plots, based on their correlation with mean_rwave: equal number of channels in each group.

    Parameters
    ----------
    artif_per_ch : list
        List of objects of class Avg_artif
    tmin : float
        Start time of the epoch (negative value)
    tmax : float
        End time of the epoch
    m_or_g : str
        Type of the channel: mag or grad
    ecg_or_eog : str
        Type of the artifact: ECG or EOG
    chs_by_lobe : dict
        Dictionary with channels split by lobe
    flip_data : bool
        Use True or False, doesnt matter here. It is only passed into the plotting function and influences the threshold presentation. But since treshold is not used in correlation method, this is not used.
    verbose_plots : bool
        If True, plots are shown in the notebook.

    Returns
    -------
    artif_per_ch : list
        List of objects of class Avg_artif
    affected_derivs : list
        List of objects of class QC_derivative (plots)
    

    """


    most_correlated, middle_correlated, least_correlated, _, _, _ = split_correlated_artifacts_into_3_groups(artif_per_ch)

    #plot using plotly: 
    # artif_per_ch.artif_data - a third of all channels that are the most correlated with mean_rwave, 
    # artif_per_ch.artif_data - a third of all channels that are the less with mean_rwave, 
    # artif_per_ch.artif_data - a third of all channels that are the least correlated with mean_rwave

    artif_time_vector = np.linspace(tmin, tmax, len(artif_per_ch[0].artif_data))

    fig_most_affected = plot_affected_channels(most_correlated, None, artif_time_vector, ch_type=m_or_g, fig_tit=ecg_or_eog+' most affected channels (smoothed): ', chs_by_lobe=chs_by_lobe, flip_data=flip_data, smoothed = True, verbose_plots=False)
    fig_middle_affected = plot_affected_channels(middle_correlated, None, artif_time_vector, ch_type=m_or_g, fig_tit=ecg_or_eog+' middle affected channels (smoothed): ', chs_by_lobe=chs_by_lobe, flip_data=flip_data, smoothed = True, verbose_plots=False)
    fig_least_affected = plot_affected_channels(least_correlated, None, artif_time_vector, ch_type=m_or_g, fig_tit=ecg_or_eog+' least affected channels (smoothed): ', chs_by_lobe=chs_by_lobe, flip_data=flip_data, smoothed = True, verbose_plots=False)

    #set the same Y axis limits for all 3 figures for clear comparison:
    
    # combine the data lists into one numpy array
    arr = np.array([ch.artif_data for ch in artif_per_ch])

    # #find the highest and lowest value in artif_per_ch.artif_data:
    ymin = np.min(arr)
    ymax = np.max(arr)

    ylim = [ymin*.95, ymax*1.05]

    # update the layout of all three figures with the same y-axis limits
    fig_most_affected.update_layout(yaxis_range=ylim)
    fig_middle_affected.update_layout(yaxis_range=ylim)
    fig_least_affected.update_layout(yaxis_range=ylim)

    if verbose_plots is True:
        fig_most_affected.show()
        fig_middle_affected.show()
        fig_least_affected.show()
    
    affected_derivs = []
    affected_derivs += [QC_derivative(fig_most_affected, ecg_or_eog+'most_affected_channels_'+m_or_g, 'plotly')]
    affected_derivs += [QC_derivative(fig_middle_affected, ecg_or_eog+'middle_affected_channels_'+m_or_g, 'plotly')]
    affected_derivs += [QC_derivative(fig_least_affected, ecg_or_eog+'least_affected_channels_'+m_or_g, 'plotly')]
        
    return affected_derivs



def find_affected_over_mean(artif_per_ch: list, ecg_or_eog: str, params_internal: dict, thresh_lvl_peakfinder: float, plotflag: bool, verbose_plots: bool, m_or_g: str, chs_by_lobe: dict, norm_lvl: float, flip_data: bool, gaussian_sigma: float, artif_time_vector: np.ndarray):
    
    """
    1. Calculate average ECG epoch on the epochs from all channels. Check if average has a wave shape. 
    If no wave shape - no need to check for affected channels further.
    If it has - check further

    2. Set a threshold which defines a high amplitude of ECG event. (All above this threshold counted as potential ECG peak.)
    Threshold is the magnitude of the peak of the average ECG/EOG epoch multiplued by norm_lvl. 
    norl_lvl is chosen by user in config file
    
    3. Find all peaks above this threshold.
    Finding approach:

    - again, set t0 actual as the time point of the peak of an average artifact (over all channels)
    - again, set a window around t0_actual. this new window is defined by how long the wave of the artifact normally is. 
        The window is centered around t0 and for ECG it will be -0.-02 to 0.02s, for EOG it will be -0.1 to 0.1s.
    - find one main peak of the epoch for each channel which would be inside this window and closest to t0.
    - if this peaks magnitude is over the threshold - this channels is considered to be affected by ECG or EOG. Otherwise - not affected.
        (The epoch has to have a wave shape).

    4. Affected and non affected channels will be plotted and outputted as lists for adding tothe json on the next step.

    Parameters
    ----------
    artif_per_ch : list 
        list of Avg_artif objects
    ecg_or_eog : str
        'ECG' or 'EOG'
    params_internal : dict
        dictionary with parameters from setings_internal file
    thresh_lvl_peakfinder : float
        threshold for peakfinder. Defines the magnitude of the peak of the average ECG/EOG epoch multiplued by norm_lvl.
    plotflag : bool
        if True - plots will be made
    verbose_plots : bool
        if True - plots will be shown in notebook
    m_or_g : str
        'mag' or 'grad'
    chs_by_lobe : dict
        dictionary with channels grouped by lobes
    norm_lvl : float
        defines the threshold for peakfinder. Threshold = mean overall artifact poch magnitude * norm_lvl
    flip_data : bool
        ifo for plotting. If data was flipped - only upper threshold will be shown on the plot, if not - both upper and lower
    gaussian_sigma : float
        sigma for gaussian smoothing
    artif_time_vector : np.ndarray
        time vector for the artifact epoch

    Returns
    -------
    affected_channels: list
        list of affected channels
    affected_derivs: list
        list of QC_derivative objects with figures for affected and not affected channels (smoothed and not soothed versions)
    bad_avg_str : str
        string about the average artifact: if it was not considered to be a wave shape
    avg_overall_obj : Avg_artif
        Avg_artif object with the average artifact

    """

    max_n_peaks_allowed_for_avg = params_internal['max_n_peaks_allowed_for_avg']
    window_size_for_mean_threshold_method = params_internal['window_size_for_mean_threshold_method']

    artif_per_ch_only_data = [ch.artif_data for ch in artif_per_ch] # USE NON SMOOTHED data. If needed, can be changed to smoothed data
    avg_overall=np.mean(artif_per_ch_only_data, axis=0) 
    # will show if there is ecg artifact present  on average. should have wave shape if yes. 
    # otherwise - it was not picked up/reconstructed correctly

    avg_overall_obj=Avg_artif(name='Mean_'+ecg_or_eog+'_overall',artif_data=avg_overall)

    #detect peaks and wave for the average overall artifact:
    avg_overall_obj.get_peaks_wave(max_n_peaks_allowed=max_n_peaks_allowed_for_avg, thresh_lvl_peakfinder=thresh_lvl_peakfinder)
    avg_overall_obj.get_peaks_wave_smoothed(gaussian_sigma = gaussian_sigma, max_n_peaks_allowed=max_n_peaks_allowed_for_avg, thresh_lvl_peakfinder=thresh_lvl_peakfinder)

    affected_derivs=[]
    affected_channels = []

    if avg_overall_obj.wave_shape is True or avg_overall_obj.wave_shape_smoothed is True: #if the average ecg artifact is good - do steps 2 and 3:

        mean_magnitude_peak=np.max(avg_overall_obj.peak_magnitude)
        mean_ecg_loc_peak = avg_overall_obj.peak_loc[np.argmax(avg_overall_obj.peak_magnitude)]
        t0_actual=artif_time_vector[mean_ecg_loc_peak]
        #set t0_actual as the time of the peak of the average ecg artifact
        
        if avg_overall_obj.wave_shape_smoothed is not None: #if smoothed average and its peaks were also calculated:
            mean_magnitude_peak_smoothed=np.max(avg_overall_obj.peak_magnitude_smoothed)
            mean_ecg_loc_peak_smoothed = avg_overall_obj.peak_loc_smoothed[np.argmax(avg_overall_obj.peak_magnitude_smoothed)]
            t0_actual_smoothed=artif_time_vector[mean_ecg_loc_peak_smoothed]
        else:
            mean_magnitude_peak_smoothed=None
            t0_actual_smoothed=None
            
        
        tit, _ = get_tit_and_unit(m_or_g)
        
        if avg_overall_obj.wave_shape is True:
            avg_artif_description1 = tit+": (original) GOOD " +ecg_or_eog+ " average. Detected " + str(len(avg_overall_obj.peak_magnitude)) + " peak(s). Expected 1-" + str(max_n_peaks_allowed_for_avg) + " peaks (pos+neg)."
        else:
            avg_artif_description1 = tit+": (original) BAD " +ecg_or_eog+ " average. Detected " + str(len(avg_overall_obj.peak_magnitude)) + " peak(s). Expected 1-" + str(max_n_peaks_allowed_for_avg) + " peaks (pos+neg). Affected channels can not be estimated."

        if avg_overall_obj.wave_shape_smoothed is True:
            avg_artif_description2 =  tit+": (smoothed) GOOD " +ecg_or_eog+ " average. Detected " + str(len(avg_overall_obj.peak_magnitude_smoothed)) + " peak(s). Expected 1-" + str(max_n_peaks_allowed_for_avg) + " peaks (pos+neg)."
        else:
            avg_artif_description2 = tit+": (smoothed) BAD " +ecg_or_eog+ " average. Detected " + str(len(avg_overall_obj.peak_magnitude_smoothed)) + " peak(s). Expected 1-" + str(max_n_peaks_allowed_for_avg) + " peaks (pos+neg). Affected channels can not be estimated."

        avg_artif_description = avg_artif_description1 + "<p></p>" + avg_artif_description2

        print('___MEG QC___: ', avg_artif_description1)
        print('___MEG QC___: ', avg_artif_description2)

        bad_avg_str = ''

        # detect channels which are over the threshold defined by mean_magnitude_peak (average overall artifact) and norm_lvl (set in config):
        affected_channels, not_affected_channels, artifact_lvl, affected_channels_smoothed, not_affected_channels_smoothed, artifact_lvl_smoothed = detect_channels_above_norm(norm_lvl=norm_lvl, list_mean_artif_epochs=artif_per_ch, mean_magnitude_peak=mean_magnitude_peak, t=artif_time_vector, t0_actual=t0_actual, window_size_for_mean_threshold_method=window_size_for_mean_threshold_method, mean_magnitude_peak_smoothed=mean_magnitude_peak_smoothed, t0_actual_smoothed=t0_actual_smoothed)

        if plotflag is True:
            fig_affected = plot_affected_channels(affected_channels, artifact_lvl, artif_time_vector, ch_type=m_or_g, fig_tit=ecg_or_eog+' affected channels (orig): ', chs_by_lobe=chs_by_lobe, flip_data=flip_data, smoothed = False, verbose_plots=verbose_plots)
            fig_affected_smoothed = plot_affected_channels(affected_channels_smoothed, artifact_lvl_smoothed, artif_time_vector, ch_type=m_or_g, fig_tit=ecg_or_eog+' affected channels (smoothed): ', chs_by_lobe=chs_by_lobe, flip_data=flip_data, smoothed = True, verbose_plots=verbose_plots)
            fig_not_affected = plot_affected_channels(not_affected_channels, artifact_lvl, artif_time_vector, ch_type=m_or_g, fig_tit=ecg_or_eog+' not affected channels (orig): ', chs_by_lobe=chs_by_lobe, flip_data=flip_data, smoothed = False, verbose_plots=verbose_plots)
            fig_not_affected_smoothed = plot_affected_channels(not_affected_channels_smoothed, artifact_lvl_smoothed, artif_time_vector, ch_type=m_or_g, fig_tit=ecg_or_eog+' not affected channels (smoothed): ', chs_by_lobe=chs_by_lobe, flip_data=flip_data, smoothed = True, verbose_plots=verbose_plots)
            
            affected_derivs += [QC_derivative(fig_affected, ecg_or_eog+'_affected_channels_'+m_or_g, 'plotly')]
            affected_derivs += [QC_derivative(fig_not_affected, ecg_or_eog+'_not_affected_channels_smooth'+m_or_g, 'plotly')]
            affected_derivs += [QC_derivative(fig_affected_smoothed, ecg_or_eog+'_affected_channels_'+m_or_g, 'plotly')]
            affected_derivs += [QC_derivative(fig_not_affected_smoothed, ecg_or_eog+'_not_affected_channels_smooth'+m_or_g, 'plotly')]

    else: #if the average artifact is bad - end processing
        tit, _ = get_tit_and_unit(m_or_g)
        avg_artif_description = tit+": BAD " +ecg_or_eog+ " average. Detected " + str(len(avg_overall_obj.peak_magnitude)) + " peak(s). Expected 1-" + str(max_n_peaks_allowed_for_avg) + " peaks (pos+neg). Affected channels can not be estimated."
        bad_avg_str = tit+": "+ ecg_or_eog+ " signal detection/reconstruction did not produce reliable results. Affected channels can not be estimated."
        print('___MEG QC___: ', bad_avg_str)


    if plotflag is True:
        fig_avg = avg_overall_obj.plot_epoch_and_peak(artif_time_vector, 'Mean '+ecg_or_eog+' artifact over all data: ', m_or_g, None, plot_original = True, plot_smoothed = True)

        if verbose_plots is True:
            fig_avg.show()
        affected_derivs.insert(0, QC_derivative(fig_avg, 'overall_average_ECG_epoch_'+m_or_g, 'plotly', description_for_user = avg_artif_description))
        #prepend the avg plot before all other plots. because they will be added to report in the order they are in list.

    return affected_channels, affected_derivs, bad_avg_str, avg_overall_obj



#%%
def make_dict_global_ECG_EOG(channels_ranked: list, use_method: str):
    """
    Make a dictionary for the global part of simple metrics for ECG/EOG artifacts.
    For ECG/EOG no local metrics are calculated, so global is the only one.
    
    Parameters
    ----------
    channels_ranked : list
        List of all affected channels
    use_method : str
        Method used for detection of ECG/EOG artifacts: correlation, correlation_reconstructed or mean_threshold.
        Depending in this the dictionary will have difefrent structure and descriptions.
        
    Returns
    -------
    metric_global_content : dict
        Dictionary with simple metrics for ECG/EOG artifacts.
   
    """

    # sort all_affected_channels by main_peak_magnitude:
    if use_method == 'mean_threshold':
        if channels_ranked:
            all_affected_channels_sorted = sorted(channels_ranked, key=lambda ch: ch.main_peak_magnitude, reverse=True)
            affected_chs = {ch.name: ch.main_peak_magnitude for ch in all_affected_channels_sorted}
            metric_global_content = {'details':  affected_chs}
        else:
            metric_global_content = {'details':  None}
    elif use_method == 'correlation' or use_method == 'correlation_reconstructed':
        all_affected_channels_sorted = sorted(channels_ranked, key=lambda ch: abs(ch.corr_coef), reverse=True)
        affected_chs = {ch.name: [ch.corr_coef, ch.p_value] for ch in all_affected_channels_sorted}
        metric_global_content = {'details':  affected_chs}
    else:
        raise ValueError('Unknown method_used: ', use_method)

    

    return metric_global_content


def make_simple_metric_ECG_EOG(channels_ranked: dict, m_or_g_chosen: list, ecg_or_eog: str, avg_artif_str: dict, use_method: str):
    
    """
    Make simple metric for ECG/EOG artifacts as a dictionary, which will further be converted into json file.
    
    Parameters
    ----------
    channels_ranked : dict
        Dictionary with lists of channels.
    m_or_g_chosen : list
        List of channel types chosen for the analysis. 
    ecg_or_eog : str
        String 'ecg' or 'eog' depending on the artifact type.
    avg_artif_str : dict
        Dict with strings with info about the ECG/EOG channel and average artifact.
    use_method : str
        Method used for detection of ECG/EOG artifacts: correlation, correlation_reconstructed or mean_threshold.
        Depending in this the dictionary will have difefrent structure and descriptions.
        
    Returns
    -------
    simple_metric : dict
        Dictionary with simple metrics for ECG/EOG artifacts.
        

    """

    metric_global_name = 'all_channels_raned_by_'+ecg_or_eog+'_contamination_level'
    metric_global_content = {'mag': None, 'grad': None}

    if use_method == 'mean_threshold':
        metric_global_description = 'Here presented the channels with average (over '+ecg_or_eog+' epochs of this channel) ' +ecg_or_eog+ ' artifact above the threshold. Channels are listed here in order from the highest to lowest artifact amplitude. Non affected channels are not listed. Threshld is defined as average '+ecg_or_eog+' artifact peak magnitude over al channels * norm_lvl. norm_lvl is defined in the config file. Channels are presented in the form: ch.name: ch.main_peak_magnitude.'
    elif use_method == 'correlation' or use_method == 'correlation_reconstructed':
        metric_global_description = 'Here the channels are ranked by correlation coefficient between the channel and the averaged '+ecg_or_eog+' channel (recorded or reconstructed). Channels are listed here in order from the highest to lowest correlation coefficient. Channels are presented in the form: ch.name: [ch.corr_coef, ch.p_value]. Sign of the correlation value is kept to reflect the position of the channel toward the magnetic fild omly, it does not reflect the level of contamination (absolute value should be considered for this).'

    for m_or_g in m_or_g_chosen:
        if channels_ranked[m_or_g]: #if there are affected channels for this channel type
            metric_global_content[m_or_g]= make_dict_global_ECG_EOG(channels_ranked[m_or_g], use_method)
        else:
            metric_global_content[m_or_g]= avg_artif_str[m_or_g]

    if use_method == 'mean_threshold':
        measurement_units = True
    else:
        measurement_units = False

    simple_metric = simple_metric_basic(metric_global_name, metric_global_description, metric_global_content['mag'], metric_global_content['grad'], display_only_global=True, measurement_units = measurement_units)

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

    #averaging the ECG epochs together:
    avg_ecg_epochs = ecg_epochs.average() #.apply_baseline((-0.5, -0.2))
    # about baseline see here: https://mne.tools/stable/auto_tutorials/preprocessing/10_preprocessing_overview.html#sphx-glr-auto-tutorials-preprocessing-10-preprocessing-overview-py

    #plot average artifact with topomap
    fig_ecg_sensors = avg_ecg_epochs.plot_joint(times=[tmin-tmin/100, tmin/2, 0, tmax/2, tmax-tmax/100], picks = m_or_g)
    # tmin+tmin/10 and tmax-tmax/10 is done because mne sometimes has a plotting issue, probably connected tosamplig rate: 
    # for example tmin is  set to -0.05 to 0.02, but it  can only plot between -0.0496 and 0.02.

    mne_ecg_derivs += [QC_derivative(fig_ecg_sensors, 'ECG_field_pattern_sensors_'+m_or_g, 'matplotlib')]

    return mne_ecg_derivs


def get_ECG_data_choose_method(raw: mne.io.Raw, ecg_params: dict, verbose_plots: bool):

    """
    Choose the method of finding affected channels based on the presense and quality of ECG channel.

    Options:
    - Channel present and good: correlation with ECG channel
    - Channel present and bad or missing:correlation with reconstructed channel
    - Use mean ECG artifact as threshold (currrently not used)
    
    Parameters
    ----------
    ecg_params : dict
        Dictionary with ECG parameters originating from config file.
    raw : mne.io.Raw
        Raw data.
    verbose_plots : bool
        If True, plots are displayed in notebook.
    
        
    Returns
    -------
    use_method : str
        String with the method chosen for the analysis.
    ecg_str : str
        String with info about the ECG channel presense.
    noisy_ch_derivs : list
        List of QC_derivative objects with plot of the ECG channel
    ecg_data:
        ECG channel data.
    event_indexes:
        Indexes of the ECG events.

    """

    picks_ECG = mne.pick_types(raw.info, ecg=True)

    ecg_ch = [raw.info['chs'][name]['ch_name'] for name in picks_ECG]

    if len(ecg_ch)>=1: #ecg channel present

        if len(ecg_ch)>1: #more than 1 ecg channel present
            ecg_str = 'More than 1 ECG channel found. The first one is used to identify hearbeats. '

        ecg_ch = ecg_ch[0]

        bad_ecg_eog, ecg_data, event_indexes, ecg_eval = detect_noisy_ecg(raw, ecg_ch,  ecg_or_eog = 'ECG', n_breaks_bursts_allowed_per_10min = ecg_params['n_breaks_bursts_allowed_per_10min'], allowed_range_of_peaks_stds = ecg_params['allowed_range_of_peaks_stds'])

        fig = plot_ECG_EOG_channel(ecg_data, event_indexes, ch_name = ecg_ch, fs = raw.info['sfreq'], verbose_plots = verbose_plots)
        noisy_ch_derivs = [QC_derivative(fig, bad_ecg_eog[ecg_ch]+' '+ecg_ch, 'plotly', description_for_user = ecg_ch+' is '+ bad_ecg_eog[ecg_ch]+ ': 1) peaks have similar amplitude: '+str(ecg_eval[0])+', 2) tolerable number of breaks: '+str(ecg_eval[1])+', 3) tolerable number of bursts: '+str(ecg_eval[2]))]

        if bad_ecg_eog[ecg_ch] == 'bad': #ecg channel present but noisy:
            ecg_str = 'ECG channel data is too noisy, cardio artifacts were reconstructed. ECG channel was dropped from the analysis. Consider checking the quality of ECG channel on your recording device. '
            print('___MEG QC___: ', ecg_str)
            raw.drop_channels(ecg_ch)
            use_method = 'correlation_reconstructed'

        elif bad_ecg_eog[ecg_ch] == 'good': #ecg channel present and good - use it
            ecg_str = ecg_ch + ' is good and is used to identify hearbeats. '
            use_method = 'correlation'

    else: #no ecg channel present

        noisy_ch_derivs, ecg_data, event_indexes = [], [], []
        ecg_str = 'No ECG channel found. The signal is reconstructed based on magnetometers data. '
        use_method = 'correlation_reconstructed'
        print('___MEG QC___: ', ecg_str)

    return use_method, ecg_str, noisy_ch_derivs, ecg_data, event_indexes

def get_EOG_data(raw: mne.io.Raw):

    """
    Find if the EOG channel is present anfd get its data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    
        
    Returns
    -------
    eog_str : str
        String with info about the EOG channel presense.
    eog_data:
        EOG channel data.
    event_indexes:
        Indexes of the ECG events.
    eog_ch_name: str
        Name of the EOG channel.

    """

    
    # Find EOG events in your data and get the name of the EOG channel

    # Select the EOG channels
    eog_channels = mne.pick_types(raw.info, meg=False, eeg=False, stim=False, eog=True)

    # Get the names of the EOG channels
    eog_channel_names = [raw.ch_names[ch] for ch in eog_channels]

    print('___MEG QC___: EOG channel names:', eog_channel_names)


    #WHY AM I DOING THIS CHECK??
    try:
        eog_events = mne.preprocessing.find_eog_events(raw)
        #eog_events_times  = (eog_events[:, 0] - raw.first_samp) / raw.info['sfreq']

        #even if 2 EOG channels are present, MNE can only detect blinks!
    except:
        noisy_ch_derivs, eog_data, event_indexes = [], [], []
        eog_str = 'No EOG channels found is this data set - EOG artifacts can not be detected.'
        print('___MEG QC___: ', eog_str)
        return eog_str, noisy_ch_derivs, eog_data, event_indexes

    # Get the data of the EOG channel as an array. MNE only sees blinks, not saccades.
    eog_data = raw.get_data(picks=eog_channel_names)

    eog_str = ', '.join(eog_channel_names)+' used to identify eye blinks. '

    height = np.mean(eog_data) + 1 * np.std(eog_data)
    fs=raw.info['sfreq']

    event_indexes_all = []
    for ch in eog_data:
        event_indexes, _ = find_peaks(ch, height=height, distance=round(0.5 * fs)) #assume there are no peaks within 0.5 seconds from each other.
        event_indexes_all += [event_indexes.tolist()]

    return eog_str, eog_data, event_indexes_all, eog_channel_names


def check_mean_wave(raw: mne.io.Raw, use_method: str, ecg_data: np.ndarray, ecg_or_eog: str, event_indexes: np.ndarray, tmin: float, tmax: float, sfreq: int, params_internal: dict, thresh_lvl_peakfinder: float, verbose_plots: bool):

    """
    Calculate mean R wave based on either real ECG channel data or on reconstructed data (depends on the method used) 
    and check if it has an R wave shape.
    Plot Rwave with peaks.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    use_method : str
        String with the method chosen for the analysis.
    ecg_data: np.ndarray
        ECG channel data. If it s empty, it will be reconstructed here
    event_indexes:
        Indexes of the ECG events.
    tmin : float
        Epoch start time before event (negative value)
    tmax : float
        Epoch end time after event (positive value)
    sfreq : float
        Sampling frequency
    params_internal : dict
        Dictionary with internal parameters originating from settings_internal.
    thresh_lvl_peakfinder : float
        Threshold level for peakfinder function.
    
    Returns
    -------
    mean_rwave_obj.wave_shape: bool
        True if the mean R wave shape is good, False if not.
    ecg_str_checked: str
        String with info about the ECG channel quality (after checking)
    mean_rwave: np.array
        Mean R wave (1 dimentional).
    
    
    """

    max_n_peaks_allowed_for_avg=params_internal['max_n_peaks_allowed_for_avg']

    if use_method == 'correlation_reconstructed':
        _, _, _, ecg_data = mne.preprocessing.find_ecg_events(raw, return_ecg=True)
        # here the RECONSTRUCTED ecg data will be outputted (based on magnetometers), and only if u set return_ecg=True and no real ec channel present).
        ecg_data = ecg_data[0]

    #Now check if ecg_data (reconstructed or original) is good enough:

    #Calculate average over the whole reconstrcted channels and check if it has an R wave shape:

    if len(event_indexes) <1:
        ecg_str_checked = 'Reconstructed ECG data can not be used for artifact detection: no expected wave shape was detected.'
        print('___MEG QC___: ', ecg_str_checked)

        return False, ecg_str_checked, np.empty((0, 0)), []

    mean_rwave = find_mean_rwave_blink(ecg_data, event_indexes, tmin, tmax, sfreq)  

    mean_rwave_obj=Avg_artif(name='Mean_rwave',artif_data=mean_rwave)

    #detect peaks and wave for the average overall artifact:
    mean_rwave_obj.get_peaks_wave(max_n_peaks_allowed=max_n_peaks_allowed_for_avg, thresh_lvl_peakfinder=thresh_lvl_peakfinder)

    if mean_rwave_obj.wave_shape is True:
        ecg_str_checked = 'Mean wave is good enough to use for artifact detection'
        print('___MEG QC___: ', ecg_str_checked)
    else:
        ecg_str_checked = 'Mean wave is not good enough to use for artifact detection. Artifact detection was not performed.'
        print('___MEG QC___: ', ecg_str_checked)


    #Plot:
    if mean_rwave.size > 0:
        t = np.linspace(tmin, tmax, len(mean_rwave))
        if use_method == 'correlation_reconstructed':
            title = 'Mean data of the RECONSTRUCTED '
        elif use_method == 'correlation':
            title = 'Mean data of the RECORDED '
        else:
            title = 'Mean data of '

        mean_rwave_fig = mean_rwave_obj.plot_epoch_and_peak(t, title, ecg_or_eog, fig = None, plot_original = True, plot_smoothed = False)
        if verbose_plots is True:
                mean_rwave_fig.show()

        fig_derivs = [QC_derivative(mean_rwave_fig, 'Mean_artifact'+ecg_or_eog, 'plotly', description_for_user = ecg_str_checked)]
    else:
        fig_derivs = []

    return mean_rwave_obj.wave_shape, ecg_str_checked, mean_rwave, fig_derivs

    

#%%
def ECG_meg_qc(ecg_params: dict, ecg_params_internal: dict, raw: mne.io.Raw, channels: list, chs_by_lobe_orig: dict, m_or_g_chosen: list, verbose_plots: bool):
    
    """
    Main ECG function. Calculates average ECG artifact and finds affected channels.
    
    Parameters
    ----------
    ecg_params : dict
        Dictionary with ECG parameters originating from config file.
    ecg_params_internal : dict
        Dictionary with ECG parameters originating from config file preset, not to be changed by user.
    raw : mne.io.Raw
        Raw data.
    channels : dict
        Dictionary with listds of channels for each channel type (mag and grad).
    chs_by_lobe : dict
        Dictionary with lists of channels by lobe.
    m_or_g_chosen : list
        List of channel types chosen for the analysis.
    verbose_plots : bool
        True for showing plot in notebook.
        
    Returns
    -------
    ecg_derivs : list
        List of all derivatives (plotly figures) as QC_derivative instances
    simple_metric_ECG : dict
        Dictionary with simple metrics for ECG artifacts to be exported into json file.
    ecg_str_for_report : str
        String with information about ECG channel used in the final report.
        

    """

    chs_by_lobe = deepcopy(chs_by_lobe_orig) 
    #in case we will change this variable in any way. If not copied it might introduce errors in parallel processing. 
    # This variable is used in all modules

    if verbose_plots is False:
        matplotlib.use('Agg') 
        #this command will suppress showing matplotlib figures produced by mne. They will still be saved for use in report but not shown when running the pipeline

    sfreq=raw.info['sfreq']
    tmin=ecg_params_internal['ecg_epoch_tmin']
    tmax=ecg_params_internal['ecg_epoch_tmax']

    #WROTE THIS BEFORE, BUT ACTUALLY NEED TO CHECK IF IT S STILL TRUE OR THE PROBLEM WAS SOLVED FRO THRESHOLD METHOD:
    #tmin, tmax can be anything from -0.1/0.1 to -0.04/0.04. for CORRELATION method. But if we do mean and threshold - time best has to be -0.04/0.04. 
    # For this method number of peaks in particular time frame is calculated and based on that good/bad rwave is decided.
    norm_lvl=ecg_params['norm_lvl']
    gaussian_sigma=ecg_params['gaussian_sigma']
    thresh_lvl_peakfinder=ecg_params['thresh_lvl_peakfinder']

    ecg_derivs = []
    use_method, ecg_str, noisy_ch_derivs, ecg_data, event_indexes = get_ECG_data_choose_method(raw, ecg_params, verbose_plots)
    ecg_derivs += noisy_ch_derivs


    mean_good, ecg_str_checked, mean_rwave, rwave_derivs = check_mean_wave(raw, use_method, ecg_data, 'ECG', event_indexes, tmin, tmax, sfreq, ecg_params_internal, thresh_lvl_peakfinder, verbose_plots)
    ecg_str += ecg_str_checked

    ecg_derivs += rwave_derivs

    if mean_good is False:
        simple_metric_ECG = {'description': ecg_str}
        return ecg_derivs, simple_metric_ECG, ecg_str, []

    
    affected_channels={}
    bad_avg_str = {}
    avg_objects_ecg =[]

    for m_or_g  in m_or_g_chosen:

        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, picks=channels[m_or_g], tmin=tmin, tmax=tmax)

        ecg_derivs += plot_ecg_eog_mne(ecg_epochs, m_or_g, tmin, tmax)

        artif_per_ch = calculate_artifacts_on_channels(ecg_epochs, channels[m_or_g], chs_by_lobe=chs_by_lobe[m_or_g], thresh_lvl_peakfinder=thresh_lvl_peakfinder, tmin=tmin, tmax=tmax, params_internal=ecg_params_internal, gaussian_sigma=gaussian_sigma)

        #use_method = 'mean_threshold' 

        #2 options:
        #1. find channels with peaks above threshold defined by average over all channels+multiplier set by user
        #2. find channels that have highest Pearson correlation with average R wave shape (if the ECG channel is present)

        if use_method == 'mean_threshold':
            artif_per_ch, artif_time_vector = flip_channels(artif_per_ch, tmin, tmax, sfreq, ecg_params_internal)
            affected_channels[m_or_g], affected_derivs, bad_avg_str[m_or_g], avg_overall_obj = find_affected_over_mean(artif_per_ch, 'ECG', ecg_params_internal, thresh_lvl_peakfinder, plotflag=True, verbose_plots=verbose_plots, m_or_g=m_or_g, chs_by_lobe=chs_by_lobe[m_or_g], norm_lvl=norm_lvl, flip_data=True, gaussian_sigma=gaussian_sigma, artif_time_vector=artif_time_vector)
            correlation_derivs = []

        elif use_method == 'correlation' or use_method == 'correlation_reconstructed':

            affected_channels[m_or_g] = find_affected_by_correlation(mean_rwave, artif_per_ch)
            affected_derivs = plot_artif_per_ch_correlated_lobes(affected_channels[m_or_g], tmin, tmax, m_or_g, 'ECG', chs_by_lobe[m_or_g], flip_data=False, verbose_plots=verbose_plots)
            correlation_derivs = plot_correlation(affected_channels[m_or_g], 'ECG', m_or_g, verbose_plots=verbose_plots)
            bad_avg_str[m_or_g] = ''
            avg_overall_obj = None

        else:
            raise ValueError('use_method should be either mean_threshold or correlation')
        

        ecg_derivs += affected_derivs+correlation_derivs
        #higher thresh_lvl_peakfinder - more peaks will be found on the eog artifact for both separate channels and average overall. As a result, average overll may change completely, since it is centered around the peaks of 5 most prominent channels.
        avg_objects_ecg.append(avg_overall_obj)


    simple_metric_ECG = make_simple_metric_ECG_EOG(affected_channels, m_or_g_chosen, 'ECG', bad_avg_str, use_method)

    return ecg_derivs, simple_metric_ECG, ecg_str, avg_objects_ecg


#%%
def EOG_meg_qc(eog_params: dict, eog_params_internal: dict, raw: mne.io.Raw, channels: dict, chs_by_lobe_orig: dict, m_or_g_chosen: list, verbose_plots: bool):
    
    """
    Main EOG function. Calculates average EOG artifact and finds affected channels.
    
    Parameters
    ----------
    eog_params : dict
        Dictionary with EOG parameters originating from the config file.
    eog_params_internal : dict
        Dictionary with EOG parameters originating from the config file - preset for internal use, not to be changed by the user.
    raw : mne.io.Raw
        Raw MEG data.
    channels : dict
        Dictionary with listds of channels for each channel type (mag and grad).
    chs_by_lobe : dict
        Dictionary with lists of channels separated by lobe.
    m_or_g_chosen : list
        List of channel types chosen for the analysis.
    verbose_plots : bool
        True for showing plot in notebook.
        
    Returns
    -------
    eog_derivs : list
        List of all derivatives (plotly figures) as QC_derivative instances
    simple_metric_EOG : dict
        Dictionary with simple metrics for ECG artifacts to be exported into json file.
    eog_str_for_report : str
        String with information about EOG channel used in the final report.
    
    """

    chs_by_lobe = deepcopy(chs_by_lobe_orig) 
    #in case we will change this variable in any way. If not copied it might introduce errors in parallel processing. 
    # This variable is used in all modules

    if verbose_plots is False:
        import matplotlib
        matplotlib.use('Agg') #this command will suppress showing matplotlib figures produced by mne. They will still be saved for use in report but not shown when running the pipeline

    sfreq=raw.info['sfreq']
    tmin=eog_params_internal['eog_epoch_tmin']
    tmax=eog_params_internal['eog_epoch_tmax']

    norm_lvl=eog_params['norm_lvl']
    gaussian_sigma=eog_params['gaussian_sigma']
    thresh_lvl_peakfinder=eog_params['thresh_lvl_peakfinder']

    eog_str, eog_data, event_indexes, eog_ch_name = get_EOG_data(raw)

    eog_derivs = []
    if len(eog_data) == 0:
        simple_metric_EOG = {'description': eog_str}
        return eog_derivs, simple_metric_EOG, eog_str, []
    
        
    for data, name in zip(eog_data, eog_ch_name):
        fig = plot_ECG_EOG_channel(data, [], name, fs= raw.info['sfreq'], verbose_plots = verbose_plots)
        eog_derivs += [QC_derivative(fig, name+' data', 'plotly')]

    # Now choose the channel with blinks only (if there are several):
    #(NEED TO FIGURE OUT HOW)
    eog_data = eog_data[0]
    eog_ch_name = eog_ch_name[0]
    event_indexes = event_indexes[0]
    print('___MEG_QC___: Blinks will be detected based on channel: ', eog_ch_name)

    
    use_method = 'correlation' #'mean_threshold' 
    #no need to choose method in EOG because we cant reconstruct channel, always correlaion (if channel present) or fail.

    mean_good, eog_str_checked, mean_blink, blink_derivs = check_mean_wave(raw, use_method, eog_data, 'EOG', event_indexes, tmin, tmax, sfreq, eog_params_internal, thresh_lvl_peakfinder, verbose_plots)
    eog_str += eog_str_checked

    eog_derivs += blink_derivs

    if mean_good is False:
        simple_metric_EOG = {'description': eog_str}
        return eog_derivs, simple_metric_EOG, eog_str, []


    affected_channels={}
    bad_avg_str = {}
    avg_objects_eog=[]
    
    for m_or_g  in m_or_g_chosen:

        eog_epochs = mne.preprocessing.create_eog_epochs(raw, picks=channels[m_or_g], tmin=tmin, tmax=tmax)

        eog_derivs += plot_ecg_eog_mne(eog_epochs, m_or_g, tmin, tmax)

        artif_per_ch = calculate_artifacts_on_channels(eog_epochs, channels[m_or_g], chs_by_lobe=chs_by_lobe[m_or_g], thresh_lvl_peakfinder=thresh_lvl_peakfinder, tmin=tmin, tmax=tmax, params_internal=eog_params_internal, gaussian_sigma=gaussian_sigma)

        #2 options:
        #1. find channels with peaks above threshold defined by average over all channels+multiplier set by user
        #2. find channels that have highest Pearson correlation with average R wave shape (if the ECG channel is present)

        if use_method == 'mean_threshold':
            artif_per_ch, artif_time_vector = flip_channels(artif_per_ch, tmin, tmax, sfreq, eog_params_internal)
            affected_channels[m_or_g], affected_derivs, bad_avg_str[m_or_g], avg_overall_obj = find_affected_over_mean(artif_per_ch, 'EOG', eog_params_internal, thresh_lvl_peakfinder, plotflag=True, verbose_plots=verbose_plots, m_or_g=m_or_g, chs_by_lobe=chs_by_lobe[m_or_g], norm_lvl=norm_lvl, flip_data=True, gaussian_sigma=gaussian_sigma, artif_time_vector=artif_time_vector)
            correlation_derivs = []

        elif use_method == 'correlation' or use_method == 'correlation_reconstructed':
            
            affected_channels[m_or_g] = find_affected_by_correlation(mean_blink, artif_per_ch)
            affected_derivs = plot_artif_per_ch_correlated_lobes(affected_channels[m_or_g], tmin, tmax, m_or_g, 'EOG', chs_by_lobe[m_or_g], flip_data=False, verbose_plots=verbose_plots)
            correlation_derivs = plot_correlation(affected_channels[m_or_g], 'EOG', m_or_g, verbose_plots=verbose_plots)
            bad_avg_str[m_or_g] = ''
            avg_overall_obj = None

        else:
            raise ValueError('use_method should be either mean_threshold or correlation')
        

        eog_derivs += affected_derivs+correlation_derivs
        #higher thresh_lvl_peakfinder - more peaks will be found on the eog artifact for both separate channels and average overall. As a result, average overll may change completely, since it is centered around the peaks of 5 most prominent channels.
        avg_objects_eog.append(avg_overall_obj)


    simple_metric_EOG = make_simple_metric_ECG_EOG(affected_channels, m_or_g_chosen, 'EOG', bad_avg_str, use_method)

    return eog_derivs, simple_metric_EOG, eog_str, avg_objects_eog

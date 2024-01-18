
# # Annotate muscle artifacts
# 
# Explanation from MNE:
# Muscle contractions produce high frequency activity that can mask brain signal
# of interest. Muscle artifacts can be produced when clenching the jaw,
# swallowing, or twitching a cranial muscle. Muscle artifacts are most
# noticeable in the range of 110-140 Hz.
# 
# This code uses :func:`~mne.preprocessing.annotate_muscle_zscore` to annotate
# segments where muscle activity is likely present. This is done by band-pass
# filtering the data in the 110-140 Hz range. Then, the envelope is taken using
# the hilbert analytical signal to only consider the absolute amplitude and not
# the phase of the high frequency signal. The envelope is z-scored and summed
# across channels and divided by the square root of the number of channels.
# Because muscle artifacts last several hundred milliseconds, a low-pass filter
# is applied on the averaged z-scores at 4 Hz, to remove transient peaks.
# Segments above a set threshold are annotated as ``BAD_muscle``. In addition,
# the ``min_length_good`` parameter determines the cutoff for whether short
# spans of "good data" in between muscle artifacts are included in the
# surrounding "BAD" annotation.
# 


import mne
mne.viz.set_browser_backend('matplotlib')
import plotly.graph_objects as go
from scipy.signal import find_peaks
import numpy as np
from mne.preprocessing import annotate_muscle_zscore
from meg_qc.source.universal_plots import QC_derivative, get_tit_and_unit

def find_powerline_noise_short(raw, psd_params, m_or_g_chosen):

    method = 'welch'
    prominence_lvl_pos = 50 #this is a good level for average psd over all channels. Same should be used for PSD module.
    #WE CAN PUT THESE 2 ABOVE INTO CONFIG AS WELL.  BUT THEY ACTUALLY SHOULD NOT BE CHANGED BY USER. OR MAKE A SEPARATE CONFIG ONLY ACCED BY DEVELOPERS FOR SETTING DEFAULTS?


    psd_step_size = psd_params['psd_step_size']
    sfreq=raw.info['sfreq']
    nfft=int(sfreq/psd_step_size)
    nperseg=int(sfreq/psd_step_size)


    noisy_freqs = {}
    for m_or_g in m_or_g_chosen:

        psds, freqs = raw.compute_psd(method=method, fmin=psd_params['freq_min'], fmax=psd_params['freq_max'], picks=m_or_g, n_jobs=-1, n_fft=nfft, n_per_seg=nperseg).get_data(return_freqs=True)
        avg_psd=np.mean(psds,axis=0) # average psd over all channels
        prominence_pos=(max(avg_psd) - min(avg_psd)) / prominence_lvl_pos

        noisy_freqs_indexes, _ = find_peaks(avg_psd, prominence=prominence_pos)
        noisy_freqs [m_or_g] = freqs[noisy_freqs_indexes]

    return noisy_freqs


def make_simple_metric_muscle(m_or_g_decided: str, z_scores_dict: dict, muscle_str_joined: str):

    """
    Make a simple metric dict for muscle events.
    
    Parameters
    ----------
    m_or_g_decided : str
        The channel type used for muscle detection: 'mag' or 'grad'.
    z_scores_dict : dict
        The z-score thresholds used for muscle detection.
    muscle_str_joined : str
        Notes about muscle detection to use as description.
        
    Returns
    -------
    simple_metric : dict
        A simple metric dict for muscle events.
        
    """

    #if the string contains <p> or </p> - remove it:
    muscle_str_joined = muscle_str_joined.replace("<p>", "").replace("</p>", "")


    simple_metric = {
    'description': muscle_str_joined+'Data below shows detected high frequency (muscle) events.',
    'muscle_calculated_using': m_or_g_decided,
    'unit_muscle_evet_times': 'seconds',
    'unit_muscle_event_zscore': 'z-score',
    'zscore_thresholds': z_scores_dict}

    return simple_metric


def plot_muscle(m_or_g: str, raw: mne.io.Raw, scores_muscle: np.ndarray, threshold_muscle: float, muscle_times: np.ndarray, high_scores_muscle: np.ndarray, verbose_plots: bool, annot_muscle: mne.Annotations = None, interactive_matplot:bool = False):

    """
    Plot the muscle events with the z-scores and the threshold.
    
    Parameters
    ----------
    m_or_g : str
        The channel type used for muscle detection: 'mag' or 'grad'.
    raw : mne.io.Raw
        The raw data.
    scores_muscle : np.ndarray
        The z-scores of the muscle events.
    threshold_muscle : float
        The z-score threshold used for muscle detection.
    muscle_times : np.ndarray
        The times of the muscle events.
    high_scores_muscle : np.ndarray
        The z-scores of the muscle events over the threshold.
    verbose_plots : bool
        True for showing plot in notebook.
    annot_muscle : mne.Annotations
        The annotations of the muscle events. Used only for interactive_matplot.
    interactive_matplot : bool
        Whether to use interactive matplotlib plots or not. Default is False because it cant be extracted into the report.

    Returns
    -------
    fig_derivs : list
        A list of QC_derivative objects with plotly figures for muscle events.

    """
    fig_derivs = []

    fig=go.Figure()
    tit, _ = get_tit_and_unit(m_or_g)
    fig.add_trace(go.Scatter(x=raw.times, y=scores_muscle, mode='lines', name='high freq (muscle scores)'))
    fig.add_trace(go.Scatter(x=muscle_times, y=high_scores_muscle, mode='markers', name='high freq (muscle) events'))
    #removed threshold, so this one is not plotted now:
    #fig.add_trace(go.Scatter(x=raw.times, y=[threshold_muscle]*len(raw.times), mode='lines', name='z score threshold: '+str(threshold_muscle)))
    fig.update_layout(xaxis_title='time, (s)', yaxis_title='zscore', title={
    'text': "Muscle z scores (high fequency artifacts) over time based on "+tit,
    'y':0.85,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'})

    if verbose_plots is True:
        fig.show()

    fig_derivs += [QC_derivative(fig, 'muscle_z_scores_over_time_based_on_'+tit, 'plotly')]

    # ## View the annotations (interactive_matplot)
    if interactive_matplot is True:
        order = np.arange(144, 164)
        raw.set_annotations(annot_muscle)
        fig2=raw.plot(start=5, duration=20, order=order)
        #Change settings to show all channels!

        # No suppressing of plots should be done here. This one is matplotlib interactive plot, so it ll only work with %matplotlib qt.
        # Makes no sense to suppress it. Also, adding to QC_derivative is just formal, cos whe extracting to html it s not interactive any more. 
        # Should not be added to report. Kept here in case mne will allow to extract interactive later.

        fig_derivs += [QC_derivative(fig2, 'muscle_annotations_'+tit, 'matplotlib')]

    return fig_derivs


def filter_noise_before_muscle_detection(raw: mne.io.Raw, noisy_freqs_global: dict, muscle_freqs: list = [110, 140]):

    """
    Filter out power line noise and other noisy freqs in range of muscle artifacts before muscle artifact detection.
    MNE advices to filter power line noise. We also filter here noisy frequencies in range of muscle artifacts.
    List of noisy frequencies for filtering come from PSD artifact detection function. If any noise peaks were found there for mags or grads 
    they will all be passed here and checked if they are in range of muscle artifacts.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data.
    noisy_freqs_global : dict
        The noisy frequencies found in PSD artifact detection function.
    muscle_freqs : list
        The frequencies of muscle artifacts, usually 110 and 140 Hz.
        
    Returns
    -------
    raw : mne.io.Raw
        The raw data with filtered noise or not filtered if no noise was found."""

    #print(noisy_freqs_global, 'noisy_freqs_global')

    #Find out if the data contains powerline noise freqs or other noisy in range of muscle artifacts - notch filter them before muscle artifact detection:

    # - collect all values in moisy_freqs_global into one list:
    noisy_freqs=[]
    for key in noisy_freqs_global.keys():
        noisy_freqs.extend(np.round(noisy_freqs_global[key], 1))
    
    #print(noisy_freqs, 'noisy_freqs')
    
    # - detect power line freqs and their harmonics
    powerline=[50, 60]

    #Were the power line freqs found in this data?
    powerline_found = [x for x in powerline if x in noisy_freqs]

    # add harmonics of powerline freqs to the list of noisy freqs IF they are in range of muscle artifacts [110-140Hz]:
    for freq in powerline_found:
        for i in range(1, 3):
            if freq*i not in powerline_found and muscle_freqs[0]<freq*i<muscle_freqs[1]:
                powerline_found.append(freq*i)

    # DELETE THESE?
    # - detect other noisy freqs in range of muscle artifacts: DECIDED NOT TO DO IT: MIGHT JUST FILTER OUT MUSCLES THIS WAY.
    # extra_noise_freqs = [x for x in noisy_freqs if muscle_freqs[0]<x<muscle_freqs[1]]
    # noisy_freqs_all = list(set(powerline_freqs+extra_noise_freqs)) #leave only unique values

    noisy_freqs_all = powerline_found

    #(issue almost never happens, but might):
    # find Nyquist frequncy for this data to check if the noisy freqs are not higher than it (otherwise filter will fail):
    noisy_freqs_all = [x for x in noisy_freqs_all if x<raw.info['sfreq']/2 - 1]


    # - notch filter the data (it has to be preloaded before. done in the parent function):
    if noisy_freqs_all==[]:
        print('___MEG QC___: ', 'No powerline noise found in data or PSD artifacts detection was not performed. Notch filtering skipped.')
    elif (len(noisy_freqs_all))>0:
        print('___MEG QC___: ', 'Powerline noise was found in data. Notch filtering at: ', noisy_freqs_all, ' Hz')
        raw.notch_filter(noisy_freqs_all)
    else:
        print('Something went wrong with powerline frequencies. Notch filtering skipped. Check parameter noisy_freqs_all')

    return raw


def attach_dummy_data(raw: mne.io.Raw, attach_seconds: int = 5):

    """
    Attach a dummy start and end to the data to avoid filtering artifacts at the beginning/end of the recording.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data.
    attach_seconds : int
        The number of seconds to attach to the start and end of the recording.
        
    Returns
    -------
    raw : mne.io.Raw
        The raw data with dummy start attached."""
    
    print('Duration original: ', raw.n_times / raw.info['sfreq'])
    # Attach a dummy start to the data to avoid filtering artifacts at the beginning of the recording:
    raw_dummy_start=raw.copy()
    raw_dummy_start_data = raw_dummy_start.crop(tmin=0, tmax=attach_seconds-1/raw.info['sfreq']).get_data()
    print('START', raw_dummy_start_data.shape)
    inverted_data_start = np.flip(raw_dummy_start_data, axis=1) # Invert the data

    # Attach a dummy end to the data to avoid filtering artifacts at the end of the recording:
    raw_dummy_end=raw.copy()
    raw_dummy_end_data = raw_dummy_end.crop(tmin=raw_dummy_end.times[int(-attach_seconds*raw.info['sfreq']-1/raw.info['sfreq'])], tmax=raw_dummy_end.times[-1]).get_data()
    print('END', raw_dummy_end_data.shape)
    inverted_data_end = np.flip(raw_dummy_end_data, axis=1) # Invert the data

    # Update the raw object with the inverted data
    raw_dummy_start._data = inverted_data_start
    raw_dummy_end._data = inverted_data_end
    print('Duration of start attached: ', raw_dummy_start.n_times / raw.info['sfreq'])
    print('Duration of end attached: ', raw_dummy_end.n_times / raw.info['sfreq'])

    # Concatenate the inverted data with the original data
    raw = mne.concatenate_raws([raw_dummy_start, raw, raw_dummy_end])
    print('Duration after attaching dummy data: ', raw.n_times / raw.info['sfreq'])

    return raw

def MUSCLE_meg_qc(muscle_params: dict, psd_params: dict, raw_orig: mne.io.Raw, noisy_freqs_global: dict, m_or_g_chosen:list, verbose_plots: bool, interactive_matplot:bool = False, attach_dummy:bool = True, cut_dummy:bool = True):

    """
    Detect muscle artifacts in MEG data. 
    Gives the number of muscle artifacts based on the set z score threshold: artifact time + artifact z score.
    Threshold  is set by the user in the config file. Several thresholds can be used on the loop.

    Notes
    -----
    The data has to first be notch filtered at powerline frequencies as suggested by mne.


    Parameters
    ----------

    muscle_params : dict
        The parameters for muscle artifact detection originally defined in the config file.
    psd_params : dict
        The parameters for PSD calculation originally defined in the config file. This in only needed to calculate powerline noise in case PSD was not calculated before.
    raw_orig : mne.io.Raw
        The raw data.
    noisy_freqs_global : list
        The powerline frequencies found in the data by previously running PSD_meg_qc.
    m_or_g_chosen : list
        The channel types chosen for the analysis: 'mag' or 'grad'.
    verbose_plots : bool
        True for showing plot in notebook.
    interactive_matplot : bool
        Whether to use interactive matplotlib plots or not. Default is False because it cant be extracted into the report. 
        But might just be useful for beter undertanding while maintaining this function.
    attach_dummy : bool
        Whether to attach dummy data to the start and end of the recording to avoid filtering artifacts. Default is True.
    cut_dummy : bool
        Whether to cut the dummy data after filtering. Default is True.

    Returns
    -------
    muscle_derivs : list
        A list of QC_derivative objects for muscle events containing figures.
    simple_metric : dict
        A simple metric dict for muscle events.
    muscle_str : str
        String with notes about muscle artifacts for report

    """

    if noisy_freqs_global is None: # if PSD was not calculated before, calculate noise frequencies now:
        noisy_freqs_global = find_powerline_noise_short(raw_orig, psd_params, m_or_g_chosen)
        print('Noisy frequencies found in data at (HZ): ', noisy_freqs_global)
    else: # if PSD was calculated before, use the frequencies from the PSD step:
        pass


    muscle_freqs = muscle_params['muscle_freqs']
   
    raw = raw_orig.copy() # make a copy of the raw data, to make sure the original data is not changed while filtering for this metric.

    if 'mag' in m_or_g_chosen:
        m_or_g_decided=['mag']
        muscle_str = 'For this data file artifact detection was performed on magnetometers, they are more sensitive to muscle activity than gradiometers. '
        print('___MEG QC___: ', muscle_str)
    elif 'grad' in m_or_g_chosen and 'mag' not in m_or_g_chosen:
        m_or_g_decided=['grad']
        muscle_str = 'For this data file artifact detection was performed on gradiometers, they are less sensitive to muscle activity than magnetometers. '
        print('___MEG QC___: ', muscle_str)
    else:
        print('___MEG QC___: ', 'No magnetometers or gradiometers found in data. Artifact detection skipped.')
        return [], []
    
    muscle_note = "This metric shows high frequency artifacts in range between 110-140 Hz. High power in this frequency band compared to the rest of the signal is strongly correlated with muscles artifacts, as suggested by MNE. However, high frequency oscillations may also occure in this range for reasons other than muscle activity (for example, in an empty room recording). "
    muscle_str_joined=muscle_note+"<p>"+muscle_str+"</p>"

    muscle_derivs=[]

    raw.load_data() #need to preload data for filtering both in notch filter and in annotate_muscle_zscore

    attach_sec = 3 # seconds

    if attach_dummy is True:
        print(raw, attach_sec)
        raw = attach_dummy_data(raw, attach_sec) #attach dummy data to avoid filtering artifacts at the beginning and end of the recording.  

    # Filter out power line noise and other noisy freqs in range of muscle artifacts before muscle artifact detection.
    raw = filter_noise_before_muscle_detection(raw, noisy_freqs_global, muscle_freqs)

    # Loop through different thresholds for muscle artifact detection:
    threshold_muscle_list = muscle_params['threshold_muscle']  # z-score
    min_distance_between_different_muscle_events = muscle_params['min_distance_between_different_muscle_events']  # seconds
    
    #muscle_derivs, simple_metric, scores_muscle = calculate_muscle_over_threshold(raw, m_or_g_decided, muscle_params, threshold_muscle_list, muscle_freqs, cut_dummy, attach_sec, min_distance_between_different_muscle_events, verbose_plots, interactive_matplot, muscle_str_joined)
    muscle_derivs, simple_metric, scores_muscle = calculate_muscle_NO_threshold(raw, m_or_g_decided, muscle_params, threshold_muscle_list[0], muscle_freqs, cut_dummy, attach_sec, min_distance_between_different_muscle_events, verbose_plots, interactive_matplot, muscle_str_joined)

    return muscle_derivs, simple_metric, muscle_str_joined, scores_muscle, raw



def calculate_muscle_over_threshold(raw, m_or_g_decided, muscle_params, threshold_muscle_list, muscle_freqs, cut_dummy, attach_sec, min_distance_between_different_muscle_events, verbose_plots, interactive_matplot, muscle_str_joined):

    muscle_derivs=[]

    for m_or_g in m_or_g_decided: #generally no need for loop, we will use just 1 type here. Left in case we change the principle.

        z_scores_dict={}
        for threshold_muscle in threshold_muscle_list:

            z_score_details={}

            annot_muscle, scores_muscle = annotate_muscle_zscore(
            raw, ch_type=m_or_g, threshold=threshold_muscle, min_length_good=muscle_params['min_length_good'],
            filter_freq=muscle_freqs)

            #cut attached beginning and end from annot_muscle, scores_muscle:
            if cut_dummy is True:
                # annot_muscle = annot_muscle[annot_muscle['onset']>attach_sec]
                # annot_muscle['onset'] = annot_muscle['onset']-attach_sec
                # annot_muscle['duration'] = annot_muscle['duration']-attach_sec
                scores_muscle = scores_muscle[int(attach_sec*raw.info['sfreq']): int(-attach_sec*raw.info['sfreq'])]
                raw = raw.crop(tmin=attach_sec, tmax=raw.times[int(-attach_sec*raw.info['sfreq'])])



            # Plot muscle z-scores across recording
            peak_locs_pos, _ = find_peaks(scores_muscle, height=threshold_muscle, distance=raw.info['sfreq']*min_distance_between_different_muscle_events)

            muscle_times = raw.times[peak_locs_pos]
            high_scores_muscle=scores_muscle[peak_locs_pos]

            muscle_derivs += plot_muscle(m_or_g, raw, scores_muscle, threshold_muscle, muscle_times, high_scores_muscle, verbose_plots, interactive_matplot, annot_muscle)

            # collect all details for simple metric:
            z_score_details['muscle_event_times'] = muscle_times.tolist()
            z_score_details['muscle_event_zscore'] = high_scores_muscle.tolist()
            z_scores_dict[threshold_muscle] = {
                'number_muscle_events': len(muscle_times), 
                'Details': z_score_details}
            
        simple_metric = make_simple_metric_muscle(m_or_g_decided[0], z_scores_dict, muscle_str_joined)

    return muscle_derivs, simple_metric, scores_muscle


def calculate_muscle_NO_threshold(raw, m_or_g_decided, muscle_params, threshold_muscle, muscle_freqs, cut_dummy, attach_sec, min_distance_between_different_muscle_events, verbose_plots, interactive_matplot, muscle_str_joined):

    """
    annotate_muscle_zscore() requires threshold_muscle so define a minimal one here: 5 z-score.
    
    """


    muscle_derivs=[]


    for m_or_g in m_or_g_decided: #generally no need for loop, we will use just 1 type here. Left in case we change the principle.

        z_scores_dict={}

        z_score_details={}

        annot_muscle, scores_muscle = annotate_muscle_zscore(
        raw, ch_type=m_or_g, threshold=threshold_muscle, min_length_good=muscle_params['min_length_good'],
        filter_freq=muscle_freqs)

        #cut attached beginning and end from annot_muscle, scores_muscle:
        if cut_dummy is True:
            # annot_muscle = annot_muscle[annot_muscle['onset']>attach_sec]
            # annot_muscle['onset'] = annot_muscle['onset']-attach_sec
            # annot_muscle['duration'] = annot_muscle['duration']-attach_sec
            scores_muscle = scores_muscle[int(attach_sec*raw.info['sfreq']): int(-attach_sec*raw.info['sfreq'])]
            raw = raw.crop(tmin=attach_sec, tmax=raw.times[int(-attach_sec*raw.info['sfreq'])])


        # Plot muscle z-scores across recording
        peak_locs_pos, _ = find_peaks(scores_muscle, height=threshold_muscle, distance=raw.info['sfreq']*min_distance_between_different_muscle_events)

        muscle_times = raw.times[peak_locs_pos]
        high_scores_muscle=scores_muscle[peak_locs_pos]

        muscle_derivs += plot_muscle(m_or_g, raw, scores_muscle, None, muscle_times, high_scores_muscle, verbose_plots, interactive_matplot, annot_muscle)

        # collect all details for simple metric:
        z_score_details['muscle_event_times'] = muscle_times.tolist()
        z_score_details['muscle_event_zscore'] = high_scores_muscle.tolist()
        z_scores_dict = {
            'number_muscle_events': len(muscle_times), 
            'Details': z_score_details}
            
        simple_metric = make_simple_metric_muscle(m_or_g_decided[0], z_scores_dict, muscle_str_joined)

    return muscle_derivs, simple_metric, scores_muscle
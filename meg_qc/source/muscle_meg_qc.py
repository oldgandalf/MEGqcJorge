
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
from universal_plots import QC_derivative, get_tit_and_unit

def make_simple_metric_muscle(m_or_g_decided: str, z_scores_dict: dict):

    """
    Make a simple metric dict for muscle events.
    
    Parameters
    ----------
    m_or_g_decided : str
        The channel type used for muscle detection: 'mag' or 'grad'.
    z_scores_dict : dict
        The z-score thresholds used for muscle detection.
        
    Returns
    -------
    simple_metric : dict
        A simple metric dict for muscle events.
        
    """

    simple_metric = {
    'description': 'Muscle artifact events at different z score thresholds.',
    'muscle_calculated_using': m_or_g_decided,
    'unit_muscle_evet_times': 'seconds',
    'unit_muscle_event_zscore': 'z-score',
    'zscore_thresholds': z_scores_dict}

    return simple_metric


def plot_muscle(m_or_g: str, raw: mne.io.Raw, scores_muscle: np.ndarray, threshold_muscle: float, muscle_times: np.ndarray, high_scores_muscle: np.ndarray, annot_muscle: mne.Annotations = None, interactive_matplot:bool = False):

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
    fig.add_trace(go.Scatter(x=raw.times, y=scores_muscle, mode='lines', name='muscle scores'))
    fig.add_trace(go.Scatter(x=muscle_times, y=high_scores_muscle, mode='markers', name='muscle events'))
    fig.add_trace(go.Scatter(x=raw.times, y=[threshold_muscle]*len(raw.times), mode='lines', name='z score threshold: '+str(threshold_muscle)))
    fig.update_layout(xaxis_title='time, (s)', yaxis_title='zscore', title={
    'text': "Muscle z scores over time based on "+tit+". Threshold zscore "+str(threshold_muscle),
    'y':0.85,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'})
    fig.show()

    fig_derivs += [QC_derivative(fig, 'muscle_z_scores_over_time_based_on_'+tit+'_threshold_zscore_'+str(threshold_muscle), 'plotly')]

    # ## View the annotations (interactive_matplot)
    if interactive_matplot is True:
        order = np.arange(144, 164)
        raw.set_annotations(annot_muscle)
        fig2=raw.plot(start=5, duration=20, order=order)
        #Change settings to show all channels!

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
    powerline_freqs = [x for x in powerline if x in np.round(noisy_freqs)]
    powerline_freqs+=[x*2 for x in powerline_freqs]+[x*3 for x in powerline_freqs]

    # - detect other noisy freqs in range of muscle artifacts:
    extra_noise_freqs = [x for x in noisy_freqs if muscle_freqs[0]<x<muscle_freqs[1]]

    noisy_freqs_all = list(set(powerline_freqs+extra_noise_freqs)) #leave only unique values

    #find Nyquist frequncy for this data to check if the noisy freqs are not higher than it (otherwise filter will fail):
    nyquist_freq = raw.info['sfreq']/2

    #remove noisy freqs that are higher than Nyquist freq-1:
    noisy_freqs_all = [x for x in noisy_freqs_all if x<nyquist_freq-1]


    # - notch filter the data (it has to be preloaded before. done in the parent function):
    if noisy_freqs_all==[]:
        print('___MEG QC___: ', 'No powerline noise found in data or PSD artifacts detection was not performed. Notch filtering skipped.')
    elif (len(noisy_freqs_all))>0:
        print('___MEG QC___: ', 'Powerline and|or other noise in range of muscle frequencies was found in data. Notch filtering at: ', noisy_freqs_all, ' Hz')
        raw.notch_filter(noisy_freqs_all)
    else:
        print('Something went wrong with powerline frequencies. Notch filtering skipped. Check parameter noisy_freqs_all')

    return raw


def MUSCLE_meg_qc(muscle_params: dict, raw: mne.io.Raw, noisy_freqs_global: dict, m_or_g_chosen:list, interactive_matplot:bool = False):

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
    raw : mne.io.Raw
        The raw data.
    noisy_freqs_global : list
        The powerline frequencies found in the data by previously running PSD_meg_qc.
    m_or_g_chosen : list
        The channel types chosen for the analysis: 'mag' or 'grad'.
    interactive_matplot : bool
        Whether to use interactive matplotlib plots or not. Default is False because it cant be extracted into the report. 
        But might just be useful for beter undertanding while maintaining this function.

    Returns
    -------
    muscle_derivs : list
        A list of QC_derivative objects for muscle events containing figures.
    simple_metric : dict
        A simple metric dict for muscle events.
    muscle_str : str
        String with notes about muscle artifacts for report

    """

    muscle_freqs = muscle_params['muscle_freqs']
   

    if 'mag' in m_or_g_chosen:
        m_or_g_decided=['mag']
        muscle_str = 'Muscle artifact detection performed on magnetometers, they are more sensitive to muscle activity than gradiometers.'
        print('___MEG QC___: ', muscle_str)
    elif 'grad' in m_or_g_chosen and 'mag' not in m_or_g_chosen:
        m_or_g_decided=['grad']
        muscle_str = 'Magnetometers are more sensitive to muscle artifacts and are recommended for artifact detection. If you only use gradiometers, some muscle events might not show. This will not be a problem if the data set only contains gradiometers. If it contains both gradiometers and magnetometers, but only gradiometers were chosen for this analysis - the results will not include an extra part of the muscle events present in magnetometers data.'
        print('___MEG QC___: ', muscle_str)
    else:
        print('___MEG QC___: ', 'No magnetometers or gradiometers found in data. Muscle artifact detection skipped.')
        return [], []

    muscle_derivs=[]

    raw.load_data() #need to preload data for filtering both in notch filter and in annotate_muscle_zscore

    # Filter out power line noise and other noisy freqs in range of muscle artifacts before muscle artifact detection.
    raw = filter_noise_before_muscle_detection(raw, noisy_freqs_global, muscle_freqs)



    # raw.notch_filter([120])
    # print('___MEG QC___: ', 'Notch filtering at 120 Hz')




    # Loop through different thresholds for muscle artifact detection:
    threshold_muscle_list = muscle_params['threshold_muscle']  # z-score
    min_distance_between_different_muscle_events = muscle_params['min_distance_between_different_muscle_events']  # seconds
    
    for m_or_g in m_or_g_decided: #generally no need for loop, we will use just 1 type here. Left in case we change the principle.

        z_scores_dict={}
        for threshold_muscle in threshold_muscle_list:

            z_score_details={}

            annot_muscle, scores_muscle = annotate_muscle_zscore(
            raw, ch_type=m_or_g, threshold=threshold_muscle, min_length_good=muscle_params['min_length_good'],
            filter_freq=muscle_freqs)

            # Plot muscle z-scores across recording
            peak_locs_pos, _ = find_peaks(scores_muscle, height=threshold_muscle, distance=raw.info['sfreq']*min_distance_between_different_muscle_events)

            muscle_times = raw.times[peak_locs_pos]
            high_scores_muscle=scores_muscle[peak_locs_pos]

            muscle_derivs += plot_muscle(m_or_g, raw, scores_muscle, threshold_muscle, muscle_times, high_scores_muscle, interactive_matplot, annot_muscle)

            # collect all details for simple metric:
            z_score_details['muscle_event_times'] = muscle_times.tolist()
            z_score_details['muscle_event_zscore'] = high_scores_muscle.tolist()
            z_scores_dict[threshold_muscle] = {
                'number_muscle_events': len(muscle_times), 
                'Details': z_score_details}
            
        simple_metric = make_simple_metric_muscle(m_or_g_decided[0], z_scores_dict)

    return muscle_derivs, simple_metric, muscle_str





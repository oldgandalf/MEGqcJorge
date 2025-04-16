import numpy as np
import pandas as pd
import mne
from typing import List
from scipy.signal import find_peaks
from meg_qc.plotting.universal_plots import assign_epoched_std_ptp_to_channels
from meg_qc.plotting.universal_html_report import simple_metric_basic
from meg_qc.calculation.metrics.STD_meg_qc import make_dict_global_std_ptp, make_dict_local_std_ptp, get_big_small_std_ptp_all_data, get_noisy_flat_std_ptp_epochs
from meg_qc.calculation.initial_meg_qc import (chs_dict_to_csv,load_data)
import copy

#The manual PtP version. 

def neighbour_peak_amplitude(max_pair_dist_sec: float, sfreq: int,
                             pos_peak_locs: np.ndarray, neg_peak_locs: np.ndarray,
                             pos_magnitudes: np.ndarray, neg_magnitudes: np.ndarray):
    """
    Vectorized version that pairs each positive peak with the nearest negative peak.
    Avoids Python loops and does not build the huge n_pos x n_neg matrix.

    Parameters
    ----------
    max_pair_dist_sec : float
        Maximum allowed distance (in seconds) for pairing +/- peaks
    sfreq : int
        Sampling frequency
    pos_peak_locs : np.ndarray (sorted)
        Indices of detected positive peaks
    neg_peak_locs : np.ndarray (sorted)
        Indices of detected negative peaks
    pos_magnitudes : np.ndarray
        Magnitude at each positive peak
    neg_magnitudes : np.ndarray
        Magnitude at each negative peak

    Returns
    -------
    mean_amp : float
        Mean amplitude of all valid pairs
    amplitudes : np.ndarray
        Amplitudes of all valid pairs
    """

    # If there are no peaks, nothing can be calculated
    if len(pos_peak_locs) == 0 or len(neg_peak_locs) == 0:
        return 0.0, None

    max_pair_dist = (max_pair_dist_sec * sfreq) / 2.0

    # 1) Use searchsorted to position each positive peak in neg_peak_locs
    #    idxs_left is the index where each positive peak would be "inserted."
    #    Thus, idxs_left[i] - 1 points to the negative peak on the left,
    #    and idxs_left[i] points to the one on the right (if it exists).
    idxs_left = np.searchsorted(neg_peak_locs, pos_peak_locs, side='left')

    # 2) Build arrays of indices for the candidate negative peak to the left (left_cand)
    #    and to the right (right_cand). Where the left one does not exist (idx=0), we set -1
    left_cand = idxs_left - 1  # could be -1 if it does not exist
    right_cand = idxs_left    # could be == len(neg_peak_locs) if it does not exist

    # Ensure they do not exceed the valid bounds [0, len(neg_peak_locs)-1]
    valid_left = (left_cand >= 0)
    valid_right = (right_cand < len(neg_peak_locs))

    # 3) Create arrays for the locations and magnitudes of each candidate
    #    If the candidate is invalid, set an 'impossible' value that we'll filter out later.
    left_locs = np.full_like(pos_peak_locs, -999999, dtype=np.int64)
    left_mags = np.zeros_like(pos_magnitudes)
    left_locs[valid_left] = neg_peak_locs[left_cand[valid_left]]
    left_mags[valid_left] = neg_magnitudes[left_cand[valid_left]]

    right_locs = np.full_like(pos_peak_locs, 999999999, dtype=np.int64)
    right_mags = np.zeros_like(pos_magnitudes)
    right_locs[valid_right] = neg_peak_locs[right_cand[valid_right]]
    right_mags[valid_right] = neg_magnitudes[right_cand[valid_right]]

    # 4) For each positive peak, compute the distance to its left and right candidate
    dist_left = np.abs(pos_peak_locs - left_locs)
    dist_right = np.abs(pos_peak_locs - right_locs)

    # Choose whichever is closer
    left_is_better = (dist_left <= dist_right)

    # Build arrays 'best_neg_loc' and 'best_neg_mag' with the best candidate for each positive peak
    best_neg_loc = np.where(left_is_better, left_locs, right_locs)
    best_neg_mag = np.where(left_is_better, left_mags, right_mags)
    best_dist = np.where(left_is_better, dist_left, dist_right)

    # 5) Filter only those pairs within max_pair_dist
    mask_valid = best_dist <= max_pair_dist
    if not np.any(mask_valid):
        # Fallback case: there are no pairs within distance, use max-min of everything
        fallback_amp = pos_magnitudes.max() - neg_magnitudes.min()
        print("___MEGqc___: No valid +/- pairs found; fallback amplitude used.")
        return fallback_amp, np.array([fallback_amp], dtype=float)

    # 6) Calculate amplitude: pos_peak - neg_peak
    amps = pos_magnitudes[mask_valid] - best_neg_mag[mask_valid]

    return amps.mean(), amps



def get_ptp_all_data(data: mne.io.Raw, channels: List, sfreq: int, ptp_thresh_lvl: float, max_pair_dist_sec: float):

    """ 
    Calculate peak-to-peak amplitude for all channels over whole data (not epoched).

    Parameters:
    -----------
    data : mne.io.Raw
        Raw data
    channels : List
        List of channel names to be used for peak-to-peak amplitude calculation
    sfreq : int
        Sampling frequency of data. Attention to which data is used! original or resampled.
    ptp_thresh_lvl : float
        The level definig how the PtP threshold will be scaled. Higher number will result in more peaks detected.
        The threshold is calculated as (max - min) / ptp_thresh_lvl
    max_pair_dist_sec : float
        Maximum distance in seconds which is allowed for negative+positive peaks to be detected as a pair

    Returns:
    --------
    peak_ampl_channels : dict
        Peak-to-peak amplitude values for each channel.

    """
        
    data_channels=data.get_data(picks = channels)

    peak_ampl_channels=[]
    for one_ch_data in data_channels: 

        thresh=(max(one_ch_data) - min(one_ch_data)) / ptp_thresh_lvl 
        #can also change the whole thresh to a single number setting

        #mne.preprocessing.peak_finder() gives error if there are no peaks detected. We use scipy.signal.find_peaks() instead here:
        pos_peak_locs, _ = find_peaks(one_ch_data, prominence=thresh) #assume there are no peaks within 0.5 seconds from each other.
        pos_peak_magnitudes = one_ch_data[pos_peak_locs]

        neg_peak_locs, _ = find_peaks(-one_ch_data, prominence=thresh) #assume there are no peaks within 0.5 seconds from each other.
        neg_peak_magnitudes = one_ch_data[neg_peak_locs]

        pp_ampl, _ = neighbour_peak_amplitude(max_pair_dist_sec, sfreq, pos_peak_locs, neg_peak_locs, pos_peak_magnitudes, neg_peak_magnitudes)
        peak_ampl_channels.append(pp_ampl)

    #add channel name for every std value:
    peak_ampl_channels_named = {}
    for i, ch in enumerate(channels):
        peak_ampl_channels_named[ch] = peak_ampl_channels[i]
        
    return peak_ampl_channels_named


def get_ptp_epochs(channels: List, epochs_mg: mne.Epochs, sfreq: int,
                   ptp_thresh_lvl: float, max_pair_dist_sec: float):
    """
    Computes the peak-to-peak (PtP) amplitude for each epoch and each channel in a vectorized manner.

    Parameters
    ----------
    channels : List
        List of channel names.
    epochs_mg : mne.Epochs
        Epoched data.
    sfreq : int
        Sampling frequency.
    ptp_thresh_lvl : float
        Threshold: (max-min)/ptp_thresh_lvl.
    max_pair_dist_sec : float
        Maximum distance in seconds for pairing peaks.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per channel and one column per epoch, containing the PtP amplitude.
    """
    # Extract the data for the selected channels
    # data shape: (n_epochs, n_channels, n_times)
    data = epochs_mg.get_data(picks=channels)
    n_epochs = data.shape[0]
    n_channels = data.shape[1]

    # Initialize a matrix to store the PtP amplitude for each channel/epoch
    ptp_matrix = np.empty((n_channels, n_epochs))

    # Loop through epochs and channels
    for ep in range(n_epochs):
        for ch in range(n_channels):
            one_ch_data = data[ep, ch, :]
            # Compute a threshold based on the difference between max and min,
            # scaled by ptp_thresh_lvl
            thresh = (one_ch_data.max() - one_ch_data.min()) / ptp_thresh_lvl

            # Find positive and negative peaks above that threshold
            pos_peaks, _ = find_peaks(one_ch_data, prominence=thresh)
            neg_peaks, _ = find_peaks(-one_ch_data, prominence=thresh)
            pos_mags = one_ch_data[pos_peaks]
            neg_mags = one_ch_data[neg_peaks]

            # Use neighbour_peak_amplitude to compute the mean amplitude
            mean_amp, _ = neighbour_peak_amplitude(
                max_pair_dist_sec, sfreq,
                pos_peaks, neg_peaks,
                pos_mags, neg_mags
            )

            # Fill in the results
            ptp_matrix[ch, ep] = mean_amp

    # Return the results in a DataFrame, indexed by channel
    return pd.DataFrame(ptp_matrix, index=channels)




def make_simple_metric_ptp_manual(ptp_manual_params: dict, big_ptp_with_value_all_data: dict, small_ptp_with_value_all_data: dict, channels: dict, deriv_epoch_ptp: dict, metric_local: bool, m_or_g_chosen: List):

    """ 
    Create a simple metric for peak-to-peak amplitude. 
    Global: The metric is calculated for all data (not epoched) and 
    Local: for each epoch.

    Parameters:
    -----------
    ptp_manual_params : dict
        Dictionary containing the parameters for the metric
    big_ptp_with_value_all_data : dict
        Dict (mag, grad) with channels with peak-to-peak amplitude higher than the threshold + the value of the peak-to-peak amplitude
    small_ptp_with_value_all_data : dict
        Dict (mag, grad) with channels with peak-to-peak amplitude lower than the threshold + the value of the peak-to-peak amplitude
    channels : dict
        Dict (mag, grad) with all channel names
    deriv_epoch_ptp : dict
        Dict (mag, grad) with peak-to-peak amplitude for each epoch for each channel
    metric_local : bool
        If True, the local metric was calculated and will be added to the simple metric
    m_or_g_chosen : List
        'mag' or 'grad' or both, chosen by user in config file

    Returns:
    --------
    simple_metric : dict
        Dict (mag, grad) with the simple metric for peak-to-peak manual amplitude

    """

    metric_global_name = 'ptp_manual_all'
    metric_global_description = 'Peak-to-peak deviation of the data over the entire time series (not epoched): ... The ptp_lvl is the peak-to-peak threshold level set by the user. Threshold = ... The channel where data is higher than this threshod is considered as noisy. Same: if the std of some channel is lower than -threshold, this channel is considered as flat. In details only the noisy/flat channels are listed. Channels with normal std are not listed. If needed to see all channels data - use csv files.'
    metric_local_name = 'ptp_manual_epoch'
    if metric_local==True:
        metric_local_description = 'Peak-to-peak deviation of the data over stimulus-based epochs. The epoch is counted as noisy (or flat) if the percentage of noisy (or flat) channels in this epoch is over allow_percent_noisy_flat. this percent is set by user, default=70%. Hense, if no epochs have over 70% of noisy channels - total number of noisy epochs will be 0. Definition of a noisy channel inside of epoch: 1)Take Peak-to-Peak amplitude (PtP) of data of THIS channel in THIS epoch. 2) Take PtP of the data of THIS channel for ALL epochs and get mean of it. 3) If (1) is higher than (2)*noisy_channel_multiplier - this channel is noisy.  If (1) is lower than (2)*flat_multiplier - this channel is flat. '
    else:
        metric_local_description = 'Not calculated. Ne epochs found'

    metric_global_content={'mag': None, 'grad': None}
    metric_local_content={'mag': None, 'grad': None}
    for m_or_g in m_or_g_chosen:
        metric_global_content[m_or_g]=make_dict_global_std_ptp(ptp_manual_params, big_ptp_with_value_all_data[m_or_g], small_ptp_with_value_all_data[m_or_g], channels[m_or_g], 'ptp')
        
        if metric_local is True:
            metric_local_content[m_or_g]=make_dict_local_std_ptp(ptp_manual_params, deriv_epoch_ptp[m_or_g][1].content, deriv_epoch_ptp[m_or_g][2].content, percent_noisy_flat_allowed=ptp_manual_params['allow_percent_noisy_flat_epochs'])
            #deriv_epoch_std[m_or_g][1].content is df with big std(noisy), df_epoch_std[m_or_g][2].content is df with small std(flat)
        else:
            metric_local_content[m_or_g]=None
    
    simple_metric = simple_metric_basic(metric_global_name, metric_global_description, metric_global_content['mag'], metric_global_content['grad'], metric_local_name, metric_local_description, metric_local_content['mag'], metric_local_content['grad'])

    return simple_metric


def PP_manual_meg_qc(ptp_manual_params: dict, channels: dict, chs_by_lobe: dict, dict_epochs_mg: dict, data_path:str, m_or_g_chosen: List):

    """
    Main Peak to peak amplitude function. Calculates:
    
    - Peak to peak amplitudes (PtP) of data for each channel over all time series.
    - Channels with big PtP (noisy) and small PtP (flat) over all time series.
    - PtP of data for each channel  in each epoch.
    - Epochs with big PtP (noisy) and small PtP (flat).

    PtP is calculated as the average amplitude between the positive and negative peaks,
    which are located at a certain distance from each other. The distance is set by the user in config file.

    Parameters:
    -----------
    ptp_manual_params : dict
        Dictionary containing the parameters for the metric
    channels : dict
        Dict (mag, grad) with all channel names
    chs_by_lobe : dict
        dictionary with channels grouped first by ch type and then by lobe: chs_by_lobe['mag']['Left Occipital'] or chs_by_lobe['grad']['Left Occipital']
    dict_epochs_mg : dict
        Dict (mag, grad) with epochs for each channel. Should be the same for both channels. Used only to check if epochs are present.
    data : mne.io.Raw
        Raw data
    m_or_g_chosen : List
        'mag' or 'grad' or both, chosen by user in config file.

    Returns:
    --------
    derivs_ptp : List
        List with QC_deriv objects for peak-to-peak amplitude (figures and csv files)
    simple_metric_ptp_manual : dict
        Simple metric for peak-to-peak amplitude
    pp+manual_str : str
        String with notes about PtP manual for report
    
    """
    # Load data
    data, shielding_str, meg_system = load_data(data_path)

    # data.load_data()

    sfreq = data.info['sfreq']

    big_ptp_with_value_all_data = {}
    small_ptp_with_value_all_data = {}
    derivs_ptp = []
    derivs_list = []
    peak_ampl = {}
    noisy_flat_epochs_derivs = {}

    chs_by_lobe_ptp=copy.deepcopy(chs_by_lobe)
    # copy here, because want to keep original dict unchanged. 
    # In principal it s good to collect all data about channel metrics there BUT if the metrics are run in parallel this might produce conflicts 
    # (without copying  dict can be chanaged both inside+outside this function even when it is not returned.)

    for m_or_g in m_or_g_chosen:

        peak_ampl[m_or_g] = get_ptp_all_data(data, channels[m_or_g], sfreq, ptp_thresh_lvl=ptp_manual_params['ptp_thresh_lvl'], max_pair_dist_sec=ptp_manual_params['max_pair_dist_sec'])
        
        #Add ptp data into channel object inside the chs_by_lobe dictionary:
        for lobe in chs_by_lobe_ptp[m_or_g]:
            for ch in chs_by_lobe_ptp[m_or_g][lobe]:
                ch.ptp_overall = peak_ampl[m_or_g][ch.name]
        
        big_ptp_with_value_all_data[m_or_g], small_ptp_with_value_all_data[m_or_g] = get_big_small_std_ptp_all_data(peak_ampl[m_or_g], channels[m_or_g], ptp_manual_params['std_lvl'])

    if dict_epochs_mg['mag'] is not None or dict_epochs_mg['grad'] is not None: #if epochs are present
        for m_or_g in m_or_g_chosen:
            df_ptp=get_ptp_epochs(channels[m_or_g], dict_epochs_mg[m_or_g], sfreq, ptp_manual_params['ptp_thresh_lvl'], ptp_manual_params['max_pair_dist_sec'])
            
            chs_by_lobe_ptp[m_or_g] = assign_epoched_std_ptp_to_channels(what_data='peaks', chs_by_lobe=chs_by_lobe_ptp[m_or_g], df_std_ptp=df_ptp) #for easier plotting

            noisy_flat_epochs_derivs[m_or_g] = get_noisy_flat_std_ptp_epochs(df_ptp, m_or_g, 'ptp', ptp_manual_params['noisy_channel_multiplier'], ptp_manual_params['flat_multiplier'], ptp_manual_params['allow_percent_noisy_flat_epochs'])
            derivs_list += noisy_flat_epochs_derivs[m_or_g]

            metric_local=True
        pp_manual_str = ''

    else:
        metric_local=False
        pp_manual_str = 'Peak-to-Peak amplitude per epoch can not be calculated because no events are present. Check stimulus channel.'
        print('___MEGqc___: ', pp_manual_str)
        
    
    simple_metric_ptp_manual = make_simple_metric_ptp_manual(ptp_manual_params, big_ptp_with_value_all_data, small_ptp_with_value_all_data, channels, noisy_flat_epochs_derivs, metric_local, m_or_g_chosen)

    #Extract chs_by_lobe into a data frame
    df_deriv = chs_dict_to_csv(chs_by_lobe_ptp,  file_name_prefix = 'PtPsManual')

    derivs_ptp += derivs_list + df_deriv

    #each deriv saved into a separate list and only at the end put together because this way they keep the right order: 
    #first everything about mags, then everything about grads. - in this order they ll be added to repot. 
    # TODO: Can use fig_order parameter of QC_derivative to adjust figure order in the report, if got nothing better to do XD.

    return derivs_ptp, simple_metric_ptp_manual, pp_manual_str
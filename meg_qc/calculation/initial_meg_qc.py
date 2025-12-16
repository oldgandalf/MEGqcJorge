import os
import re
import shutil
import gc
import mne
import configparser
import numpy as np
import pandas as pd
import random
import copy
import warnings
from typing import List
from meg_qc.calculation.objects import QC_derivative, MEG_channel


def get_all_config_params(config_file_path: str):
    """
    Parse all the parameters from config and put into a python dictionary
    divided by sections. Parsing approach can be changed here, which
    will not affect working of other fucntions.


    Parameters
    ----------
    config_file_path: str
        The path to the config file.

    Returns
    -------
    all_qc_params: dict
        A dictionary with all the parameters from the config file.

    """

    all_qc_params = {}

    config = configparser.ConfigParser()
    config.read(config_file_path)

    default_section = config['DEFAULT']

    m_or_g_chosen = default_section['ch_types']
    m_or_g_chosen = [chosen.strip() for chosen in m_or_g_chosen.split(",")]
    if 'mag' not in m_or_g_chosen and 'grad' not in m_or_g_chosen:
        print('___MEGqc___: ', 'No channels to analyze. Check parameter ch_types in config file.')
        return None

    # TODO: save list of mags and grads here and use later everywhere? because for CTF types are messed up.

    run_STD = default_section.getboolean('STD')
    run_PSD = default_section.getboolean('PSD')
    run_PTP_manual = default_section.getboolean('PTP_manual')
    run_PTP_auto_mne = default_section.getboolean('PTP_auto_mne')
    run_ECG = default_section.getboolean('ECG')
    run_EOG = default_section.getboolean('EOG')
    run_Head = default_section.getboolean('Head')
    run_Muscle = default_section.getboolean('Muscle')

    tmin = default_section['data_crop_tmin']
    tmax = default_section['data_crop_tmax']
    try:
        if not tmin:
            tmin = 0
        else:
            tmin = float(tmin)
        if not tmax:
            tmax = None
        else:
            tmax = float(tmax)

        default_params = dict({
            'm_or_g_chosen': m_or_g_chosen,
            'run_STD': run_STD,
            'run_PSD': run_PSD,
            'run_PTP_manual': run_PTP_manual,
            'run_PTP_auto_mne': run_PTP_auto_mne,
            'run_ECG': run_ECG,
            'run_EOG': run_EOG,
            'run_Head': run_Head,
            'run_Muscle': run_Muscle,
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
        if stim_channel == ['']:
            stim_channel = None

        epoching_params = dict({
            'event_dur': epoching_section.getfloat('event_dur'),
            'epoch_tmin': epoching_section.getfloat('epoch_tmin'),
            'epoch_tmax': epoching_section.getfloat('epoch_tmax'),
            'stim_channel': stim_channel,
            'event_repeated': epoching_section['event_repeated']})
        all_qc_params['Epoching'] = epoching_params

        std_section = config['STD']
        all_qc_params['STD'] = dict({
            'std_lvl': std_section.getint('std_lvl'),
            'allow_percent_noisy_flat_epochs': std_section.getfloat('allow_percent_noisy_flat_epochs'),
            'noisy_channel_multiplier': std_section.getfloat('noisy_channel_multiplier'),
            'flat_multiplier': std_section.getfloat('flat_multiplier'), })

        psd_section = config['PSD']
        freq_min = psd_section['freq_min']
        freq_max = psd_section['freq_max']
        if not freq_min:
            freq_min = 0
        else:
            freq_min = float(freq_min)
        if not freq_max:
            freq_max = np.inf
        else:
            freq_max = float(freq_max)

        all_qc_params['PSD'] = dict({
            'freq_min': freq_min,
            'freq_max': freq_max,
            'psd_step_size': psd_section.getfloat('psd_step_size')})

        ptp_manual_section = config['PTP_manual']
        all_qc_params['PTP_manual'] = dict({
            'numba_version': ptp_manual_section.getboolean('numba_version'),
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
            'thresh_lvl_peakfinder': eog_section.getfloat('thresh_lvl_peakfinder'), })

        head_section = config['Head_movement']
        all_qc_params['Head'] = dict({})

        muscle_section = config['Muscle']
        list_thresholds = muscle_section['threshold_muscle']
        # separate values in list_thresholds based on coma, remove spaces and convert them to floats:
        list_thresholds = [float(i) for i in list_thresholds.split(',')]
        muscle_freqs = [float(i) for i in muscle_section['muscle_freqs'].split(',')]

        all_qc_params['Muscle'] = dict({
            'threshold_muscle': list_thresholds,
            'min_distance_between_different_muscle_events': muscle_section.getfloat(
                'min_distance_between_different_muscle_events'),
            'muscle_freqs': muscle_freqs,
            'min_length_good': muscle_section.getfloat('min_length_good')})

        gqi_section = config['GlobalQualityIndex']

        compute_gqi = gqi_section.getboolean('compute_gqi', fallback=True)
        include_corr = gqi_section.getboolean('include_ecg_eog', fallback=True)

        weights = {
            'ch': gqi_section.getfloat('bad_ch_weight'),
            'corr': gqi_section.getfloat('correlation_weight'),
            'mus': gqi_section.getfloat('muscle_weight'),
            'psd': gqi_section.getfloat('psd_noise_weight'),
        }
        total_w = sum(weights.values())
        if total_w == 0:
            total_w = 1
        weights = {k: v / total_w for k, v in weights.items()}
        all_qc_params['GlobalQualityIndex'] = {
            'compute_gqi': compute_gqi,
            'include_ecg_eog': include_corr,
            'ch':   {
                'start': gqi_section.getfloat('bad_ch_start'),
                'end': gqi_section.getfloat('bad_ch_end'),
                'weight': weights['ch']
            },
            'corr': {
                'start': gqi_section.getfloat('correlation_start'),
                'end': gqi_section.getfloat('correlation_end'),
                'weight': weights['corr']
            },
            'mus':  {
                'start': gqi_section.getfloat('muscle_start'),
                'end': gqi_section.getfloat('muscle_end'),
                'weight': weights['mus']
            },
            'psd':  {
                'start': gqi_section.getfloat('psd_noise_start'),
                'end': gqi_section.getfloat('psd_noise_end'),
                'weight': weights['psd']
            },
        }

    except:
        print('___MEGqc___: ',
              'Invalid setting in config file! Please check instructions for each setting. \nGeneral directions: \nDon`t write any parameter as None. Don`t use quotes.\nLeaving blank is only allowed for parameters: \n- stim_channel, \n- data_crop_tmin, data_crop_tmax, \n- freq_min and freq_max in Filtering section, \n- all parameters of Filtering section if apply_filtering is set to False.')
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


def stim_data_to_df(raw: mne.io.Raw):
    """
    Extract stimulus data from MEG data and put it into a pandas DataFrame.

    Parameters
    ----------
    raw : mne.io.Raw
        MEG data.

    Returns
    -------
    stim_deriv : list
        List with QC_derivative object with stimulus data.

    """

    stim_channels = mne.pick_types(raw.info, stim=True)

    if len(stim_channels) == 0:
        print('___MEGqc___: ', 'No stimulus channels found.')
        stim_df = pd.DataFrame()
    else:
        stim_channel_names = [raw.info['ch_names'][ch] for ch in stim_channels]
        # Extract data for stimulus channels
        stim_data, times = raw[stim_channels, :]
        # Create a DataFrame with the stimulus data
        stim_df = pd.DataFrame(stim_data.T, columns=stim_channel_names)
        stim_df['time'] = times

    # save df as QC_derivative object
    stim_deriv = [QC_derivative(stim_df, 'stimulus', 'df')]

    return stim_deriv


def robust_epoching(data, events, sfreq, tmin, tmax,
                    baseline=None, reject=None, picks=None,
                    preload=True, verbose=True, ch_types='eeg'):
    """
    Robust epoch extraction with complete control over the process.

    This function manually extracts epochs from continuous data, providing
    more reliability than mne.Epochs by explicitly handling edge cases
    and giving transparent feedback about valid/invalid events.

    Parameters
    ----------
    data : array (n_channels, n_times) or mne.io.Raw
        Continuous data to epoch
    events : array (n_events, 3)
        Event matrix [sample, 0, event_id]
    sfreq : float
        Sampling frequency in Hz
    tmin, tmax : float
        Start and end time of epochs relative to events (seconds)
    baseline : tuple or None
        Baseline period for correction (start, end) in seconds
        If None, no baseline correction is applied
    reject : dict or None
        Amplitude rejection thresholds (e.g., {'grad': 4000e-13})
        If None, no amplitude-based rejection is performed
    picks : list or None
        Channel names to select (e.g., ['EEG FP1-Ref', 'EEG FP2-Ref', ...])
        If None, all channels are used
    preload : bool
        If True, returns an mne.EpochsArray object
        If False, returns a dictionary with epoch data and metadata
    verbose : bool
        If True, print progress and validation information
    ch_types : str or list
        Channel types. Can be a string for uniform types or
        a list of specific types per channel

    Returns
    -------
    epochs : mne.EpochsArray or dict
        If preload=True: mne.EpochsArray object with metadata stored in comment field
        If preload=False: dictionary containing:
            - 'data': epoch data array (n_epochs, n_channels, n_times)
            - 'times': time vector for each sample
            - 'events': valid events array
            - 'valid_indices': indices of valid events
            - 'sfreq': sampling frequency
            - Additional metadata

    Raises
    ------
    ValueError
        If no valid epochs can be created with given parameters
        If picks contains channel names not found in data
    """

    # ===== 1. PREPARE DATA WITH PICKS HANDLING =====
    # Handle both numpy arrays and MNE Raw objects
    if isinstance(data, mne.io.BaseRaw):
        # Get channel names from Raw object
        all_ch_names = data.ch_names

        # If picks are specified, validate and select channels
        if picks is not None:
            # Validate that all requested channels exist
            missing_channels = [ch for ch in picks if ch not in all_ch_names]
            if missing_channels:
                raise ValueError(
                    f"Channel(s) not found in data: {missing_channels}. "
                    f"Available channels: {all_ch_names}"
                )

            # Get indices of requested channels
            picks_indices = [all_ch_names.index(ch) for ch in picks]

            # Extract data for selected channels
            data_array = data.get_data(picks=picks_indices)

            # Use the requested channel names
            ch_names = picks

            if verbose:
                print(f"Selected {len(picks)} channels: {', '.join(picks[:5])}"
                      f"{'...' if len(picks) > 5 else ''}")
        else:
            # No picks specified, use all channels
            data_array = data.get_data()
            ch_names = all_ch_names
            picks_indices = list(range(len(all_ch_names)))

        # Verify sampling frequency matches
        data_sfreq = data.info['sfreq']
        if abs(data_sfreq - sfreq) > 0.1:
            warnings.warn(f"Sampling frequency mismatch: "
                          f"data={data_sfreq}Hz, parameter={sfreq}Hz")

    else:
        # Data is a numpy array - handle picks differently
        data_array = data
        n_channels_total = data_array.shape[0]

        # Create default channel names if not provided
        all_ch_names = [f'CH{i}' for i in range(n_channels_total)]

        if picks is not None:
            # For numpy arrays, picks should be indices or we need mapping
            if all(isinstance(pick, str) for pick in picks):
                # Picks are channel names - need to map to indices
                # This assumes the numpy array follows the same order
                # as the provided channel names
                ch_names = picks
                picks_indices = list(range(len(picks)))

                if len(picks) != n_channels_total:
                    warnings.warn(
                        f"Number of picks ({len(picks)}) doesn't match "
                        f"data channels ({n_channels_total}). Using first "
                        f"{min(len(picks), n_channels_total)} channels."
                    )
                    # Use whichever is smaller
                    n_to_use = min(len(picks), n_channels_total)
                    data_array = data_array[:n_to_use, :]
                    ch_names = picks[:n_to_use]
                    picks_indices = list(range(n_to_use))
            else:
                # Picks are indices
                picks_indices = picks
                ch_names = [all_ch_names[i] for i in picks_indices]
                data_array = data_array[picks_indices, :]
        else:
            # No picks specified, use all channels
            ch_names = all_ch_names
            picks_indices = list(range(n_channels_total))

    # Get data dimensions after picks selection
    n_channels, n_times_total = data_array.shape

    # ===== 2. CALCULATE SAMPLES FOR EPOCH =====
    # Convert time values to sample counts
    n_pre_samples = int(-tmin * sfreq) if tmin < 0 else 0
    n_post_samples = int(tmax * sfreq)
    n_samples_per_epoch = n_pre_samples + n_post_samples

    # Create time vector for the epoch
    times = np.linspace(tmin, tmax, n_samples_per_epoch, endpoint=False)

    # ===== 3. VALIDATE EVENTS =====
    # Extract event sample positions (first column of events matrix)
    event_samples = events[:, 0].astype(int)

    # Track which events are valid (have enough data before and after)
    valid_indices = []  # Indices of valid events in the original events array
    valid_start_samples = []  # Starting sample position for each valid epoch

    if verbose:
        print(f"Total events to process: {len(events)}")
        print(f"Epoch duration: {tmax - tmin:.3f}s = {n_samples_per_epoch} samples")
        print(f"Data available: {n_times_total} samples ({n_times_total / sfreq:.2f}s)")
        print(f"Channels selected: {n_channels}")

    # Check each event for data availability
    for event_idx, event_sample in enumerate(event_samples):
        # Calculate data range needed for this epoch
        epoch_start_sample = event_sample - n_pre_samples
        epoch_end_sample = event_sample + n_post_samples

        # Check if the entire epoch fits within available data
        if epoch_start_sample >= 0 and epoch_end_sample <= n_times_total:
            valid_indices.append(event_idx)
            valid_start_samples.append(epoch_start_sample)
        elif verbose:
            # Only print warning for invalid events if verbose mode is on
            print(f"  Event {event_idx} (sample {event_sample}) skipped: "
                  f"requires samples [{epoch_start_sample}, {epoch_end_sample}), "
                  f"but data has [0, {n_times_total})")

    n_valid_epochs = len(valid_indices)

    # Check if any valid epochs were found
    if n_valid_epochs == 0:
        raise ValueError(
            f"No valid epochs could be created. "
            f"Check that events have enough data before/after. "
            f"tmin={tmin}s, tmax={tmax}s, sfreq={sfreq}Hz"
        )

    if verbose:
        print(f"Valid epochs found: {n_valid_epochs}/{len(events)}")
        if n_valid_epochs < len(events):
            print(f"Removed {len(events) - n_valid_epochs} events without sufficient data")

    # ===== 4. EXTRACT EPOCH DATA =====
    # Pre-allocate array for epoch data
    epochs_data = np.zeros((n_valid_epochs, n_channels, n_samples_per_epoch))

    # Extract data for each valid epoch
    for epoch_idx, (event_idx, start_sample) in enumerate(zip(valid_indices, valid_start_samples)):
        epochs_data[epoch_idx] = data_array[:, start_sample:start_sample + n_samples_per_epoch]

    # ===== 5. APPLY BASELINE CORRECTION =====
    if baseline is not None:
        baseline_start = int((baseline[0] - tmin) * sfreq)
        baseline_end = int((baseline[1] - tmin) * sfreq)

        # Validate baseline range
        if baseline_start < 0 or baseline_end > n_samples_per_epoch:
            warnings.warn(
                f"Baseline period [{baseline[0]}, {baseline[1]}]s "
                f"is outside epoch range [{tmin}, {tmax}]s. "
                f"No baseline correction applied."
            )
        else:
            # Calculate mean for each channel in baseline period
            baseline_mean = np.mean(
                epochs_data[:, :, baseline_start:baseline_end],
                axis=2, keepdims=True  # Keep dimensions for broadcasting
            )
            # Subtract baseline from entire epoch
            epochs_data -= baseline_mean

            if verbose:
                print(f"Applied baseline correction: [{baseline[0]}, {baseline[1]}]s")

    # ===== 6. APPLY AMPLITUDE REJECTION =====
    if reject is not None:
        # Track which epochs to keep after rejection
        keep_indices = []

        for epoch_idx in range(n_valid_epochs):
            epoch = epochs_data[epoch_idx]
            keep_epoch = True

            # Check each channel against rejection thresholds
            for ch_idx in range(n_channels):
                # Calculate maximum absolute amplitude in this channel
                ch_max_amplitude = np.max(np.abs(epoch[ch_idx]))

                # Simple rejection logic - can be customized
                # Example for MEG gradiometers
                if 'grad' in reject and ch_max_amplitude > reject['grad']:
                    keep_epoch = False
                    if verbose:
                        print(f"  Reject epoch {epoch_idx}, channel {ch_names[ch_idx]}: "
                              f"amplitude {ch_max_amplitude:.2e} > {reject['grad']:.2e}")
                    break
                # Example for EEG
                elif 'eeg' in reject and ch_max_amplitude > reject['eeg']:
                    keep_epoch = False
                    if verbose:
                        print(f"  Reject epoch {epoch_idx}, channel {ch_names[ch_idx]}: "
                              f"amplitude {ch_max_amplitude:.2e} > {reject['eeg']:.2e}")
                    break
                # Example for generic threshold
                elif 'amplitude' in reject and ch_max_amplitude > reject['amplitude']:
                    keep_epoch = False
                    if verbose:
                        print(f"  Reject epoch {epoch_idx}, channel {ch_names[ch_idx]}: "
                              f"amplitude {ch_max_amplitude:.2e} > {reject['amplitude']:.2e}")
                    break

            if keep_epoch:
                keep_indices.append(epoch_idx)

        # Apply rejection by keeping only selected epochs
        if len(keep_indices) < n_valid_epochs:
            epochs_data = epochs_data[keep_indices]
            # Update valid indices to match original events array
            original_valid_indices = [valid_indices[i] for i in keep_indices]
            valid_indices = original_valid_indices
            n_valid_epochs = len(epochs_data)

            if verbose:
                print(f"After amplitude rejection: {n_valid_epochs} epochs remaining")

    # ===== 7. CREATE MNE EPOCHS OBJECT =====
    if preload:
        # Prepare channel types
        if isinstance(ch_types, str):
            # All channels same type
            ch_types_list = [ch_types] * n_channels
        elif isinstance(ch_types, list) and len(ch_types) == n_channels:
            # Specific type for each channel
            ch_types_list = ch_types
        else:
            # Default to EEG
            ch_types_list = ['eeg'] * n_channels
            if verbose and ch_types != 'eeg':
                warnings.warn(f"ch_types parameter format not recognized. Defaulting to 'eeg' for all channels")

        # Create MNE Info object
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=ch_types_list
        )

        # Extract valid events
        valid_events = events[valid_indices]

        # Create MNE EpochsArray object
        epochs = mne.EpochsArray(
            data=epochs_data,
            info=info,
            events=valid_events,
            tmin=tmin,
            event_id=None,  # Can be customized with a dictionary
            reject=None,  # Rejection already handled above
            flat=None,
            verbose=False  # Control verbosity at function level
        )

        # Store metadata in a way that's compatible with MNE
        # Option 1: Add to the description field (safe for MNE)
        if baseline is not None:
            baseline_str = f"baseline={baseline}"
        else:
            baseline_str = "no baseline"

        if picks is not None:
            picks_str = f"picks={len(picks)} channels"
        else:
            picks_str = "all channels"

        description = (f"Created with robust_epoching | "
                       f"{n_valid_epochs} epochs | {picks_str} | {baseline_str}")
        epochs.info['description'] = description

        # Option 2: Create a custom attribute on the epochs object itself
        # (not in info, but directly on the object)
        epochs.robust_epoching_metadata = {
            'valid_events_indices': valid_indices,
            'original_n_events': len(events),
            'picks_applied': picks,
            'tmin': tmin,
            'tmax': tmax,
            'sfreq': sfreq,
            'baseline_applied': baseline,
            'rejection_applied': reject is not None
        }

        if verbose:
            print(f"\nCreated MNE EpochsArray with {len(epochs)} epochs")
            print(f"Data shape: {epochs.get_data().shape}")
            print(f"Time range: [{epochs.times[0]:.3f}, {epochs.times[-1]:.3f}]s")
            print(f"Channels: {len(epochs.ch_names)} channels")
            print(f"Metadata stored in: epochs.robust_epoching_metadata")

        return epochs

    # ===== 8. RETURN AS DICTIONARY (if preload=False) =====
    else:
        metadata = {
            'data': epochs_data,
            'times': times,
            'events': events[valid_indices],
            'valid_indices': valid_indices,
            'sfreq': sfreq,
            'tmin': tmin,
            'tmax': tmax,
            'n_epochs': n_valid_epochs,
            'n_channels': n_channels,
            'n_samples_per_epoch': n_samples_per_epoch,
            'ch_names': ch_names,
            'picks_applied': picks,
            'description': 'Epoch data extracted with robust_epoching()'
        }

        if verbose:
            print(f"\nReturning epoch data as dictionary")
            print(f"Epochs: {metadata['n_epochs']}")
            print(f"Shape: {metadata['data'].shape}")
            print(f"Channels: {metadata['ch_names']}")

        return metadata


# ===== EXAMPLE USAGE WITH ACCESSING METADATA =====
# def example_with_metadata():
#     """Show how to access the stored metadata"""
#
#     # Create example data
#     sfreq = 250
#     n_channels = 10
#     n_times = 3000
#
#     # Create channel names
#     ch_names = [f'EEG{i}-Ref' for i in range(1, n_channels + 1)]
#
#     # Create data
#     data_array = np.random.randn(n_channels, n_times)
#     info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
#     raw = mne.io.RawArray(data_array, info)
#
#     # Create events
#     events = np.array([
#         [500, 0, 1],
#         [1000, 0, 1],
#         [1500, 0, 1],
#         [2800, 0, 1],  # This will be skipped (not enough data for tmax=1)
#     ])
#
#     # Select specific channels
#     selected_channels = ['EEG1-Ref', 'EEG2-Ref', 'EEG3-Ref', 'EEG4-Ref', 'EEG5-Ref']
#
#     print("=== Creating epochs with metadata storage ===")
#
#     try:
#         # Create epochs
#         epochs = robust_epoching(
#             data=raw,
#             events=events,
#             sfreq=sfreq,
#             tmin=0,
#             tmax=1,
#             baseline=None,
#             reject=None,
#             picks=selected_channels,
#             preload=True,
#             verbose=True
#         )
#
#         print(f"\n=== Accessing metadata ===")
#
#         # Access the custom metadata
#         if hasattr(epochs, 'robust_epoching_metadata'):
#             meta = epochs.robust_epoching_metadata
#             print(f"Valid event indices: {meta['valid_events_indices']}")
#             print(f"Original events: {meta['original_n_events']}")
#             print(f"Valid epochs: {len(epochs)}")
#             print(f"Picks applied: {meta['picks_applied']}")
#             print(f"Time window: [{meta['tmin']}, {meta['tmax']}]s")
#             print(f"Sampling freq: {meta['sfreq']}Hz")
#             print(f"Baseline applied: {meta['baseline_applied']}")
#             print(f"Rejection applied: {meta['rejection_applied']}")
#
#         # Also check info description
#         print(f"\nInfo description: {epochs.info.get('description', 'No description')}")
#
#         # Verify we can use the epochs normally
#         print(f"\n=== Using epochs normally ===")
#         print(f"Number of epochs: {len(epochs)}")
#         print(f"Data shape: {epochs.get_data().shape}")
#         print(f"Channel names: {epochs.ch_names}")
#
#         # Example: Calculate mean across epochs
#         mean_data = epochs.get_data().mean(axis=0)
#         print(f"Mean data shape: {mean_data.shape}")
#
#         # Example: Get event information
#         print(f"\nEvent information:")
#         print(f"Event samples: {epochs.events[:, 0]}")
#         print(f"Event IDs: {np.unique(epochs.events[:, 2])}")
#
#         # Verify which events were used
#         print(f"\nVerification of valid events:")
#         print(f"Total events provided: {len(events)}")
#         print(f"Events used (samples): {epochs.events[:, 0]}")
#
#         # Check which event was skipped
#         all_event_samples = events[:, 0]
#         used_event_samples = epochs.events[:, 0]
#         skipped = [s for s in all_event_samples if s not in used_event_samples]
#         print(f"Event(s) skipped (not enough data): {skipped}")
#
#     except ValueError as e:
#         print(f"Error: {e}")
#
#
# # ===== HOW TO USE WITH YOUR SPECIFIC DATA =====
# def use_with_your_data():
#     """
#     Template for your specific use case with your channel list
#     """
#
#     # Your specific channel list
#     your_channels = [
#         'EEG FP1-Ref', 'EEG FP2-Ref', 'EEG F3-Ref', 'EEG F4-Ref',
#         'EEG C3-Ref', 'EEG C4-Ref', 'EEG P3-Ref', 'EEG P4-Ref',
#         'EEG O1-Ref', 'EEG O2-Ref', 'EEG F7-Ref', 'EEG F8-Ref',
#         'EEG T3-Ref', 'EEG T4-Ref', 'EEG T5-Ref', 'EEG T6-Ref',
#         'EEG FZ-Ref', 'EEG CZ-Ref', 'EEG PZ-Ref', 'EEG FT9-Ref',
#         'EEG FT10-Ref'
#     ]
#
#     # Your data loading code here
#     # raw = your_function_to_load_data()
#     # events = your_events_array
#
#     epochs = robust_epoching(
#         data=raw,  # Your MNE Raw object
#         events=events,
#         sfreq=250,  # Your sampling frequency
#         tmin=0,
#         tmax=1,
#         baseline=None,
#         reject=None,
#         picks=your_channels,  # Your specific channels
#         preload=True,
#         verbose=True
#     )
#
#     # Access metadata about which events were valid
#     print(f"\nMetadata:")
#     print(f"Valid event indices: {epochs.robust_epoching_metadata['valid_events_indices']}")
#     print(f"Channels selected: {len(epochs.robust_epoching_metadata['picks_applied'])}")
#
#     return epochs
#
#
# if __name__ == "__main__":
#     example_with_metadata()
#
def Epoch_meg(epoching_params, data: mne.io.Raw):
    """
    Epoch MEG data based on the parameters provided in the config file.

    Parameters
    ----------
    epoching_params : dict
        Dictionary with parameters for epoching.
    data : mne.io.Raw
        MEG data to be epoched.

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

    if picks_magn is None:
        picks_magn = data.copy().pick('eeg').ch_names if 'eeg' in data else None

    if not stim_channel:
        print('___MEGqc___: ',
              'No stimulus channel detected. Setting stimulus channel to None to allow mne to detect events autamtically.')
        stim_channel = None
        # here for info on how None is handled by mne:
        # even if stim is None, mne will check once more when creating events.

    epochs_grad, epochs_mag = None, None

    try:
        if stim_channel:
            # Try to find events if stimulus channels exist
            events = mne.find_events(data, stim_channel=stim_channel, min_duration=event_dur)
            print('___MEGqc___: ', 'Stimulus Events found:', len(events))
            # Use real events
            epochs_mag = mne.Epochs(data, events, picks=picks_magn, tmin=epoch_tmin, tmax=epoch_tmax, preload=True,
                                    baseline=None, event_repeated=epoching_params['event_repeated'])
            epochs_grad = mne.Epochs(data, events, picks=picks_grad, tmin=epoch_tmin, tmax=epoch_tmax, preload=True,
                                     baseline=None, event_repeated=epoching_params['event_repeated'])
        else:
            print('___MEGqc___: ',
                  'No events with set minimum duration were found using all stimulus channels. No epoching can be done. Try different event duration in config file.')
            events = np.array([], dtype=int)
            # Create artificial events for 1-second adjacent epochs
            sfreq = data.info['sfreq']
            duration_samples = int(1.0 * sfreq)  # 1-second epochs
            n_epochs = data.n_times // duration_samples

            # Create events at 1-second intervals
            events = []
            for i in range(n_epochs):
                events.append([i * duration_samples, 0, 1])  # event_id = 1
            events = np.array(events, dtype=int)

            # Create epochs with these artificial events
            epoch_tmax = 1.0 #hardcoding epoch length of 1 secs.
            epoch_tmin = 0.0
            # Use this function since mne.Epochs has shown to be unstable when no stimulus is given
            # epochs_mag = mne.Epochs(data, events, picks=picks_magn, tmin=epoch_tmin, tmax=epoch_tmax,  preload=True,
            #                         baseline=None, event_repeated=epoching_params['event_repeated'])
            epochs_mag = robust_epoching(
                data=data, events=events, sfreq=sfreq, tmin=epoch_tmin, tmax=epoch_tmax, baseline=None,
                reject=None, picks=picks_magn, preload=True, verbose=True, ch_types='eeg')

            if picks_grad:
                # epochs_grad = mne.Epochs(data, events, picks=picks_grad, tmin=epoch_tmin, tmax=epoch_tmax, preload=True,
                #                      baseline=None, event_repeated=epoching_params['event_repeated'])
                # Use this function since mne.Epochs has shown to be unstable when no stimulus is given
                epochs_grad = robust_epoching(
                    data=data, events=events, sfreq=sfreq, tmin=epoch_tmin, tmax=epoch_tmax, baseline=None,
                    reject=None, preload=True, verbose=True)
            else:
                epochs_grad = None
            print(f"Created {len(epochs_mag)} adjacent 1-second epochs")

        # Now epochs_mag exists regardless of whether there were stimulus channels
    except:  # case when we use stim_channel=None, mne checks once more,  finds no other stim ch and no events and throws error:
        print('___MEGqc___: ', 'No stim channels detected, no events found.')
        pass  # go to returning empty dict

    dict_epochs_mg = {
        'mag': epochs_mag,
        'grad': epochs_grad}

    return dict_epochs_mg


def check_chosen_ch_types(m_or_g_chosen: List, channels_objs: dict):
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

    skipped_str = ''

    if not any(ch in m_or_g_chosen for ch in ['mag', 'grad', 'eeg']):
        skipped_str = "No channels to analyze. Check parameter ch_types in config file."
        raise ValueError(skipped_str)

    skipped_msgs = {
        'mag': "There are no magnetometers in this data set: check parameter ch_types in config file. Analysis will be done only for gradiometers.",
        'grad': "There are no gradiometers in this data set: check parameter ch_types in config file. Analysis will be done only for magnetometers.",
        'eeg': "There are no EEG electrodes in this data set: check parameter ch_types in config file."
    }

    for ch in ['mag', 'grad', 'eeg']:
        if len(channels_objs[ch]) == 0 and ch in m_or_g_chosen:
            skipped_str = skipped_msgs[ch]
            print(f'___MEGqc___: {skipped_str}')
            m_or_g_chosen.remove(ch)

    if not any(channels_objs[ch] for ch in ['mag', 'grad', 'eeg']):
        skipped_str = "There are no magnetometers, no gradiometers, no EEG electrodes in this data set. Analysis will not be done."
        raise ValueError(skipped_str)

    # Now m_or_g_chosen contain only those channel types which are present in the data set and were chosen by the user.

    return m_or_g_chosen, skipped_str


def choose_channels(raw: mne.io.Raw):
    """
    Separate channels by 'mag' and 'grad'.
    Done this way, because pick() or pick_types() sometimes gets wrong results, especialy for CTF data.

    Parameters
    ----------
    raw : mne.io.Raw
        MEG data

    Returns
    -------
    channels : dict
        dict with ch names separated by mag and grad

    """

    channels = {'mag': [], 'grad': [], 'eeg': []}

    # Loop over all channel indexes
    for ch_idx, ch_name in enumerate(raw.info['ch_names']):
        ch_type = mne.channel_type(raw.info, ch_idx)
        if ch_type in channels:
            if ch_type == 'eeg':
                #ATTENTION: here assigning EEG as MAG
                channels['mag'].append(ch_name)
            else:
                channels[ch_type].append(ch_name)

    return channels


def change_ch_type_CTF(raw, channels):
    """
    For CTF data channels types and units need to be chnaged from mag to grad.

    Parameters
    ----------
    channels : dict
        dict with ch names separated by mag and grad

    Returns
    -------
    channels : dict
        dict with ch names separated by mag and grad UPDATED

    """

    # Create a copy of the channels['mag'] list to iterate over
    mag_channels_copy = channels['mag'][:]

    for ch_name in mag_channels_copy:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_channel_types({ch_name: 'grad'})
        channels['grad'].append(ch_name)
        # Remove from mag list
        channels['mag'].remove(ch_name)

    print('___MEGqc___: Types of channels changed from mag to grad for CTF data.')

    return channels, raw


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
        raw = mne.io.read_raw_ctf(file_path, preload=True, verbose='ERROR')
        meg_system = 'CTF'

    elif os.path.isfile(file_path) and file_path.endswith('.fif'):
        # It's a FIF file
        meg_system = 'Triux'

        print("___MEGqc___: ", "Loading FIF data...")
        try:
            raw = mne.io.read_raw_fif(file_path, on_split_missing='ignore', verbose='ERROR')
        except Exception: # noqa
            raw = mne.io.read_raw_fif(file_path, allow_maxshield=True, on_split_missing='ignore', verbose='ERROR')
            shielding_str = ''' <p>This fif file contains Internal Active Shielding data. Quality measurements calculated on this data should not be compared to the measuremnts calculated on the data without active shileding, since in the current case invironmental noise reduction was already partially performed by shileding, which normally should not be done before assesing the quality.</p>'''

    elif os.path.isfile(file_path) and file_path.endswith('.edf'):
        # print("___MEGqc___: ", "Loading EEG data...")
        meg_system = 'EEG'
        raw = load_eeg_meg(file_path, bids_root=None, ses=None, task=None, run=None, datatype='eeg')

    return raw, shielding_str, meg_system


##changes to update in calculation
import re

EEG_BASE_RE = re.compile(
    r"^(?:Fp|AF|F|FC|C|CP|T|TP|FT|P|PO|O|I)\d{1,2}$"
    r"|^(?:Cz|Fz|Pz|Oz|POz|AFz|FCz|CPz|Iz)$"
    r"|^(?:T3|T4|T5|T6)$",
    re.IGNORECASE
)

def _strip_suffix_ref(name: str):
    m = re.search(r"(-\s*[Rr][Ee][Ff])\s*$", name)
    if m:
        return name[:m.start()].strip(), True
    return name.strip(), False

def _base_token(name_no_suffix: str):
    s = name_no_suffix.strip()
    # Detect 'EEG ' prefix so we can remove it cleanly later for EXT
    eeg_prefixed = s.upper().startswith("EEG ")
    if eeg_prefixed:
        s = s[4:].strip()
    base = re.split(r"[-\s]+", s, maxsplit=1)[0].strip()
    return base, eeg_prefixed

def _map_t1_t2(tok: str):
    up = tok.upper()
    if up == "T1":  return "FT9"
    if up == "T2":  return "FT10"
    return tok

def normalize_channels(ch_names):
    out = []

    for ch in ch_names:
        original = ch.strip()

        # Extract REF suffix
        base_stripped, had_ref = _strip_suffix_ref(original)
        base, eeg_prefixed = _base_token(base_stripped)
        base_up = base.upper()
        suffix = "-Ref" if had_ref else ""

        # PHOTIC / IBI / SUPPR / BURSTS
        if base_up == "PHOTIC":
            out.append(f"PHOTIC {base_up}{suffix}")
            continue
        if base_up == "IBI":
            out.append(f"IBI {base_up}{suffix}")
            continue
        if base_up in {"SUPR", "SUPPR"}:
            out.append(f"SUPPR {base_up}{suffix}")
            continue
        if base_up in {"BURST", "BURSTS"}:
            out.append(f"BURSTS {base_up}{suffix}")
            continue

        # EMG
        if base_up.startswith("EMG"):
            out.append(f"EMG {base_up}{suffix}")
            continue

        # EOG
        if base_up in {"ROC", "LOC", "EOG", "HEOG", "VEOG", "REOG", "LEOG"}:
            out.append(f"EOG {base_up}{suffix}")
            continue

        # ECG
        if base_up.startswith("EKG") or base_up.startswith("ECG"):
            out.append(f"ECG {base_up}{suffix}")
            continue

        # Reference electrodes
        if base_up in {"A1", "A2", "M1", "M2", "REF"}:
            out.append(f"REF {base_up}{suffix}")
            continue

        # EEG
        mapped = _map_t1_t2(base)
        if EEG_BASE_RE.match(mapped):
            out.append(f"EEG {mapped.upper()}{suffix}")
            continue

        # EXT (remove EEG prefix if present!)
        if eeg_prefixed:
            # Remove 'EEG ' prefix properly
            cleaned = base_stripped[4:].strip()
            out.append(f"EXT {cleaned}{suffix}")
        else:
            out.append(f"EXT {original}")

    return out
######

# def load_data(file_path):
#     """
#     Load MEG data from a file. It can be a CTF data or a FIF file.
#
#     Parameters
#     ----------
#     file_path : str
#         Path to the fif file with MEG data.
#
#     Returns
#     -------
#     raw : mne.io.Raw
#         MEG data.
#     shielding_str : str
#         String with information about active shielding.
#
#     """
#
#     shielding_str = ''
#
#     meg_system = None
#
#     if os.path.isdir(file_path) and file_path.endswith('.ds'):
#         # It's a CTF data directory
#         print("___MEGqc___: ", "Loading CTF data...")
#         raw = mne.io.read_raw_ctf(file_path, preload=True, verbose='ERROR')
#         meg_system = 'CTF'
#
#     elif os.path.isfile(file_path) and file_path.endswith('.fif'):
#         # It's a FIF file
#         meg_system = 'Triux'
#
#         print("___MEGqc___: ", "Loading FIF data...")
#         try:
#             raw = mne.io.read_raw_fif(file_path, on_split_missing='ignore', verbose='ERROR')
#         except:
#             raw = mne.io.read_raw_fif(file_path, allow_maxshield=True, on_split_missing='ignore', verbose='ERROR')
#             shielding_str = ''' <p>This fif file contains Internal Active Shielding data. Quality measurements calculated on this data should not be compared to the measuremnts calculated on the data without active shileding, since in the current case invironmental noise reduction was already partially performed by shileding, which normally should not be done before assesing the quality.</p>'''
#
#     elif os.path.isfile(file_path) and file_path.endswith('.edf'):
#         # print("___MEGqc___: ", "Loading EEG data...")
#         raw = load_eeg_meg(file_path)
#         meg_system = 'EEG'
#
#     return raw, shielding_str, meg_system


##changes to update in calculation
from mne_bids import BIDSPath, read_raw_bids

from mne.io.constants import FIFF
def fix_channel_info_directly(raw):
    """
    Directly modify raw.info['chs'] based on channel name prefixes
    """

    for idx, ch in enumerate(raw.info['chs']):
        ch_name = raw.info['ch_names'][idx].upper()

        # EEG channels
        if (ch_name.startswith('EEG')):
                # or
                # ch_name.startswith('C') or
                # ch_name.startswith('F') or
                # ch_name.startswith('P') or
                # ch_name.startswith('O') or
                # ch_name.startswith('T') or
                # ch_name.startswith('A') or
                # ch_name.startswith('FP')):
                #
            ch['kind'] = FIFF.FIFFV_EEG_CH
            ch['coil_type'] = FIFF.FIFFV_COIL_EEG

        # EOG channels
        elif 'EOG' in ch_name or 'LOC' in ch_name or 'ROC' in ch_name:
            ch['kind'] = FIFF.FIFFV_EOG_CH
            ch['coil_type'] = FIFF.FIFFV_COIL_NONE

        # ECG channels
        elif 'ECG' in ch_name or 'EKG' in ch_name:
            ch['kind'] = FIFF.FIFFV_ECG_CH
            ch['coil_type'] = FIFF.FIFFV_COIL_NONE

        # EMG channels
        elif 'EMG' in ch_name:
            ch['kind'] = FIFF.FIFFV_EMG_CH
            ch['coil_type'] = FIFF.FIFFV_COIL_NONE

        # Stimulus channels
        elif 'STIM' in ch_name or 'TRIG' in ch_name or 'STATUS' in ch_name:
            ch['kind'] = FIFF.FIFFV_STIM_CH
            ch['coil_type'] = FIFF.FIFFV_COIL_NONE

        # Default to MISC
        else:
            ch['kind'] = FIFF.FIFFV_MISC_CH
            ch['coil_type'] = FIFF.FIFFV_COIL_NONE

    print("Channel info updated directly in raw.info['chs']")
    return raw

def load_eeg_meg(file_path, bids_root=None, ses=None, task=None, run=None, datatype=None):
    """
    Load EEG/MEG data from a file, supporting multiple formats including BIDS and XDF.

    Parameters:
    - file_path (str): Path to the EEG/MEG file.
    - bids_root (str, optional): Path to the root BIDS directory (for BIDS-EEG).

    Returns:
    - raw (mne.io.Raw or dict): Loaded EEG/MEG data or XDF streams.
    """
    if "_eeg." in file_path:
        dtype = 'eeg'
    else:
        dtype = None

    try:
        # Handle BIDS dataset
        if bids_root:
            # Define subject, session, and task (use None if not applicable)
            bids_path = BIDSPath(subject=file_path,  # Subject ID (without 'sub-')
                     session=ses,  # Session ID (None if not applicable)
                     task=task,  # Task name
                     run=run,  # Task name
                     datatype=datatype,  # Specify EEG/MEG data type
                     root=bids_root)
            # Load the dataset
            raw = read_raw_bids(bids_path)
            # bids_path = BIDSPath(root=bids_root, subject=file_path)
            # raw = read_raw_bids(bids_path)

        else:
            # Detect file extension
            ext = file_path.split('.')[-1].lower()

            dtype = 'eeg'
            # Load based on extension
            if ext in ['edf']:  # EDF
                raw = mne.io.read_raw_edf(file_path, preload=True)
            elif ext in ['bdf']:  # BDF (Biosemi)
                raw = mne.io.read_raw_bdf(file_path, preload=True)
            elif ext in ['vhdr']:  # BrainVision format. Includes .vmrk and .eeg
                raw = mne.io.read_raw_brainvision(file_path, preload=True)
            elif ext in ['cnt']:  # Neuroscan CNT
                raw = mne.io.read_raw_cnt(file_path, preload=True)
            elif ext in ['mff']:  # EGI MFF
                raw = mne.io.read_raw_egi(file_path, preload=True)
            elif ext in ['set']:  # EEGLAB .set files
                raw = mne.io.read_raw_eeglab(file_path, preload=True)
            elif ext in ['fif']:  # MEG FIF format (Neuromag)
                raw = mne.io.read_raw_fif(file_path, preload=True)
                dtype = 'meg'
            elif ext in ['nxe']:  # Nicolet EEG
                raw = mne.io.read_raw_nicolet(file_path, preload=True)
            elif ext in ['eeg']:  # Nihon Kohden
                raw = mne.io.read_raw_nihon(file_path, preload=True)
            elif ext in ['mef', 'mefd']:  # MEF (Multi-Scale EEG Format)
                raw = mne.io.read_raw_mef(file_path, preload=True)
            elif ext in ['snirf']:  # fNIRS format
                raw = mne.io.read_raw_snirf(file_path, preload=True)
                dtype = 'nirs'
            elif ext in ['xdf']:  # LabRecorder / mBrainTrain .xdf files
                streams, header = pyxdf.load_xdf(file_path)
                raw = {"streams": streams, "header": header}  # Return raw XDF data
            elif file_path.endswith('.ds'):  # CTF MEG directory
                raw = mne.io.read_raw_ctf(file_path, preload=True)
                dtype = 'meg'
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            ##changes to update in calculation
            if datatype == 'eeg' or dtype == 'eeg':
                # Normalize all names
                normalized = normalize_channels(raw.ch_names)
                mapping = dict(zip(raw.ch_names, normalized))
                raw.rename_channels(mapping)
                raw = fix_channel_info_directly(raw)
                ###
            print(f"Successfully loaded {file_path}")
            return raw

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
######

def add_3d_ch_locations(raw, channels_objs):
    """
    Add channel locations to the MEG channels objects.

    Parameters
    ----------
    raw : mne.io.Raw
        MEG data.
    channels_objs : dict
        Dictionary with MEG channels.

    Returns
    -------
    channels_objs : dict
        Dictionary with MEG channels with added locations.

    """

    # Create a dictionary to store the channel locations
    ch_locs = {ch['ch_name']: ch['loc'][:3] for ch in raw.info['chs']}
    # why [:3]?  to Get only the x, y, z coordinates (first 3 values), theer are also rotations, etc storred in loc.

    # Iterate through the channel names and add the locations
    for key, value in channels_objs.items():
        for ch in value:
            ch.loc = ch_locs[ch.name]

    return channels_objs


def add_CTF_lobes(channels_objs):
    # Initialize dictionary to store channels by lobe and side
    lobes_ctf = {
        'Left Frontal': [],
        'Right Frontal': [],
        'Left Temporal': [],
        'Right Temporal': [],
        'Left Parietal': [],
        'Right Parietal': [],
        'Left Occipital': [],
        'Right Occipital': [],
        'Central': [],
        'Reference': [],
        'EEG/EOG/ECG': [],
        'Extra': []  # Add 'Extra' lobe
    }

    # Iterate through the channel names and categorize them
    for key, value in channels_objs.items():
        for ch in value:
            categorized = False  # Track if the channel is categorized
            # Magnetometers (assuming they start with 'M')
            # Even though they all have to be grads for CTF!!!
            if ch.name.startswith('MLF'):  # Left Frontal
                lobes_ctf['Left Frontal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MRF'):  # Right Frontal
                lobes_ctf['Right Frontal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MLT'):  # Left Temporal
                lobes_ctf['Left Temporal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MRT'):  # Right Temporal
                lobes_ctf['Right Temporal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MLP'):  # Left Parietal
                lobes_ctf['Left Parietal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MRP'):  # Right Parietal
                lobes_ctf['Right Parietal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MLO'):  # Left Occipital
                lobes_ctf['Left Occipital'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MRO'):  # Right Occipital
                lobes_ctf['Right Occipital'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MLC') or ch.name.startswith('MRC'):  # Central (Midline)
                lobes_ctf['Central'].append(ch.name)
                categorized = True
            elif ch.name.startswith('MZ'):  # Reference Sensors
                lobes_ctf['Reference'].append(ch.name)
                categorized = True
            elif ch.name in ['Cz', 'Pz', 'ECG', 'VEOG', 'HEOG']:  # EEG/EOG/ECG channels
                lobes_ctf['EEG/EOG/ECG'].append(ch.name)
                categorized = True

            # Gradiometers (assuming they have a different prefix or suffix, such as 'G')
            elif ch.name.startswith('GLF'):  # Left Frontal Gradiometers
                lobes_ctf['Left Frontal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GRF'):  # Right Frontal Gradiometers
                lobes_ctf['Right Frontal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GLT'):  # Left Temporal Gradiometers
                lobes_ctf['Left Temporal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GRT'):  # Right Temporal Gradiometers
                lobes_ctf['Right Temporal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GLP'):  # Left Parietal Gradiometers
                lobes_ctf['Left Parietal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GRP'):  # Right Parietal Gradiometers
                lobes_ctf['Right Parietal'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GLO'):  # Left Occipital Gradiometers
                lobes_ctf['Left Occipital'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GRO'):  # Right Occipital Gradiometers
                lobes_ctf['Right Occipital'].append(ch.name)
                categorized = True
            elif ch.name.startswith('GLC') or ch.name.startswith('GRC'):  # Central (Midline) Gradiometers
                lobes_ctf['Central'].append(ch.name)
                categorized = True

            # If the channel was not categorized, add it to 'Extra'
            if not categorized:
                lobes_ctf['Extra'].append(ch.name)

    lobe_colors = {
        'Left Frontal': '#1f77b4',
        'Right Frontal': '#ff7f0e',
        'Left Temporal': '#2ca02c',
        'Right Temporal': '#9467bd',
        'Left Parietal': '#e377c2',
        'Right Parietal': '#d62728',
        'Left Occipital': '#bcbd22',
        'Right Occipital': '#17becf',
        'Central': '#8c564b',
        'Reference': '#7f7f7f',
        'EEG/EOG/ECG': '#bcbd22',
        'Extra': '#d3d3d3'
    }

    lobes_color_coding_str = 'Color coding by lobe is applied as per CTF system.'
    for key, value in channels_objs.items():
        for ch in value:
            for lobe in lobes_ctf.keys():
                if ch.name in lobes_ctf[lobe]:
                    ch.lobe = lobe
                    ch.lobe_color = lobe_colors[lobe]

    return channels_objs, lobes_color_coding_str


def add_Triux_lobes(channels_objs):
    lobes_treux = {
        'Left Frontal': ['MEG0621', 'MEG0622', 'MEG0623', 'MEG0821', 'MEG0822', 'MEG0823', 'MEG0121', 'MEG0122',
                         'MEG0123', 'MEG0341', 'MEG0342', 'MEG0343', 'MEG0321', 'MEG0322', 'MEG0323', 'MEG0331',
                         'MEG0332', 'MEG0333', 'MEG0643', 'MEG0642', 'MEG0641', 'MEG0611', 'MEG0612', 'MEG0613',
                         'MEG0541', 'MEG0542', 'MEG0543', 'MEG0311', 'MEG0312', 'MEG0313', 'MEG0511', 'MEG0512',
                         'MEG0513', 'MEG0521', 'MEG0522', 'MEG0523', 'MEG0531', 'MEG0532', 'MEG0533'],
        'Right Frontal': ['MEG0811', 'MEG0812', 'MEG0813', 'MEG0911', 'MEG0912', 'MEG0913', 'MEG0921', 'MEG0922',
                          'MEG0923', 'MEG0931', 'MEG0932', 'MEG0933', 'MEG0941', 'MEG0942', 'MEG0943', 'MEG1011',
                          'MEG1012', 'MEG1013', 'MEG1021', 'MEG1022', 'MEG1023', 'MEG1031', 'MEG1032', 'MEG1033',
                          'MEG1211', 'MEG1212', 'MEG1213', 'MEG1221', 'MEG1222', 'MEG1223', 'MEG1231', 'MEG1232',
                          'MEG1233', 'MEG1241', 'MEG1242', 'MEG1243', 'MEG1411', 'MEG1412', 'MEG1413'],
        'Left Temporal': ['MEG0111', 'MEG0112', 'MEG0113', 'MEG0131', 'MEG0132', 'MEG0133', 'MEG0141', 'MEG0142',
                          'MEG0143', 'MEG0211', 'MEG0212', 'MEG0213', 'MEG0221', 'MEG0222', 'MEG0223', 'MEG0231',
                          'MEG0232', 'MEG0233', 'MEG0241', 'MEG0242', 'MEG0243', 'MEG1511', 'MEG1512', 'MEG1513',
                          'MEG1521', 'MEG1522', 'MEG1523', 'MEG1531', 'MEG1532', 'MEG1533', 'MEG1541', 'MEG1542',
                          'MEG1543', 'MEG1611', 'MEG1612', 'MEG1613', 'MEG1621', 'MEG1622', 'MEG1623'],
        'Right Temporal': ['MEG1311', 'MEG1312', 'MEG1313', 'MEG1321', 'MEG1322', 'MEG1323', 'MEG1421', 'MEG1422',
                           'MEG1423', 'MEG1431', 'MEG1432', 'MEG1433', 'MEG1441', 'MEG1442', 'MEG1443', 'MEG1341',
                           'MEG1342', 'MEG1343', 'MEG1331', 'MEG1332', 'MEG1333', 'MEG2611', 'MEG2612', 'MEG2613',
                           'MEG2621', 'MEG2622', 'MEG2623', 'MEG2631', 'MEG2632', 'MEG2633', 'MEG2641', 'MEG2642',
                           'MEG2643', 'MEG2411', 'MEG2412', 'MEG2413', 'MEG2421', 'MEG2422', 'MEG2423'],
        'Left Parietal': ['MEG0411', 'MEG0412', 'MEG0413', 'MEG0421', 'MEG0422', 'MEG0423', 'MEG0431', 'MEG0432',
                          'MEG0433', 'MEG0441', 'MEG0442', 'MEG0443', 'MEG0711', 'MEG0712', 'MEG0713', 'MEG0741',
                          'MEG0742', 'MEG0743', 'MEG1811', 'MEG1812', 'MEG1813', 'MEG1821', 'MEG1822', 'MEG1823',
                          'MEG1831', 'MEG1832', 'MEG1833', 'MEG1841', 'MEG1842', 'MEG1843', 'MEG0631', 'MEG0632',
                          'MEG0633', 'MEG1631', 'MEG1632', 'MEG1633', 'MEG2011', 'MEG2012', 'MEG2013'],
        'Right Parietal': ['MEG1041', 'MEG1042', 'MEG1043', 'MEG1111', 'MEG1112', 'MEG1113', 'MEG1121', 'MEG1122',
                           'MEG1123', 'MEG1131', 'MEG1132', 'MEG1133', 'MEG2233', 'MEG1141', 'MEG1142', 'MEG1143',
                           'MEG2243', 'MEG0721', 'MEG0722', 'MEG0723', 'MEG0731', 'MEG0732', 'MEG0733', 'MEG2211',
                           'MEG2212', 'MEG2213', 'MEG2221', 'MEG2222', 'MEG2223', 'MEG2231', 'MEG2232', 'MEG2233',
                           'MEG2241', 'MEG2242', 'MEG2243', 'MEG2021', 'MEG2022', 'MEG2023', 'MEG2441', 'MEG2442',
                           'MEG2443'],
        'Left Occipital': ['MEG1641', 'MEG1642', 'MEG1643', 'MEG1711', 'MEG1712', 'MEG1713', 'MEG1721', 'MEG1722',
                           'MEG1723', 'MEG1731', 'MEG1732', 'MEG1733', 'MEG1741', 'MEG1742', 'MEG1743', 'MEG1911',
                           'MEG1912', 'MEG1913', 'MEG1921', 'MEG1922', 'MEG1923', 'MEG1931', 'MEG1932', 'MEG1933',
                           'MEG1941', 'MEG1942', 'MEG1943', 'MEG2041', 'MEG2042', 'MEG2043', 'MEG2111', 'MEG2112',
                           'MEG2113', 'MEG2141', 'MEG2142', 'MEG2143'],
        'Right Occipital': ['MEG2031', 'MEG2032', 'MEG2033', 'MEG2121', 'MEG2122', 'MEG2123', 'MEG2311', 'MEG2312',
                            'MEG2313', 'MEG2321', 'MEG2322', 'MEG2323', 'MEG2331', 'MEG2332', 'MEG2333', 'MEG2341',
                            'MEG2342', 'MEG2343', 'MEG2511', 'MEG2512', 'MEG2513', 'MEG2521', 'MEG2522', 'MEG2523',
                            'MEG2531', 'MEG2532', 'MEG2533', 'MEG2541', 'MEG2542', 'MEG2543', 'MEG2431', 'MEG2432',
                            'MEG2433', 'MEG2131', 'MEG2132', 'MEG2133'],
        'Extra': []}  # Add 'Extra' lobe

    # These were just for Aarons presentation:
    # lobes_treux = {
    #         'Left Frontal': ['MEG0621', 'MEG0622', 'MEG0623', 'MEG0821', 'MEG0822', 'MEG0823', 'MEG0121', 'MEG0122', 'MEG0123', 'MEG0341', 'MEG0342', 'MEG0343', 'MEG0321', 'MEG0322', 'MEG0323', 'MEG0331',  'MEG0332', 'MEG0333', 'MEG0643', 'MEG0642', 'MEG0641', 'MEG0541', 'MEG0542', 'MEG0543', 'MEG0311', 'MEG0312', 'MEG0313', 'MEG0511', 'MEG0512', 'MEG0513', 'MEG0521', 'MEG0522', 'MEG0523', 'MEG0531', 'MEG0532', 'MEG0533'],
    #         'Right Frontal': ['MEG0811', 'MEG0812', 'MEG0813', 'MEG0911', 'MEG0912', 'MEG0913', 'MEG0921', 'MEG0922', 'MEG0923', 'MEG0931', 'MEG0932', 'MEG0933', 'MEG0941', 'MEG0942', 'MEG0943', 'MEG1011', 'MEG1012', 'MEG1013', 'MEG1021', 'MEG1022', 'MEG1023', 'MEG1031', 'MEG1032', 'MEG1033', 'MEG1211', 'MEG1212', 'MEG1213', 'MEG1221', 'MEG1222', 'MEG1223', 'MEG1231', 'MEG1232', 'MEG1233', 'MEG1241', 'MEG1242', 'MEG1243', 'MEG1411', 'MEG1412', 'MEG1413'],
    #         'Left Temporal': ['MEG0111', 'MEG0112', 'MEG0113', 'MEG0131', 'MEG0132', 'MEG0133', 'MEG0141', 'MEG0142', 'MEG0143', 'MEG0211', 'MEG0212', 'MEG0213', 'MEG0221', 'MEG0222', 'MEG0223', 'MEG0231', 'MEG0232', 'MEG0233', 'MEG0241', 'MEG0242', 'MEG0243', 'MEG1511', 'MEG1512', 'MEG1513', 'MEG1521', 'MEG1522', 'MEG1523', 'MEG1531', 'MEG1532', 'MEG1533', 'MEG1541', 'MEG1542', 'MEG1543', 'MEG1611', 'MEG1612', 'MEG1613', 'MEG1621', 'MEG1622', 'MEG1623'],
    #         'Right Temporal': ['MEG1311', 'MEG1312', 'MEG1313', 'MEG1321', 'MEG1322', 'MEG1323', 'MEG1421', 'MEG1422', 'MEG1423', 'MEG1431', 'MEG1432', 'MEG1433', 'MEG1441', 'MEG1442', 'MEG1443', 'MEG1341', 'MEG1342', 'MEG1343', 'MEG1331', 'MEG1332', 'MEG1333', 'MEG2611', 'MEG2612', 'MEG2613', 'MEG2621', 'MEG2622', 'MEG2623', 'MEG2631', 'MEG2632', 'MEG2633', 'MEG2641', 'MEG2642', 'MEG2643', 'MEG2411', 'MEG2412', 'MEG2413', 'MEG2421', 'MEG2422', 'MEG2423'],
    #         'Left Parietal': ['MEG0411', 'MEG0412', 'MEG0413', 'MEG0421', 'MEG0422', 'MEG0423', 'MEG0431', 'MEG0432', 'MEG0433', 'MEG0441', 'MEG0442', 'MEG0443', 'MEG0711', 'MEG0712', 'MEG0713', 'MEG0741', 'MEG0742', 'MEG0743', 'MEG1811', 'MEG1812', 'MEG1813', 'MEG1821', 'MEG1822', 'MEG1823', 'MEG1831', 'MEG1832', 'MEG1833', 'MEG1841', 'MEG1842', 'MEG1843', 'MEG0631', 'MEG0632', 'MEG0633', 'MEG1631', 'MEG1632', 'MEG1633', 'MEG2011', 'MEG2012', 'MEG2013'],
    #         'Right Parietal': ['MEG1041', 'MEG1042', 'MEG1043', 'MEG1111', 'MEG1112', 'MEG1113', 'MEG1121', 'MEG1122', 'MEG1123', 'MEG1131', 'MEG1132', 'MEG1133', 'MEG2233', 'MEG1141', 'MEG1142', 'MEG1143', 'MEG2243', 'MEG0721', 'MEG0722', 'MEG0723', 'MEG0731', 'MEG0732', 'MEG0733', 'MEG2211', 'MEG2212', 'MEG2213', 'MEG2221', 'MEG2222', 'MEG2223', 'MEG2231', 'MEG2232', 'MEG2233', 'MEG2241', 'MEG2242', 'MEG2243', 'MEG2021', 'MEG2022', 'MEG2023', 'MEG2441', 'MEG2442', 'MEG2443'],
    #         'Left Occipital': ['MEG1641', 'MEG1642', 'MEG1643', 'MEG1711', 'MEG1712', 'MEG1713', 'MEG1721', 'MEG1722', 'MEG1723', 'MEG1731', 'MEG1732', 'MEG1733', 'MEG1741', 'MEG1742', 'MEG1743', 'MEG1911', 'MEG1912', 'MEG1913', 'MEG1921', 'MEG1922', 'MEG1923', 'MEG1931', 'MEG1932', 'MEG1933', 'MEG1941', 'MEG1942', 'MEG1943', 'MEG2041', 'MEG2042', 'MEG2043', 'MEG2111', 'MEG2112', 'MEG2113', 'MEG2141', 'MEG2142', 'MEG2143', 'MEG2031', 'MEG2032', 'MEG2033', 'MEG2121', 'MEG2122', 'MEG2123', 'MEG2311', 'MEG2312', 'MEG2313', 'MEG2321', 'MEG2322', 'MEG2323', 'MEG2331', 'MEG2332', 'MEG2333', 'MEG2341', 'MEG2342', 'MEG2343', 'MEG2511', 'MEG2512', 'MEG2513', 'MEG2521', 'MEG2522', 'MEG2523', 'MEG2531', 'MEG2532', 'MEG2533', 'MEG2541', 'MEG2542', 'MEG2543', 'MEG2431', 'MEG2432', 'MEG2433', 'MEG2131', 'MEG2132', 'MEG2133'],
    #         'Right Occipital': ['MEG0611', 'MEG0612', 'MEG0613']}

    # #Now add to lobes_treux also the name of each channel with space in the middle:
    for lobe in lobes_treux.keys():
        lobes_treux[lobe] += [channel[:-4] + ' ' + channel[-4:] for channel in lobes_treux[lobe]]

    lobe_colors = {
        'Left Frontal': '#1f77b4',
        'Right Frontal': '#ff7f0e',
        'Left Temporal': '#2ca02c',
        'Right Temporal': '#9467bd',
        'Left Parietal': '#e377c2',
        'Right Parietal': '#d62728',
        'Left Occipital': '#bcbd22',
        'Right Occipital': '#17becf',
        'Extra': '#d3d3d3'}

    # These were just for Aarons presentation:
    # lobe_colors = {
    #     'Left Frontal': '#2ca02c',
    #     'Right Frontal': '#2ca02c',
    #     'Left Temporal': '#2ca02c',
    #     'Right Temporal': '#2ca02c',
    #     'Left Parietal': '#2ca02c',
    #     'Right Parietal': '#2ca02c',
    #     'Left Occipital': '#2ca02c',
    #     'Right Occipital': '#d62728'}

    # loop over all values in the dictionary:
    lobes_color_coding_str = 'Color coding by lobe is applied as per Treux system. Separation by lobes based on Y. Hu et al. "Partial Least Square Aided Beamforming Algorithm in Magnetoencephalography Source Imaging", 2018. '
    for key, value in channels_objs.items():
        for ch in value:
            categorized = False
            for lobe in lobes_treux.keys():
                if ch.name in lobes_treux[lobe]:
                    ch.lobe = lobe
                    ch.lobe_color = lobe_colors[lobe]
                    categorized = True
                    break
            # If the channel was not categorized, assign it to 'extra' lobe
            if not categorized:
                ch.lobe = 'Extra'
                ch.lobe_color = lobe_colors[lobe]

    return channels_objs, lobes_color_coding_str


def assign_channels_properties(channels_short: dict, meg_system: str):
    """
    Assign lobe area to each channel according to the lobe area dictionary + the color for plotting + channel location.

    Can later try to make this function a method of the MEG_channels class.
    At the moment not possible because it needs to know the total number of channels to figure which meg system to use for locations. And MEG_channels class is created for each channel separately.

    Parameters
    ----------
    channels : dict
        dict with channels names like: {'mag': [...], 'grad': [...]}
    meg_system: str
        CTF, Triux, None...

    Returns
    -------
    channels_objs : dict
        Dictionary with channel names for each channel type: mag, grad. Each channel has assigned lobe area and color for plotting + channel location.
    lobes_color_coding_str : str
        A string with information about the color coding of the lobes.

    """

    channels_full = copy.deepcopy(channels_short)

    # for understanding how the locations are obtained. They can be extracted as:
    # mag_locs = raw.copy().pick('mag').info['chs']
    # mag_pos = [ch['loc'][:3] for ch in mag_locs]
    # (XYZ locations are first 3 digit in the ch['loc']  where ch is 1 sensor in raw.info['chs'])

    # Assign lobe labels to the channels:

    if meg_system.upper() == 'TRIUX' and len(channels_full['mag']) == 102 and len(channels_full['grad']) == 204:
        # for 306 channel data in Elekta/Neuromag Treux system
        channels_full, lobes_color_coding_str = add_Triux_lobes(channels_full)

        # assign 'TRIUX' to all channels:
        for key, value in channels_full.items():
            for ch in value:
                ch.system = 'TRIUX'

    elif meg_system.upper() == 'CTF':
        channels_full, lobes_color_coding_str = add_CTF_lobes(channels_full)

        # assign 'CTF' to all channels:
        for key, value in channels_full.items():
            for ch in value:
                ch.system = 'CTF'

    else:
        channels_full, lobes_color_coding_str = map_channels_to_lobes_and_colors(channels_full)
        # assign 'EEG' to all channels:
        for key, value in channels_full.items():
            for ch in value:
                ch.system = 'EEG'

    # sort channels by name:
    if meg_system.upper() != 'EEG':
        for key, value in channels_full.items():
                channels_full[key] = sorted(value, key=lambda x: x.name)

    return channels_full, lobes_color_coding_str

# def assign_channels_properties(channels_short: dict, meg_system: str):
#     """
#     Assign lobe area to each channel according to the lobe area dictionary + the color for plotting + channel location.
#
#     Can later try to make this function a method of the MEG_channels class.
#     At the moment not possible because it needs to know the total number of channels to figure which meg system to use for locations. And MEG_channels class is created for each channel separately.
#
#     Parameters
#     ----------
#     channels : dict
#         dict with channels names like: {'mag': [...], 'grad': [...]}
#     meg_system: str
#         CTF, Triux, None...
#
#     Returns
#     -------
#     channels_objs : dict
#         Dictionary with channel names for each channel type: mag, grad. Each channel has assigned lobe area and color for plotting + channel location.
#     lobes_color_coding_str : str
#         A string with information about the color coding of the lobes.
#
#     """
#
#     channels_full = copy.deepcopy(channels_short)
#
#     # for understanding how the locations are obtained. They can be extracted as:
#     # mag_locs = raw.copy().pick('mag').info['chs']
#     # mag_pos = [ch['loc'][:3] for ch in mag_locs]
#     # (XYZ locations are first 3 digit in the ch['loc']  where ch is 1 sensor in raw.info['chs'])
#
#     # Assign lobe labels to the channels:
#
#     if meg_system.upper() == 'TRIUX' and len(channels_full['mag']) == 102 and len(channels_full['grad']) == 204:
#         # for 306 channel data in Elekta/Neuromag Treux system
#         channels_full, lobes_color_coding_str = add_Triux_lobes(channels_full)
#
#         # assign 'TRIUX' to all channels:
#         for key, value in channels_full.items():
#             for ch in value:
#                 ch.system = 'TRIUX'
#
#     elif meg_system.upper() == 'CTF':
#         channels_full, lobes_color_coding_str = add_CTF_lobes(channels_full)
#
#         # assign 'CTF' to all channels:
#         for key, value in channels_full.items():
#             for ch in value:
#                 ch.system = 'CTF'
#
#     else:
#         lobes_color_coding_str = 'For MEG systems other than MEGIN Triux or CTF color coding by lobe is not applied.'
#         lobe_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#e377c2', '#d62728', '#bcbd22', '#17becf']
#         print('___MEGqc___: ' + lobes_color_coding_str)
#
#         for key, value in channels_full.items():
#             for ch in value:
#                 ch.lobe = 'All channels'
#                 # take random color from lobe_colors:
#                 ch.lobe_color = random.choice(lobe_colors)
#                 ch.system = 'OTHER'
#
#     # sort channels by name:
#     for key, value in channels_full.items():
#         channels_full[key] = sorted(value, key=lambda x: x.name)
#
#     return channels_full, lobes_color_coding_str
#

def sort_channels_by_lobe(channels_objs: dict):
    """ Sorts channels by lobes.

    Parameters
    ----------
    channels_objs : dict
        A dictionary of channel objects.

    Returns
    -------
    chs_by_lobe : dict
        A dictionary of channels sorted by ch type and lobe.

    """
    chs_by_lobe = {}
    for m_or_g in channels_objs:

        # put all channels into separate lists based on their lobes:
        lobes_names = list(set([ch.lobe for ch in channels_objs[m_or_g]]))

        lobes_dict = {key: [] for key in lobes_names}
        # fill the dict with channels:
        for ch in channels_objs[m_or_g]:
            lobes_dict[ch.lobe].append(ch)

            # Sort the dictionary by lobes names (by the second word in the key, if it exists)
        chs_by_lobe[m_or_g] = dict(
            sorted(lobes_dict.items(), key=lambda x: x[0].split()[1] if len(x[0].split()) > 1 else ''))

    return chs_by_lobe


def save_meg_with_suffix(file_path: str, dataset_path: str, raw, final_suffix: str = "FILTERED") -> str:
    """
    Given the original file_path (MEG data) and an MNE raw object,
    this function creates an output directory based on the file_path
    and saves the raw data in FIF format with the user-provided suffix.

    The output directory is constructed as:
        <base_dir>/derivatives/temp/<subject>
    where:
        - base_dir is the portion of file_path up to ds_name
        - subject is the folder immediately after ds_name
          (plus a small offset if 'temp' is in the path)
        - ds_name is extracted from dataset_path (e.g., the basename "ds_orig").

    Logic for Windows:
     - If the path starts with "K:" or "C:" but not "K:\", we add a backslash
       so Windows recognizes it as an absolute drive path.

    Logic for Linux:
     - If the first component is "", it means an absolute path like "/home/..."
       so we strip that "" and eventually re-add the leading slash when building base_dir.

    Everything else remains as in your original code.
    """

    # 1) Normalize and split
    norm_path = os.path.normpath(file_path)
    components = norm_path.split(os.sep)

    # 2) Minimal fix for Windows drive letter "K:", "C:", etc.
    #    If the first component is just <Letter>:, add a backslash to make it absolute.
    if components and re.match(r'^[A-Za-z]:$', components[0]):
        # e.g. components[0] == "K:"
        components[0] += "\\"  # becomes "K:\"
        use_windows_join = True

    # 3) Linux absolute path => first component might be "", remove it as you did originally.
    elif components and components[0] == '':
        # It's an absolute path on Linux, e.g. "/home/..."
        components = components[1:]
        use_windows_join = False

    else:
        # Possibly a relative path or something else => treat like Linux
        use_windows_join = False

    # 4) ds_name from dataset_path
    ds_name = os.path.basename(os.path.normpath(dataset_path))
    print("ds_name:", ds_name)

    # 5) Find ds_name in components
    idx = components.index(ds_name)

    # 6) Determine subject after ds_name (plus small offset if 'temp' is in the path)
    if 'temp' in components:
        subject = components[idx + 3]
    else:
        subject = components[idx + 1]

    # 7) Build base_dir from everything up to ds_name
    #    If Windows drive letter is in [0], we skip adding os.sep at the front.
    if use_windows_join:
        # e.g. "K:\ds_orig" => we join normally
        base_dir = os.path.join(*components[:idx + 1])
    else:
        # Linux or relative => replicate your old logic: leading slash
        base_dir = os.path.join(os.sep, *components[:idx + 1])

    # 8) Construct the output directory
    output_dir = os.path.join(base_dir, 'derivatives', 'temp', subject)
    output_dir = os.path.abspath(output_dir)
    print("Output directory:", output_dir)

    # 9) Create if doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print("Directory created (or already exists):", output_dir)

    # 10) Build new filename with suffix.
    #    If the data is CTF, it will replace .ds with .fif
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)

    if ext.lower() != '.fif':
        ext = '.fif'

    new_filename = f"{name}_{final_suffix}{ext}"
    new_file_path = os.path.join(output_dir, new_filename)
    print("New file path:", new_file_path)

    # 11) Save
    raw.save(new_file_path, overwrite=True, verbose='ERROR')

    return new_file_path


def delete_temp_folder(dataset_path: str) -> str:
    """
    Given the original dataset_path, this function re-creates the temporary written files
    directory and then delete it.

    The output directory is constructed as:
         <base_dir>/derivatives/temp/<subject>
    where:
         - base_dir is the portion of file_path up to and including 'ds_orig'
         - subject is the folder immediately after 'ds_orig'

    Parameters
    ----------
    dataset_path : str
         Absolute path to the dataset folder.
    """
    temp_dir = os.path.join(dataset_path, 'derivatives', 'temp')
    temp_dir = os.path.abspath(temp_dir)
    shutil.rmtree(temp_dir)
    print("Removing directory:", temp_dir)

    return

##changes to update in calculation
import re
def replace_t1_t2_preserve_suffix(channels):
    """
    Replace only the T1/T2 base label with T9/T10 while keeping
    the rest of the channel name (suffixes/prefixes) unchanged.
    """

    new_channels = []

    for ch in channels:
        original = ch

        # Normalize spaces
        s = ch.strip()
        s = re.sub(r"\s+", " ", s)

        # Split by common separators but KEEP separators (using capturing group)
        parts = re.split(r"([ \t]*[-–—_/][ \t]*|[ \t]+)", s)

        # Extract only the text pieces (odd positions = separators)
        tokens = parts[::2]
        seps   = parts[1::2]

        # Identify main token
        if not tokens:
            new_channels.append(original)
            continue

        # If label starts with "EEG", then target is next token (if exists)
        if tokens[0].upper() == "EEG" and len(tokens) >= 2:
            base_idx = 1
        else:
            base_idx = 0

        base = tokens[base_idx].strip()

        # Replace only the base token
        if base.upper() == "T1":
            tokens[base_idx] = "T9"
        elif base.upper() == "T2":
            tokens[base_idx] = "T10"

        # Reconstruct preserving all separators EXACTLY as they were
        rebuilt = []
        for t, ssep in zip(tokens, seps + [""]):
            rebuilt.append(t + ssep)

        new_channels.append("".join(rebuilt).strip())

    return new_channels
####

def initial_processing(default_settings: dict, filtering_settings: dict, epoching_params: dict, file_path: str,
                       dataset_path: str):
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
    file_path : str
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
    m_or_g_chosen : list
        List with channel types to analize: mag, grad.
    m_or_g_skipped_str : str
        String with information about which channel types were skipped.
    lobes_color_coding_str : str
        String with information about color coding for lobes.
    resample_str : str
        String with information about resampling.

    """

    print('___MEGqc___: ', 'Reading data from file:', file_path)

    raw, shielding_str, meg_system = load_data(file_path)

    ##changes to update in calculation
    # Working with channels:
    if meg_system == 'EEG':  # Different processing for EEG channels.
        channels = {
            "mag": [ch for ch in raw.ch_names if ch.upper().startswith("EEG")],
            "grad": [],
            "eeg": []
        }
    else:
        channels = choose_channels(raw)
    ######

    if meg_system == 'CTF':  # ONLY FOR CTF we do this! Return raw with changed channel types.
        channels, raw = change_ch_type_CTF(raw, channels)

    # Turn channel names into objects:
    channels_objs = {key: [MEG_channel(name=ch_name, type=key) for ch_name in value] for key, value in channels.items()}

    # Assign channels properties:
    channels_objs, lobes_color_coding_str = assign_channels_properties(channels_objs, meg_system)

    # Add channel locations:
    if meg_system != 'EEG':
        channels_objs = add_3d_ch_locations(raw, channels_objs)

    # Check if there are channels to analyze according to info in config file:
    m_or_g_chosen, m_or_g_skipped_str = check_chosen_ch_types(m_or_g_chosen=default_settings['m_or_g_chosen'],
                                                              channels_objs=channels_objs)

    # Sort channels by lobe - this will be used often for plotting
    chs_by_lobe = sort_channels_by_lobe(channels_objs)
    print('___MEGqc___: ', 'Channels sorted by lobe.')

    info = raw.info
    info_derivs = [QC_derivative(content=info, name='RawInfo', content_type='info', fig_order=-1)]

    # crop the data to calculate faster:
    tmax_possible = raw.times[-1]
    tmax = default_settings['crop_tmax']
    if tmax is None or tmax > tmax_possible:
        tmax = tmax_possible
    raw_cropped = raw.copy().crop(tmin=default_settings['crop_tmin'], tmax=tmax)
    # When resampling for plotting, cropping or anything else you don't need permanent in raw inside any functions - always do raw_new=raw.copy() not just raw_new=raw. The last command doesn't create a new object, the whole raw will be changed and this will also be passed to other functions even if you don't return the raw.

    stim_deriv = stim_data_to_df(raw_cropped)

    # Data filtering:
    raw_cropped_filtered = raw_cropped.copy()
    if filtering_settings['apply_filtering'] is True:
        raw_cropped.load_data()  # Data has to be loaded into memory before filtering:
        # Save raw_cropped
        raw_cropped_path = save_meg_with_suffix(file_path, dataset_path, raw_cropped, final_suffix="CROPPED")

        raw_cropped_filtered = raw_cropped

        # if filtering_settings['h_freq'] is higher than the Nyquist frequency, set it to Nyquist frequency:
        if filtering_settings['h_freq'] > raw_cropped_filtered.info['sfreq'] / 2 - 5:
            filtering_settings['h_freq'] = raw_cropped_filtered.info['sfreq'] / 2 - 5
            filtering_settings['downsample_to_hz'] = filtering_settings['h_freq']
            print('___MEGqc___: ',
                  'High frequency for filtering is higher than Nyquist frequency. High frequency was set to Nyquist frequency:',
                  filtering_settings['h_freq'])

        if meg_system == 'EEG':
            filtering_settings['downsample_to_hz'] = False
            raw_cropped_filtered.filter(l_freq=filtering_settings['l_freq'], h_freq=filtering_settings['h_freq'],
                                    picks='all', method=filtering_settings['method'], iir_params=None)
        else:
            raw_cropped_filtered.filter(l_freq=filtering_settings['l_freq'], h_freq=filtering_settings['h_freq'],
                                    picks='meg', method=filtering_settings['method'], iir_params=None)
        print('___MEGqc___: ', 'Data filtered from', filtering_settings['l_freq'], 'to', filtering_settings['h_freq'],
              'Hz.')

        # Save filtered signal
        raw_cropped_filtered_path = save_meg_with_suffix(file_path, dataset_path, raw_cropped_filtered,
                                                         final_suffix="FILTERED")

        if filtering_settings['downsample_to_hz'] is False:
            raw_cropped_filtered_resampled = raw_cropped_filtered
            raw_cropped_filtered_resampled_path = raw_cropped_filtered_path
            resample_str = 'Data not resampled. '
            print('___MEGqc___: ', resample_str)
        elif filtering_settings['downsample_to_hz'] >= filtering_settings['h_freq'] * 5:
            raw_cropped_filtered_resampled = raw_cropped_filtered.resample(sfreq=filtering_settings['downsample_to_hz'])
            raw_cropped_filtered_resampled_path = save_meg_with_suffix(file_path, dataset_path,
                                                                       raw_cropped_filtered_resampled,
                                                                       final_suffix="FILTERED_RESAMPLED")
            resample_str = 'Data resampled to ' + str(filtering_settings['downsample_to_hz']) + ' Hz. '
            print('___MEGqc___: ', resample_str)
        else:
            raw_cropped_filtered_resampled = raw_cropped_filtered.resample(sfreq=filtering_settings['h_freq'] * 5)
            raw_cropped_filtered_resampled_path = save_meg_with_suffix(file_path, dataset_path,
                                                                       raw_cropped_filtered_resampled,
                                                                       final_suffix="FILTERED_RESAMPLED")
            # frequency to resample is 5 times higher than the maximum chosen frequency of the function
            resample_str = 'Chosen "downsample_to_hz" value set was too low, it must be at least 5 time higher than the highest filer frequency. Data resampled to ' + str(
                filtering_settings['h_freq'] * 5) + ' Hz. '
            print('___MEGqc___: ', resample_str)


    else:
        print('___MEGqc___: ', 'Data not filtered.')
        # And downsample:
        if filtering_settings['downsample_to_hz'] is not False:
            raw_cropped_filtered_resampled = raw_cropped_filtered.resample(sfreq=filtering_settings['downsample_to_hz'])
            raw_cropped_filtered_resampled_path = save_meg_with_suffix(file_path, dataset_path,
                                                                       raw_cropped_filtered_resampled,
                                                                       final_suffix="FILTERED_RESAMPLED")
            if filtering_settings['downsample_to_hz'] < 500:
                resample_str = 'Data resampled to ' + str(filtering_settings[
                                                              'downsample_to_hz']) + ' Hz. Keep in mind: resampling to less than 500Hz is not recommended, since it might result in high frequency data loss (for example of the CHPI coils signal. '
                print('___MEGqc___: ', resample_str)
            else:
                resample_str = 'Data resampled to ' + str(filtering_settings['downsample_to_hz']) + ' Hz. '
                print('___MEGqc___: ', resample_str)
        else:
            raw_cropped_filtered_resampled = raw_cropped_filtered
            raw_cropped_filtered_resampled_path = save_meg_with_suffix(file_path, dataset_path,
                                                                       raw_cropped_filtered_resampled,
                                                                       final_suffix="FILTERED_RESAMPLED")
            resample_str = 'Data not resampled. '
            print('___MEGqc___: ', resample_str)

    del raw_cropped_filtered, raw_cropped_filtered_resampled, raw_cropped, raw
    gc.collect()

    # Load data
    orig_meg_system = meg_system
    raw_cropped_filtered, shielding_str, meg_system = load_data(raw_cropped_filtered_path)

    # Apply epoching: USE NON RESAMPLED DATA. Or should we resample after epoching?
    # Since sampling freq is 1kHz and resampling is 500Hz, it s not that much of a win...

    dict_epochs_mg = Epoch_meg(epoching_params, data=raw_cropped_filtered)

    epoching_str = ''
    if dict_epochs_mg['mag'] is None and dict_epochs_mg['grad'] is None:
        epoching_str = ''' <p>No epoching could be done in this data set: no events found. Quality measurement were only performed on the entire time series. If this was not expected, try: 1) checking the presence of stimulus channel in the data set, 2) setting stimulus channel explicitly in config file, 3) setting different event duration in config file.</p><br></br>'''

    resample_str = '<p>' + resample_str + '</p>'

    # Extract chs_by_lobe into a data frame
    sensors_derivs = chs_dict_to_csv(chs_by_lobe, file_name_prefix='Sensors')

    raw_path = file_path

    return meg_system, dict_epochs_mg, chs_by_lobe, channels, raw_cropped_filtered_path, raw_cropped_filtered_resampled_path, raw_cropped_path, raw_path, info_derivs, stim_deriv, shielding_str, epoching_str, sensors_derivs, m_or_g_chosen, m_or_g_skipped_str, lobes_color_coding_str, resample_str,orig_meg_system


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

    # Extract chs_by_lobe into a data frame
    chs_by_lobe_df = {k1: {k2: pd.concat([channel.to_df() for channel in v2]) for k2, v2 in v1.items()} for k1, v1 in
                      chs_by_lobe.items()}

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

    df_deriv = [QC_derivative(content=df_fin, name=file_name_prefix, content_type='df')]

    return df_deriv

import re

# Color mapping based on CTF lobe convention
lobe_colors = {
    'Left Frontal': '#1f77b4',
    'Right Frontal': '#ff7f0e',
    'Left Temporal': '#2ca02c',
    'Right Temporal': '#9467bd',
    'Left Parietal': '#e377c2',
    'Right Parietal': '#d62728',
    'Left Occipital': '#bcbd22',
    'Right Occipital': '#17becf',
    'Left Central': '#8c564b',
    'Right Central': '#8c564b',
    'Central': '#8c564b',
    'Reference': '#7f7f7f',
    'EEG/EOG/ECG': '#bcbd22',
    'Extra': '#d3d3d3'
}

# Electrode to lobe mapping for 10-5 system
electrode_to_lobe = {
    'FP1': 'Left Frontal', 'FP2': 'Right Frontal',
    'AF3': 'Left Frontal', 'AF4': 'Right Frontal', 'AF7': 'Left Frontal', 'AF8': 'Right Frontal',
    'F1': 'Left Frontal', 'F2': 'Right Frontal', 'F3': 'Left Frontal', 'F4': 'Right Frontal',
    'F5': 'Left Frontal', 'F6': 'Right Frontal', 'F7': 'Left Frontal', 'F8': 'Right Frontal',
    'F9': 'Left Frontal', 'F10': 'Right Frontal',
    'FT7': 'Left Frontal', 'FT8': 'Right Frontal',
    'FC1': 'Left Frontal', 'FC2': 'Right Frontal', 'FC3': 'Left Frontal', 'FC4': 'Right Frontal',
    'FC5': 'Left Frontal', 'FC6': 'Right Frontal',
    'FZ': 'Central',
    'C1': 'Left Central', 'C2': 'Right Central',
    'C3': 'Left Central', 'C4': 'Right Central',
    'C5': 'Left Central', 'C6': 'Right Central',
    'CZ': 'Central',
    'P1': 'Left Parietal', 'P2': 'Right Parietal',
    'P3': 'Left Parietal', 'P4': 'Right Parietal',
    'P5': 'Left Parietal', 'P6': 'Right Parietal',
    'T5': 'Left Parietal', 'T6': 'Right Parietal',
    'P7': 'Left Parietal', 'P8': 'Right Parietal',
    'P9': 'Left Parietal', 'P10': 'Right Parietal',
    'CP1': 'Left Parietal', 'CP2': 'Right Parietal',
    'CP3': 'Left Parietal', 'CP4': 'Right Parietal',
    'CP5': 'Left Parietal', 'CP6': 'Right Parietal',
    'PZ': 'Central',
    'T3': 'Left Temporal', 'T4': 'Right Temporal',
    'T7': 'Left Temporal', 'T8': 'Right Temporal',
    'T9': 'Left Temporal', 'T10': 'Right Temporal',
    'TP7': 'Left Temporal', 'TP8': 'Right Temporal',
    'FT9': 'Left Temporal', 'FT10': 'Right Temporal',
    'TP9': 'Left Temporal', 'TP10': 'Right Temporal',
    'O1': 'Left Occipital', 'O2': 'Right Occipital',
    'OZ': 'Central',
    'PO3': 'Left Occipital', 'PO4': 'Right Occipital',
    'PO7': 'Left Occipital', 'PO8': 'Right Occipital',
    'I1': 'Left Occipital', 'I2': 'Right Occipital',
    'A1': 'Reference', 'A2': 'Reference', 'M1': 'Reference', 'M2': 'Reference',
    'EOG': 'EEG/EOG/ECG', 'ECG': 'EEG/EOG/ECG', 'EMG': 'EEG/EOG/ECG',
    'LOC': 'EEG/EOG/ECG', 'ROC': 'EEG/EOG/ECG', 'EKG1': 'EEG/EOG/ECG'
}

def extract_base_electrode(name):
    """Extracts base electrode name from noisy EEG channel labels."""
    name = name.upper()
    name = re.sub(r'(EEG|CH|ECOG|REF|AUX|POL|TRIG|EMG|EKG|ROC|LOC|PHOTIC)[:\-_ ]*', '', name)
    parts = re.split(r'[-_:\s]', name)
    for part in parts:
        if part in electrode_to_lobe:
            return part
    return None

def assign_lobe_color_location(channel_names, mni_coord_dict):
    """
    Assigns lobe, color, and MNI coordinates to each EEG channel in-place.

    Parameters:
    - channel_names: dict with 'eeg' key mapping to list of EEG channel objects
    - mni_coord_dict: dict mapping electrode name → [X, Y, Z] list in mm

    Returns:
    - Updated channel_names dict
    """
    # for ch in channel_names.get('eeg', []):
    # ATTENTION: here treating EEG as MAG channels
    for ch in channel_names.get('mag', []):
        base = extract_base_electrode(ch.name)
        lobe = electrode_to_lobe.get(base, 'Extra') if base else 'Extra'
        color = lobe_colors.get(lobe, '#000000')
        coords = mni_coord_dict.get(base, None)
        ch.lobe_area = lobe
        ch.lobe = lobe
        ch.color_code = color
        ch.lobe_color = color
        ch.location = coords
        ch.loc = coords
    return channel_names

import mne
import pandas as pd

def load_10_5_montage_info():
    # Load the standard 10-5 montage
    montage = mne.channels.make_standard_montage('standard_1005')

    # Get electrode positions in meters (we'll convert to millimeters)
    positions = montage.get_positions()['ch_pos']

    # Convert to DataFrame and filter only desired electrodes (optional)
    mni_coords = pd.DataFrame([
        {'Electrode': name.upper(), 'X': pos[0] * 1000, 'Y': pos[1] * 1000, 'Z': pos[2] * 1000}
        for name, pos in positions.items()
    ])

    # Round coordinates and build location list
    mni_coords['location'] = mni_coords[['X', 'Y', 'Z']].round(2).values.tolist()

    # Optionally filter to electrodes you care about
    # from your 10-5 list, e.g.:
    # mni_coords_filtered = mni_coords[mni_coords['Electrode'].isin(ten_five_electrodes)]
    mni_coords_filtered = mni_coords.reset_index(drop=True)

    # Create the dictionary required by assign_lobe_color_location
    mni_coord_dict = dict(zip(mni_coords_filtered['Electrode'], mni_coords_filtered['location']))
    return mni_coord_dict

def map_channels_to_lobes_and_colors(channel_names):
    # Round coordinates and build dict once
    mni_coord_dict = load_10_5_montage_info()
    channel_names = assign_lobe_color_location(channel_names, mni_coord_dict)
    lobes_color_coding_str = 'Color coding by lobe is applied as per CTF system.'
    return channel_names, lobes_color_coding_str

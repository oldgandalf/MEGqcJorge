import os
import mne
import pandas as pd

def load_meg_data(data_file) -> list([mne.io.Raw, list, list]):
    '''Load data and separate magnetometers and gradiometers.
    
    Args:
    data_file: data file .fif (or other format)
    
    Returns:
    raw(mne.io.Raw): data in raw format, 
    mags: list of tuples: magnetometer channel name + its index
    grads: list of tuples: gradiometer channel name + its index'''

    #data_file = os.path.join('Katharinas_Data','sub_HT05ND16', '210811', 'mikado-1.fif')                               
    raw = mne.io.read_raw_fif(data_file)

    #Separate mags and grads:
    # mags = [(chs['ch_name'], i) for i, chs in enumerate(raw.info['chs']) if str(chs['unit']).endswith('UNIT_T)')]
    # grads = [(chs['ch_name'], i) for i, chs in enumerate(raw.info['chs']) if str(chs['unit']).endswith('UNIT_T_M)')]

    channels = {'mags': raw.copy().pick_types(meg='mag').ch_names, 'grads': raw.copy().pick_types(meg='grad').ch_names}

    return(raw, channels)


def make_folders_meg(sid: str):
    '''Create folders (if they dont exist yet).

    Folders are created in BIDS-compliant directory order: 
    Working directory - Subject - derivtaives - megQC - csvs and figures

    Args:
    sid (int): subject Id, must be a string, like '1'. '''


    #Make sure to add subfolders on the list here AFTER the parent folder.
    path_list = [f'../derivatives', 
    f'../derivatives/sub-{sid}',
    f'../derivatives/sub-{sid}/megqc',
    f'../derivatives/sub-{sid}/megqc/csv files',
    f'../derivatives/sub-{sid}/megqc/figures',
    f'../derivatives/sub-{sid}/megqc/reports']

    for path in path_list:
        if os.path.isdir(path)==False: #if directory doesnt exist yet - create
            os.mkdir(path)


# def filter_and_resample_data(data: mne.io.Raw, l_freq: float, h_freq: float, method: str) -> list([mne.io.Raw, mne.io.Raw]):

#     '''Filtering and resampling the data. 
#       Commented out as it was copied to main, easier.
    
#     Filtering: Recommended: 1-100Hz bandpass or 0.5-100 Hz - better for frequency spectrum
#     https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter
#     method='iir' - using here the Butterworth filter similar to filtfilt in matlab, 
#     dont know if this one is the best possible option. 
    
#     Resampling:
#     LOOK AT THE WARNING HERE https://mne.tools/stable/generated/mne.io.Raw.html?highlight=resample#mne.io.Raw.resample
#     It s not recommended to epoch resampled data as it can mess up the triggers.
#     We can either downsample after epoching - if needed. https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.resample
#     And here downsample the continuous data - and use it further only in continuouys form, no epoching. This is why 2 options returned.

#     Frequency to resample is 5 times higher than the maximum chosen frequency of the function (MAKE THIS AS AN INPUT?)
    
#     Args:
#     data (mne.io.Raw): data in raw format 
#     l_freq (float): lowest frequency used for filtering 
#     h_freq: float: highest frequency used for filtering 
#     method(str): filtering method, example: 'iir'

#     Returns:
#     raw_bandpass(mne.raw): data only filtered
#     raw_bandpass_resamp(mne.raw): data filtered and resampled
#     '''

#     #Data has to be loaded into mememory before filetering:
#     data.load_data(verbose=True)
#     raw_bandpass = data.copy()
#     raw_bandpass.filter(l_freq=l_freq, h_freq=h_freq, picks='meg', method=method, iir_params=None)

#     #And resample:
#     raw_bandpass_resamp=raw_bandpass.copy()
#     raw_bandpass_resamp.resample(sfreq=h_freq*5)
#     #frequency to resample is 5 times higher than the maximum chosen frequency of the function

#     return(raw_bandpass, raw_bandpass_resamp)
#     #JOCHEM SAID: Try turning off the aliasing filter in downsampling. Not sure how?


def Epoch_meg(config, data: mne.io.Raw):

    '''Gives epoched data in 2 separated data frames: mags and grads + as epoch objects.
    
    Args:
    config
    data (mne.io.Raw): data in raw format
    
    Returns: 
    n_events (int): number of events(=number of epochs)
    df_epochs_mags (pd. Dataframe): data frame containing data for all epochs for mags 
    df_epochs_grads (pd. Dataframe): data frame containing data for all epochs for grads 
    epochs_mags (mne. Epochs): epochs as mne data structure for magnetometers
    epochs_grads (mne. Epochs): epochs as mne data structure for gradiometers '''

    # picks_grad = mne.pick_types(data.info, meg='grad', eeg=False, eog=False, stim=False)
    # picks_magn = mne.pick_types(data.info, meg='mag', eeg=False, eog=False, stim=False)

    epoching_section = config['Epoching']
    event_dur = epoching_section.getfloat('event_dur') 
    epoch_tmin = epoching_section.getfloat('epoch_tmin') 
    epoch_tmax = epoching_section.getfloat('epoch_tmax') 

    default_section = config['DEFAULT']
    stim_channel = default_section['stim_channel'] 
    stim_channel = stim_channel.replace(" ", "")
    stim_channel = stim_channel.split(",")

    if len(stim_channel) == 0:
        picks_stim = mne.pick_types(data.info, stim=True)
        stim_channel = []
        for ch in picks_stim:
            stim_channel.append(data.info['chs'][ch]['ch_name'])

    picks_magn = data.copy().pick_types(meg='mag').ch_names if 'mag' in data else None
    picks_grad = data.copy().pick_types(meg='grad').ch_names if 'grad' in data else None

    events = mne.find_events(data, stim_channel=stim_channel, min_duration=event_dur)
    n_events=len(events)

    if n_events == 0:
        print('No events with set minimum duration were found using all stimulus channels. No epoching can be done. Try different event duration in config file.')
        return(None, None, None, None)

    epochs_mags = mne.Epochs(data, events, picks=picks_magn, tmin=epoch_tmin, tmax=epoch_tmax, preload=True, baseline = None)
    epochs_grads = mne.Epochs(data, events, picks=picks_grad, tmin=epoch_tmin, tmax=epoch_tmax, preload=True, baseline = None)

    df_epochs_mags = epochs_mags.to_data_frame(time_format=None, scalings=dict(mag=1, grad=1))
    df_epochs_grads = epochs_grads.to_data_frame(time_format=None, scalings=dict(mag=1, grad=1))

    dict_of_dfs_epoch = {
    'grads': df_epochs_grads,
    'mags': df_epochs_mags}

    epochs_mg = {
    'grads': epochs_grads,
    'mags': epochs_mags}

    return dict_of_dfs_epoch, epochs_mg



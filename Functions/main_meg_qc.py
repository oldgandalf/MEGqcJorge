# Main script calling all other functions. 
# Will add imports here when other functions are done and moved from notebooks into py files

# For now it s wrapped into a function to be called in RMSE and Freq spectrum. When all done function will be removed

def initial_stuff(duration: int or None):

    '''Here all the initial actions need to work with MEG data are done: 
    - load fif file and convert into raw,
    - create folders in BIDS compliant format,
    - crop the data if needed,
    - filter and downsample the data,
    - epoch the data.

    Args:
    duration (int): how long the cropped data should be, in seconds

    Returns: 
    n_events (int): number of events(=number of epochs)
    df_epochs_mags (pd. Dataframe): data frame containing data for all epochs for mags 
    df_epochs_grads (pd. Dataframe): data frame containing data for all epochs for grads 
    epochs_mags (mne. Epochs): epochs as mne data structure for magnetometers
    epochs_grads (mne. Epochs): epochs as mne data structure for gradiometers 
    mags (list of tuples): magnetometer channel name + its index
    grads (list of tuples): gradiometer channel name + its index
    raw_bandpass(mne.raw): data only filtered, cropped (*)
    raw_bandpass_resamp(mne.raw): data filtered and resampled, cropped (*)
    raw_cropped(mne.io.Raw): data in raw format, cropped, not filtered, not resampled (*)
    raw(mne.io.Raw): original data in raw format, not cropped, not filtered, not resampled.
    (*): if duration was set to None - the data will not be cropped and these outputs 
    will return what is stated, but in origibal duration.

    Yes, these are a lot  of data output option, we can reduce them later when we know what will not be used.
    '''

    from data_load_and_folders import load_meg_data, make_folders_meg, filter_and_resample_data, Epoch_meg

    #Load data
    data_file = '../data/sub_HT05ND16/210811/mikado-1.fif/'
    raw, mags, grads=load_meg_data(data_file)

    #Create folders:
    make_folders_meg(sid='1')

    #crop the data to calculate faster
    raw_cropped = raw.copy()
    if duration is not None:
        raw_cropped.crop(0, duration) 

    #apply filtering and downsampling:
    raw_bandpass, raw_bandpass_resamp=filter_and_resample_data(data=raw_cropped,l_freq=0.5, h_freq=100, method='iir')

    #Apply epoching: USE NON RESAMPLED DATA. Or should we resample after epoching? 
    # Since sampling freq is 1kHz and resampling is 500Hz, it s not that much of a win...
    n_events, df_epochs_mags, df_epochs_grads, epochs_mags, epochs_grads=Epoch_meg(data=raw_bandpass, 
        stim_channel='STI101', event_dur=1.2, epoch_tmin=-0.2, epoch_tmax=1)

    return n_events, df_epochs_mags, df_epochs_grads, epochs_mags, epochs_grads, mags, grads, raw_bandpass, raw_bandpass_resamp, raw_cropped, raw

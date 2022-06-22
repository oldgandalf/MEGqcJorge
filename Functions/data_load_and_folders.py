import os
import mne

def load_meg_data(raw_file=None):
    #Load data AND SEPARATE MAGS AND GRADS. How do we want to input the path file here?

    #raw_file = os.path.join('Katharinas_Data','sub_HT05ND16', '210811', 'mikado-1.fif')                               
    raw = mne.io.read_raw_fif(raw_file)

    #Separate mags and grads:
    mags = [(chs['ch_name'], i) for i, chs in enumerate(raw.info['chs']) if str(chs['unit']).endswith('UNIT_T)')]
    grads = [(chs['ch_name'], i) for i, chs in enumerate(raw.info['chs']) if str(chs['unit']).endswith('UNIT_T_M)')]

    return(raw, mags, grads)


def make_folders_meg(sid='1'):
#Create folders (if they dont exist yet)

#sid is subject Id, must be a string.
#Folders are created in BIDS-compliant directory order: 
#Working directory - Subject - derivtaives - megQC - csvs and figures

    #This is the list of folders and subfolders to be created. Loop checks if directory already exists, if not - create.
    #Make sure to add subfolders on the list here AFTER the parent folder.

    #DO WE NEED TO CREATE IT ACTUALLY NOT IN CURRENT DIRECTORY, BUT GO ONE STEP UP FROM CURRENT DIRECTORY AND THEN CREAT DERIVATIVES?

    path_list = [f'../derivatives', 
    f'../derivatives/sub-{sid}',
    f'../derivatives/sub-{sid}/megqc',
    f'../derivatives/sub-{sid}/megqc/csv files',
    f'../derivatives/sub-{sid}/megqc/figures',
    f'../derivatives/sub-{sid}/megqc/reports']

    print(path_list)

    for path in path_list:
        if os.path.isdir(path)==False: #if directory doesnt exist yet - create
            os.mkdir(path)


#Filter the data and downsampling. see comments!

def filter_and_resample_data(data=None,l_freq=None, h_freq=None, method='iir'):
    # Filtering the data. Recommended: 1-100Hz bandpass or 0.5-100 Hz - better for frequency spectrum
    # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter

    # method='iir' - I m using here the Butterworth filter similar to filtfilt in matlab, like we  
    # did in the course with eeg data. such filter creates no time shift, since it filters forward and backward.
    # But we might use a different filter as well. I dont know if this one is the best possible option.

    #Data has to be loaded into mememory before filetering:
    data.load_data(verbose=True)
    raw_bandpass = data.copy()
    raw_bandpass.filter(l_freq=l_freq, h_freq=h_freq, picks='meg', method=method, iir_params=None)

    #And resample:
    #LOOK AT THE WARNING HERE https://mne.tools/stable/generated/mne.io.Raw.html?highlight=resample#mne.io.Raw.resample
    #It s not recommended to epoch resampled data as it can mess up the triggers.
    #We can either downsample after epoching - if needed. https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.resample
    #And here downsample the continuous data - and use it further only in continuouys form, no epoching. This is why 2 options returned.

    raw_bandpass_resamp=raw_bandpass.copy()
    raw_bandpass_resamp.resample(sfreq=h_freq*5)
    #frequency to resample is 5 times higher than the maximum chosen frequency of the function

    return(raw_bandpass, raw_bandpass_resamp)

    #JOCHEM SAID: Try turning off the aliasing filter in downsampling. Not sure how?


def Epoch_meg(data=None, stim_channel='STI101', event_dur=1.2, epoch_tmin=-0.2, epoch_tmax=1):
#Gives epoched data i2 separatet data frames: mags and grads


       picks_grad = mne.pick_types(data.info, meg='grad', eeg=False, eog=False, stim=False)
       picks_magn = mne.pick_types(data.info, meg='mag', eeg=False, eog=False, stim=False)

       events = mne.find_events(data, stim_channel=stim_channel, min_duration=event_dur)
       n_events=len(events)

       epochs_mags = mne.Epochs(data, events, picks=picks_magn, tmin=epoch_tmin, tmax=epoch_tmax, preload=True, baseline = None)
       epochs_grads = mne.Epochs(data, events, picks=picks_grad, tmin=epoch_tmin, tmax=epoch_tmax, preload=True, baseline = None)

       #Present epochs as data frame - separately for mags and grads
       df_epochs_mags = epochs_mags.to_data_frame(time_format=None, scalings=dict(mag=1, grad=1))
       df_epochs_grads = epochs_grads.to_data_frame(time_format=None, scalings=dict(mag=1, grad=1))

       return(n_events, df_epochs_mags, df_epochs_grads, epochs_mags, epochs_grads)
       #Returns: 
       # number of events(=number of epochs), 
       # data frame containing data for all epochs: mags and grads separately
       # epochs as mne data structure (not used anywhere, we may use it for something in the future)


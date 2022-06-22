# Main script calling all other functions. 
# Will add imports here when other functions are done and moved from notebooks into py files

# For now it s wrapped into a function to be called in RMSE and Freq spectrum. When all done function will be removed

def initial_stuff(duration):
    from data_load_and_folders import load_meg_data, make_folders_meg, filter_and_resample_data, Epoch_meg

    #Load data
    raw_file = '../data/sub_HT05ND16/210811/mikado-1.fif/'
    #raw_file = data
    raw, mags, grads=load_meg_data(raw_file=raw_file)

    #Create folders:
    make_folders_meg(sid='1')

    #crop the data to calculate faster
    raw_cropped = raw.copy()
    raw_cropped.crop(0, duration*60) 

    #apply filtering and downsampling:
    filtered_d, filtered_d_resamp=filter_and_resample_data(data=raw_cropped,l_freq=0.5, h_freq=100, method='iir')

    #Apply epoching: USE NON RESAMPLED DATA. Or should we resample after epoching? 
    # Since sampling freq is 1kHz and resampling is 500Hz, it s not that much of a win...
    n_events, df_epochs_mags, df_epochs_grads, epochs_mags, epochs_grads=Epoch_meg(data=filtered_d, 
        stim_channel='STI101', event_dur=1.2, epoch_tmin=-0.2, epoch_tmax=1)

    return n_events, df_epochs_mags, df_epochs_grads, epochs_mags, epochs_grads, mags, grads, filtered_d, filtered_d_resamp, raw_cropped, raw

# Main script calling all other functions. 
# Will add imports here when other functions are done and moved from notebooks into py files

# For now it s wrapped into a function to be called in RMSE and Freq spectrum. When all done function will be removed

#def initial_stuff(duration: int or None, config: dict):
def initial_stuff():


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

    import configparser
    config = configparser.ConfigParser()
    config.read('settings.ini')
    #config.sections()
    #config['DEFAULT']['data_file']
    #type(config)
    default_section = config['DEFAULT']
    data_file = default_section['data_file']
    duration = default_section.getint('duration') #int(config['DEFAULT']['duration'])
    sid = default_section['sid']

    from data_load_and_folders import load_meg_data, make_folders_meg, filter_and_resample_data, Epoch_meg

    #Load data
    #data_file = '../data/sub_HT05ND16/210811/mikado-1.fif/'
    #data_file = config['DEFAULT']['data_file']

    raw, mags, grads=load_meg_data(data_file)

    #Create folders:
    make_folders_meg(sid)

    #crop the data to calculate faster
    raw_cropped = raw.copy()
    if duration is not None:
        raw_cropped.crop(0, duration) 

    #apply filtering and downsampling:

    filtering_section = config['Filter_and_resample']
    l_freq = filtering_section.getint('l_freq') 
    h_freq = filtering_section.getint('l_freq') 
    method = filtering_section['method']
    raw_bandpass, raw_bandpass_resamp=filter_and_resample_data(data=raw_cropped,l_freq=l_freq, h_freq=h_freq, method=method)

    #Apply epoching: USE NON RESAMPLED DATA. Or should we resample after epoching? 
    # Since sampling freq is 1kHz and resampling is 500Hz, it s not that much of a win...

    epoching_section = config['Epoching']
    stim_channel = default_section['stim_channel'] #DO WE ALWAYS HAVE A STIM CHANNEL?
    event_dur = epoching_section.getint('event_dur') 
    epoch_tmin = epoching_section.getint('epoch_tmin') 
    epoch_tmax = epoching_section.getint('epoch_tmax') 

    n_events, df_epochs_mags, df_epochs_grads, epochs_mags, epochs_grads=Epoch_meg(data=raw_bandpass, 
        stim_channel=stim_channel, event_dur=event_dur, epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax)

    return n_events, df_epochs_mags, df_epochs_grads, epochs_mags, epochs_grads, mags, grads, raw_bandpass, raw_bandpass_resamp, raw_cropped, raw


def MEG_QC_measures():

    """This function will call all the QC functions.
    Here goes several sections which will in the future be called over main, but are not yet, since they are in the notebooks"""

    n_events, df_epochs_mags, df_epochs_grads, epochs_mags, epochs_grads, mags, grads, filtered_d, filtered_d_resamp, raw_cropped, raw=initial_stuff()

    import configparser
    config = configparser.ConfigParser()
    config.read('settings.ini')
    default_section = config['DEFAULT']
    sid = default_section['sid']

    # RMSE:

    # import RMSE_meg_qc as rmse #or smth like this - when it's extracted to .py
    rmse_section = config['RMSE']
    std_lvl = rmse_section.getint('std_lvl') 

    m_big_std_with_value, g_big_std_with_value, m_small_std_with_value, g_small_std_with_value, rmse_mags, rmse_grads = RMSE_meg_all(data = filtered_d_resamp, 
    mags=mags, grads=grads, std_lvl=std_lvl)


    fig_m, fig_path_m=boxplot_std_hovering_plotly(std_data=rmse_mags, tit='Magnetometers', channel_names=mags, sid=sid)
    fig_g, fig_path_g=boxplot_std_hovering_plotly(std_data=rmse_grads, tit='Gradiometers', channel_names=grads, sid=sid)

    df_std_mags, df_std_grads=RMSE_meg_epoch(mags=mags, grads=grads, std_lvl=std_lvl, n_events=n_events, df_epochs_mags=df_epochs_mags, df_epochs_grads=df_epochs_grads, sid=sid) 

    from universal_plots import boxplot_channel_epoch_hovering_plotly
    fig_std_epoch_m, fig_path_m_std_epoch = boxplot_channel_epoch_hovering_plotly(df_mg=df_std_mags, ch_type='Magnetometers', sid=sid, what_data='stds')
    fig_std_epoch_g, fig_path_g_std_epoch =boxplot_channel_epoch_hovering_plotly(df_mg=df_std_grads, ch_type='Gradiometers', sid=sid, what_data='stds')

    from universal_html_report import make_RMSE_html_report
    list_of_figure_paths=[fig_path_m, fig_path_g, fig_path_m_std_epoch, fig_path_g_std_epoch]
    make_RMSE_html_report(sid=sid, what_data='stds', list_of_figure_paths=list_of_figure_paths)

    # Frequency spectrum
    
import numpy as np
import pandas as pd
import mne
from meg_qc.source.universal_plots import boxplot_all_time, boxplot_all_time_csv, boxplot_epochs, QC_derivative, boxplot_epoched_xaxis_channels, boxplot_epoched_xaxis_epochs, boxplot_epoched_xaxis_epochs_csv, boxplot_epoched_xaxis_channels_csv, assign_epoched_std_ptp_to_channels
from meg_qc.source.universal_html_report import simple_metric_basic
from meg_qc.source.initial_meg_qc import chs_dict_to_csv
from IPython.display import display

# In[2]:

def RMSE(data_m_or_g: np.array or list):

    """ 
    RMSE - general root means squared error. Currently NOT USED, as np.std is slightly faster.
    Was used before as alternative to std calculation, was faster.
    
    Parameters
    ----------
    data_m_or_g : np.array or list 
        data for magnetometer or gradiometer given as np array or list 
        (it can be 1 or several channels data  as 2 dimentional array or as list of lists)
        
    Returns
    -------
    np.ndarray 
        rmse as numpy array (1-dimentional if 1 channel was given, 2-dim if more channels)
    """

    data_m_or_g=np.array(data_m_or_g) #convert to numpy array if it s not
    rmse_list=[]

    data_dimentions=len(data_m_or_g.shape)
    if data_dimentions==2: #if the data has raws and columns - iterate over raws. if input is 1 dimentional - just calculate the whole thing.
        for dat_raw in data_m_or_g:
            y_actual=dat_raw
            y_pred=y_actual.mean()
            rmse_data=np.sqrt(((y_pred - y_actual) ** 2).mean())
            rmse_list.append(rmse_data)
    elif data_dimentions==1:
        y_actual=data_m_or_g
        y_pred=data_m_or_g.mean()
        rmse_data=np.sqrt(((y_pred - y_actual) ** 2).mean())
        rmse_list.append(rmse_data)
    else:
        print('___MEG QC___: ', 'Only 1 or 2 dimentional data is accepted, not more!')
        return

    rmse_np=np.array(rmse_list) #conver to numpy array

    return rmse_np

def get_std_all_data(data: mne.io.Raw, channels: list):

    """
    Calculate RMSE/std (same mathematically) for each channel - for the entire time duration.

    Parameters
    ----------
    data : mne.io.Raw
        raw data 
    channels : list 
        list of channel names

    Returns
    -------
    dict
        rmse/std for each channel
    
    """

    data_channels=data.get_data(picks = channels)

    #std_channels = RMSE(data_channels)
    std_channels = np.std(data_channels, axis=1)

    #add channel name for every std value:
    std_channels_named = {}
    for i, ch in enumerate(channels):
        std_channels_named[ch] = std_channels[i]

    return std_channels_named


def get_big_small_std_ptp_all_data(ptp_or_std_channels_named: dict, channels: list, std_multiplier: float):

    """
    Function calculates peak-to-peak amplitude or STDs over the entire data set for each channel.
    Threshold for noisy = mean + multiplier*std, above it - noisy,
    Threshold for flat = mean - multiplier*std, below it - flat,
    where:

    - mean is mean stds/ptp values over all channels
    - std is standard deviation of stds values over all channels
    - multiplier is a parameter set in config, defines how many stds/ptp above or below mean should be taken as threshold.


    Parameters
    ----------
    ptp_or_std_channels_named : dict
        peak-to-peak amplitude or std for each channel
    channels : list
        list of channel names
    std_multiplier : float
        multipliar for std, used to define thresholds for noisy and flat channels

    Returns
    -------
    dict
        dictionary with channel names and their stds/ptp values. Noisy channels.
    dict
        dictionary with channel names and their stds/ptp values. Flat channels.

    """
    # Put all values in 1 array from the dictionsry:
    ptp_or_std_channels = np.array(list(ptp_or_std_channels_named.values()))

    ## Check if channel data is within std level of PtP/std.
    std_of_measure_channels=np.std(ptp_or_std_channels)
    mean_of_measure_channels=np.mean(ptp_or_std_channels)

    print('___MEG QC___: ', mean_of_measure_channels + std_multiplier*std_of_measure_channels, ' threshold for NOISY. ')
    print('___MEG QC___: ', mean_of_measure_channels - std_multiplier*std_of_measure_channels, ' threshold for FLAT. ')

    # Find the index of channels with biggest and smallest std:
    ch_ind_big_measure = [index for (index, item) in enumerate(ptp_or_std_channels) if item > mean_of_measure_channels + std_multiplier*std_of_measure_channels] #find index with bigst std
    ch_ind_small_measure = [index for (index, item) in enumerate(ptp_or_std_channels) if item < mean_of_measure_channels - std_multiplier*std_of_measure_channels] #find index with smallest std

    #make dictionaries with channel names and their std values:
    noisy_channels = {}
    flat_channels = {}

    for index in ch_ind_big_measure:
        ch_name = np.array(channels)[index]
        noisy_channels[ch_name] = ptp_or_std_channels[index]

    for index in ch_ind_small_measure:
        ch_name = np.array(channels)[index]
        flat_channels[ch_name] = ptp_or_std_channels[index]

    return noisy_channels, flat_channels

#%%
def get_std_epochs(channels: list, epochs_mg: mne.Epochs):

    """ 
    Calculate std for multiple epochs for a list of channels.
    Used as internal function in std_meg_epoch()

    Parameters
    ----------
    channels : list
        list of channel names
    epochs_mg : mne.Epochs
        epochs data as mne.Epochs object

    Returns
    -------
    pd.DataFrame
        dataframe with std values for each channel and each epoch
    """
    
    dict_ep = {}

    #get 1 epoch, 1 channel and calculate std of its data:
    for ep in range(0, len(epochs_mg)):
        std_epoch=[]
        for ch_name in channels: 
            data_ch_epoch=epochs_mg[ep].get_data(picks=ch_name)[0][0] 
            #[0][0] is because get_data creats array in array in array, it expects several epochs, several channels, but we only need  one.

            #std_ch_ep = RMSE(data_ch_epoch)
            std_ch_ep = np.std(data_ch_epoch) #if want to use std instead
            std_epoch.append(np.float64(std_ch_ep))

        dict_ep[ep] = std_epoch

    return pd.DataFrame(dict_ep, index=channels)


def get_big_small_std_ptp_epochs(df_std: pd.DataFrame, ch_type: str, std_lvl: int, std_or_ptp: str):

    """ NOT USED ANY MORE

    - Calculate std for every separate epoch of a given list of channels
    - Find which channels in which epochs have too high/too small stds or PtP amplitudes
    - Create MEG_QC_derivative as dfs

    Parameters
    ----------

    """

    # Check (which epochs for which channel) are over set STD_level (1 or 2, 3, etc STDs) for this epoch for all channels

    std_std_per_epoch=[]
    mean_std_per_epoch=[]

    epochs = df_std.columns.tolist()
    epochs = [int(ep) for ep in epochs]

    for ep in epochs:  
        std_std_per_epoch.append(np.std(df_std.iloc[:, ep])) #std of stds of all channels of every single epoch
        mean_std_per_epoch.append(np.mean(df_std.iloc[:, ep])) #mean of stds of all channels of every single epoch

    df_ch_ep_big_std=df_std.copy()
    df_ch_ep_small_std=df_std.copy()

    # Now see which channles in epoch are over std_level or under -std_level:
    for ep in epochs:   
        df_ch_ep_big_std.iloc[:,ep] = df_ch_ep_big_std.iloc[:,ep] > mean_std_per_epoch[ep]+std_lvl*std_std_per_epoch[ep] 
        df_ch_ep_small_std.iloc[:,ep] = df_ch_ep_small_std.iloc[:,ep] < mean_std_per_epoch[ep]-std_lvl*std_std_per_epoch[ep] 

    # Create derivatives:
    dfs_deriv = [
        QC_derivative(df_std, std_or_ptp+'_per_epoch_'+ch_type, 'df'),
        QC_derivative(df_ch_ep_big_std, 'big_'+std_or_ptp+'_per_epoch_'+ch_type, 'df'),
        QC_derivative(df_ch_ep_small_std, 'Small_'+std_or_ptp+'_per_epoch_'+ch_type, 'df')]


    return dfs_deriv

#%% All about simple metrc jsons:

def get_noisy_flat_std_ptp_epochs(df_std: pd.DataFrame, ch_type: str, std_or_ptp: str, noisy_channel_multiplier: float, flat_multiplier: float, percent_noisy_flat_allowed: float):
    
    """
    1. Define if the channels data inside the epoch is noisy or flat:
    Compare the std of this channel for this epoch (df_std) to the mean STD of this particular channel or over all epchs.

    - If std of this channel for this epoch is over the mean std of this channel for all epochs together * flat multiplyer, then  the data for this channel in this epoch is noisy.
    - If std of this channel for this epoch is under the mean std of this channel for all epochs together * noisy multiplyer, then this the data for this channel in this epoch is flat.
    
    Multiplyer is set by user in the config file.

    2. Count how many channels are noisy/flat in each epoch. 
    If more than percent_noisy_flat_allowed of channels are noisy/flat, then this epoch is noisy/flat.
    Percent is set by user in the config file.

    3. Create MEG_QC_derivative as 3 dfs:

    - df_epoch_vs_mean: ratio of std of this channel for this epoch to the mean std of this channel over all epochs together
    - df_noisy_epoch: df with True/False values for each channel in each epoch, True if this channel is noisy in this epoch
    - df_flat_epoch: df with True/False values for each channel in each epoch, True if this channel is flat in this epoch

    Parameters
    ----------
    df_std : pd.DataFrame
        dataframe with std/ptp values for each channel and each epoch
    ch_type : str
        channel type, 'mag', 'grad'
    std_or_ptp : str
        'std' or 'ptp' - to use std or peak to peak amplitude as a metric
    noisy_channel_multiplier : float
        multiplier to define noisy channel, if std of this channel for this epoch is over (the mean std of this channel for all epochs together*multipliar), then this channel is noisy
        set by user in the config file
    flat_multiplier : float
        multiplier to define flat channel, if std of this channel for this epoch is under (the mean std of this channel for all epochs together*multipliar), then this channel is flat
        set by user in the config file
    percent_noisy_flat_allowed : float
        percent of noisy/flat channels allowed in each epoch, if more than this percent, then this epoch is noisy/flat. Example: 70
        Means that if more than 70% of channels are noisy/flat in this epoch, then this epoch is noisy/flat.

    Returns
    -------
    list
        list of 3 MEG_QC_derivative objects:
        - df_epoch_vs_mean: ratio of std of this channel for this epoch to the mean std of this channel over all epochs together
        - df_noisy_epoch: df with True/False values for each channel in each epoch, True if this channel is noisy in this epoch
        - df_flat_epoch: df with True/False values for each channel in each epoch, True if this channel is flat in this epoch
    
    """

    epochs = df_std.columns.tolist() #get epoch numbers
    epochs = [int(ep) for ep in epochs]

    df_std_with_mean=df_std.copy() #make a separate df, because it also changes this variable utside this function, to avoid messing up tye data.
    df_std_with_mean['mean'] = df_std_with_mean.mean(axis=1) #mean of stds for each separate channel over all epochs together

    #compare mean std of each channel to std of this channel for every epoch:
    df_noisy_epoch=df_std_with_mean.copy()
    df_flat_epoch=df_std_with_mean.copy()
    df_epoch_vs_mean=df_std_with_mean.copy()

    # Now see which channles in epoch are over std_level or under -std_level:
    
    #append raws to df_noisy_epoch to hold the % of noisy/flat channels in each epoch:


    df_noisy_epoch.loc['number noisy channels'] = np.nan
    df_noisy_epoch.loc['% noisy channels'] = np.nan
    df_noisy_epoch.loc['noisy > %s perc' % percent_noisy_flat_allowed] = np.nan

    df_flat_epoch.loc['number flat channels'] = np.nan
    df_flat_epoch.loc['% flat channels'] = np.nan
    df_flat_epoch.loc['flat < %s perc' % percent_noisy_flat_allowed] = np.nan


    for ep in epochs:  

        df_epoch_vs_mean.iloc[:,ep] = df_epoch_vs_mean.iloc[:,ep]/ df_std_with_mean.iloc[:, -1] #divide std of this channel for this epoch by mean std of this channel over all epochs

        df_noisy_epoch.iloc[:,ep] = df_noisy_epoch.iloc[:,ep]/ df_std_with_mean.iloc[:, -1] > noisy_channel_multiplier #if std of this channel for this epoch is over the mean std of this channel for all epochs together*multiplyer
        df_flat_epoch.iloc[:,ep] = df_flat_epoch.iloc[:,ep]/ df_std_with_mean.iloc[:, -1] < flat_multiplier #if std of this channel for this epoch is under the mean std of this channel for all epochs together*multiplyer

        # Calculate the number of noisy/flat channels in this epoch:
        df_noisy_epoch.iloc[-3,ep] = df_noisy_epoch.iloc[:-3,ep].sum()
        df_flat_epoch.iloc[-3,ep] = df_flat_epoch.iloc[:-3,ep].sum()

        # Calculate percent of noisy channels in this epoch:
        df_noisy_epoch.iloc[-2,ep] = round(df_noisy_epoch.iloc[:-3,ep].sum()/len(df_noisy_epoch)*100, 1)
        df_flat_epoch.iloc[-2,ep] = round(df_flat_epoch.iloc[:-3,ep].sum()/len(df_flat_epoch)*100, 1)

        # Now check if the epoch has over 70% of noisy/flat channels in it -> it is a noisy/flat epoch:
        df_noisy_epoch.iloc[-1,ep] = df_noisy_epoch.iloc[:-3,ep].sum() > len(df_noisy_epoch)*percent_noisy_flat_allowed/100
        df_flat_epoch.iloc[-1,ep] = df_flat_epoch.iloc[:-3,ep].sum() > len(df_flat_epoch)*percent_noisy_flat_allowed/100


    # Create derivatives:
    noisy_flat_epochs_derivs = [
        QC_derivative(df_epoch_vs_mean, std_or_ptp+'_per_epoch_vs_mean_ratio_'+ch_type, 'df'),
        QC_derivative(df_noisy_epoch, 'Noisy_epochs_on_'+std_or_ptp+'_base_'+ch_type, 'df'),
        QC_derivative(df_flat_epoch, 'Flat_epochs_on_'+std_or_ptp+'_base_'+ch_type, 'df')]

    return noisy_flat_epochs_derivs



# def make_dict_global_std_ptp(std_ptp_params: dict, big_std_with_value_all_data: list, small_std_with_value_all_data: list, channels: list[str], std_or_ptp: str):

from typing import List

def make_dict_global_std_ptp(std_ptp_params: dict, big_std_with_value_all_data: List[dict], small_std_with_value_all_data: List[dict], channels: List[str], std_or_ptp: str):
    # rest of the code here

    """Make a dictionary with global metric content for std or ptp metric.
    Global means that it is calculated over entire data series, not over epochs.
    
    Parameters
    ----------
    std_ptp_params : dict
        dictionary with parameters for std or ptp metric
    big_std_with_value_all_data : list
        list of dictionaries (channel_name: value) for channels with big std or ptp
    small_std_with_value_all_data : list
        list of dictionaries (channel_name: value) for channels with small std or ptp
    channels : list
        list of channel names
    std_or_ptp : str
        'std' or 'ptp': use STD or Peak-to-peak metric

    Returns
    -------
    dict
        dictionary with global metric content for std or ptp metric
    """

    global_details = {
        'noisy_ch': big_std_with_value_all_data,
        'flat_ch': small_std_with_value_all_data}

    metric_global_content = {
        'number_of_noisy_ch': len(big_std_with_value_all_data),
        'percent_of_noisy_ch': round(len(big_std_with_value_all_data)/len(channels)*100, 1), 
        'number_of_flat_ch': len(small_std_with_value_all_data),
        'percent_of_flat_ch': round(len(small_std_with_value_all_data)/len(channels)*100, 1), 
        std_or_ptp+'_lvl': std_ptp_params['std_lvl'],
        'details': global_details}

    return metric_global_content


def make_dict_local_std_ptp(std_ptp_params: dict, noisy_epochs_df: pd.DataFrame, flat_epochs_df: pd.DataFrame):

    """Make a dictionary with local metric content for std or ptp metric.
    Local means that it is calculated over epochs.

    Parameters
    ----------
    std_ptp_params : dict
        dictionary with parameters for std or ptp metric, originally from config file
    noisy_epochs_df : pd.DataFrame
        dataframe with True/False values for noisy channels in each epoch
    flat_epochs_df : pd.DataFrame
        dataframe with True/False values for flat channels in each epoch
    
    Returns
    -------
    dict
        dictionary with local metric content for std or ptp metric

    """
        
    epochs = noisy_epochs_df.columns.tolist()
    epochs = [int(ep) for ep in epochs[:-1]]

    epochs_details = []
    for ep in epochs:
        epochs_details += [{'epoch': ep, 'number_of_noisy_ch': int(noisy_epochs_df.iloc[-3,ep]), 'perc_of_noisy_ch': float(noisy_epochs_df.iloc[-2,ep]), 'epoch_too_noisy': noisy_epochs_df.iloc[-1,ep], 'number_of_flat_ch': int(flat_epochs_df.iloc[-3,ep]), 'perc_of_flat_ch': float(flat_epochs_df.iloc[-2,ep]), 'epoch_too_flat': flat_epochs_df.iloc[-1,ep]}]

    total_num_noisy_ep=len([ep for ep in epochs_details if ep['epoch_too_noisy'] is True])
    total_perc_noisy_ep=round(total_num_noisy_ep/len(epochs)*100)

    total_num_flat_ep=len([ep for ep in epochs_details if ep['epoch_too_flat'] is True])
    total_perc_flat_ep=round(total_num_flat_ep/len(epochs)*100)

    metric_local_content={
        'allow_percent_noisy_flat_epochs': std_ptp_params['allow_percent_noisy_flat_epochs'],
        'noisy_channel_multiplier': std_ptp_params['noisy_channel_multiplier'],
        'flat_multiplier': std_ptp_params['flat_multiplier'],
        'total_num_noisy_ep': total_num_noisy_ep, 
        'total_perc_noisy_ep': total_perc_noisy_ep, 
        'total_num_flat_ep': total_num_flat_ep,
        'total_perc_flat_ep': total_perc_flat_ep,
        'details': epochs_details}

    return metric_local_content



def make_simple_metric_std(std_params:  dict, big_std_with_value_all_data: List[dict], small_std_with_value_all_data: List[dict], channels: List[str], deriv_epoch_std: dict, metric_local_present: bool, m_or_g_chosen: List[dict]):

    """Make simple metric for STD.

    Parameters
    ----------
    std_params : dict
        dictionary with parameters for std metric, originally from config file
    big_std_with_value_all_data : list
        list of dictionaries (channel_name: value) for channels with big std
    small_std_with_value_all_data : list
        list of dictionaries (channel_name: value) for channels with small std
    channels : list
        list of channel names
    deriv_epoch_std : dict
        dictionary with QC_derivative objects containing data frames. 
        Used only data frame 1 and 2. 
        1: contains True/False values for noisy channels in each epoch. 
        2: contains True/False values for flat channels in each epoch.
    metric_local_present : bool
        True if local metric was calculated (epochs present). False if not calculated (epochs were not detected).
    m_or_g_chosen : list
        list of strings with channel types chosen by user: ['mag', 'grad'] or ['mag'] or ['grad']

    Returns
    -------
    dict
        dictionary with simple metric for std/ptp

"""

    metric_global_name = 'STD_all_time_series'
    metric_global_description = 'Standard deviation of the data over the entire time series (not epoched): the number of noisy channels depends on the std of the data over all channels. The std level is set by the user. Noisy channel: The channel where std of data is higher than threshod: mean_over_all_stds_channel + (std_of_all_channels*std_lvl). Flat: where std of data is lower than threshld: mean_over_all_stds_channel - (std_of_all_channels*std_lvl). In details only the noisy/flat channels are listed. Channels with normal std are not listed. If needed to see all channels data - use csv files.'
    metric_local_name = 'STD_epoch'
    if metric_local_present==True:
        metric_local_description = 'Standard deviation of the data over stimulus-based epochs. The epoch is counted as noisy (or flat) if the percentage of noisy (or flat) channels in this epoch is over allow_percent_noisy_flat. this percent is set by user, default=70%. Hense, if no epochs have over 70% of noisy channels - total number of noisy epochs will be 0. Definition of a noisy channel inside of epoch: 1)Take std of data of THIS channel in THIS epoch. 2) Take std of the data of THIS channel for ALL epochs and get mean of it. 3) If (1) is higher than (2)*noisy_channel_multiplier - this channel is noisy.  If (1) is lower than (2)*flat_multiplier - this channel is flat.'
    else:
        metric_local_description = 'Not calculated. No epochs found'

    metric_global_content={'mag': None, 'grad': None}
    metric_local_content={'mag': None, 'grad': None}
    for m_or_g in m_or_g_chosen:

        metric_global_content[m_or_g]=make_dict_global_std_ptp(std_params, big_std_with_value_all_data[m_or_g], small_std_with_value_all_data[m_or_g], channels[m_or_g], 'std')
        
        if metric_local_present is True:
            metric_local_content[m_or_g]=make_dict_local_std_ptp(std_params, deriv_epoch_std[m_or_g][1].content, deriv_epoch_std[m_or_g][2].content)
            #deriv_epoch_std[m_or_g][1].content is df with big std(noisy), df_epoch_std[m_or_g][2].content is df with small std(flat)
        else:
            metric_local_content[m_or_g]=None
    
    simple_metric = simple_metric_basic(metric_global_name, metric_global_description, metric_global_content['mag'], metric_global_content['grad'], metric_local_name, metric_local_description, metric_local_content['mag'], metric_local_content['grad'])

    return simple_metric

#%%
def STD_meg_qc(std_params: dict, channels: dict, chs_by_lobe: dict, dict_epochs_mg: dict, data: mne.io.Raw, m_or_g_chosen: list, verbose_plots: bool):

    """
    Main STD function. Calculates:
    - Std of data for each channel over all time series.
    - Channels with big std (noisy) and small std (flat) over all time series.
    - Std of data for each channel  in each epoch.
    - Epochs with big std (noisy) and small std (flat).

    Parameters
    ----------
    std_params : dict
        dictionary with parameters for std metric, originally from config file
    channels : dict
        dictionary with channel names for each channel type: channels['mag'] or channels['grad']
    chs_by_lobe : dict
        dictionary with channels grouped first by ch type and then by lobe: chs_by_lobe['mag']['Left Occipital'] or chs_by_lobe['grad']['Left Occipital']
    dict_epochs_mg : dict
        dictionary with epochs for each channel type: dict_epochs_mg['mag'] or dict_epochs_mg['grad']
    data : mne.io.Raw
        raw data
    m_or_g_chosen : list
        list of strings with channel types chosen by user: ['mag', 'grad'] or ['mag'] or ['grad']
    verbose_plots : bool
        True for showing plot in notebook.

    Returns
    -------
    list
        list of QC_derivative objects containing data frames and figures for std metric.
    dict
        dictionary with simple metric for std/ptp.
    std_str : str
        String with notes about STD for report
    
    """


    big_std_with_value_all_data = {}
    small_std_with_value_all_data = {}
    std_all_data = {}
    derivs_std = []
    fig_std_epoch0 = []
    fig_std_epoch1 = []
    fig_std_epoch2 = []
    derivs_list = []
    noisy_flat_epochs_derivs={}

    chs_by_lobe_std=chs_by_lobe.copy()
    # copy here, because want to keep original dict unchanged. 
    # In principal it s good to collect all data about channel metrics there BUT if the metrics are run in parallel this might produce conflicts 
    # (without copying  dict can be chanaged both inside+outside this function even when it is not returned.)

    for m_or_g in m_or_g_chosen:

        std_all_data[m_or_g] = get_std_all_data(data, channels[m_or_g])

        #Add std data into channel object inside the chs_by_lobe dictionary:
        for lobe in chs_by_lobe_std[m_or_g]:
            for ch in chs_by_lobe_std[m_or_g][lobe]:
                ch.std_overall = std_all_data[m_or_g][ch.name]
                #print(ch.__dict__) #will print all the info saved in the object, more than just simply printing the object

        #derivs_std += [boxplot_all_time_csv(chs_by_lobe_std[m_or_g], ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)]

        #TODO: make flexible file name
        derivs_std += [boxplot_all_time_csv('/Volumes/M2_DATA/STDs_by_lobe.csv', ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)]

        big_std_with_value_all_data[m_or_g], small_std_with_value_all_data[m_or_g] = get_big_small_std_ptp_all_data(std_all_data[m_or_g], channels[m_or_g], std_params['std_lvl'])

    if dict_epochs_mg['mag'] is not None or dict_epochs_mg['grad'] is not None: #If epochs are present
        for m_or_g in m_or_g_chosen:
            df_std=get_std_epochs(channels[m_or_g], dict_epochs_mg[m_or_g])

            chs_by_lobe_std[m_or_g] = assign_epoched_std_ptp_to_channels(what_data='stds', chs_by_lobe=chs_by_lobe_std[m_or_g], df_std_ptp=df_std) #for easier plotting

            # fig_std_epoch0 += [boxplot_epoched_xaxis_channels(chs_by_lobe_copy[m_or_g], df_std, ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)]
            fig_std_epoch0 += [boxplot_epoched_xaxis_channels_csv('/Volumes/M2_DATA/STDs_by_lobe.csv', ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)]


            #TODO: make flexible file name
            fig_std_epoch1 += [boxplot_epoched_xaxis_epochs_csv('/Volumes/M2_DATA/STDs_by_lobe.csv', ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)]

            #older versions, no color coding:
            #fig_std_epoch1 += [boxplot_epochs(df_mg=df_std, ch_type=m_or_g, what_data='stds', x_axis_boxes='channels', verbose_plots=verbose_plots)] #old version
            #fig_std_epoch2 += [boxplot_epochs(df_mg=df_std, ch_type=m_or_g, what_data='stds', x_axis_boxes='epochs', verbose_plots=verbose_plots)]


            noisy_flat_epochs_derivs[m_or_g] = get_noisy_flat_std_ptp_epochs(df_std, m_or_g, 'std', std_params['noisy_channel_multiplier'], std_params['flat_multiplier'], std_params['allow_percent_noisy_flat_epochs'])
            derivs_list += noisy_flat_epochs_derivs[m_or_g]

        metric_local=True
        std_str = ''
    else:
        metric_local=False
        std_str = 'STD per epoch can not be calculated because no events are present. Check stimulus channel.'
        print('___MEG QC___: ', std_str)


    simple_metric_std = make_simple_metric_std(std_params, big_std_with_value_all_data, small_std_with_value_all_data, channels, noisy_flat_epochs_derivs, metric_local, m_or_g_chosen)

    derivs_std += fig_std_epoch0+fig_std_epoch1 + fig_std_epoch2 + derivs_list 
    #each of them saved into a separate list and only at the end put together because this way they keep the right order: 
    #first everything about mags, then everything about grads. - in this ordr they ll be added to repot. 
    #Report funcion doesnt have sorting inside 1 measurement. It only separates derivs by measurement.

    #Extract chs_by_lobe into a data frame
    f_path = chs_dict_to_csv(chs_by_lobe,  file_name_prefix = 'STDs')

    return derivs_std, simple_metric_std, std_str, f_path

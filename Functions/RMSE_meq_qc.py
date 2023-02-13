import numpy as np
import pandas as pd
import mne
from universal_plots import boxplot_std_hovering_plotly, boxplot_channel_epoch_hovering_plotly, QC_derivative
from universal_html_report import simple_metric_basic

# In[2]:

def RMSE(data_m_or_g: np.array or list):
    ''' RMSE - general root means squared error function to use in other functions of this module.
    Alternatively std could be used, but the calculation time of std is longer, result is the same.
    
    Args:
    data_m_or_g (np.array or list): data for magnetometer or gradiometer given as np array or list 
        (it can be 1 or several channels data  as 2 dimentional array or as list of lists)
        
    Returns:
    rmse_np (np.array): rmse as numpy array (1-dimentional if 1 channel was given, 2-dim if more channels)'''

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

def get_rmse_all_data(data: mne.io.Raw, channels: list):

    '''Calculate RMSE for each channel - for the entire time duration'''
    data_channels=data.get_data(picks = channels)

    std_channels = RMSE(data_channels)

    return std_channels

def get_big_small_std_ptp_all_data(peak_ampl_channels, channels: list, std_ptp_lvl: float):

    '''Function calculates peak-to-peak amplitude over the entire data set for every channel (mag or grad).

    Args:
    mg_names (list of tuples): channel name + its index
    df_epoch_mg (pd. Dataframe): data frame containing data for all epochs for mag  or grad
    sfreq: sampling frequency of data. Attention to which data is used! original or resampled.
    n_events (int): number of events in this peace of data
    ptp_thresh_lvl (float): defines how high or low need to peak to be to be detected, this can also be changed into a sigle value later
        used in: max(data_ch_epoch) - min(data_ch_epoch)) / ptp_thresh_lvl 
    max_pair_dist_sec (float): maximum distance in seconds which is allowed for negative+positive peaks to be detected as a pair 

    Returns:
    peak_ampl (list): contains the mean peak-to-peak amplitude for all time for each channel

    '''
    
    
    ## Check if channel data is within std level of PtP amplitudes.
    std_of_ptp_channels=np.std(peak_ampl_channels)
    mean_ptp_channels=np.mean(peak_ampl_channels)

    # Find the index of channels with largest and smallest std:
    ch_ind_large_std = [index for (index, item) in enumerate(peak_ampl_channels) if item > mean_ptp_channels + std_ptp_lvl*std_of_ptp_channels] #find index with largest std
    ch_ind_small_std = [index for (index, item) in enumerate(peak_ampl_channels) if item < mean_ptp_channels - std_ptp_lvl*std_of_ptp_channels] #find index with smallest std

    #make dictionaries with channel names and std values:
    big_ptp_with_value = {}
    for index in ch_ind_large_std:
        ch_name = np.array(channels)[index] #find the names of the channels with large std 
        ch_std = peak_ampl_channels[index]
        big_ptp_with_value[ch_name] = ch_std

    small_ptp_with_value = {}
    for index in ch_ind_small_std:
        ch_name = np.array(channels)[index]
        ch_std = peak_ampl_channels[index]
        small_ptp_with_value[ch_name] = ch_std

    return big_ptp_with_value, small_ptp_with_value

#%%
def get_std_epochs(channels: list, epochs_mg: mne.Epochs):

    ''' --fastest  and cleanest version, no need to use data frames--

    Calculate std for multiple epochs for a list of channels.
    Used as internal function in RMSE_meg_epoch

    Args:
    channels (list): channel name, 
    df_mg (pd.DataFrame): data frame containing data for all epochs for mag or for grad
    epoch_numbers (list): list of epoch numbers

    Returns:
    df_std_mg (pd.DataFrame): data frame containing stds for all epoch for each channel
    '''
    
    dict_ep = {}

    #get 1 epoch, 1 channel and calculate rmse of its data:
    for ep in range(0, len(epochs_mg)):
        rmse_epoch=[]
        for ch_name in channels: 
            data_ch_epoch=epochs_mg[ep].get_data(picks=ch_name)[0][0] 
            #[0][0] is because get_data creats array in array in array, it expects several epochs, several channels, but we only need  one.
            rmse_ch_ep = RMSE(data_ch_epoch)
            rmse_epoch.append(np.float64(rmse_ch_ep))

            #std_ch_ep = np.std(data_ch_epoch) #if want to use std instead
        dict_ep[ep] = rmse_epoch

    return pd.DataFrame(dict_ep, index=channels)


def get_large_small_RMSE_PtP_epochs(df_std: pd.DataFrame, ch_type: str, std_lvl: int, epochs_mg: mne.Epochs, std_or_ptp: str):

    '''
    - Calculate std for every separate epoch of a given list of channels
    - Find which channels in which epochs have too high/too small stds
    - Create MEG_QC_derivative as dfs

    Args:
    channels (list of tuples): channel name + its index as list,
    std_lvl (int): how many standard deviations from the mean are acceptable in the data. 
        data variability over this setting will be concedered as too too noisy, under -std_lvl as too flat 
    epoch_numbers(list): list of event numbers, 
    df_epochs (pd.DataFrame): data frame containing stds for all epoch for each channel

    Returns:
    dfs_deriv(list of 3 pd.DataFrame): 1 data frame containing std data for each epoch/each channel, 1df with too high and 1 too low stds.

    '''

    # Check (which epochs for which channel) are over set STD_level (1 or 2, 3, etc STDs) for this epoch for all channels

    std_std_per_epoch=[]
    mean_std_per_epoch=[]

    for ep in range(0, len(epochs_mg)):
        std_std_per_epoch.append(np.std(df_std.iloc[:, ep])) #std of stds of all channels of every single epoch
        mean_std_per_epoch.append(np.mean(df_std.iloc[:, ep])) #mean of stds of all channels of every single epoch

    df_ch_ep_large_std=df_std.copy()
    df_ch_ep_small_std=df_std.copy()

    # Now see which channles in epoch are over std_level or under -std_level:
    for ep in range(0, len(epochs_mg)):  
        df_ch_ep_large_std.iloc[:,ep] = df_ch_ep_large_std.iloc[:,ep] > mean_std_per_epoch[ep]+std_lvl*std_std_per_epoch[ep] 
        df_ch_ep_small_std.iloc[:,ep] = df_ch_ep_small_std.iloc[:,ep] < mean_std_per_epoch[ep]-std_lvl*std_std_per_epoch[ep] 

    # Create derivatives:
    dfs_deriv = [
        QC_derivative(df_std, std_or_ptp+'_per_epoch_'+ch_type, 'df'),
        QC_derivative(df_ch_ep_large_std, 'Large_'+std_or_ptp+'_per_epoch_'+ch_type, 'df'),
        QC_derivative(df_ch_ep_small_std, 'Small_'+std_or_ptp+'_per_epoch_'+ch_type, 'df')]


    return dfs_deriv

#%% All about simple metrc jsons:

def make_dict_global_rmse_ptp(std_lvl, big_rmse_with_value_all_data, small_rmse_with_value_all_data, channels, std_or_ptp):

    global_details = {
        'noisy_ch': big_rmse_with_value_all_data,
        'flat_ch': small_rmse_with_value_all_data}

    metric_global_content = {
        'number_of_noisy_ch': len(big_rmse_with_value_all_data),
        'percent_of_noisy_ch': round(len(big_rmse_with_value_all_data)/len(channels)*100, 1), 
        'number_of_flat_ch': len(small_rmse_with_value_all_data),
        'percent_of_flat_ch': round(len(small_rmse_with_value_all_data)/len(channels)*100, 1), 
        std_or_ptp+'_lvl': std_lvl,
        'Details': global_details}

    return metric_global_content


def make_dict_local_rmse_ptp(std_lvl, df_std_noisy: pd.DataFrame, df_std_flat: pd.DataFrame, epochs_mg, std_or_ptp, allow_percent_noisy: float=70, allow_percent_flat: float=70):
        
    eps=[ep for ep in range(0, len(epochs_mg))] #list of epoch numbers

    epochs_details = []
    for ep in eps:
        number_noisy_ch=int(df_std_noisy.loc[:,ep].sum()) 
        number_flat_ch=int(df_std_flat.loc[:,ep].sum()) 
        #count the number of TRUE in data frame. meaning the number of channels with high std for given epoch
        #converted to int becuse json doesnt undertand numpy.int64

        perc_noisy_ch=round(number_noisy_ch/len(df_std_noisy)*100, 1)
        perc_flat_ch=round(number_flat_ch/len(df_std_flat)*100, 1)

        if perc_noisy_ch>allow_percent_noisy:
            ep_too_noisy=True
        else:
            ep_too_noisy=False

        if perc_flat_ch>allow_percent_flat:
            ep_too_flat=True
        else:
            ep_too_flat=False
            
        epochs_details += [{'epoch': ep, 'number_of_noisy_ch': number_noisy_ch, 'perc_of_noisy_ch': perc_noisy_ch, 'epoch_too_noisy': ep_too_noisy, 'number_of_flat_ch': number_flat_ch, 'perc_of_flat_ch': perc_flat_ch, 'epoch_too_flat': ep_too_flat}]

    total_num_noisy_ep=sum([ep for ep in epochs_details if ep['epoch_too_noisy'] is True])
    total_perc_noisy_ep=round(total_num_noisy_ep/len(eps)*100)

    total_num_flat_ep=sum([ep for ep in epochs_details if ep['epoch_too_flat'] is True])
    total_perc_flat_ep=round(total_num_flat_ep/len(eps)*100)

    metric_local_content={
        std_or_ptp+'_lvl': std_lvl,
        'total_num_noisy_ep': total_num_noisy_ep, 
        'total_perc_noisy_ep': total_perc_noisy_ep, 
        'total_num_flat_ep': total_num_flat_ep,
        'total_perc_flat_ep': total_perc_flat_ep,
        'Details': epochs_details}

    return metric_local_content



def make_simple_metric_rmse(std_lvl, big_rmse_with_value_all_data, small_rmse_with_value_all_data, channels, deriv_epoch_rmse, dict_epochs_mg, allow_percent_noisy, allow_percent_flat, metric_local, m_or_g_chosen):

    """Make simple metric for RMSE.

    Parameters
    ----------
    noise_ampl_global : dict
        DESCRIPTION.
    noise_ampl_relative_to_all_signal_global : dict
        DESCRIPTION.
    noise_peaks_global : dict
        DESCRIPTION.
    noise_ampl_local : dict
        DESCRIPTION.
    noise_ampl_relative_to_all_signal_local : dict
        DESCRIPTION.
    noise_peaks_local : dict
        DESCRIPTION.
    m_or_g_chosen : list
        DESCRIPTION.
    freqs : dict
        DESCRIPTION.

    Returns
    -------
    simple_metric: dict
        DESCRIPTION.
    

"""

    metric_global_name = 'RMSE_all'
    metric_global_description = 'Standard deviation of the data over the entire time series (not epoched): the number of noisy channels depends on the std of the data over all channels. The std level is set by the user. Threshold = mean_over_ all_data + (std_of_all_data*std_lvl). The channel where data is higher than this threshod is considered as noisy. Same: if the std of some channel is lower than -threshold, this channel is considered as flat. In details only the noisy/flat channels are listed. Channels with normal std are not listed. If needed to see all channels data - use csv files.'
    metric_local_name = 'RMSE_epoch'
    if metric_local==True:
        metric_local_description = 'Standard deviation of the data over stimulus-based epochs. The epoch is counted as noisy (or flat) if the percentage of noisy (or flat) channels in this epoch is over allow_percent_noisy (or allow_percent_flat). this percent is set by user, default=70%. Hense, if no epochs have over 70% of noisy channels - total number of noisy epochs will be 0. Definition of a noisy channel here: if std of the chanels data in given epoch is higher than threshold - this channel is noisy. Threshold is:  mean of all channels data in this epoch + (std of all channels data in this epoch * std_lvl). std_lvl is set by user.'
    else:
        metric_local_description = 'Not calculated. No epochs found'

    metric_global_content={'mag': None, 'grad': None}
    metric_local_content={'mag': None, 'grad': None}
    for m_or_g in m_or_g_chosen:

        metric_global_content[m_or_g]=make_dict_global_rmse_ptp(std_lvl, big_rmse_with_value_all_data[m_or_g], small_rmse_with_value_all_data[m_or_g], channels[m_or_g], 'std')
        
        if metric_local is True:
            metric_local_content[m_or_g]=make_dict_local_rmse_ptp(std_lvl, deriv_epoch_rmse[m_or_g][1].content, deriv_epoch_rmse[m_or_g][2].content, dict_epochs_mg[m_or_g], 'std', allow_percent_noisy, allow_percent_flat)
            #deriv_epoch_rmse[m_or_g][1].content is df with big rmse(noisy), df_epoch_rmse[m_or_g][2].content is df with small rmse(flat)
        else:
            metric_local_content[m_or_g]=None
    
    simple_metric = simple_metric_basic(metric_global_name, metric_global_description, metric_global_content['mag'], metric_global_content['grad'], metric_local_name, metric_local_description, metric_local_content['mag'], metric_local_content['grad'])

    return simple_metric

#%%
def RMSE_meg_qc(rmse_params:  dict, channels: dict, dict_epochs_mg: dict, data: mne.io.Raw, m_or_g_chosen: list):

    """Main RMSE function
    
    Args:
    channels (dict): channel names
    dict_of_dfs_epoch (dict of pd.DataFrame-s): data frames with epoched data per channels
    data(mne.io.Raw): raw non-epoched data
    m_or_g_chosen (list): mag or grad or both are chosen for analysis
    
    Returns:
    derivs_rmse: list of tuples QC_derivative objects: figures and data frames. Exact number of derivatives depends on: 
        - was data epoched (*2 derivs) or not (*1 derivs)
        - were both mag and grad analyzed (*2 derivs) or only one type of channels(*1 derivs)
    big_rmse_with_value_all_data + small_rmse_with_value_all_data: 2 lists of tuples: channel+ std value of calculsted value is too high and to low"""

    big_rmse_with_value_all_data = {}
    small_rmse_with_value_all_data = {}
    rmse = {}
    derivs_rmse = []
    fig_std_epoch_with_name = []
    derivs_list = []

    for m_or_g in m_or_g_chosen:

        rmse[m_or_g] = get_rmse_all_data(data, channels[m_or_g])
        big_rmse_with_value_all_data[m_or_g], small_rmse_with_value_all_data[m_or_g] = get_big_small_std_ptp_all_data(rmse[m_or_g], channels[m_or_g], rmse_params['std_lvl'])
      
        derivs_rmse += [boxplot_std_hovering_plotly(std_data=rmse[m_or_g], ch_type=m_or_g, channels=channels[m_or_g], what_data='stds')]

    deriv_epoch_rmse={}
    if dict_epochs_mg['mag'] is not None or dict_epochs_mg['grad'] is not None:
        for m_or_g in m_or_g_chosen:
            df_std=get_std_epochs(channels[m_or_g], dict_epochs_mg[m_or_g])
            deriv_epoch_rmse[m_or_g] = get_large_small_RMSE_PtP_epochs(df_std, m_or_g, rmse_params['std_lvl'], dict_epochs_mg[m_or_g], 'std') 
            derivs_list += deriv_epoch_rmse[m_or_g] # dont delete/change line, otherwise it will mess up the order of df_epoch_rmse list at the next line.

            fig_std_epoch_with_name += [boxplot_channel_epoch_hovering_plotly(df_mg=deriv_epoch_rmse[m_or_g][0].content, ch_type=m_or_g, what_data='stds')]
            #df_epoch_rmse[0].content - df with stds per channel per epoch, other 2 dfs have True/False values calculated on base of 1st df.
        metric_local=True
    else:
        metric_local=False
        print('___MEG QC___: ', 'RMSE per epoch can not be calculated because no events are present. Check stimulus channel.')

    simple_metric_rmse = make_simple_metric_rmse(rmse_params['std_lvl'], big_rmse_with_value_all_data, small_rmse_with_value_all_data, channels, deriv_epoch_rmse, dict_epochs_mg, rmse_params['allow_percent_noisy'], rmse_params['allow_percent_flat'], metric_local, m_or_g_chosen)
    
    derivs_rmse += fig_std_epoch_with_name + derivs_list 
    return derivs_rmse, simple_metric_rmse

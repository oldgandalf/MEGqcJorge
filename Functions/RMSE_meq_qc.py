import numpy as np
import pandas as pd
import mne
from universal_plots import boxplot_std_hovering_plotly, boxplot_channel_epoch_hovering_plotly, QC_derivative, boxplot_epochs
from universal_html_report import simple_metric_basic

# In[2]:

def RMSE(data_m_or_g: np.array or list):
    ''' RMSE - general root means squared error function to use in other functions of this module.
    Used before as alternative to std calculation, as my func was faster. Currently not used, as now std is slightly faster.
    
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

    #std_channels = RMSE(data_channels)

    std_channels = np.std(data_channels, axis=1)


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
    
    ## Check if channel data is within std level of PtP/RMSE.
    std_of_measure_channels=np.std(peak_ampl_channels)
    mean_of_measure_channels=np.mean(peak_ampl_channels)

    print('___MEG QC___: ', mean_of_measure_channels + std_ptp_lvl*std_of_measure_channels, ' threshold for NOISY. ')
    print('___MEG QC___: ', mean_of_measure_channels - std_ptp_lvl*std_of_measure_channels, ' threshold for FLAT. ')

    # Find the index of channels with biggest and smallest std:
    ch_ind_big_measure = [index for (index, item) in enumerate(peak_ampl_channels) if item > mean_of_measure_channels + std_ptp_lvl*std_of_measure_channels] #find index with bigst std
    ch_ind_small_measure = [index for (index, item) in enumerate(peak_ampl_channels) if item < mean_of_measure_channels - std_ptp_lvl*std_of_measure_channels] #find index with smallest std

    #make dictionaries with channel names and their std values:
    noisy_channels = {}
    flat_channels = {}

    for index in ch_ind_big_measure:
        ch_name = np.array(channels)[index]
        noisy_channels[ch_name] = peak_ampl_channels[index]

    for index in ch_ind_small_measure:
        ch_name = np.array(channels)[index]
        flat_channels[ch_name] = peak_ampl_channels[index]

    return noisy_channels, flat_channels

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

            #rmse_ch_ep = RMSE(data_ch_epoch)
            rmse_ch_ep = np.std(data_ch_epoch) #if want to use std instead
            rmse_epoch.append(np.float64(rmse_ch_ep))

        dict_ep[ep] = rmse_epoch

    return pd.DataFrame(dict_ep, index=channels)


def get_big_small_RMSE_PtP_epochs(df_std: pd.DataFrame, ch_type: str, std_lvl: int, std_or_ptp: str):

    ''' NOT USED ANY MORE

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

def get_noisy_flat_rmse_ptp_epochs(df_std: pd.DataFrame, ch_type: str, std_or_ptp: str, noisy_multiplier: float, flat_multiplier: float, percent_noisy_flat_allowed: float):
    
    """Compare the std of this channel for this epoch (df_std) TO the mean STD of this particular channel over all time. (or over all epchs!)
    Use some multiplier to figure out by how much it is noisier."""

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
    df_noisy_epoch.loc['number noisy channels'] = None
    df_noisy_epoch.loc['% noisy channels'] = None
    df_noisy_epoch.loc['noisy > %s perc' % percent_noisy_flat_allowed] = None

    df_flat_epoch.loc['number flat channels'] = None
    df_flat_epoch.loc['% flat channels'] = None
    df_flat_epoch.loc['flat < %s perc' % percent_noisy_flat_allowed] = None

    for ep in epochs:  

        df_epoch_vs_mean.iloc[:,ep] = df_epoch_vs_mean.iloc[:,ep]/ df_std_with_mean.iloc[:, -1] #divide std of this channel for this epoch by mean std of this channel over all epochs

        df_noisy_epoch.iloc[:,ep] = df_noisy_epoch.iloc[:,ep]/ df_std_with_mean.iloc[:, -1] > noisy_multiplier #if std of this channel for this epoch is over the mean std of this channel for all epochs together*multiplyer
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



def make_dict_global_rmse_ptp(rmse_params: dict, big_rmse_with_value_all_data, small_rmse_with_value_all_data, channels, std_or_ptp):

    global_details = {
        'noisy_ch': big_rmse_with_value_all_data,
        'flat_ch': small_rmse_with_value_all_data}

    metric_global_content = {
        'number_of_noisy_ch': len(big_rmse_with_value_all_data),
        'percent_of_noisy_ch': round(len(big_rmse_with_value_all_data)/len(channels)*100, 1), 
        'number_of_flat_ch': len(small_rmse_with_value_all_data),
        'percent_of_flat_ch': round(len(small_rmse_with_value_all_data)/len(channels)*100, 1), 
        std_or_ptp+'_lvl': rmse_params['std_lvl'],
        'details': global_details}

    return metric_global_content


def make_dict_local_rmse_ptp(rmse_params: dict, noisy_epochs_df: pd.DataFrame, flat_epochs_df: pd.DataFrame):
        
    epochs = noisy_epochs_df.columns.tolist()
    epochs = [int(ep) for ep in epochs[:-1]]

    epochs_details = []
    for ep in epochs:
        epochs_details += [{'epoch': ep, 'number_of_noisy_ch': int(noisy_epochs_df.iloc[-3,ep]), 'perc_of_noisy_ch': float(noisy_epochs_df.iloc[-2,ep]), 'epoch_too_noisy': noisy_epochs_df.iloc[-1,ep], 'number_of_flat_ch': int(flat_epochs_df.iloc[-3,ep]), 'perc_of_flat_ch': float(flat_epochs_df.iloc[-2,ep]), 'epoch_too_flat': flat_epochs_df.iloc[-1,ep]}]

    total_num_noisy_ep=sum([ep for ep in epochs_details if ep['epoch_too_noisy'] is True])
    total_perc_noisy_ep=round(total_num_noisy_ep/len(epochs)*100)

    total_num_flat_ep=sum([ep for ep in epochs_details if ep['epoch_too_flat'] is True])
    total_perc_flat_ep=round(total_num_flat_ep/len(epochs)*100)

    metric_local_content={
        'allow_percent_noisy_flat_epochs': rmse_params['allow_percent_noisy_flat_epochs'],
        'noisy_multiplier': rmse_params['noisy_multiplier'],
        'flat_multiplier': rmse_params['flat_multiplier'],
        'total_num_noisy_ep': total_num_noisy_ep, 
        'total_perc_noisy_ep': total_perc_noisy_ep, 
        'total_num_flat_ep': total_num_flat_ep,
        'total_perc_flat_ep': total_perc_flat_ep,
        'details': epochs_details}

    return metric_local_content



def make_simple_metric_rmse(rmse_params:  dict, big_rmse_with_value_all_data, small_rmse_with_value_all_data, channels, deriv_epoch_rmse, metric_local, m_or_g_chosen):

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
    metric_global_description = 'Standard deviation of the data over the entire time series (not epoched): the number of noisy channels depends on the std of the data over all channels. The std level is set by the user. Noisy channel: The channel where std of data is higher than threshod: mean_over_all_stds_channel + (std_of_all_channels*std_lvl). Flat: where std of data is lower than threshld: mean_over_all_stds_channel - (std_of_all_channels*std_lvl). In details only the noisy/flat channels are listed. Channels with normal std are not listed. If needed to see all channels data - use csv files.'
    metric_local_name = 'RMSE_epoch'
    if metric_local==True:
        metric_local_description = 'Standard deviation of the data over stimulus-based epochs. The epoch is counted as noisy (or flat) if the percentage of noisy (or flat) channels in this epoch is over allow_percent_noisy_flat. this percent is set by user, default=70%. Hense, if no epochs have over 70% of noisy channels - total number of noisy epochs will be 0. Definition of a noisy channel inside of epoch: 1)Take std of data of THIS channel in THIS epoch. 2) Take std of the data of THIS channel for ALL epochs and get mean of it. 3) If (1) is higher than (2)*noisy_multiplier - this channel is noisy.  If (1) is lower than (2)*flat_multiplier - this channel is flat.'
    else:
        metric_local_description = 'Not calculated. No epochs found'

    metric_global_content={'mag': None, 'grad': None}
    metric_local_content={'mag': None, 'grad': None}
    for m_or_g in m_or_g_chosen:

        metric_global_content[m_or_g]=make_dict_global_rmse_ptp(rmse_params, big_rmse_with_value_all_data[m_or_g], small_rmse_with_value_all_data[m_or_g], channels[m_or_g], 'std')
        
        if metric_local is True:
            metric_local_content[m_or_g]=make_dict_local_rmse_ptp(rmse_params, deriv_epoch_rmse[m_or_g][1].content, deriv_epoch_rmse[m_or_g][2].content)
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
    fig_std_epoch = []
    fig_std_epoch2 = []
    derivs_list = []
    deriv_epoch_rmse={}
    noisy_flat_epochs_derivs={}

    for m_or_g in m_or_g_chosen:

        rmse[m_or_g] = get_rmse_all_data(data, channels[m_or_g])
        big_rmse_with_value_all_data[m_or_g], small_rmse_with_value_all_data[m_or_g] = get_big_small_std_ptp_all_data(rmse[m_or_g], channels[m_or_g], rmse_params['std_lvl'])
      
        derivs_rmse += [boxplot_std_hovering_plotly(std_data=rmse[m_or_g], ch_type=m_or_g, channels=channels[m_or_g], what_data='stds')]


    if dict_epochs_mg['mag'] is not None or dict_epochs_mg['grad'] is not None:
        for m_or_g in m_or_g_chosen:
            df_std=get_std_epochs(channels[m_or_g], dict_epochs_mg[m_or_g])

            fig_std_epoch += [boxplot_channel_epoch_hovering_plotly(df_mg=df_std, ch_type=m_or_g, what_data='stds')]
            fig_std_epoch2 += [boxplot_epochs(df_mg=df_std, ch_type=m_or_g, what_data='stds')]

            #deriv_epoch_rmse[m_or_g] = get_big_small_RMSE_PtP_epochs(df_std, m_or_g, rmse_params['std_lvl'], 'std') 
            #derivs_list += deriv_epoch_rmse[m_or_g] # dont delete/change line, otherwise it will mess up the order of df_epoch_rmse list at the next line.

            noisy_flat_epochs_derivs[m_or_g] = get_noisy_flat_rmse_ptp_epochs(df_std, m_or_g, 'std', rmse_params['noisy_multiplier'], rmse_params['flat_multiplier'], rmse_params['allow_percent_noisy_flat_epochs'])
            derivs_list += noisy_flat_epochs_derivs[m_or_g]

            #df_epoch_rmse[0].content - df with stds per channel per epoch, other 2 dfs have True/False values calculated on base of 1st df.
        metric_local=True
    else:
        metric_local=False
        print('___MEG QC___: ', 'RMSE per epoch can not be calculated because no events are present. Check stimulus channel.')

    simple_metric_rmse = make_simple_metric_rmse(rmse_params, big_rmse_with_value_all_data, small_rmse_with_value_all_data, channels, noisy_flat_epochs_derivs, metric_local, m_or_g_chosen)
    
    derivs_rmse += fig_std_epoch + fig_std_epoch2 + derivs_list 
    return derivs_rmse, simple_metric_rmse

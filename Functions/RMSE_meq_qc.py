import numpy as np
import pandas as pd
import mne
from universal_plots import boxplot_std_hovering_plotly, boxplot_channel_epoch_hovering_plotly, QC_derivative

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
        print('Only 1 or 2 dimentional data is accepted, not more!')
        return

    rmse_np=np.array(rmse_list) #conver to numpy array

    return rmse_np

# In[6]:
def RMSE_meg_all(data: mne.io.Raw, channels: list, std_lvl: int): 

    '''Root mean squared error calculated over ALL data (not epoched)
    
    Args:
    data (mne.raw): data in raw format
    channels (list of tuples): list of channel names + their indexes (mags or grads)
    std_lvl (int): how many standard deviations from the mean are acceptable. 
        data variability over this setting will be considered as too too noisy, under -std_lvl as too flat 
    
    Returns:
    big_std_with_value (list of tuples): list of channels with too high std value, 
    small_std_with_value (list of tuples): list of channels with too low std value
    std_channels (np.array): std values for channels
    '''

    data_channels=data.get_data(picks = channels)

    # Calculate STD or RMSE of each channel

    #Calculate RMSE for each channel (separated mags and grads) - for the entire time duration:
    std_channels = RMSE(data_channels)

    #STD (if wanna use insted of RMSE. it will exactly replace the RMSE function above):
    #std_channels=np.std(data_channels, axis=1) 

    # Check if channel data is within std_lvl over all channels.
    std_std_channels=np.std(std_channels)
    mean_std_channels=np.mean(std_channels)

    ch_ind_large_std= np.where(std_channels > mean_std_channels+std_lvl*std_std_channels) #find channels with largest std
    ch_ind_small_std= np.where(std_channels < mean_std_channels-std_lvl*std_std_channels) #findchannels with smallest std

    channel_big_std_names=np.array(channels)[ch_ind_large_std] #find the name of the channel with largest std 
    channel_small_std_names=np.array(channels)[ch_ind_small_std]


    def Channels_with_nonnormal_stds(ch_ind, all_stds_m_or_g, channels_big_std_names):
        #This function simply makes a list of tuples. Each tuple is: name of channel, std value.
        #Each tuple represents channel with too big or too small std, calculated over whole data.
        channel_big_std_vals=all_stds_m_or_g[ch_ind]
        nonnormal_std_with_value=[]
        for id, val in enumerate (ch_ind[0]):
            new_tuple=(channels_big_std_names[id],  channel_big_std_vals[id])
            nonnormal_std_with_value.append(new_tuple)
        return(nonnormal_std_with_value)

    big_std_with_value=Channels_with_nonnormal_stds(ch_ind_large_std, std_channels, channel_big_std_names)
    small_std_with_value=Channels_with_nonnormal_stds(ch_ind_small_std, std_channels, channel_small_std_names)
        
    #Return the channel names with STD over the set STD level and under the set negative STD level.
    return big_std_with_value, small_std_with_value, std_channels


# In[11]:

def std_of_epochs_slow(mg_names: list, epochs_mg: mne.Epochs, df_mg: pd.DataFrame,):

    '''Calculate std for multiple epochs for a list of channels.
    Used as internal function in RMSE_meg_epoch

    Args:
    mg_names (list of tuples): channel name + its index, 
    df_mg (pd.DataFrame): data frame containing data for all epochs for mags or for grads
    epoch_numbers (list): list of epoch numbers

    Returns:
    df_std_mg (pd.DataFrame): data frame containing stds for all epoch for each channel
    '''
    
    dict_mg = {}

    for ep in range(0, len(epochs_mg)):
        rows_for_ep = [row for row in df_mg.iloc if row.epoch == ep] #take all rows of 1 epoch, all channels.
        #std_epoch = [] #list with stds
        rmse_epoch=[]

        for ch_name in mg_names: #loop over channel names
            data_ch_epoch = [row_mg[ch_name] for row_mg in rows_for_ep] #take the data for 1 epoch for 1 channel
            rmse_epoch.append(np.float64(RMSE(data_ch_epoch)))

            #std_epoch.append(np.std(data_ch_epoch)) #if want to use std instead - but it will take longer.
            
        dict_mg[ep] = rmse_epoch

    df_std_mg = pd.DataFrame(dict_mg, index=mg_names)

    return(df_std_mg)

#%%
def std_of_epochs_dfs_fast(mg_names: list, epochs_mg: mne.Epochs, df_mg: pd.DataFrame):

    '''Calculate std for multiple epochs for a list of channels.
    Used as internal function in RMSE_meg_epoch

    Args:
    mg_names (list of tuples): channel name + its index, 
    df_mg (pd.DataFrame): data frame containing data for all epochs for mags or for grads
    epoch_numbers (list): list of epoch numbers

    Returns:
    df_std_mg (pd.DataFrame): data frame containing stds for all epoch for each channel
    '''
    
    dict_mg = {}

    for ep in range(0, len(epochs_mg)):
        df_one_ep=df_mg.loc[df_mg['epoch'] == ep]
        
        #std_epoch = [] #list with stds
        rmse_epoch=[]

        for ch_name in mg_names: #loop over channel names
            data_ch_epoch=list(df_one_ep.loc[:,ch_name])
            rmse_ch_ep = RMSE(data_ch_epoch)
            rmse_ch_ep=np.float64(rmse_ch_ep) #convert from ndarray to float
            rmse_epoch.append(rmse_ch_ep)

            #std_ch_ep = np.std(data_ch_epoch) #if want to use std instead
            
        dict_mg[ep] = rmse_epoch

    df_std_mg = pd.DataFrame(dict_mg, index=mg_names)

    return(df_std_mg)

#%%

def std_of_epochs(channels: list, epochs_mg: mne.Epochs, df_mg: pd.DataFrame):

    ''' --fastest  and cleanest version, no need to use data frames--

    Calculate std for multiple epochs for a list of channels.
    Used as internal function in RMSE_meg_epoch

    Args:
    channels (list): channel name, 
    df_mg (pd.DataFrame): data frame containing data for all epochs for mags or for grads
    epoch_numbers (list): list of epoch numbers

    Returns:
    df_std_mg (pd.DataFrame): data frame containing stds for all epoch for each channel
    '''
    
    dict_mg = {}

    #get 1 epoch, 1 channel and calculate rmse of its data:
    for ep in range(0, len(epochs_mg)):
        rmse_epoch=[]
        for ch_name in channels: 
            data_ch_epoch=epochs_mg[ep].get_data(picks=ch_name)[0][0] 
            #[0][0] is because get_data creats array in array in array, it expects several epochs, several channels, but we only need  one.
            rmse_ch_ep = RMSE(data_ch_epoch)
            rmse_epoch.append(np.float64(rmse_ch_ep))

            #std_ch_ep = np.std(data_ch_epoch) #if want to use std instead
            

        dict_mg[ep] = rmse_epoch

    df_std_mg = pd.DataFrame(dict_mg, index=channels)

    return(df_std_mg)


#%% STD over epochs: use 2 separate data frames for mags and grads in calculations:

def RMSE_meg_epoch(ch_type: str, channels: list, std_lvl: int, epochs_mg: mne.Epochs, df_epochs: pd.DataFrame):

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

    # 1) Find std for every channel for every epoch:

    df_std=std_of_epochs(channels=channels, epochs_mg=epochs_mg, df_mg=df_epochs)

    # 2) Check (which epochs for which channel) are over set STD_level (1 or 2, 3, etc STDs) for this epoch for all channels

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

    # 3) Create derivatives:
    dfs_deriv = [
        QC_derivative(df_std,'std_per_epoch_'+ch_type, None, 'df'),
        QC_derivative(df_ch_ep_large_std, 'Large_std_per_epoch_'+ch_type, None, 'df'),
        QC_derivative(df_ch_ep_small_std, 'Small_std_per_epoch_'+ch_type, None, 'df')]

    return dfs_deriv

#%%
def RMSE_meg_qc(rmse_params:  dict, channels: dict, dict_epochs_mg: dict, dict_of_dfs_epoch:dict, data: mne.io.Raw, m_or_g_chosen: list):

    """Main RMSE function
    
    Args:
    channels (dict): channel names
    dict_of_dfs_epoch (dict of pd.DataFrame-s): data frames with epoched data per channels
    data(mne.io.Raw): raw non-epoched data
    m_or_g_chosen (list): mags or grads or both are chosen for analysis
    
    Returns:
    derivs_rmse: list of tuples QC_derivative objects: figures and data frames. Exact number of derivatives depends on: 
        - was data epoched (*2 derivs) or not (*1 derivs)
        - were both mags and grads analyzed (*2 derivs) or only one type of channels(*1 derivs)
    big_std_with_value_all_data + small_std_with_value_all_data: 2 lists of tuples: channel+ std value of calculsted value is too high and to low"""


    m_or_g_title = {
    'grads': 'Gradiometers',
    'mags': 'Magnetometers'}

    big_std_with_value_all_data = {}
    small_std_with_value_all_data = {}
    rmse = {}
    derivs_rmse = []
    fig_std_epoch_with_name = []
    dfs_list = []

    for m_or_g in m_or_g_chosen:

        big_std_with_value_all_data[m_or_g], small_std_with_value_all_data[m_or_g], rmse[m_or_g] = RMSE_meg_all(data=data, channels=channels[m_or_g], std_lvl=1)
        derivs_rmse += [boxplot_std_hovering_plotly(std_data=rmse[m_or_g], ch_type=m_or_g_title[m_or_g], channels=channels[m_or_g], what_data='stds')]

    if dict_of_dfs_epoch['mags'] is not None and dict_of_dfs_epoch['grads'] is not None:

        for m_or_g in m_or_g_chosen:

            df_epoch_rmse = RMSE_meg_epoch(ch_type=m_or_g, channels=channels[m_or_g], std_lvl=rmse_params['std_lvl'], epochs_mg=dict_epochs_mg[m_or_g], df_epochs=dict_of_dfs_epoch[m_or_g]) 
            dfs_list += df_epoch_rmse
            fig_std_epoch_with_name += [boxplot_channel_epoch_hovering_plotly(df_mg=df_epoch_rmse[0].content, ch_type=m_or_g_title[m_or_g], what_data='stds')]
            #df_epoch_rmse[0].content - take from list the first obj, from there the content which is the df with stds
    else:
        print('RMSE per epoch can not be calculated because no events are present. Check stimulus channel.')
        
    derivs_rmse += fig_std_epoch_with_name + dfs_list
    
    return derivs_rmse, big_std_with_value_all_data, small_std_with_value_all_data

#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import mne

from universal_plots import boxplot_std_hovering_plotly, boxplot_channel_epoch_hovering_plotly
from universal_html_report import make_std_peak_report

# In[2]:

def RMSE(data_m_or_g: np.array or list):
    ''' RMSE - general root means squared error function to use in other functions of this module.
    Alternatively std could be used, but the calculation time of std is lower, result is the same.
    
    Args:
    data_m_or_g (np.array or list): data for magnetometer or gradiometer given as np array or list 
        (it can be 1 channel or several as 2 dimentional array or list of lists
        
    Returns:
    rmse_np (np.array): rmse as numpy array (1 if 1 channel was given, more if more channels)'''

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
    std_lvl (int): how many standard deviations from the mean are acceptable in the data. 
        data variability over this setting will be concedered as too too noisy, under -std_lvl as too flat 
    
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
    return(big_std_with_value, small_std_with_value, std_channels)


# In[11]:

def std_mg(mg_names: list, df_mg: pd.DataFrame, epoch_numbers: list):

    '''Calculate std for every separate epoch of mags or grads.
    Used as internal function in RMSE_meg_epoch

    Args:
    mg_names (list of tuples): channel name + its index, 
    df_mg (pd.DataFrame): data frame containing data for all epochs for mags or for grads
    epoch_numbers (list): list of epoch numbers

    Returns:
    df_std_mg (pd.DataFrame): data frame containing stds for all epoch for each channel
    '''
    
    dict_mg = {}

    for ep in epoch_numbers: #loop over each epoch
        rows_for_ep = [row for row in df_mg.iloc if row.epoch == ep] #take all rows of 1 epoch, all channels.
        #std_epoch = [] #list with stds
        rmse_epoch=[]

        for ch_name in mg_names: #loop over channel names
            data_ch_epoch = [row_mg[ch_name] for row_mg in rows_for_ep] #take the data for 1 epoch for 1 channel
            rmse_ch_ep = RMSE(data_ch_epoch)
            rmse_ch_ep=np.float64(rmse_ch_ep) #convert from ndarray to float
            rmse_epoch.append(rmse_ch_ep)

            #std_ch_ep = np.std(data_ch_epoch) #if want to use std instead
            

        dict_mg[ep] = rmse_epoch

    df_std_mg = pd.DataFrame(dict_mg, index=mg_names)

    return(df_std_mg)


#%% STD over epochs: use 2 separate data frames for mags and grads in calculations:

def RMSE_meg_epoch(ch_type: str, channels: list, std_lvl: int, epoch_numbers: list, df_epochs: pd.DataFrame, sid: str):

    '''
    - Calculate std for every separate epoch of mags + grads
    - Find which channels in which epochs have too high/too small stds
    - Extract all these data for the user as csc files.

    Args:
    channels (list of tuples): channel name + its index as list,
    std_lvl (int): how many standard deviations from the mean are acceptable in the data. 
        data variability over this setting will be concedered as too too noisy, under -std_lvl as too flat 
    epoch_numbers(list): list of event numbers, 
    df_epochs (pd.DataFrame): data frame containing stds for all epoch for each channel
    sid (str): subject id number, like '1'

    Returns:
    df_std_epoch(pd.DataFrame): data frame containing std data for each epoch, each channel 
    + csv file of this df
    '''

    # 1) Loop over the epochs of each channel and check for every separate magn and grad and calculate std

    #Apply function from above for mags and grads:
    df_std=std_mg(df_mg=df_epochs, mg_names=channels, epoch_numbers=epoch_numbers)

    # 2) Check (which epochs for which channel) are over 1STD (or 2, 3, etc STDs) for this epoch for all channels

    #Find what is 1 std over all channels per 1 epoch:
    std_std_per_epoch=[]
    mean_std_per_epoch=[]

    for ep in epoch_numbers: #goes over each epoch
        std_std_per_epoch.append(np.std(df_std.iloc[:, ep])) #std of stds of all channels of every single epoch
        mean_std_per_epoch.append(np.mean(df_std.iloc[:, ep])) #mean of stds of all channels of every single epoch

    df_ch_ep_large_std=df_std.copy()
    df_ch_ep_small_std=df_std.copy()

    #Now see which channles in epoch are over 1 std or under -1 std:
    for ep in epoch_numbers: #goes over each epoch   
        df_ch_ep_large_std.iloc[:,ep] = df_ch_ep_large_std.iloc[:,ep] > mean_std_per_epoch[ep]+std_lvl*std_std_per_epoch[ep] 
        df_ch_ep_small_std.iloc[:,ep] = df_ch_ep_small_std.iloc[:,ep] < mean_std_per_epoch[ep]-std_lvl*std_std_per_epoch[ep] 


    # Create csv files  for the user:
    df_std.to_csv('../derivatives/sub-'+sid+'/megqc/csv files/std_per_epoch_'+ch_type+'.csv')
    df_ch_ep_large_std.to_csv('../derivatives/sub-'+sid+'/megqc/csv files/Large_std_per_epoch_'+ch_type+'.csv')
    df_ch_ep_small_std.to_csv('../derivatives/sub-'+sid+'/megqc/csv files/Small_std_per_epoch_'+ch_type+'.csv')

    return df_std


#%%
def MEG_QC_rmse(sid: str, config, channels: dict, df_epochs:pd.DataFrame, filtered_d_resamp, m_or_g_chosen):

    m_or_g_title = {
    'grads': 'Gradiometers',
    'mags': 'Magnetometers'}

    rmse_section = config['RMSE']
    std_lvl = rmse_section.getint('std_lvl')

    epoch_numbers = df_epochs[m_or_g_chosen[0]]['epoch'].unique()

    list_of_figure_paths = []
    list_of_figures = []
    list_of_figure_descriptions = []
    list_of_figure_paths_std_epoch = []
    list_of_figures_std_epoch = []
    list_of_figure_descriptions_std_epoch = []
    big_std_with_value = {}
    small_std_with_value = {}
    figs = {}
    fig_path = {}
    fig_name = {}
    df_std = {}
    fig_std_epoch = {}
    fig_path_std_epoch = {}
    fig_name_epoch = {}
    rmse = {}

    # will run for both if mags+grads are chosen,otherwise just for one of them:
    for m_or_g in m_or_g_chosen:

        big_std_with_value[m_or_g], small_std_with_value[m_or_g], rmse[m_or_g] = RMSE_meg_all(data=filtered_d_resamp, channels=channels[m_or_g], std_lvl=1)

        figs[m_or_g], fig_path[m_or_g], fig_name[m_or_g] = boxplot_std_hovering_plotly(std_data=rmse[m_or_g], ch_type=m_or_g_title[m_or_g], channels=channels[m_or_g], sid=sid, what_data='stds')
        list_of_figure_paths.append(fig_path[m_or_g])
        list_of_figures.append(figs[m_or_g])
        list_of_figure_descriptions.append(fig_name[m_or_g])

        if df_epochs[m_or_g] is not None:
            df_std[m_or_g] = RMSE_meg_epoch(ch_type=m_or_g, channels=channels[m_or_g], std_lvl=std_lvl, epoch_numbers=epoch_numbers, df_epochs=df_epochs[m_or_g], sid=sid) 
            fig_std_epoch[m_or_g], fig_path_std_epoch[m_or_g], fig_name_epoch[m_or_g] = boxplot_channel_epoch_hovering_plotly(df_mg=df_std[m_or_g], ch_type=m_or_g_title[m_or_g], sid=sid, what_data='stds')
            list_of_figure_paths_std_epoch.append(fig_path_std_epoch[m_or_g])
            list_of_figures_std_epoch.append(fig_std_epoch[m_or_g])
            list_of_figure_descriptions_std_epoch.append(fig_name_epoch[m_or_g])
        else:
            print('RMSE per epoch can not be calculated because no events are present. Check stimulus channel.')
        
    list_of_figure_paths += list_of_figure_paths_std_epoch
    list_of_figures += list_of_figures_std_epoch
    list_of_figure_descriptions += list_of_figure_descriptions_std_epoch
    
    # make_std_peak_report(sid=sid, what_data='stds', list_of_figure_paths=list_of_figure_paths, config=config)

    return list_of_figures, list_of_figure_paths, list_of_figure_descriptions
    
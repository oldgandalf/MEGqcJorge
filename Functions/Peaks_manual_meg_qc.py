# Hand-written peak to peak detection.
# MNE has own peak to peak detection in annotate_amplitude, but it used different settings and the process 
# of calculation there is not yet clear. We may use their way or may keep this one.

import numpy as np
import pandas as pd
import mne
from universal_plots import boxplot_std_hovering_plotly, boxplot_channel_epoch_hovering_plotly, QC_derivative


def neighbour_peak_amplitude(pair_dist_sec: float, sfreq: int, pos_peak_locs:np.ndarray, neg_peak_locs:np.ndarray, pos_peak_magnitudes: np.ndarray, neg_peak_magnitudes: np.ndarray) -> float:

    ''' Function finds a pair: postive+negative peak and calculates the amplitude between them. 
    If no neighbour is found withing given distance - this peak is skipped. 
    If several neighbours are found - several pairs are created. 
    As the result a mean peak-to-peak distance is calculated over all detected pairs for given chunck of data
    
    Args:
    pair_dist_sec (float): maximum distance in seconds which is allowed for negative+positive peaks to be detected as a pair 
    sfreq: sampling frequency of data. Attention to which data is used! original or resampled.
    pos_peak_locs (np.ndarray): output of peak_finder (Python) function - positions of detected Positive peaks
    neg_peak_locs (np.ndarray): output of peak_finder (Python) function - positions of detected Negative peaks
    pos_peak_magnitudes (np.ndarray): output of peak_finder (Python) function - magnitudes of detected Positive peaks
    neg_peak_magnitudes (np.ndarray): output of peak_finder (Python) function - magnitudes of detected Negative peaks

    Returns:
    (float): mean value over all detected peak pairs for this chunck of data.
    '''

    pair_dist=pair_dist_sec*sfreq
    pairs_magnitudes=[]
    pairs_locs=[]

    for posit_peak_ind, posit_peak_loc in enumerate(pos_peak_locs):
        neg_peak_ind=np.where(np.logical_and(neg_peak_locs>=posit_peak_loc-pair_dist/2, neg_peak_locs<=posit_peak_loc+pair_dist/2))
        if neg_peak_ind[0].size != 0:

            #find the negative peak  which is located at a half of pair_dist from positive peak -> they will for a pair
            pairs_locs.append([pos_peak_locs[posit_peak_ind], neg_peak_locs[neg_peak_ind[0][0]]])
            pairs_magnitudes.append([pos_peak_magnitudes[posit_peak_ind], neg_peak_magnitudes[neg_peak_ind[0][0]]])

    # if no positive+negative pairs were fould (no corresponding peaks at given distamce to each other) -> 
    # peak amplitude will be given as 0 (THINK MAYBE GIVE SOMETHING DIFFERENT INSTEAD? 
    # FOR EXAMPLE JUST AMPLITIDU OF MOST POSITIVE AND MOST NEGATIVE VALUE OVER ALL GIVEN TIME?
    # HOWEVER THIS WILL NOT CORRESPOND TO PEAK TO PEAK IDEA).

    if len(pairs_magnitudes)==0:
        return 0

    amplitude=np.zeros(len(pairs_magnitudes),)
    for i, pair in enumerate(pairs_magnitudes):
        amplitude[i]=pair[0]-pair[1]

    return np.mean(amplitude)

#%%
def peak_amplitude_all_data(data: mne.io.Raw, channels: list, sfreq: int, thresh_lvl: float, pair_dist_sec: float) -> pd.DataFrame:

    '''Function calculates peak-to-peak amplitude over the entire data set for every channel (mag or grad).

    Args:
    mg_names (list of tuples): channel name + its index
    df_epoch_mg (pd. Dataframe): data frame containing data for all epochs for mags  or grads
    sfreq: sampling frequency of data. Attention to which data is used! original or resampled.
    n_events (int): number of events in this peace of data
    thresh_lvl (float): defines how high or low need to peak to be to be detected, this can also be changed into a sigle value later
        used in: max(data_ch_epoch) - min(data_ch_epoch)) / thresh_lvl 
    pair_dist_sec (float): maximum distance in seconds which is allowed for negative+positive peaks to be detected as a pair 

    Returns:
    df_pp_ampl_all (pd.DataFrame): contains the mean peak-to-peak amplitude for each epoch for each channel

    '''
    
    dict_mg = {}
    #print(channels, len(channels))

    data_channels=data.get_data(picks = channels)

    peak_ampl=[]
    for one_ch_data in data_channels: 

        thresh=(max(one_ch_data) - min(one_ch_data)) / thresh_lvl 
        #can also change the whole thresh to a single number setting

        pos_peak_locs, pos_peak_magnitudes = mne.preprocessing.peak_finder(one_ch_data, extrema=1, thresh=thresh, verbose=False) #positive peaks
        neg_peak_locs, neg_peak_magnitudes = mne.preprocessing.peak_finder(one_ch_data, extrema=-1, thresh=thresh, verbose=False) #negative peaks

        pp_ampl=neighbour_peak_amplitude(pair_dist_sec, sfreq, pos_peak_locs, neg_peak_locs, pos_peak_magnitudes, neg_peak_magnitudes)
        peak_ampl.append(pp_ampl)

    #df_pp_ampl_all = pd.DataFrame(peak_ampl, index=channels)

    return peak_ampl

# In[7]:

def peak_amplitude_per_epoch(channels: list, df_epoch: dict, sfreq: int, thresh_lvl: float, pair_dist_sec: float, epoch_numbers:list, ch_type:  str):

    '''Function calculates peak-to-peak amplitude for every epoch and every channel (mag or grad).

    Args:
    mg_names (list of tuples): channel name + its index
    df_epoch_mg (pd. Dataframe): data frame containing data for all epochs for mags  or grads
    sfreq: sampling frequency of data. Attention to which data is used! original or resampled.
    n_events (int): number of events in this peace of data
    thresh_lvl (float): defines how high or low need to peak to be to be detected, this can also be changed into a sigle value later
        used in: max(data_ch_epoch) - min(data_ch_epoch)) / thresh_lvl 
    pair_dist_sec (float): maximum distance in seconds which is allowed for negative+positive peaks to be detected as a pair 

    Returns:
    df_pp_ampl_mg (pd.DataFrame): contains the mean peak-to-peak aplitude for each epoch for each channel

    '''

    dict_ep = {}

    for ep in epoch_numbers: #loop over each epoch

        rows_for_ep = [row for row in df_epoch.iloc if row.epoch == ep] #take all rows of 1 epoch, all channels.

        peak_ampl_epoch=[]
        for ch_name in channels: 
            data_ch_epoch = [row_mg[ch_name] for row_mg in rows_for_ep] #take the data for 1 epoch for 1 channel
            
            thresh=(max(data_ch_epoch) - min(data_ch_epoch)) / thresh_lvl 
            #can also change the whole thresh to a single number setting

            pos_peak_locs, pos_peak_magnitudes = mne.preprocessing.peak_finder(data_ch_epoch, extrema=1, thresh=thresh, verbose=False) #positive peaks
            neg_peak_locs, neg_peak_magnitudes = mne.preprocessing.peak_finder(data_ch_epoch, extrema=-1, thresh=thresh, verbose=False) #negative peaks
            
            pp_ampl=neighbour_peak_amplitude(pair_dist_sec, sfreq, pos_peak_locs, neg_peak_locs, pos_peak_magnitudes, neg_peak_magnitudes)
            peak_ampl_epoch.append(pp_ampl)

        dict_ep[ep] = peak_ampl_epoch
    df_pp_ampl_mg = pd.DataFrame(dict_ep, index=channels)
    df_pp_name = 'Peak_to_Peak_per_epoch_'+ch_type

    file_path = None
    dfs_with_name = [QC_derivative(df_pp_ampl_mg, df_pp_name, file_path, 'df')]

    return dfs_with_name


def PP_manual_meg_qc(ptp_manual_params, channels: dict, dict_of_dfs_epoch: dict, data: mne.io.Raw, m_or_g_chosen: list):


    """Main Peak to peak amplitude function.
    
    Output:
    out_with_name_and_format: list of tuples(figure, fig_name, fig_path, format_of_output_content)"""

    m_or_g_title = {
    'grads': 'Gradiometers',
    'mags': 'Magnetometers'}


    sfreq = data.info['sfreq']

    derivs_ptp = []
    fig_ptp_epoch_with_name = []
    dfs_list = []
    peak_ampl = {}

    # will run for both if mags+grads are chosen,otherwise just for one of them:
    for m_or_g in m_or_g_chosen:

        peak_ampl[m_or_g] = peak_amplitude_all_data(data, channels[m_or_g], sfreq, thresh_lvl=ptp_manual_params['thresh_lvl'], pair_dist_sec=ptp_manual_params['pair_dist_sec'])
        derivs_ptp += [boxplot_std_hovering_plotly(peak_ampl[m_or_g], ch_type=m_or_g_title[m_or_g], channels=channels[m_or_g], what_data='peaks')]

    if dict_of_dfs_epoch['mags'] is not None and dict_of_dfs_epoch['grads'] is not None:

        epoch_numbers = dict_of_dfs_epoch[m_or_g_chosen[0]]['epoch'].unique()
        for m_or_g in m_or_g_chosen:
            df_ptp=peak_amplitude_per_epoch(channels[m_or_g], dict_of_dfs_epoch[m_or_g], sfreq, thresh_lvl=ptp_manual_params['thresh_lvl'], pair_dist_sec=ptp_manual_params['pair_dist_sec'], epoch_numbers=epoch_numbers, ch_type=m_or_g)
            dfs_list += df_ptp

            fig_ptp_epoch_with_name += [boxplot_channel_epoch_hovering_plotly(df_mg=df_ptp[0].content, ch_type=m_or_g_title[m_or_g], what_data='peaks')]
            #df_epoch_rmse[0].content - take from list the first obj, from there the content which is the df with ptp
    else:
        print('Peak-to-Peak per epoch can not be calculated because no events are present. Check stimulus channel.')
        
    derivs_ptp += fig_ptp_epoch_with_name + dfs_list
    
    return derivs_ptp
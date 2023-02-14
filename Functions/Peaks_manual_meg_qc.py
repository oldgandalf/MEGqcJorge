# Hand-written peak to peak detection.
# MNE has own peak to peak detection in annotate_amplitude, but it used different settings and the process 
# of calculation there is not yet clear. We may use their way or may keep this one.

import numpy as np
import pandas as pd
import mne
from universal_plots import boxplot_std_hovering_plotly, boxplot_channel_epoch_hovering_plotly, QC_derivative, boxplot_epochs
from universal_html_report import simple_metric_basic
from RMSE_meq_qc import get_big_small_RMSE_PtP_epochs, make_dict_global_rmse_ptp, make_dict_local_rmse_ptp, get_big_small_std_ptp_all_data, get_noisy_flat_rmse_ptp_epochs


def neighbour_peak_amplitude(max_pair_dist_sec: float, sfreq: int, pos_peak_locs:np.ndarray, neg_peak_locs:np.ndarray, pos_peak_magnitudes: np.ndarray, neg_peak_magnitudes: np.ndarray) -> float:

    ''' Function finds a pair: postive+negative peak and calculates the amplitude between them. 
    If no neighbour is found withing given distance - this peak is skipped. 
    If several neighbours are found - several pairs are created. 
    As the result a mean peak-to-peak distance is calculated over all detected pairs for given chunck of data
    
    Args:
    max_pair_dist_sec (float): maximum distance in seconds which is allowed for negative+positive peaks to be detected as a pair 
    sfreq: sampling frequency of data. Attention to which data is used! original or resampled.
    pos_peak_locs (np.ndarray): output of peak_finder (Python) function - positions of detected Positive peaks
    neg_peak_locs (np.ndarray): output of peak_finder (Python) function - positions of detected Negative peaks
    pos_peak_magnitudes (np.ndarray): output of peak_finder (Python) function - magnitudes of detected Positive peaks
    neg_peak_magnitudes (np.ndarray): output of peak_finder (Python) function - magnitudes of detected Negative peaks

    Returns:
    (float): mean value over all detected peak pairs for this chunck of data.
    '''

    pair_dist=max_pair_dist_sec*sfreq
    pairs_magnitudes=[]
    pairs_locs=[]

    # Looping over all positive peaks
    for posit_peak_ind, posit_peak_loc in enumerate(pos_peak_locs):
        
        # Finding the value in neg_peak_locs which is closest to posit_peak_loc
        closest_negative_peak_index = np.abs(neg_peak_locs - posit_peak_loc).argmin()

        # Check if the closest negative peak is within the given distance
        if np.abs(neg_peak_locs[closest_negative_peak_index] - posit_peak_loc) <= pair_dist / 2:
            pairs_locs.append([pos_peak_locs[posit_peak_ind], neg_peak_locs[closest_negative_peak_index]])
            pairs_magnitudes.append([pos_peak_magnitudes[posit_peak_ind], neg_peak_magnitudes[closest_negative_peak_index]])
        
        # print("||:", neg_peak_ind)
        # if neg_peak_ind[0].size != 0:
        #     print("||: A")
        #     #find the negative peak  which is located at a half of pair_dist from positive peak -> they will for a pair
        #     pairs_locs.append([pos_peak_locs[posit_peak_ind], neg_peak_locs[neg_peak_ind[0][0]]])
        #     pairs_magnitudes.append([pos_peak_magnitudes[posit_peak_ind], neg_peak_magnitudes[neg_peak_ind[0][0]]])

    # if no positive+negative pairs were fould (no corresponding peaks at given distamce to each other) -> 
    # peak amplitude will be given as 0 (THINK MAYBE GIVE SOMETHING DIFFERENT INSTEAD? 
    # FOR EXAMPLE JUST AMPLITIDU OF MOST POSITIVE AND MOST NEGATIVE VALUE OVER ALL GIVEN TIME?
    # HOWEVER THIS WILL NOT CORRESPOND TO PEAK TO PEAK IDEA).

    if len(pairs_magnitudes)==0:
        return 0, None

    amplitude=np.zeros(len(pairs_magnitudes),)
    for i, pair in enumerate(pairs_magnitudes):
        amplitude[i]=pair[0]-pair[1]

    return np.mean(amplitude), amplitude

def get_ptp_all_data(data: mne.io.Raw, channels: list, sfreq: int, ptp_thresh_lvl: float, max_pair_dist_sec: float):
        
    data_channels=data.get_data(picks = channels)

    peak_ampl_channels=[]
    for one_ch_data in data_channels: 

        thresh=(max(one_ch_data) - min(one_ch_data)) / ptp_thresh_lvl 
        #can also change the whole thresh to a single number setting

        pos_peak_locs, pos_peak_magnitudes = mne.preprocessing.peak_finder(one_ch_data, extrema=1, thresh=thresh, verbose=False) #positive peaks
        neg_peak_locs, neg_peak_magnitudes = mne.preprocessing.peak_finder(one_ch_data, extrema=-1, thresh=thresh, verbose=False) #negative peaks

        pp_ampl, _ = neighbour_peak_amplitude(max_pair_dist_sec, sfreq, pos_peak_locs, neg_peak_locs, pos_peak_magnitudes, neg_peak_magnitudes)
        peak_ampl_channels.append(pp_ampl)
        
    return peak_ampl_channels


# In[7]:

def get_ptp_epochs(channels: list, epochs_mg: mne.Epochs, sfreq: int, ptp_thresh_lvl: float, max_pair_dist_sec: float):

    ''' --fastest  and cleanest version, no need to use data frames--

    Function calculates peak-to-peak amplitude for every epoch and every channel (mag or grad).

    Args:
    channels (list of tuples): channel name + its index
    df_epoch_mg (pd. Dataframe): data frame containing data for all epochs for mag  or grad
    sfreq: sampling frequency of data. Attention to which data is used! original or resampled.
    n_events (int): number of events in this peace of data
    ptp_thresh_lvl (float): defines how high or low need to peak to be to be detected, this can also be changed into a sigle value later
        used in: max(data_ch_epoch) - min(data_ch_epoch)) / ptp_thresh_lvl 
    max_pair_dist_sec (float): maximum distance in seconds which is allowed for negative+positive peaks to be detected as a pair 

    Returns:
    df_pp_ampl_mg (pd.DataFrame): contains the mean peak-to-peak aplitude for each epoch for each channel

    '''
    dict_ep = {}

    #get 1 epoch, 1 channel and calculate PtP on its data:
    for ep in range(0, len(epochs_mg)):
        peak_ampl_epoch=[]
        for ch_name in channels: 
            data_ch_epoch=epochs_mg[ep].get_data(picks=ch_name)[0][0]
            #[0][0] is because get_data creats array in array in array, it expects several epochs, several channels, but we only need  one.
            
            thresh=(max(data_ch_epoch) - min(data_ch_epoch)) / ptp_thresh_lvl 
            #can also change the whole thresh to a single number setting

            pos_peak_locs, pos_peak_magnitudes = mne.preprocessing.peak_finder(data_ch_epoch, extrema=1, thresh=thresh, verbose=False) #positive peaks
            neg_peak_locs, neg_peak_magnitudes = mne.preprocessing.peak_finder(data_ch_epoch, extrema=-1, thresh=thresh, verbose=False) #negative peaks
            pp_ampl,_=neighbour_peak_amplitude(max_pair_dist_sec, sfreq, pos_peak_locs, neg_peak_locs, pos_peak_magnitudes, neg_peak_magnitudes)
            peak_ampl_epoch.append(pp_ampl)

        dict_ep[ep] = peak_ampl_epoch

    return pd.DataFrame(dict_ep, index=channels)



def make_simple_metric_ptp_manual(ptp_manual_params: dict, big_ptp_with_value_all_data, small_ptp_with_value_all_data, channels, deriv_epoch_ptp, metric_local, m_or_g_chosen):

    metric_global_name = 'ptp_manual_all'
    metric_global_description = 'Peak-to-peak deviation of the data over the entire time series (not epoched): ... The ptp_lvl is the peak-to-peak threshold level set by the user. Threshold = ... The channel where data is higher than this threshod is considered as noisy. Same: if the std of some channel is lower than -threshold, this channel is considered as flat. In details only the noisy/flat channels are listed. Channels with normal std are not listed. If needed to see all channels data - use csv files.'
    metric_local_name = 'ptp_manual_epoch'
    if metric_local==True:
        metric_local_description = 'Peak-to-peak deviation of the data over stimulus-based epochs. The epoch is counted as noisy (or flat) if the percentage of noisy (or flat) channels in this epoch is over allow_percent_noisy_flat. this percent is set by user, default=70%. Hense, if no epochs have over 70% of noisy channels - total number of noisy epochs will be 0. Definition of a noisy channel inside of epoch: 1)Take Peak-to-Peak amplitude (PtP) of data of THIS channel in THIS epoch. 2) Take PtP of the data of THIS channel for ALL epochs and get mean of it. 3) If (1) is higher than (2)*noisy_multiplier - this channel is noisy.  If (1) is lower than (2)*flat_multiplier - this channel is flat. '
    else:
        metric_local_description = 'Not calculated. Ne epochs found'

    metric_global_content={'mag': None, 'grad': None}
    metric_local_content={'mag': None, 'grad': None}
    for m_or_g in m_or_g_chosen:
        metric_global_content[m_or_g]=make_dict_global_rmse_ptp(ptp_manual_params, big_ptp_with_value_all_data[m_or_g], small_ptp_with_value_all_data[m_or_g], channels[m_or_g], 'ptp')
        
        if metric_local is True:
            metric_local_content[m_or_g]=make_dict_local_rmse_ptp(ptp_manual_params, deriv_epoch_ptp[m_or_g][1].content, deriv_epoch_ptp[m_or_g][2].content)
            #deriv_epoch_rmse[m_or_g][1].content is df with big rmse(noisy), df_epoch_rmse[m_or_g][2].content is df with small rmse(flat)
        else:
            metric_local_content[m_or_g]=None
    
    simple_metric = simple_metric_basic(metric_global_name, metric_global_description, metric_global_content['mag'], metric_global_content['grad'], metric_local_name, metric_local_description, metric_local_content['mag'], metric_local_content['grad'])

    return simple_metric


def PP_manual_meg_qc(ptp_manual_params, channels: dict, dict_epochs_mg: dict, data: mne.io.Raw, m_or_g_chosen: list):


    """Main Peak to peak amplitude function.
    
    Output:
    out_with_name_and_format: list of tuples(figure, fig_name, fig_path, format_of_output_content)"""


    sfreq = data.info['sfreq']

    big_ptp_with_value_all_data = {}
    small_ptp_with_value_all_data = {}
    derivs_ptp = []
    fig_ptp_epoch = []
    fig_ptp_epoch2 = []
    derivs_list = []
    peak_ampl = {}
    noisy_flat_epochs_derivs = {}

    # will run for both if mag+grad are chosen,otherwise just for one of them:
    for m_or_g in m_or_g_chosen:
        peak_ampl[m_or_g] = get_ptp_all_data(data, channels[m_or_g], sfreq, ptp_thresh_lvl=ptp_manual_params['ptp_thresh_lvl'], max_pair_dist_sec=ptp_manual_params['max_pair_dist_sec'])
        big_ptp_with_value_all_data[m_or_g], small_ptp_with_value_all_data[m_or_g] = get_big_small_std_ptp_all_data(peak_ampl[m_or_g], channels[m_or_g], ptp_manual_params['std_lvl'])
        derivs_ptp += [boxplot_std_hovering_plotly(peak_ampl[m_or_g], ch_type=m_or_g, channels=channels[m_or_g], what_data='peaks')]

    if dict_epochs_mg['mag'] is not None or dict_epochs_mg['grad'] is not None:
        for m_or_g in m_or_g_chosen:
            df_pp_ampl=get_ptp_epochs(channels[m_or_g], dict_epochs_mg[m_or_g], sfreq, ptp_manual_params['ptp_thresh_lvl'], ptp_manual_params['max_pair_dist_sec'])
            #deriv_epoch_ptp[m_or_g] = get_big_small_RMSE_PtP_epochs(df_pp_ampl, m_or_g, ptp_manual_params['std_lvl'], 'ptp') 
            #derivs_list += deriv_epoch_ptp[m_or_g]

            noisy_flat_epochs_derivs[m_or_g] = get_noisy_flat_rmse_ptp_epochs(df_pp_ampl, m_or_g, 'ptp', ptp_manual_params['noisy_multiplier'], ptp_manual_params['flat_multiplier'], ptp_manual_params['allow_percent_noisy_flat_epochs'])
            derivs_list += noisy_flat_epochs_derivs[m_or_g]

            fig_ptp_epoch += [boxplot_channel_epoch_hovering_plotly(df_mg=df_pp_ampl, ch_type=m_or_g, what_data='peaks')]
            fig_ptp_epoch2 += [boxplot_epochs(df_mg=df_pp_ampl, ch_type=m_or_g, what_data='stds')]
            metric_local=True
    else:
        metric_local=False
        print('___MEG QC___: ', 'Peak-to-Peak per epoch can not be calculated because no events are present. Check stimulus channel.')
        
    derivs_ptp += fig_ptp_epoch + fig_ptp_epoch2 + derivs_list

    simple_metric_ptp_manual = make_simple_metric_ptp_manual(ptp_manual_params, big_ptp_with_value_all_data, small_ptp_with_value_all_data, channels, noisy_flat_epochs_derivs, metric_local, m_or_g_chosen)

    return derivs_ptp, simple_metric_ptp_manual
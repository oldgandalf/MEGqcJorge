#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import mne
import plotly.graph_objects as go
from scipy.integrate import simpson
from universal_plots import Plot_periodogram, plot_pie_chart_freq, QC_derivative, get_tit_and_unit
from scipy.signal import find_peaks, peak_widths
from universal_html_report import simple_metric_basic

# ISSUE IN /Volumes/M2_DATA/MEG_QC_stuff/data/from openneuro/ds004107/sub-mind004/ses-01/meg/sub-mind004_ses-01_task-auditory_meg.fif...
# COULDNT SPLIT  when filtered data
# In[42]:

def Power_of_band(freqs: np.ndarray, f_low: float, f_high: float, psds: np.ndarray):

    '''Calculates the area under the curve of one chosen band (e.g. alpha, beta, gamma, delta, ...) for mag or grad.
    (Named here as power, but in fact it s amplitude of the signal, since psds are turned into amplitudes already.)
    Adopted from: https://raphaelvallat.com/bandpower.html

    This function is called in Power_of_freq_meg
    
    Args:
    freqs (np.ndarray): numpy array of frequencies,
    psds (np.ndarray): numpy array of power spectrum dencities. Expects array of arrays: channels*psds. !
        Will not work properly if it is 1 dimentional array give. In this case do: np.array([your_1d_array])
    f_low (float): minimal frequency of the chosend band, in Hz (For delta it would be: 0.5),
    f_high (float): maximal frequency of the chosend band, in Hz (For delta it would be: 4).


    Returns:
    bandpower_per_ch_list (list): list of powers of each band like: [abs_power_of_delta, abs_power_of_gamma, etc...] - in absolute values
    power_by_Nfreq_per_ch_list (list): list of powers of bands divided by the  number of frequencies in the band - to compare 
        with RMSE later. Like: [power_of_delta/n_freqs, power_of_gamma/n_freqs, etc...]
    rel_bandpower_per_ch_list (list): list of power of each band like: [rel_power_of_delta, rel_power_of_gamma, etc...] - in relative  
        (percentage) values: what percentage of the total power does this band take.

    '''

    bandpower_per_ch_list=[]
    rel_bandpower_per_ch_list=[]
    power_by_Nfreq_per_ch_list=[]

    idx_band = np.logical_and(freqs >= f_low, freqs <= f_high) 
    # Find closest indices of band in frequency vector so idx_band is a list of indices frequencies that 
    # correspond to this band. F.e. for delta band these would be the indices of 0.5 ... 4 Hz)

    for ch in psds: 
    #loop over channels. psd_ch is psd of partigular channel

        psd_ch=np.array(ch)

        # Compute Area under the curve (power):
        # Frequency resolution
        freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

        # Compute the absolute power by approximating the area under the curve:
        band_power = simpson(psd_ch[idx_band], dx=freq_res) #power of chosen band
        total_power = simpson(psd_ch, dx=freq_res) # power of all bands
        band_rel_power = band_power / total_power # relative power: % of this band in the total bands power for this channel:

        #devide the power of band by the  number of frequencies in the band, to compare with RMSE later:
        power_compare=band_power/sum(idx_band) 

        bandpower_per_ch_list.append(band_power)
        rel_bandpower_per_ch_list.append(band_rel_power)
        power_by_Nfreq_per_ch_list.append(power_compare)

    return(bandpower_per_ch_list, power_by_Nfreq_per_ch_list, rel_bandpower_per_ch_list, total_power)


    
# In[53]:

def Power_of_freq_meg(ch_names: list, m_or_g: str, freqs: np.ndarray, psds: np.ndarray, mean_power_per_band_needed: bool, plotflag: bool):

    '''
    - Power of frequencies calculation for all mag + grad channels separately, 
    - Saving power + power/freq value into data frames.
    - If desired: creating a pie chart of mean power of every band over the entire data (all channels of 1 type together)
    
    Args:
    ch_names (list of tuples): channel names + index as list, 
    freqs (np.ndarray): numpy array of frequencies for mag  or grad
    psds (np.ndarray): numpy array of power spectrum dencities for mag or grad
    mean_power_per_band_needed (bool): need to calculate mean band power in the ENTIRE signal (averaged over all channels) or not.
        if True, results will also be printed.
    plotflag (bool): need to plot pie chart of mean_power_per_band_needed or not

    Returns:
    data frames as csv files saved:
    absolute power of each frequency band in each channel (mag or grad)
    relative power of each frequency band in each channel (mag or grad)
    absolute power of each frequency band in each channel (mag or grad) divided by the number of frequencies in this band
    + if plotflag is True:
    fig: plottly piechart figure 
    fig_path: path where the figure is saved as html file 
    '''
    
    # Calculate the band power:
    wave_bands=[[0.5, 4], [4, 8], [8, 12], [12, 30], [30, 100]]
    #delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–12 Hz), beta (12–30 Hz), and gamma (30–100 Hz) bands

    dict_power = {}
    dict_power_freq = {}
    dict_rel_power = {}

    for w in enumerate(wave_bands): #loop over bands
        
        f_low, f_high = w[1] # Define band lower and upper limits

        #loop over mag or grad:
        bandpower_per_ch_list, power_by_Nfreq_per_ch_list, rel_bandpower_per_ch_list, _ =Power_of_band(freqs, f_low, f_high, psds)

        dict_power[w[0]] = bandpower_per_ch_list
        dict_power_freq[w[0]] = power_by_Nfreq_per_ch_list
        dict_rel_power[w[0]] = rel_bandpower_per_ch_list


    # Save all to data frames:
    df_power = pd.DataFrame(dict_power, index=ch_names)
    df_power_freq = pd.DataFrame(dict_power_freq, index=ch_names)
    df_rel_power = pd.DataFrame(dict_rel_power, index=ch_names)

    # Rename columns and extract to csv:

    renamed_df_power = df_power.rename(columns={0: "delta (0.5-4 Hz)", 1: "theta (4-8 Hz)", 2: "alpha (8-12 Hz)", 3: "beta (12-30 Hz)", 4: "gamma (30-100 Hz)"})
    renamed_df_power_name = 'abs_power_'+m_or_g
    renamed_df_power_freq = df_power_freq.rename(columns={0: "delta (0.5-4 Hz)", 1: "theta (4-8 Hz)", 2: "alpha (8-12 Hz)", 3: "beta (12-30 Hz)", 4: "gamma (30-100 Hz)"})
    renamed_df_power_freq_name = 'power_by_Nfreq_'+m_or_g
    renamed_df_rel_power = df_rel_power.rename(columns={0: "delta (0.5-4 Hz)", 1: "theta (4-8 Hz)", 2: "alpha (8-12 Hz)", 3: "beta (12-30 Hz)", 4: "gamma (30-100 Hz)"})
    renamed_df_rel_power_name = 'relative_power_'+m_or_g


    dfs_with_name = [
        QC_derivative(renamed_df_power,renamed_df_power_name, 'df'),
        QC_derivative(renamed_df_power_freq, renamed_df_power_freq_name, 'df'),
        QC_derivative(renamed_df_rel_power, renamed_df_rel_power_name, 'df')
        ]


    if mean_power_per_band_needed is True: #if user wants to see average power per band over all channels - calculate and plot here:

        #Calculate power per band over all mag and all grad

        import statistics 

        power_dfs=[df_power, df_rel_power, df_power_freq] #keep them in this order!  

        bands_names=['delta', 'theta', 'alpha', 'beta', 'gamma']
        measure_title=['Average absolute power per band:', 'Average relative power per band:',
        'Average power/freq per band:']

        mean_abs=[]
        mean_relative=[]
        mean_power_nfreq=[]

        if m_or_g == 'mag':
            tit='Magnetometers'
        elif m_or_g == 'grad':
            tit='Gradiometers'
        else:
            TypeError ("Check channel type!")

        print('___MEG QC___: ', tit)
        for d in enumerate(power_dfs):
            print('___MEG QC___: ', '  \n'+measure_title[d[0]])

            for band in enumerate(bands_names): #loop over bands
                mean_power_per_band = statistics.mean(d[1].loc[:,band[0]])
                
                if d[0]==0: #df_power_mag:
                    mean_abs.append(mean_power_per_band) 
                elif d[0]==1: #df_rel_power_mag:
                    mean_relative.append(mean_power_per_band) 
                elif d[0]==2: #df_power_freq_mag:
                    mean_power_nfreq.append(mean_power_per_band)

                print('___MEG QC___: ', band[1], mean_power_per_band)


        if plotflag is True: 
            psd_pie_derivative = plot_pie_chart_freq(mean_relative_freq=mean_relative, tit=tit, bands_names=bands_names)
        else:
            psd_pie_derivative = []

    
    return psd_pie_derivative, dfs_with_name




#%% Final simple metrics: number of noise frequencies + aea ubnder the curve for each of them. How to:

def split_blended_freqs_old(noisy_freq_bands_idx, width_heights):

    band = 0
    while band < len(noisy_freq_bands_idx):

        # Checking if the last element of every band is contained in the current band
        last = 0
        while last < len(noisy_freq_bands_idx):

            if (noisy_freq_bands_idx[last] != noisy_freq_bands_idx[band]) and (noisy_freq_bands_idx[last][-1] in noisy_freq_bands_idx[band]):
                
                #if yes - split the biggest band at the split point and also assign the same heights of peaks to both parts.

                split_index = noisy_freq_bands_idx[band].index(noisy_freq_bands_idx[last][-1])

                split_band_left = noisy_freq_bands_idx[band][:split_index+1]
                split_band_right = noisy_freq_bands_idx[band][split_index+1:]


                noisy_freq_bands_idx[last] = split_band_left
                noisy_freq_bands_idx[band] = split_band_right

                min_width_heights = min(width_heights[last],width_heights[band])
                width_heights[band] = min_width_heights
                width_heights[last] = min_width_heights


                #set both bands to 0, so next time  the check will be done for all the bands from the beginning, 
                # concedering new state of noisy_freq_bands_idx:
                band = 0
                last = 0

            last += 1
        band += 1

    return noisy_freq_bands_idx, width_heights


def split_blended_freqs_at_negative_peaks(noisy_bands_indexes, peaks, peaks_neg):


    split_points = []
    noisy_bends_indexes_after_split = []
    for n_peak, _ in enumerate(peaks):

        #find negative peaks before and after closest to the found positive noise peak.
  
        neg_peak_before=peaks_neg[np.argwhere(peaks_neg<peaks[n_peak])[-1][0]]
        neg_peak_after=peaks_neg[np.argwhere(peaks_neg>peaks[n_peak])[0][0]]
     
        band_indexes=noisy_bands_indexes[n_peak]

        if band_indexes[0] < neg_peak_before: #if the band extends over the negative peak to the left -> cut it at the negative peak
            band_indexes = [neg_peak_before, band_indexes[1]]

            split_points += [neg_peak_before]

        if band_indexes[1] > neg_peak_after: #if the band extends over the negative peak to the right -> cut it at the negative peak
            band_indexes = [i for i in range(noisy_bands_indexes[n_peak][0], neg_peak_after)]

            split_points += [neg_peak_after]

        noisy_bends_indexes_after_split.append(band_indexes)

    return noisy_bends_indexes_after_split, split_points


def cut_the_noise_from_psd(noisy_bends_indexes, freqs, one_psd, helper_plots: bool):

    #band height will be chosen as average between the height of the limits of this bend.
    width_heights_split = []
    for band_indexes in enumerate(noisy_bends_indexes):
        width_heights_split.append(np.mean([one_psd[band_indexes][0], one_psd[band_indexes][-1]]))

    psd_only_signal=one_psd.copy()
    psd_only_peaks=one_psd.copy()
    psd_only_peaks[:]=None
    psd_only_peaks_baselined=one_psd.copy()
    psd_only_peaks_baselined[:]=0
    
    bend_start, bend_end = [], []
    for fr_n, fr_b in enumerate(noisy_bends_indexes):
        bend_start.append(freqs[fr_b][0])
        bend_end.append(freqs[fr_b][-1])
        
        psd_only_signal[fr_b]=None #keep only main psd, remove noise bands, just for visual
        psd_only_peaks[fr_b]=one_psd[fr_b].copy() #keep only noise bands, remove psd, again for visual
        psd_only_peaks_baselined[fr_b]=one_psd[fr_b].copy()-[width_heights_split[fr_n]]*len(psd_only_peaks[fr_b])
        #keep only noise bands and baseline them to 0 (remove the signal which is under the noise line)

        # clip the values to 0 if they are negative, they might appear in the beginning of psd curve
        psd_only_peaks_baselined=np.array(psd_only_peaks_baselined) 
        psd_only_peaks_baselined = np.clip(psd_only_peaks_baselined, 0, None) 


    return psd_only_peaks_baselined, psd_only_peaks, psd_only_signal, bend_start, bend_end, width_heights_split

def make_helper_plots(freqs, avg_psd, peaks, peaks_neg, left_ips, right_ips, split_points, ips_l, ips_r, width_heights, avg_psd_only_signal, avg_psd_only_peaks, avg_psd_only_peaks_baselined):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    axs[0, 0].plot(freqs,avg_psd)
    axs[0, 0].plot(freqs[peaks], avg_psd[peaks], 'x', label='central noise frequencies')
    axs[0, 0].plot(freqs[peaks_neg], avg_psd[peaks_neg], '*',label='split points')
    xmin_f=[round(l) for l in left_ips]
    xmax_f=[round(r) for r in right_ips]
    xmin=[freqs[i] for i in xmin_f]
    xmax=[freqs[i] for i in xmax_f]
    axs[0, 0].hlines(y=width_heights, xmin=xmin, xmax=xmax, color="C3", label='detected peak bottom')
    axs[0, 0].set_title('1. PSD Welch with peaks, blended freqs not split yet. \n Shown as detected by peak_widths')
    axs[0, 0].set_xlim(freqs[0], freqs[-1])
    axs[0, 0].set_ylim(min(avg_psd)*-1.05, max(avg_psd)*1.05)
    axs[0, 0].legend()

    axs[0, 1].plot(freqs,avg_psd_only_signal)
    axs[0, 1].plot(freqs[peaks], avg_psd_only_signal[peaks], "x", label='central noise frequencies')
    axs[0, 1].hlines(y=width_heights, xmin=ips_l, xmax=ips_r, color="C3", label='detected peak bottom')
    axs[0, 1].set_title('2. PSD without noise, split blended freqs')
    axs[0, 1].vlines(x=freqs[split_points], color="k", ymin=min(avg_psd)*-1, ymax=max(avg_psd)*0.7, linestyle="dashed", linewidth=0.5, label='split peaks')
    axs[0, 1].set_xlim(freqs[0], freqs[-1])
    axs[0, 1].set_ylim(min(avg_psd)*-1.05, max(avg_psd)*1.05)
    axs[0, 1].legend()

    axs[1, 0].plot(freqs,avg_psd_only_peaks)
    axs[1, 0].plot(freqs[peaks], avg_psd_only_peaks[peaks], "x", label='central noise frequencies')
    axs[1, 0].hlines(y=width_heights, xmin=ips_l, xmax=ips_r, color="C3",label='detected peak bottom')
    axs[1, 0].set_title('3. Only noise peaks, split blended freqs')
    axs[1, 0].vlines(x=freqs[split_points], color="k", ymin=min(avg_psd)*-1, ymax=max(avg_psd)*0.7, linestyle="dashed", linewidth=0.5, label='split peaks')
    axs[1, 0].set_xlim(freqs[0], freqs[-1])
    axs[1, 0].set_ylim(min(avg_psd)*-1.05, max(avg_psd)*1.05)
    axs[1, 0].legend()

    axs[1, 1].plot(freqs,avg_psd_only_peaks_baselined)
    axs[1, 1].plot(freqs[peaks], avg_psd_only_peaks_baselined[peaks], "x", label='central noise frequencies')
    axs[1, 1].hlines(y=[0]*len(freqs[peaks]), xmin=ips_l, xmax=ips_r, color="C3",label='baselined peak bottom')
    axs[1, 1].set_title('4. Noise peaks brought to basline, split blended freqs')
    axs[1, 1].vlines(x=freqs[split_points], color="k", ymin=min(avg_psd)*-1, ymax=max(avg_psd)*0.7, linestyle="dashed", linewidth=0.5,label='split peaks')
    axs[1, 1].set_xlim(freqs[0], freqs[-1])
    axs[1, 1].set_ylim(min(avg_psd)*-1.05, max(avg_psd)*1.05)
    axs[1, 1].legend()

    #plt.tight_layout()

    fig.suptitle('PSD: detecting noise peaks, splitting blended freqs, defining area under the curve.')

    return fig

def plot_one_psd(ch_name, freqs, one_psd, peak_indexes, peak_neg_indexes, noisy_freq_bands, unit):
    '''plot avg_psd with peaks and split points using plotly'''
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=one_psd, name=ch_name+' psd'))
    fig.add_trace(go.Scatter(x=freqs[peak_indexes], y=one_psd[peak_indexes], mode='markers', name='peaks'))
    fig.add_trace(go.Scatter(x=freqs[peak_neg_indexes], y=one_psd[peak_neg_indexes], mode='markers', name='peaks_neg'))
    fig.update_layout(title=ch_name+' PSD with noise peaks and split edges', xaxis_title='Frequency', yaxis_title='Amplitude ('+unit+')')
    #plot split points as vertical lines and noise bands as red rectangles:
    for fr_b in noisy_freq_bands:
        fig.add_vrect(x0=fr_b[0], x1=fr_b[-1], line_width=0.8, fillcolor="red", opacity=0.2, layer="below")
        #fig.add_vline(x=fr_b[0], line_width=0.5, line_dash="dash", line_color="black") #, annotation_text="split point", annotation_position="top right")
        #fig.add_vline(x=fr_b[-1], line_width=0.5, line_dash="dash", line_color="black") #, annotation_text="split point", annotation_position="top right")
    fig.update_yaxes(type="log")
    
    return fig


def find_noisy_freq_bands_complex(ch_name, freqs, one_psd, helper_plots: bool, m_or_g, prominence_lvl_pos: int, prominence_lvl_neg: int = 60):

    _, unit = get_tit_and_unit(m_or_g)
    # Run peak detection on psd -> get number of noise freqs, define freq bands around them
     
    prominence_pos=(max(one_psd) - min(one_psd)) / prominence_lvl_pos
    prominence_neg=(max(one_psd) - min(one_psd)) / prominence_lvl_neg
    noisy_freqs_indexes, _ = find_peaks(one_psd, prominence=prominence_pos)
    peaks_neg, _ = find_peaks(-one_psd, prominence=prominence_neg)
    peaks_neg = np.insert(peaks_neg, 0, 0, axis=0)
    peaks_neg = np.append(peaks_neg, len(freqs)-1)
    #insert 0 as index of first negative peak and last index as ind of last negative peak.

    noisy_freqs=freqs[noisy_freqs_indexes]

    _, _, left_ips, right_ips = peak_widths(one_psd, noisy_freqs_indexes, rel_height=1)


    # print('___MEG QC___: ', 'Central Freqs: ', freqs[noise_peaks])
    # print('___MEG QC___: ', 'Central Amplitudes: ', one_psd[noise_peaks])

    #turn found noisy segments into frequency bands around the central noise frequency:
    noisy_freq_bands_idx=[]
    for ip_n, _ in enumerate(noisy_freqs_indexes):
        #+1 here because I  will use these values as range,and range in python is usually "up to the value but not including", this should fix it to the right rang
        noisy_freq_bands_idx.append([round(left_ips[ip_n]), round(right_ips[ip_n])+1])

    #2* Split the blended frequency bands into separate bands based on the negative peaks:

    noisy_bands_indexes_final, split_points = split_blended_freqs_at_negative_peaks(noisy_freq_bands_idx, noisy_freqs_indexes, peaks_neg)

    #Get actual freq bands from their indexes:
    noisy_bands_final=[]
    for fr_b in noisy_bands_indexes_final:
        noisy_bands_final.append([freqs[fr_b][0], freqs[fr_b][1]])

    if helper_plots is True: #visual of the split
        fig = plot_one_psd(ch_name, freqs, one_psd, noisy_freqs_indexes, peaks_neg, noisy_bands_final, unit)
        fig.show()

    
    return noisy_freqs, noisy_freqs_indexes, noisy_bands_final, noisy_bands_indexes_final, peaks_neg


def find_noisy_freq_bands_simple(ch_name, freqs, one_psd, helper_plots: bool, m_or_g, prominence_lvl_pos: int, band_length: float):
    '''In this approach create frequency band around central noise frequency just by adding -x...+x Hz around.'''

    prominence_pos=(max(one_psd) - min(one_psd)) / prominence_lvl_pos
    noisy_freqs_indexes, _ = find_peaks(one_psd, prominence=prominence_pos)

    if noisy_freqs_indexes.size==0:

        if helper_plots is True: #visual
            _, unit = get_tit_and_unit(m_or_g)
            fig = plot_one_psd(ch_name, freqs, one_psd, [], [], [], unit)
            fig.show()

        return [], [], [], [], []

    noisy_freqs=freqs[noisy_freqs_indexes]

    freq_res = freqs[1] - freqs[0]
    #make frequency bands around the central noise frequency (-2...+2 Hz band around the peak):
    noisy_bands_final=[]
    split_points=[]
    for iter, peak in enumerate(noisy_freqs):
        noisy_bands_final.append([peak - band_length, peak + band_length])
        #if bands overlap - split them:
        if iter>0:  #on the second iteration:
            if noisy_bands_final[-1][0]<=noisy_bands_final[-2][1]: #if the end of the previous band is after the start of the current band
                overlap_amount=noisy_bands_final[-2][1]-noisy_bands_final[-1][0]
                split_point=noisy_bands_final[-1][0]+overlap_amount/2
                split_point=round(split_point/freq_res)*freq_res #round the split point to freq_res:
                noisy_bands_final[-2][1]=split_point #split the previous band
                noisy_bands_final[-1][0]=split_point #split the current band
                split_points.append(split_point)

    # It might happen that when the band was created around the noise frequency, it is outside of the freqs range.
    # For example noisy freq is 1Hz, band is -2..+2Hz, freq rage was 0.5...100Hz. 
    # In this case we need to set the first and last bands to the edge of the freq range:

    if noisy_bands_final[0][0]<freqs[0]: #if the first band starts before the first freq in freqs:
        noisy_bands_final[0][0]=freqs[0]
        print('___MEG QC___: ', 'First band starts before the first freq in freqs, setting it to the first freq in freqs')
    if noisy_bands_final[-1][1]>freqs[-1]: #if the last band ends after the last freq in freqs:
        noisy_bands_final[-1][1]=freqs[-1]
        print('___MEG QC___: ', 'Last band ends after the last freq in freqs, setting it to the last freq in freqs')

    if helper_plots is True: #visual of the split
        _, unit = get_tit_and_unit(m_or_g)
        fig = plot_one_psd(ch_name, freqs, one_psd, noisy_freqs_indexes, [], noisy_bands_final, unit)
        fig.show()

    #find indexes of noisy_bands_final in freqs:
    noisy_bands_indexes_final=[]
    for band in noisy_bands_final:
        noisy_bands_indexes_final.append([np.where(freqs==band[0])[0][0], np.where(freqs==band[1])[0][0]])

    return noisy_freqs, noisy_freqs_indexes, noisy_bands_final, noisy_bands_indexes_final, split_points


def find_number_and_power_of_noise_freqs(ch_name: str, freqs: list, one_psd: list, pie_plotflag: bool, helper_plots: bool, m_or_g: str, cut_noise_from_psd: bool, prominence_lvl_pos: int, prominence_lvl_neg: int = 60, simple_or_complex='simple'):

    """
    1. Calculate average psd curve over all channels
    2. Run peak detection on it -> get number of noise freqs. Split blended freqs
    3*. Fit a curve to the general psd OR cut the noise peaks at the point they start and baseline them to 0.
    4. Calculate area under the curve for each noisy peak: area is limited to where amplitude crosses the fitted curve. - count from there.
    
    
    prominence_lvl will be different for average psd and psd of 1 channel, because average has small peaks smoothed.
    higher prominence_lvl means more peaks will be detected.
    prominence_lvl_pos is used to detect positive peaks - central frequencies of noise bands (recommended: 50 for average, 15 for 1 channel)
    prominence_lvl_neg is used only to find the beginnning of the noise band. it should always be a large numbe,\r, for both cases average or individual channel
        small number will make it collect smaller peaks into the same band. (recommended 60 for both cases)
    """

    m_or_g_tit, unit = get_tit_and_unit(m_or_g)

    #Total amplitude of the signal together with noise:
    freq_res = freqs[1] - freqs[0]
    total_amplitude = simpson(one_psd, dx=freq_res) 

    if simple_or_complex == 'simple':
        noisy_freqs, noisy_freqs_indexes, noisy_bands_final, noisy_bands_indexes_final, split_points = find_noisy_freq_bands_simple(ch_name, freqs, one_psd, helper_plots, m_or_g, prominence_lvl_pos, band_length=1)
        peaks_neg = []
    elif simple_or_complex == 'complex':
        noisy_freqs, noisy_freqs_indexes, noisy_bands_final, noisy_bands_indexes_final, peaks_neg = find_noisy_freq_bands_complex(ch_name, freqs, one_psd, helper_plots, m_or_g, prominence_lvl_pos, prominence_lvl_neg)
    else:
        print('simple_or_complex should be either "simple" or "complex"')
        return

    if cut_noise_from_psd is True:
        #3. Fit the curve to the general psd OR cut the noise peaks at the point they start and baseline them to 0.
        psd_of_noise_peaks = cut_the_noise_from_psd(noisy_bands_indexes_final, freqs, one_psd, helper_plots)
        # if helper_plots is True: #visual of the split and cut
        #     fig = make_helper_plots(freqs, one_psd, noisy_freqs_indexes, peaks_neg, bend_start, bend_end, split_points, ips_l, ips_r, width_heights, avg_psd_only_signal, avg_psd_only_peaks, psd_of_noise_peaks)
        #     fig.show()

        #Plot psd before and after cutting the noise:
        if helper_plots is True:
            fig = plot_one_psd(ch_name, freqs, one_psd, noisy_freqs_indexes, peaks_neg, noisy_bands_final, unit)
            fig.show()
            fig2 = plot_one_psd(ch_name, freqs, psd_of_noise_peaks, noisy_freqs_indexes, peaks_neg, noisy_bands_final, unit)
            fig2.show()

        #print('___MEG QC___: ', 'Total amplitude: ', total_amplitude)
    else:
        psd_of_noise_peaks = one_psd


    #4. Calculate area under the curve for each noisy peak: 
    # if cut the noise -> area is limited to where amplitude crosses the fitted curve. - count from there to the peak amplitude.
    # if dont cut the noise -> area is calculated from 0 to the peak amplitude.
    

    noise_ampl=[]
    noise_ampl_relative_to_signal=[]

    for band in noisy_bands_final:

        idx_band = np.logical_and(freqs >= band[0], freqs <= band[-1]) 
        # Find closest indices of band in frequency vector so idx_band is a list of indices frequencies that 
        # correspond to this band.

        # Compute the absolute power of the band by approximating the area under the curve:
        band_ampl = simpson(psd_of_noise_peaks[idx_band], dx=freq_res) #power of chosen band
        noise_ampl+= [band_ampl] 

        #Calculate how much of the total power of the average signal goes into each of the noise freqs:
        noise_ampl_relative_to_signal.append(band_ampl / total_amplitude) # relative power: % of this band in the total bands power for this channel:

    #noise_ampl_relative_to_signal=[r[0] for r in noise_ampl_relative_to_signal]

    #print('___MEG QC___: ', 'BP', noise_ampl)
    #print('___MEG QC___: ', 'Amount of noisy freq in total signal in percent', [b*100 for b in noise_ampl_relative_to_signal])


    if pie_plotflag is True: # Plot pie chart of SNR:
        #Legend for the pie chart:
        bands_legend=[]
        for fr_n, fr in enumerate(noisy_freqs):
            bands_legend.append(str(round(fr,1))+' Hz noise: '+str("%.2e" % noise_ampl[fr_n])+' '+unit) # "%.2e" % removes too many digits after coma
        main_signal_ampl = total_amplitude-sum(noise_ampl)
        #print('___MEG QC___: ', 'Main signal amplitude: ', main_signal_ampl, unit)
        main_signal_legend='Main signal: '+str("%.2e" % main_signal_ampl)+' '+unit
        bands_legend.append(main_signal_legend)

        Snr=noise_ampl_relative_to_signal+[1-sum(noise_ampl_relative_to_signal)]
        noise_pie_derivative = plot_pie_chart_freq(mean_relative_freq=Snr, tit='Signal and Noise. '+m_or_g_tit, bands_names=bands_legend)
        noise_pie_derivative.content.show()
    else:
        noise_pie_derivative = []

    #find out if the data contains powerline noise freqs - later to notch filter them before muscle artifact detection:
    powerline=[50, 60]
    powerline_freqs = [x for x in powerline if x in np.round(noisy_freqs)]

    return noise_pie_derivative, powerline_freqs, noise_ampl, noise_ampl_relative_to_signal, noisy_freqs

#%%
def make_simple_metric_psd_old(noise_ampl_global:dict, noise_ampl_relative_to_all_signal_global:dict, noisy_freqs_global:dict, noise_ampl_local:dict, noise_ampl_relative_to_all_signal_local:dict, noisy_freqs_local:dict, m_or_g_chosen:list, freqs:dict, channels: dict):
    """Make simple metric for psd.

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

    simple_metric_global={'mag':{}, 'grad':{}}
    for m_or_g in m_or_g_chosen:
        
        noisy_freqs_dict={}
        central_freqs=noisy_freqs_global[m_or_g]
        for fr_n, fr in enumerate(central_freqs):
            noisy_freqs_dict[fr]={'noise_ampl_global': float(noise_ampl_global[m_or_g][fr_n]), 'noise_ampl_relative_to_all_signal_global': round(float(noise_ampl_relative_to_all_signal_global[m_or_g][fr_n]*100), 2)}

        #need to convert to float, cos json doesnt understand numpy floats
        simple_metric_global[m_or_g] = noisy_freqs_dict


    simple_metric_local={'mag':{}, 'grad':{}}
    for m_or_g in m_or_g_chosen:

        noisy_freqs_dict_all_ch={}
        for ch in channels[m_or_g]:
            central_freqs=noisy_freqs_local[m_or_g][ch]
            noisy_freqs_dict={}     
            for fr_n, fr in enumerate(central_freqs):
                noisy_freqs_dict[fr]={'noise_ampl_local': float(noise_ampl_local[m_or_g][ch][fr_n]), 'noise_ampl_relative_to_all_signal_local':  round(float(noise_ampl_relative_to_all_signal_local[m_or_g][ch][fr_n]*100), 2)}
            noisy_freqs_dict_all_ch[ch]=noisy_freqs_dict

        simple_metric_local[m_or_g] = noisy_freqs_dict_all_ch


    _, unit_mag = get_tit_and_unit('mag')
    _, unit_grad = get_tit_and_unit('grad')

    simple_metric={
        "PSD_global": {
            "description": "Noise frequencies detected globally (based on average over all channels in this data file)",
            "mag": {
                "noisy_frequencies_count: ": len(noisy_freqs_global['mag']),
                "description": "Details show each detected noisy frequency in Hz with info about its amplitude and this amplitude relative to the whole signal amplitude",
                "noise_ampl_global_unit": unit_mag,
                "noise_ampl_relative_to_all_signal_global_unit": "%",
                "details": simple_metric_global['mag']},
            "grad": {
                "noisy_frequencies_count: ": len(noisy_freqs_global['grad']),
                "description": "Details show each detected noisy frequency in Hz with info about its amplitude and this amplitude relative to the whole signal amplitude",
                "noise_ampl_global_unit": unit_grad,
                "noise_ampl_relative_to_all_signal_global_unit": "%",
                "details": simple_metric_global['grad']}
            },  

        "PSD_local": {
            "description": "Noise frequencies detected locally (present only on individual channels)",
            "mag": {
                "description": "Details show each detected noisy frequency in Hz with info about its amplitude and this amplitude relative to the whole signal amplitude",
                "noise_ampl_local_unit": unit_mag,
                "noise_ampl_relative_to_all_signal_local_unit": "%",
                "details": simple_metric_local['mag']},
            "grad": {
                "description": "Details show each detected noisy frequency in Hz with info about its amplitude and this amplitude relative to the whole signal amplitude",
                "noise_ampl_local_unit": unit_grad,
                "noise_ampl_relative_to_all_signal_local_unit": "%",
                "details": simple_metric_local['grad']}
            }
        }

    return simple_metric

def make_dict_global_psd(noisy_freqs_global, noise_ampl_global, noise_ampl_relative_to_all_signal_global):
        
    noisy_freqs_dict={}
    for fr_n, fr in enumerate(noisy_freqs_global):
        noisy_freqs_dict[fr]={'noise_ampl_global': float(noise_ampl_global[fr_n]), 'percent_of_this_noise_ampl_relative_to_all_signal_global': round(float(noise_ampl_relative_to_all_signal_global[fr_n]*100), 2)}

    dict_global = {
        "noisy_frequencies_count: ": len(noisy_freqs_global),
        "details": noisy_freqs_dict}

    return dict_global


def make_dict_local_psd(noisy_freqs_local, noise_ampl_local, noise_ampl_relative_to_all_signal_local, channels):

    noisy_freqs_dict_all_ch={}
    for ch in channels:
        central_freqs=noisy_freqs_local[ch]
        noisy_freqs_dict={}     
        for fr_n, fr in enumerate(central_freqs):
            noisy_freqs_dict[fr]={'noise_ampl_local': float(noise_ampl_local[ch][fr_n]), 'percent_of_ths_noise_ampl_relative_to_all_signal_local':  round(float(noise_ampl_relative_to_all_signal_local[ch][fr_n]*100), 2)}
        noisy_freqs_dict_all_ch[ch]=noisy_freqs_dict

    dict_local = {"details": noisy_freqs_dict_all_ch}

    return dict_local


def make_simple_metric_psd(noise_ampl_global:dict, noise_ampl_relative_to_all_signal_global:dict, noisy_freqs_global:dict, noise_ampl_local:dict, noise_ampl_relative_to_all_signal_local:dict, noisy_freqs_local:dict, m_or_g_chosen:list, freqs:dict, channels: dict):

    metric_global_name = 'PSD_global'
    metric_global_description = 'Noise frequencies detected globally (based on average over all channels in this data file). Details show each detected noisy frequency in Hz with info about its amplitude and this amplitude relative to the whole signal amplitude.'
    metric_local_name = 'PSD_local'
    metric_local_description = 'Noise frequencies detected locally (present only on individual channels). Details show each detected noisy frequency in Hz with info about its amplitude and this amplitude relative to the whole signal amplitude'

    metric_global_content={'mag': None, 'grad': None}
    metric_local_content={'mag': None, 'grad': None}

    for m_or_g in m_or_g_chosen:

        metric_global_content[m_or_g]=make_dict_global_psd(noisy_freqs_global[m_or_g], noise_ampl_global[m_or_g], noise_ampl_relative_to_all_signal_global[m_or_g])
        metric_local_content[m_or_g]=make_dict_local_psd(noisy_freqs_local[m_or_g], noise_ampl_local[m_or_g], noise_ampl_relative_to_all_signal_local[m_or_g], channels[m_or_g])
        
    simple_metric = simple_metric_basic(metric_global_name, metric_global_description, metric_global_content['mag'], metric_global_content['grad'], metric_local_name, metric_local_description, metric_local_content['mag'], metric_local_content['grad'])

    return simple_metric


def get_nfft_nperseg(raw: mne.io.Raw, psd_step_size: float):
    '''Get nfft and nperseg parameters for welch psd function. 
    Allowes to always have the step size in psd wjich is chosen by the user. Recommended 0.5 Hz'''

    sfreq=raw.info['sfreq']
    nfft=int(sfreq/psd_step_size)
    nperseg=int(sfreq/psd_step_size)
    return nfft, nperseg

#%%
def PSD_meg_qc(psd_params: dict, channels:dict, raw: mne.io.Raw, m_or_g_chosen, helperplots: bool):
    """Main psd function.

    Freq spectrum peaks we see (visible on shorter interval, ALMOST NONE SEEN when Welch is done over all time):
    50, 100, 150 - powerline EU
    60, 120, 180 - powerline US
    6 - noise of shielding chambers 
    44 - meg noise
    17 - train station 
    10 - Secret :)
    1hz - highpass filter.
    flat spectrum is white noise process. Has same energy in every frequency (starts around 50Hz or even below)

    Output:
    derivs_psd: list of QC_derivative instances like figures and data frames."""
    
    # these parameters will be saved into a dictionary. this allowes to calculate for mag or grad or both:
    freqs = {}
    psds = {}
    derivs_psd = []
    noise_ampl_global={'mag':[], 'grad':[]}
    noise_ampl_relative_to_all_signal_global={'mag':[], 'grad':[]}
    noisy_freqs_global={'mag':[], 'grad':[]}
    noise_ampl_local={'mag':[], 'grad':[]}
    noise_ampl_relative_to_all_signal_local={'mag':[], 'grad':[]}
    noisy_freqs_local={'mag':[], 'grad':[]}

    powerline_freqs = []

    method = 'welch'
    nfft, nperseg = get_nfft_nperseg(raw, psd_params['psd_step_size'])

    for m_or_g in m_or_g_chosen:

        psds[m_or_g], freqs[m_or_g] = raw.compute_psd(method=method, fmin=psd_params['freq_min'], fmax=psd_params['freq_max'], picks=m_or_g, n_jobs=-1, n_fft=nfft, n_per_seg=nperseg).get_data(return_freqs=True)
        psds[m_or_g]=np.sqrt(psds[m_or_g]) # amplitude of the noise in this band. without sqrt it is power.

        psd_derivative=Plot_periodogram(m_or_g, freqs[m_or_g], psds[m_or_g], channels[m_or_g], method) 
        
        fig_power_with_name, dfs_with_name = Power_of_freq_meg(ch_names=channels[m_or_g], m_or_g = m_or_g, freqs = freqs[m_or_g], psds = psds[m_or_g], mean_power_per_band_needed = psd_params['mean_power_per_band_needed'], plotflag = True)

        #Calculate noise freqs globally: on the average psd curve over all channels together:
        avg_psd=np.mean(psds[m_or_g],axis=0) 
        noise_pie_derivative, powerline_freqs, noise_ampl_global[m_or_g], noise_ampl_relative_to_all_signal_global[m_or_g], noisy_freqs_global[m_or_g] = find_number_and_power_of_noise_freqs('Average', freqs[m_or_g], avg_psd, True, True, m_or_g, cut_noise_from_psd=False, prominence_lvl_pos=50, prominence_lvl_neg=100, simple_or_complex='simple')

        powerline_freqs += powerline_freqs

        derivs_psd += [psd_derivative] + [fig_power_with_name] + dfs_with_name +[noise_pie_derivative] 

        #Calculate noise freqs locally: on the psd curve of each channel separately:
        noise_ampl_local_all_ch={}
        noise_ampl_relative_to_all_signal_local_all_ch={}
        noisy_freqs_local_all_ch={}

        for ch_n, ch in enumerate(channels[m_or_g]): #plot only for some channels

            if (ch_n==1 or ch_n==35 or ch_n==70 or ch_n==92) and helperplots is True:
                helper_plotflag=True
            else:
                helper_plotflag=False

            _, _, noise_ampl_local_all_ch[ch], noise_ampl_relative_to_all_signal_local_all_ch[ch], noisy_freqs_local_all_ch[ch] = find_number_and_power_of_noise_freqs(ch, freqs[m_or_g], psds[m_or_g][ch_n,:], False, helper_plotflag, m_or_g, cut_noise_from_psd=False, prominence_lvl_pos=15, prominence_lvl_neg=150)
        
        noisy_freqs_local[m_or_g]=noisy_freqs_local_all_ch
        noise_ampl_local[m_or_g]=noise_ampl_local_all_ch
        noise_ampl_relative_to_all_signal_local[m_or_g]=noise_ampl_relative_to_all_signal_local_all_ch

        #collect all noise freqs from each channel, then find which freqs there are in total. Make a list for each freq: affected cannels, power of this freq in this channel, power of this freq relative to the main signal power in this channel



    # Make a simple metric for SNR:
    simple_metric=make_simple_metric_psd(noise_ampl_global, noise_ampl_relative_to_all_signal_global, noisy_freqs_global, noise_ampl_local, noise_ampl_relative_to_all_signal_local, noisy_freqs_local, m_or_g_chosen, freqs, channels)

    return derivs_psd, simple_metric, list(set(powerline_freqs)) #take only unique freqs if they are repeated for mags, grads


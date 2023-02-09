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


def split_blended_freqs(noisy_freq_bands_idx, peaks, peaks_neg, width_heights):

    '''Here should not use width_heights any more, but rather peaks and peaks_neg.
    Width heights should instead be calculated for each band as average between 
    the height of the limits of this bend after split is done.
    Probaly afer this fuction.'''

    split_points = []
    for n_peak, _ in enumerate(peaks):

        #find negative peaks before and after closest to the found positive noise peak.
  
        neg_peak_before=peaks_neg[np.argwhere(peaks_neg<peaks[n_peak])[-1][0]]
        neg_peak_after=peaks_neg[np.argwhere(peaks_neg>peaks[n_peak])[0][0]]
     
        if noisy_freq_bands_idx[n_peak][0] < neg_peak_before:
            noisy_freq_bands_idx[n_peak] = [i for i in range(neg_peak_before, noisy_freq_bands_idx[n_peak][-1])]

            split_points += [neg_peak_before]
            #if true, then this peak was blended with another one, 
            # so the bottom of both peaks (this and previous) needs to be brought 
            # to the same value.
            # (except the case when there were no peaks before)
            if n_peak>0:
                min_width_heights = min(width_heights[n_peak-1],width_heights[[n_peak]])
                width_heights[n_peak-1] = min_width_heights
                width_heights[n_peak] = min_width_heights

        if noisy_freq_bands_idx[n_peak][-1] > neg_peak_after:
            noisy_freq_bands_idx[n_peak] = [i for i in range(noisy_freq_bands_idx[n_peak][0], neg_peak_after)]

            split_points += [neg_peak_after]

            #if true, then this peak was blended with another one, 
            # so the bottom of both peaks (this and next) needs to be brought 
            # to the same value.
            # (except the case when there are no peaks after)
            if n_peak<len(peaks)-1:
                min_width_heights = min(width_heights[n_peak],width_heights[[n_peak+1]])
                width_heights[n_peak] = min_width_heights
                width_heights[n_peak+1] = min_width_heights

    return noisy_freq_bands_idx, width_heights, split_points

def cut_the_noise_from_psd(noisy_freq_bands_idx_split, width_heights_split, freqs, avg_psd):
    
    avg_psd_only_signal=avg_psd.copy()
    avg_psd_only_peaks=avg_psd.copy()
    avg_psd_only_peaks[:]=None
    avg_psd_only_peaks_baselined=avg_psd.copy()
    avg_psd_only_peaks_baselined[:]=0
    
    ips_l, ips_r = [], []
    for fr_n, fr_b in enumerate(noisy_freq_bands_idx_split):
        ips_l.append(freqs[fr_b][0])
        ips_r.append(freqs[fr_b][-1])
        
        avg_psd_only_signal[fr_b]=None #keep only main psd, remove noise bands, just for visual
        avg_psd_only_peaks[fr_b]=avg_psd[fr_b].copy() #keep only noise bands, remove psd, again for visual
        avg_psd_only_peaks_baselined[fr_b]=avg_psd[fr_b].copy()-[width_heights_split[fr_n]]*len(avg_psd_only_peaks[fr_b])
        #keep only noise bands and baseline them to 0 (remove the signal which is under the noise line)

        # clip the values to 0 if they are negative, they might appear in the beginning of psd curve, 
        # because the first peak might be above even the higher part of psd. should look intp it? 
        # maybe that pesk should not be seen as peak at all?
        avg_psd_only_peaks_baselined=np.array(avg_psd_only_peaks_baselined) 
        avg_psd_only_peaks_baselined = np.clip(avg_psd_only_peaks_baselined, 0, None) 

    return avg_psd_only_peaks_baselined, ips_l, ips_r, avg_psd_only_signal, avg_psd_only_peaks

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

def plot_one_psd(ch_name, freqs, avg_psd, peaks, peaks_neg, noisy_freq_bands_idx_split, unit):
    '''plot avg_psd with peaks and split points using plotly'''
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=avg_psd, name=ch_name+' psd'))
    fig.add_trace(go.Scatter(x=freqs[peaks], y=avg_psd[peaks], mode='markers', name='peaks'))
    fig.add_trace(go.Scatter(x=freqs[peaks_neg], y=avg_psd[peaks_neg], mode='markers', name='peaks_neg'))
    fig.update_layout(title=ch_name+' PSD with noise peaks and split points', xaxis_title='Frequency', yaxis_title='Amplitude ('+unit+')')
    #plot split points as vertical lines:
    for fr_b in noisy_freq_bands_idx_split:
        fig.add_vrect(x0=freqs[fr_b][0], x1=freqs[fr_b][-1], line_width=0, fillcolor="red", opacity=0.2, layer="below")
        fig.add_vline(x=freqs[fr_b][0], line_width=0.5, line_dash="dash", line_color="black") #, annotation_text="split point", annotation_position="top right")
        fig.add_vline(x=freqs[fr_b][-1], line_width=0.5, line_dash="dash", line_color="black") #, annotation_text="split point", annotation_position="top right")
    fig.update_yaxes(type="log")
    
    return fig

def find_number_and_power_of_noise_freqs(ch_name, freqs, one_psd, plotflag: bool, helper_plots: bool, m_or_g, cut_noise_from_psd: bool, prominence_lvl_pos: int, prominence_lvl_neg: int = 60):

    """
    1. Calculate average psd curve over all channels
    2. Run peak detection on it -> get number of noise freqs
    2*. Split blended freqs
    3. Fit curve to the general psd OR cut the noise peaks at the point they start and baseline them to 0.
    4. Calculate area under the curve for each noisy peak: area is limited to where amplitude crosses the fitted curve. - count from there.
    
    
    prominence_lvl will be different for average psd and psd of 1 channel, because average has small peaks smoothed.
    higher prominence_lvl means more peaks will be detected.
    prominence_lvl_pos is used to detect positive peaks - central frequencies of noise bands (recommended: 50 for average, 15 for 1 channel)
    prominence_lvl_neg is used only to find the beginnning of the noise band. it should always be a large numbe,\r, for both cases average or individual channel
        small number will make it collect smaller peaks into the same band. (recommended 60 for both cases)
    """

    m_or_g_tit, unit = get_tit_and_unit(m_or_g)

    #2. Run peak detection on it -> get number of noise freqs
     
    prominence_pos=(max(one_psd) - min(one_psd)) / prominence_lvl_pos
    prominence_neg=(max(one_psd) - min(one_psd)) / prominence_lvl_neg
    noise_peaks, _ = find_peaks(one_psd, prominence=prominence_pos)
    peaks_neg, _ = find_peaks(-one_psd, prominence=prominence_neg)
    peaks_neg = np.insert(peaks_neg, 0, 0, axis=0)
    peaks_neg = np.append(peaks_neg, len(freqs)-1)
    #insert 0 as index of first negative peak and last index as ind of lastr negative peak.


    _, width_heights, left_ips, right_ips = peak_widths(one_psd, noise_peaks, rel_height=1)


    # print('___MEG QC___: ', 'Central Freqs: ', freqs[noise_peaks])
    # print('___MEG QC___: ', 'Central Amplitudes: ', one_psd[noise_peaks])
    # print('___MEG QC___: ', 'width_heights: ', width_heights)

    #turn found noisy segments into frequency bands around the central noise frequency:
    noisy_freq_bands_idx=[]
    for ip_n, _ in enumerate(noise_peaks):
        #+1 here because I  will use these values as range,and range in python is usually "up to the value but not including", this should fix it to the right rang
        noisy_freq_bands_idx.append([fr for fr in np.arange((round(left_ips[ip_n])), round(right_ips[ip_n])+1)])
        if noisy_freq_bands_idx[ip_n][0]==noisy_freq_bands_idx[ip_n-1][-1]:
            noisy_freq_bands_idx[ip_n-1].pop(-1)
        #in case the last  element of one band is the same as first of another band, remove the last  elemnt of previos.So bands dont cross.

    #2* Split the blended frequency bands into separate bands:

    noisy_freq_bands_idx_split, width_heights_split, split_points = split_blended_freqs(noisy_freq_bands_idx, noise_peaks, peaks_neg, width_heights)

    if helper_plots is True: #visual of the split
        fig = plot_one_psd(ch_name, freqs, one_psd, noise_peaks, peaks_neg, noisy_freq_bands_idx_split, unit)
        fig.show()


    if cut_noise_from_psd is True:
        #3. Fit the curve to the general psd OR cut the noise peaks at the point they start and baseline them to 0.
        avg_psd_only_peaks_final, ips_l, ips_r, avg_psd_only_signal, avg_psd_only_peaks = cut_the_noise_from_psd(noisy_freq_bands_idx_split, width_heights_split, freqs, one_psd)

        if helper_plots is True: #visual of the split and cut
            fig = make_helper_plots(freqs, one_psd, noise_peaks, peaks_neg, left_ips, right_ips, split_points, ips_l, ips_r, width_heights, avg_psd_only_signal, avg_psd_only_peaks, avg_psd_only_peaks_final)
            fig.show()

        #Total amplitude of the signal together with noise:
        freq_res = freqs[1] - freqs[0]
        total_amplitude = simpson(one_psd, dx=freq_res) 
        #print('___MEG QC___: ', 'Total amplitude: ', total_amplitude)


    #4. Calculate area under the curve for each noisy peak: 
    # if cut the noise -> area is limited to where amplitude crosses the fitted curve. - count from there to the peak amplitude.
    # if dont cut the noise -> area is calculated from 0 to the peak amplitude.
    

    noise_ampl=[]
    noise_ampl_relative_to_signal=[]
 
    for fr_n, fr_b in enumerate(noisy_freq_bands_idx_split):

        if cut_noise_from_psd is True:
            bp_noise, _, _, _ = Power_of_band(freqs=freqs, f_low = freqs[fr_b][0], f_high= freqs[fr_b][-1], psds=np.array([avg_psd_only_peaks_final]))
        else: #if dont cut out peaks, calculate amplitude of noise from 0, not above the main psd curve:
            bp_noise, _, _, total_amplitude = Power_of_band(freqs=freqs, f_low = freqs[fr_b][0], f_high= freqs[fr_b][-1], psds=np.array([one_psd]))

        #print('___MEG QC___: ', 'Band: ', freqs[fr_b][0], freqs[fr_b][-1], ' ,total amplitude:', total_amplitude)

        noise_ampl+=bp_noise

        #Calculate how much of the total power of the average signal goes into each of the noise freqs:
        noise_ampl_relative_to_signal.append(bp_noise / total_amplitude) # relative power: % of this band in the total bands power for this channel:

    noise_ampl_relative_to_signal=[r[0] for r in noise_ampl_relative_to_signal]

    #print('___MEG QC___: ', 'BP', noise_ampl)
    #print('___MEG QC___: ', 'Amount of noisy freq in total signal in percent', [b*100 for b in noise_ampl_relative_to_signal])


    if plotflag is True: # Plot pie chart of SNR:
        #Legend for the pie chart:
        bands_legend=[]
        for fr_n, fr in enumerate(freqs[noise_peaks]):
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
    powerline_freqs = [x for x in powerline if x in np.round(freqs[noise_peaks])]

    return noise_pie_derivative, powerline_freqs, noise_ampl, noise_ampl_relative_to_signal, noise_peaks

#%%
def make_simple_metric_psd(noise_ampl_global:dict, noise_ampl_relative_to_all_signal_global:dict, noise_peaks_global:dict, noise_ampl_local:dict, noise_ampl_relative_to_all_signal_local:dict, noise_peaks_local:dict, m_or_g_chosen:list, freqs:dict, channels: dict):
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
        central_freqs=freqs[m_or_g][noise_peaks_global[m_or_g]]
        for fr_n, fr in enumerate(central_freqs):
            noisy_freqs_dict[fr]={'noise_ampl_global': float(noise_ampl_global[m_or_g][fr_n]), 'noise_ampl_relative_to_all_signal_global': round(float(noise_ampl_relative_to_all_signal_global[m_or_g][fr_n]*100), 2)}

        #need to convert to float, cos json doesnt understand numpy floats
        simple_metric_global[m_or_g] = noisy_freqs_dict


    simple_metric_local={'mag':{}, 'grad':{}}
    for m_or_g in m_or_g_chosen:

        noisy_freqs_dict_all_ch={}
        for ch in channels[m_or_g]:
            central_freqs=freqs[m_or_g][noise_peaks_local[m_or_g][ch]]
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
                "noisy_frequencies_count: ": len(noise_peaks_global['mag']),
                "description": "Details show each detected noisy frequency in Hz with info about its amplitude and this amplitude relative to the whole signal amplitude",
                "noise_ampl_global_unit": unit_mag,
                "noise_ampl_relative_to_all_signal_global_unit": "%",
                "Details": simple_metric_global['mag']},
            "grad": {
                "noisy_frequencies_count: ": len(noise_peaks_global['grad']),
                "description": "Details show each detected noisy frequency in Hz with info about its amplitude and this amplitude relative to the whole signal amplitude",
                "noise_ampl_global_unit": unit_grad,
                "noise_ampl_relative_to_all_signal_global_unit": "%",
                "Details": simple_metric_global['grad']}
            },  

        "PSD_local": {
            "description": "Noise frequencies detected locally (present only on individual channels)",
            "mag": {
                "description": "Details show each detected noisy frequency in Hz with info about its amplitude and this amplitude relative to the whole signal amplitude",
                "noise_ampl_local_unit": unit_mag,
                "noise_ampl_relative_to_all_signal_local_unit": "%",
                "Details": simple_metric_local['mag']},
            "grad": {
                "description": "Details show each detected noisy frequency in Hz with info about its amplitude and this amplitude relative to the whole signal amplitude",
                "noise_ampl_local_unit": unit_grad,
                "noise_ampl_relative_to_all_signal_local_unit": "%",
                "Details": simple_metric_local['grad']}
            }
        }

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
    noise_peaks_global={'mag':[], 'grad':[]}
    noise_ampl_local={'mag':[], 'grad':[]}
    noise_ampl_relative_to_all_signal_local={'mag':[], 'grad':[]}
    noise_peaks_local={'mag':[], 'grad':[]}

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
        noise_pie_derivative, powerline_freqs, noise_ampl_global[m_or_g], noise_ampl_relative_to_all_signal_global[m_or_g], noise_peaks_global[m_or_g] = find_number_and_power_of_noise_freqs('Average', freqs[m_or_g], avg_psd, True, True, m_or_g, cut_noise_from_psd=False, prominence_lvl_pos=50, prominence_lvl_neg=100)

        powerline_freqs += powerline_freqs

        derivs_psd += [psd_derivative] + [fig_power_with_name] + dfs_with_name +[noise_pie_derivative] 

        #Calculate noise freqs locally: on the psd curve of each channel separately:
        noise_ampl_local_all_ch={}
        noise_ampl_relative_to_all_signal_local_all_ch={}
        noise_peaks_local_all_ch={}

        for ch_n, ch in enumerate(channels[m_or_g]): #plot only for some channels

            if (ch_n==1 or ch_n==35 or ch_n==70 or ch_n==92) and helperplots is True:
                helper_plotflag=True
            else:
                helper_plotflag=False
            _, _, noise_ampl_local_all_ch[ch], noise_ampl_relative_to_all_signal_local_all_ch[ch], noise_peaks_local_all_ch[ch] = find_number_and_power_of_noise_freqs(ch, freqs[m_or_g], psds[m_or_g][ch_n,:], False, helper_plotflag, m_or_g, cut_noise_from_psd=False, prominence_lvl_pos=15, prominence_lvl_neg=150)
        
        noise_peaks_local[m_or_g]=noise_peaks_local_all_ch
        noise_ampl_local[m_or_g]=noise_ampl_local_all_ch
        noise_ampl_relative_to_all_signal_local[m_or_g]=noise_ampl_relative_to_all_signal_local_all_ch

        #collect all noise freqs from each channel, then find which freqs there are in total. Make a list for each freq: affected cannels, power of this freq in this channel, power of this freq relative to the main signal power in this channel



    # Make a simple metric for SNR:
    simple_metric=make_simple_metric_psd(noise_ampl_global, noise_ampl_relative_to_all_signal_global, noise_peaks_global, noise_ampl_local, noise_ampl_relative_to_all_signal_local, noise_peaks_local, m_or_g_chosen, freqs, channels)

    return derivs_psd, simple_metric, list(set(powerline_freqs)) #take only unique freqs if they are repeated for mags, grads


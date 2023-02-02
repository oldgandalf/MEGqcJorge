#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import mne
from scipy.integrate import simps
from universal_plots import Plot_periodogram, plot_pie_chart_freq, QC_derivative, get_tit_and_unit
from scipy.signal import find_peaks, peak_widths

# ISSUE IN /Volumes/M2_DATA/MEG_QC_stuff/data/from openneuro/ds004107/sub-mind004/ses-01/meg/sub-mind004_ses-01_task-auditory_meg.fif...
# COULDNT SPLIT  when filtered data
# In[42]:

def Power_of_band(freqs: np.ndarray, f_low: float, f_high: float, psds: np.ndarray):

    '''Calculates the power (area under the curve) of one chosen band (e.g. alpha, beta, gamma, delta, ...) for mag or grad.
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
        band_power = simps(psd_ch[idx_band], dx=freq_res) #power of chosen band
        total_power = simps(psd_ch, dx=freq_res) # power of all bands
        band_rel_power = band_power / total_power # relative power: % of this band in the total bands power for this channel:

        #devide the power of band by the  number of frequencies in the band, to compare with RMSE later:
        power_compare=band_power/sum(idx_band) 

        bandpower_per_ch_list.append(band_power)
        rel_bandpower_per_ch_list.append(band_rel_power)
        power_by_Nfreq_per_ch_list.append(power_compare)

    return(bandpower_per_ch_list, power_by_Nfreq_per_ch_list, rel_bandpower_per_ch_list)


    
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
        bandpower_per_ch_list, power_by_Nfreq_per_ch_list, rel_bandpower_per_ch_list=Power_of_band(freqs, f_low, f_high, psds)

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


    # renamed_df_power.to_csv('../derivatives/megqc/csv files/abs_power_'+m_or_g+'.csv')
    # renamed_df_power_freq.to_csv('../derivatives/megqc/csv files/power_by_Nfreq_'+m_or_g+'.csv')
    # renamed_df_rel_power.to_csv('../derivatives/megqc/csv files/relative_power_'+m_or_g+'.csv')


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


#%%
def make_simple_metric_psd(all_bp_noise, bp_noise_relative_to_signal, m_or_g, freqs, peaks):
    """Make simple metric for psd.

    Parameters
    ----------
    all_bp_noise : list
        noise in each band.
    bp_noise_relative_to_signal : list
        relative noise in each band."""

    m_or_g_tit, unit = get_tit_and_unit(m_or_g)
    
    noisy_freqs_dict={}
    for fr_n, fr in enumerate(freqs[peaks]):
        noisy_freqs_dict[fr]=['Power of noise: '+str(all_bp_noise[fr_n])+' '+unit, 'Power  of noise relative to signal power in percent: '+ str(bp_noise_relative_to_signal[fr_n]*100)]

    simple_metric={
        'Metric name': 'PSD '+m_or_g_tit, 
        'Number of noisy frequencies': len(peaks), 
        'Details': noisy_freqs_dict}


    return simple_metric

#%% Final simple metrics: number of noise frequencies + aea ubnder the curve for each of them. How to:

def split_blended_freqs_old(noisy_freq_bands_idx, width_heights, freqs):

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


def split_blended_freqs(noisy_freq_bands_idx, peaks, peaks_neg, width_heights, freqs):

    # print('___MEG QC___: ', 'peaks_neg', peaks_neg)
    # print('___MEG QC___: ', 'width_weights:', width_heights)

    split_points = []
    for n_peak, _ in enumerate(peaks):

        #find negative peaks before and after closest to the found positive noise peak.
  
        neg_peak_before=peaks_neg[np.argwhere(peaks_neg<peaks[n_peak])[-1][0]]
        neg_peak_after=peaks_neg[np.argwhere(peaks_neg>peaks[n_peak])[0][0]]

        #print('___MEG QC___: ', 'target peak', peaks[n_peak])
        #print('___MEG QC___: ', 'before and after', neg_peak_before, neg_peak_after)
     
        if noisy_freq_bands_idx[n_peak][0] < neg_peak_before:
            noisy_freq_bands_idx[n_peak] = [i for i in range(neg_peak_before, noisy_freq_bands_idx[n_peak][-1])]
            #print('___MEG QC___: ', 'new band', noisy_freq_bands_idx[n_peak])

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


def find_number_and_power_of_noise_freqs(freqs, psds, helper_plots: bool, m_or_g):

    """
    # 1. Calculate average psd curve over all channels
    # 2. Run peak detection on it -> get number of noise freqs
    # 2*. Split blended freqs
    # 3. Fit curve to the general psd OR cut the noise peaks at the point they start and baseline them to 0.
    # 4. Calculate area under the curve for each noisy peak: area is limited to where amplitude crosses the fitted curve. - count from there."""

    m_or_g_tit, unit = get_tit_and_unit(m_or_g)

    #1.
    avg_psd=np.mean(psds,axis=0)

    #2. 
     
    prominence=(max(avg_psd) - min(avg_psd)) / 50
    peaks, _ = find_peaks(avg_psd, prominence=prominence)
    peaks_neg, _ = find_peaks(-avg_psd, prominence=prominence)
    peaks_neg = np.insert(peaks_neg, 0, 0, axis=0)
    peaks_neg = np.append(peaks_neg, len(freqs)-1)
    #insert 0 as index of first negative peak and last index as ind of lastr negative peak.


    widths, width_heights, left_ips, right_ips = peak_widths(avg_psd, peaks, rel_height=1)

    #Plot signal, peaks and contour lines at which the widths where calculated
    from PSD_meg_qc import Power_of_band
    from universal_plots import plot_pie_chart_freq
    from scipy.integrate import simps

    print('___MEG QC___: ', 'Central Freqs: ', freqs[peaks])
    print('___MEG QC___: ', 'Central Amplitudes: ', avg_psd[peaks])
    print('___MEG QC___: ', 'width_heights: ', width_heights)


    avg_psd_only_signal=avg_psd.copy()
    avg_psd_only_peaks=avg_psd.copy()
    avg_psd_only_peaks[:]=None
    avg_psd_only_peaks_baselined=avg_psd.copy()
    avg_psd_only_peaks_baselined[:]=0

    noisy_freq_bands_idx=[]
    for ip_n, _ in enumerate(peaks):
        #noisy_freq_bands_idx.append([fr for fr in np.arange((round(left_ips[ip_n])), round(right_ips[ip_n]))])

        #+1 here because I  will use these values as range,and range in python is usually "up to the value but not including", this should fix it to the right rang
        noisy_freq_bands_idx.append([fr for fr in np.arange((round(left_ips[ip_n])), round(right_ips[ip_n])+1)])
        if noisy_freq_bands_idx[ip_n][0]==noisy_freq_bands_idx[ip_n-1][-1]:
            noisy_freq_bands_idx[ip_n-1].pop(-1)
        #in case the las  element of one band is the same as first of another band, remove the last  elemnt of previos.So bands dont cross.

    #2*
    #print('___MEG QC___: ', 'HERE! BEFORE SPLIT')
    #print('___MEG QC___: ', noisy_freq_bands_idx)
    #noisy_freq_bands_idx_split, width_heights_split = split_blended_freqs(noisy_freq_bands_idx, width_heights, freqs)

    noisy_freq_bands_idx_split, width_heights_split, split_points = split_blended_freqs(noisy_freq_bands_idx, peaks, peaks_neg, width_heights, freqs)
    #print('___MEG QC___: ', 'HERE! AFTER SPLIT')
    #print('___MEG QC___: ', noisy_freq_bands_idx_split)


    #3.
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


    if helper_plots is True:
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
        fig.show()


    #4.
    freq_res = freqs[1] - freqs[0]
    total_power = simps(avg_psd, dx=freq_res) # power of all signal
    print('___MEG QC___: ', 'Total power: ', total_power)

    all_bp_noise=[]
    all_bp_relative=[]
    bp_noise_relative_to_signal=[]

    avg_psd_only_peaks_baselined_new=np.array([avg_psd_only_peaks_baselined]) 

    for fr_n, fr_b in enumerate(noisy_freq_bands_idx_split):

        #print('___MEG QC___: ', 'band',  freqs[fr_b][0], freqs[fr_b][-1])
        bp_noise, _, bp_relative = Power_of_band(freqs=freqs, f_low = freqs[fr_b][0], f_high= freqs[fr_b][-1], psds=avg_psd_only_peaks_baselined_new)

        all_bp_noise+=bp_noise
        all_bp_relative+=bp_relative

        #Calculate how much of the total power of the average signal goes into each of the noise freqs:
        bp_noise_relative_to_signal.append(bp_noise / total_power) # relative power: % of this band in the total bands power for this channel:

    bp_noise_relative_to_signal=[r[0] for r in bp_noise_relative_to_signal]

    #print('___MEG QC___: ', 'Freq band for each peak:', ips_pair)
    print('___MEG QC___: ', 'BP', all_bp_noise)
    print('___MEG QC___: ', 'relative BP', all_bp_relative)
    print('___MEG QC___: ', 'Amount of noisy freq in total signal', bp_noise_relative_to_signal)


    #Legend for the pie chart:
    bands_legend=[]
    for fr_n, fr in enumerate(freqs[peaks]):
        bands_legend.append(str(fr)+' Hz noise: '+str(all_bp_noise[fr_n])+' '+unit)
    main_signal_legend='Main signal: '+str(total_power-sum(all_bp_noise))+' '+unit
    bands_legend.append(main_signal_legend)
    #bands_legend=[str(fr)+' Hz noise' for fr in freqs[peaks]]+['Main signal'] #legend version without showing the abs power


    Snr=bp_noise_relative_to_signal+[1-sum(bp_noise_relative_to_signal)]
    noise_pie_derivative = plot_pie_chart_freq(mean_relative_freq=Snr, tit='Signal and Noise. '+m_or_g_tit, bands_names=bands_legend)
    noise_pie_derivative.content.show()

    simple_metric_deriv=make_simple_metric_psd(all_bp_noise, bp_noise_relative_to_signal, m_or_g, freqs, peaks)

    #find out if the data contains powerline noise freqs - sed later to notch filter them before muscle artofact detection:
    powerline=[50, 60]
    powerline_freqs = [x for x in powerline if x in np.round(freqs[peaks])]

    return noise_pie_derivative, simple_metric_deriv, powerline_freqs

#%%
def PSD_meg_qc(psd_params: dict, channels:dict, raw: mne.io.Raw, m_or_g_chosen):
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
    simple_metrics_psd = []

    powerline_freqs = []
    for m_or_g in m_or_g_chosen:

        method = 'welch'
        psds[m_or_g], freqs[m_or_g] = raw.compute_psd(method=method, fmin=psd_params['freq_min'], fmax=psd_params['freq_max'], picks=m_or_g, n_jobs=-1, n_fft=psd_params['n_fft'], n_per_seg=psd_params['n_per_seg']).get_data(return_freqs=True)

        psd_derivative=Plot_periodogram(m_or_g, freqs[m_or_g], psds[m_or_g], channels[m_or_g], method) 
        
        fig_power_with_name, dfs_with_name = Power_of_freq_meg(ch_names=channels[m_or_g], m_or_g = m_or_g, freqs = freqs[m_or_g], psds = psds[m_or_g], mean_power_per_band_needed = psd_params['mean_power_per_band_needed'], plotflag = True)

        noise_pie_derivative, simple_metric_deriv, pf = find_number_and_power_of_noise_freqs(freqs[m_or_g], psds[m_or_g], True, m_or_g)
        powerline_freqs += pf

        simple_metrics_psd += [simple_metric_deriv]

        derivs_psd += [psd_derivative] + [fig_power_with_name] + dfs_with_name +[noise_pie_derivative] #+[simple_metric_deriv]

    return derivs_psd, simple_metrics_psd, list(set(powerline_freqs)) #take only unique freqs if they are repeted for mags, grads


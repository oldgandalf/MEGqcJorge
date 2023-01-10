#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_welch #tfr_morlet, psd_multitaper
from scipy.integrate import simps
from universal_plots import Plot_periodogram, plot_pie_chart_freq, QC_derivative, get_tit_and_unit
from scipy.signal import find_peaks, peak_widths

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

    file_path = None
    dfs_with_name = [
        QC_derivative(renamed_df_power,renamed_df_power_name,file_path, 'df'),
        QC_derivative(renamed_df_power_freq, renamed_df_power_freq_name, file_path, 'df'),
        QC_derivative(renamed_df_rel_power, renamed_df_rel_power_name, file_path, 'df')
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

        print(tit)
        for d in enumerate(power_dfs):
            print('  \n'+measure_title[d[0]])

            for band in enumerate(bands_names): #loop over bands
                mean_power_per_band = statistics.mean(d[1].loc[:,band[0]])
                
                if d[0]==0: #df_power_mag:
                    mean_abs.append(mean_power_per_band) 
                elif d[0]==1: #df_rel_power_mag:
                    mean_relative.append(mean_power_per_band) 
                elif d[0]==2: #df_power_freq_mag:
                    mean_power_nfreq.append(mean_power_per_band)

                print(band[1], mean_power_per_band)


        if plotflag is True: 
            psd_pie_derivative = plot_pie_chart_freq(mean_relative_freq=mean_relative, tit=tit, bands_names=bands_names)
        else:
            psd_pie_derivative = []

    
    return psd_pie_derivative, dfs_with_name

#%% Final simple metrics: number of noise frequencies + aea ubnder the curve for each of them. How to:

def split_blended_freqs(noisy_freq_bands_idx, width_heights, freqs):
    band = 0
    while band < len(noisy_freq_bands_idx):

        # Checking if the last element of every band is contained in the current band
        last = 0
        while last < len(noisy_freq_bands_idx):

            if (noisy_freq_bands_idx[last] != noisy_freq_bands_idx[band]) and (noisy_freq_bands_idx[last][-1] in noisy_freq_bands_idx[band]):
                
                print('split freq', noisy_freq_bands_idx[last][-1])
                split_index = noisy_freq_bands_idx[band].index(noisy_freq_bands_idx[last][-1])

                split_band_left = noisy_freq_bands_idx[band][:split_index]
                split_band_right = noisy_freq_bands_idx[band][split_index:]

                noisy_freq_bands_idx[last] = split_band_left
                noisy_freq_bands_idx[band] = split_band_right

                min_width_heights = min(width_heights[last],width_heights[band])
                width_heights[band] = min_width_heights
                width_heights[last] = min_width_heights

                band = 0
                last = 0

            last += 1
        band += 1


    return noisy_freq_bands_idx, width_heights

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

    simple_metric_deriv=QC_derivative(simple_metric,'PSD_'+m_or_g_tit, None, 'json')

    return simple_metric_deriv


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

    widths, width_heights, left_ips, right_ips = peak_widths(avg_psd, peaks, rel_height=1)

    #Plot signal, peaks and contour lines at which the widths where calculated
    from PSD_meg_qc import Power_of_band
    from universal_plots import plot_pie_chart_freq
    from scipy.integrate import simps

    print('Central Freqs: ', freqs[peaks])
    print('Central Amplitudes: ', avg_psd[peaks])
    print('width_heights: ', width_heights)

    #ips_pair=[]
    ips_l=[]
    ips_r=[]
    avg_psd_only_signal=avg_psd.copy()
    avg_psd_only_peaks=avg_psd.copy()
    avg_psd_only_peaks[:]=None
    avg_psd_only_peaks_baselined=avg_psd.copy()
    avg_psd_only_peaks_baselined[:]=0

    noisy_freq_bands_idx=[]
    for ip_n, _ in enumerate(peaks):
        #+1 here because I  will use these values as range,and range in pythonis usually "up to the value but not including", this should fix it to the right rang
        noisy_freq_bands_idx.append([fr for fr in np.arange((round(left_ips[ip_n])), round(right_ips[ip_n])+1)])

    #2*
    noisy_freq_bands_idx_split, width_heights_split = split_blended_freqs(noisy_freq_bands_idx, width_heights, freqs)

    #3.
    ips_l, ips_r = [], []
    for fr_n, fr_b in enumerate(noisy_freq_bands_idx_split):
        ips_l.append(freqs[fr_b][0])
        ips_r.append(freqs[fr_b][-1])
        
        avg_psd_only_signal[fr_b]=None #keep only main psd, remove noise bands, just for visual
        avg_psd_only_peaks[fr_b]=avg_psd[fr_b].copy() #keep only noise bands, remove psd, again for visual
        avg_psd_only_peaks_baselined[fr_b]=avg_psd[fr_b].copy()-[width_heights[fr_n]]*len(avg_psd_only_peaks[fr_b])
        #keep only noise bands and baseline them to 0 (remove the signal which is under the noise line)

        # clip the values to 0 if they are negative, they might appear in the beginning of psd curve, 
        # because the first peak might be above even the higher part of psd. should look intp it? 
        # maybe that pesk shoukld not be seen as peak at all?
        avg_psd_only_peaks_baselined=np.array(avg_psd_only_peaks_baselined) 
        avg_psd_only_peaks_baselined = np.clip(avg_psd_only_peaks_baselined, 0, None) 

    #4.
    freq_res = freqs[1] - freqs[0]
    total_power = simps(avg_psd, dx=freq_res) # power of all signal
    print('Total power: ', total_power)

    all_bp_noise=[]
    all_bp_relative=[]
    bp_noise_relative_to_signal=[]

    avg_psd_only_peaks_baselined_new=np.array([avg_psd_only_peaks_baselined]) 

    for fr_n, fr_b in enumerate(noisy_freq_bands_idx_split):

        #print('band',  freqs[fr_b][0], freqs[fr_b][-1])
        bp_noise, _, bp_relative = Power_of_band(freqs=freqs, f_low = freqs[fr_b][0], f_high= freqs[fr_b][-1], psds=avg_psd_only_peaks_baselined_new)

        all_bp_noise+=bp_noise
        all_bp_relative+=bp_relative

        #Calculate how much of the total power of the average signal goes into each of the noise freqs:
        bp_noise_relative_to_signal.append(bp_noise / total_power) # relative power: % of this band in the total bands power for this channel:

    bp_noise_relative_to_signal=[r[0] for r in bp_noise_relative_to_signal]

    #print('Freq band for each peak:', ips_pair)
    print('BP', all_bp_noise)
    print('relative BP', all_bp_relative)
    print('Amount of noisy freq in total signal', bp_noise_relative_to_signal)

    if helper_plots is True:
        import matplotlib.pyplot as plt

        plt.plot(freqs,avg_psd)
        plt.plot(freqs[peaks], avg_psd[peaks], 'x')
        plt.hlines(y=width_heights, xmin=ips_l, xmax=ips_r, color="C3")
        plt.title('PSD with peaks')
        plt.show()

        plt.plot(freqs,avg_psd_only_signal)
        plt.plot(freqs[peaks], avg_psd_only_signal[peaks], "x")
        plt.hlines(y=width_heights, xmin=ips_l, xmax=ips_r, color="C3")
        plt.title('PSD without noise')
        plt.show()

        plt.plot(freqs,avg_psd_only_peaks)
        plt.plot(freqs[peaks], avg_psd_only_peaks[peaks], "x")
        plt.hlines(y=width_heights, xmin=ips_l, xmax=ips_r, color="C3")
        plt.title('Only noise peaks')
        plt.show()

        plt.plot(freqs,avg_psd_only_peaks_baselined)
        plt.plot(freqs[peaks], avg_psd_only_peaks_baselined[peaks], "x")
        plt.title('Noise peaks brought to basline')
        plt.show()

    bands_names=[str(fr)+' Hz noise' for fr in freqs[peaks]]+['Main signal']
    Snr=bp_noise_relative_to_signal+[1-sum(bp_noise_relative_to_signal)]
    noise_pie_derivative = plot_pie_chart_freq(mean_relative_freq=Snr, tit='Signal and Noise. '+m_or_g_tit, bands_names=bands_names)
    noise_pie_derivative.content.show()


    simple_metric_deriv=make_simple_metric_psd(all_bp_noise, bp_noise_relative_to_signal, m_or_g, freqs, peaks)

    return noise_pie_derivative, simple_metric_deriv

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

    for m_or_g in m_or_g_chosen:

        psds[m_or_g], freqs[m_or_g] = raw.compute_psd(method='welch', fmin=psd_params['freq_min'], fmax=psd_params['freq_max'], picks=m_or_g, n_jobs=-1, n_fft=psd_params['n_fft'], n_per_seg=psd_params['n_per_seg']).get_data(return_freqs=True)

        psd_derivative=Plot_periodogram(m_or_g, freqs[m_or_g], psds[m_or_g], channels[m_or_g]) 
        
        fig_power_with_name, dfs_with_name = Power_of_freq_meg(ch_names=channels[m_or_g], m_or_g = m_or_g, freqs = freqs[m_or_g], psds = psds[m_or_g], mean_power_per_band_needed = psd_params['mean_power_per_band_needed'], plotflag = True)

        noise_pie_derivative, simple_metric_deriv = find_number_and_power_of_noise_freqs(freqs[m_or_g], psds[m_or_g], True, m_or_g)

        derivs_psd += [psd_derivative] + [fig_power_with_name] + dfs_with_name +[noise_pie_derivative]+[simple_metric_deriv]

    return derivs_psd

# In[56]:
# This command was used to convert notebook to this .py file:

# !jupyter nbconvert PSD_meg_qc.ipynb --to python


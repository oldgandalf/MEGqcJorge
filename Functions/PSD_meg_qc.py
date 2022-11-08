#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import mne
from mne.time_frequency import psd_welch #tfr_morlet, psd_multitaper

from universal_plots import Plot_periodogram, plot_pie_chart_freq, add_output_format
from universal_html_report import make_PSD_report, make_std_peak_report

# In[40]:

#Calculate frequency spectrum:
#UPD: as discussed with Jochem, only calculate over whole time, no over concatenated epochs. For concatenated version see Funks_old notebook.


def Freq_Spectrum_meg(data: mne.io.Raw, m_or_g: str, sid:str, freq_min:float or None, freq_max:float or None, n_fft: int, n_per_seg: int or None, freq_tmin: float or None, freq_tmax: float or None, ch_names: list):

    '''Calculates frequency spectrum of the data and if desired - plots them.

    Freq spectrum peaks we see (visible on shorter interval, ALMOST NONE SEEN when Welch is done over all time):
    50, 100, 150 - powerline EU
    6 noise of shielding chambers 
    44 meg noise
    17 - was it the train station near by?
    10 Secret :)
    1hz - highpass filter.
    flat spectrum is white noise process. Has same energy in every frequency (starts around 50Hz or even below)
    
    Args:
    data (mne.raw): data in raw format
    m_or_g (str): which channel type to use
    plotflag (bool): do you need plot or not
    sid (str): subject id number, like '1'
    freq_min (float): minimal frequency of interest for frequency spectrum decomposition
    freq_max (float): maximal frequency of interest for frequency spectrum decomposition
    n_fft (float): The length of FFT used, must be >= n_per_seg (default: 256). The segments will be zero-padded if n_fft > n_per_seg. 
        If n_per_seg is None, n_fft must be <= number of time points in the data. (*)
    n_per_seg (float): Length of each Welch segment (windowed with a Hamming window). Defaults to None, which sets n_per_seg equal to n_fft. (*)
    (*) These influence the bandwidth.
    freq_tmin (float): crop a chun of data for psd calculation: start time (instead could just pass the already cropped data). 
        If None - calculates over whole data
    freq_tmax (float): crop a chun of data for psd calculation: end time (instead could just pass the already cropped data). 
        If None - calculates over whole data
    ch_names (list of tuples): mag or grad channel names + their indexes

    Returns:
    freqs (np.ndarray): numpy array of frequencies 
    psds (np.ndarray): numpy array of power spectrum dencities 
    + if plotflaf is True:
    PSD plot + saves them as html files
    '''

    if m_or_g == 'mags':
        picks = mne.pick_types(data.info, meg='mag', eeg=False, eog=False, stim=False)
        tit = 'Magnetometers'
    elif m_or_g == 'grads':
        picks = mne.pick_types(data.info, meg='grad', eeg=False, eog=False, stim=False)
        tit = 'Gradiometers'
    else:
        TypeError('Check channel type')

    psds, freqs = psd_welch(data, fmin=freq_min, fmax=freq_max, n_jobs=-1, picks=picks, n_fft=n_fft, n_per_seg=n_per_seg, tmin=freq_tmin, tmax=freq_tmax, verbose=False)
    
    fig, fig_path, fig_desc=Plot_periodogram(tit, freqs, psds, sid, ch_names) 

    return freqs, psds, fig, fig_path, fig_desc
    

# In[42]:

def Power_of_band(freqs: np.ndarray, f_low: np.ndarray, f_high: float, psds: float):

    '''Calculates the power (area under the curve) of one chosen band (e.g. alpha, beta, gamma, delta, ...) for mags or grads.
    Adopted from: https://raphaelvallat.com/bandpower.html

    This function is called in Power_of_freq_meg
    
    Args:
    freqs (np.ndarray): numpy array of frequencies,
    psds (np.ndarray): numpy array of power spectrum dencities,
    f_low (float): minimal frequency of the chosend band, in Hz (For dekta it would be: 0.5),
    f_high (float): maximal frequency of the chosend band, in Hz (For dekta it would be: 4).


    Returns:
    power_per_band_list (list): list of powers of each band like: [abs_power_of_delta, abs_power_of_gamma, etc...] - in absolute values
    power_by_Nfreq_per_band_list (list): list of powers of bands divided by the  number of frequencies in the band - to compare 
        with RMSE later. Like: [power_of_delta/n_freqs, power_of_gamma/n_freqs, etc...]
    rel_power_per_band_list (list): list of power of each band like: [rel_power_of_delta, rel_power_of_gamma, etc...] - in relative  
        (percentage) values: what percentage of the total power does this band take.

    '''
    
    from scipy.integrate import simps

    power_per_band_list=[]
    rel_power_per_band_list=[]
    power_by_Nfreq_per_band_list=[]

    idx_band = np.logical_and(freqs >= f_low, freqs <= f_high) 
    # Find closest indices of band in frequency vector so idx_band is a list of indices frequencies that 
    # correspond to this band. F.e. for delta band these would be the indices of 0.5 ... 4 Hz)

    for ch in enumerate(psds): 
    #loop over channels. psd_ch_m is psd of partigular channel

        psd_ch=np.array(ch[1])

        # Compute Area under the curve (power):
        # Frequency resolution
        freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

        # Compute the absolute power by approximating the area under the curve:
        band_power = simps(psd_ch[idx_band], dx=freq_res) #power of chosen band
        total_power = simps(psd_ch, dx=freq_res) # power of all bands
        band_rel_power = band_power / total_power # relative power: % of this band in the total bands power for this channel:

        #devide the power of band by the  number of frequencies in the band, to compare with RMSE later:
        power_compare=band_power/sum(idx_band) 

        power_per_band_list.append(band_power)
        rel_power_per_band_list.append(band_rel_power)
        power_by_Nfreq_per_band_list.append(power_compare)

    return(power_per_band_list, power_by_Nfreq_per_band_list, rel_power_per_band_list)


    
# In[53]:

def Power_of_freq_meg(ch_names: list, m_or_g: str, freqs: np.ndarray, psds: np.ndarray, mean_power_per_band_needed: bool, plotflag: bool, sid: str):

    '''
    - Power of frequencies calculation for all mags + grads channels separately, 
    - Saving power + power/freq value into data frames.
    - If desired: creating a pie chart of mean power of every band over the entire data (all channels of 1 type together)
    
    Args:
    ch_names (list of tuples): channel names + index as list, 
    freqs (np.ndarray): numpy array of frequencies for mags  or grads
    psds (np.ndarray): numpy array of power spectrum dencities for mags or grads
    mean_power_per_band_needed (bool): need to calculate mean band power in the ENTIRE signal (averaged over all channels) or not.
        if True, results will also be printed.
    plotflag (bool): need to plot pie chart of mean_power_per_band_needed or not
    sid (str): subject id number, like '1'

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

    channel = [name[0] for name in ch_names]

    dict_power = {}
    dict_power_freq = {}
    dict_rel_power = {}


    for w in enumerate(wave_bands): #loop over bands
        
        f_low, f_high = w[1] # Define band lower and upper limits

        #loop over mags or grads:
        power_per_band_list, power_by_Nfreq_per_band_list, rel_power_per_band_list=Power_of_band(freqs, f_low, f_high, psds)

        dict_power[w[0]] = power_per_band_list
        dict_power_freq[w[0]] = power_by_Nfreq_per_band_list
        dict_rel_power[w[0]] = rel_power_per_band_list


    # Save all to data frames:
    df_power = pd.DataFrame(dict_power, index=channel)
    df_power_freq = pd.DataFrame(dict_power_freq, index=channel)
    df_rel_power = pd.DataFrame(dict_rel_power, index=channel)

    # Rename columns and extract to csv:

    renamed_df_power = df_power.rename(columns={0: "delta (0.5-4 Hz)", 1: "theta (4-8 Hz)", 2: "alpha (8-12 Hz)", 3: "beta (12-30 Hz)", 4: "gamma (30-100 Hz)"})
    renamed_df_power_freq = df_power_freq.rename(columns={0: "delta (0.5-4 Hz)", 1: "theta (4-8 Hz)", 2: "alpha (8-12 Hz)", 3: "beta (12-30 Hz)", 4: "gamma (30-100 Hz)"})
    renamed_df_rel_power = df_rel_power.rename(columns={0: "delta (0.5-4 Hz)", 1: "theta (4-8 Hz)", 2: "alpha (8-12 Hz)", 3: "beta (12-30 Hz)", 4: "gamma (30-100 Hz)"})

    # Create csv file  for the user:
    if sid=='001':
        renamed_df_power.to_csv('../derivatives/sub-'+sid+'/megqc/csv files/abs_power_'+m_or_g+'.csv')
        renamed_df_power_freq.to_csv('../derivatives/sub-'+sid+'/megqc/csv files/power_by_Nfreq_'+m_or_g+'.csv')
        renamed_df_rel_power.to_csv('../derivatives/sub-'+sid+'/megqc/csv files/relative_power_'+m_or_g+'.csv')

    #preassiign to have some returns in case plotting is not needed:

    fig = None
    fig_path = None

    if mean_power_per_band_needed is True: #if user wants to see average power per band over all channels - calculate and plot here:

        #Calculate power per band over all mags and all grads

        import statistics 

        power_dfs=[df_power, df_rel_power, df_power_freq] #keep them in this order!  

        bands_names=['delta', 'theta', 'alpha', 'beta', 'gamma']
        measure_title=['Average absolute power per band:', 'Average relative power per band:',
        'Average power/freq per band:']

        mean_abs=[]
        mean_relative=[]
        mean_power_nfreq=[]

        if m_or_g == 'mags':
            tit='Magnetometers'
        elif m_or_g == 'grads':
            tit='Gradiometers'
        else:
            TypeError ("Check channel type!")

        print(tit)
        for d in enumerate(power_dfs):
            print('  \n'+measure_title[d[0]])

            for band in enumerate(bands_names): #loop over bands
                mean_power_per_band = statistics.mean(d[1].loc[:,band[0]])
                
                if d[0]==0: #df_power_mags:
                    mean_abs.append(mean_power_per_band) 
                elif d[0]==1: #df_rel_power_mags:
                    mean_relative.append(mean_power_per_band) 
                elif d[0]==2: #df_power_freq_mags:
                    mean_power_nfreq.append(mean_power_per_band)

                print(band[1], mean_power_per_band)


        if plotflag is True: 
            fig, fig_path, fig_desc = plot_pie_chart_freq(mean_relative_freq=mean_relative, tit=tit, sid=sid)
            return fig, fig_path, fig_desc
        else:
            return None, None, None

#%%

def PSD_meg_qc(sid:str, config, channels:dict, filtered_d_resamp: mne.io.Raw, m_or_g_chosen):
    """Main psd function"""

    psd_section = config['PSD']
    freq_min = psd_section.getfloat('freq_min') 
    freq_max = psd_section.getfloat('freq_max') 
    mean_power_per_band_needed = psd_section.getboolean('mean_power_per_band_needed')
    n_fft = psd_section.getint('n_fft')
    n_per_seg = psd_section.getint('n_per_seg')
    
    # these parameters will be saved into a dictionary. this allowes to calculate for mags or grads or both:
    freqs = {}
    psds = {}
    fig_psd = {}
    fig_pie ={}
    fig_path_psd = {}
    fig_path_pie ={}
    fig_desc = {}
    fig_desc_pie = {}
    list_of_figures = []
    list_of_figures_pie = []
    list_of_figure_paths = []
    list_of_figure_paths_pie = []
    list_of_fig_descriptions = []
    list_of_fig_descriptions_pie = []

    for m_or_g in m_or_g_chosen:
        freqs[m_or_g], psds[m_or_g], fig_psd[m_or_g], fig_path_psd[m_or_g], fig_desc[m_or_g] = Freq_Spectrum_meg(data=filtered_d_resamp, m_or_g = m_or_g, sid=sid, freq_min=freq_min, freq_max=freq_max, 
        n_fft=n_fft, n_per_seg=n_per_seg, freq_tmin=None, freq_tmax=None, ch_names=channels[m_or_g])

        fig_pie[m_or_g],fig_path_pie[m_or_g], fig_desc_pie[m_or_g] = Power_of_freq_meg(ch_names=channels[m_or_g], m_or_g = m_or_g, freqs = freqs[m_or_g], psds = psds[m_or_g], mean_power_per_band_needed = mean_power_per_band_needed, plotflag = True, sid = sid)

        list_of_figures.append(fig_psd[m_or_g])
        list_of_figures_pie.append(fig_pie[m_or_g])
        list_of_fig_descriptions.append(fig_desc[m_or_g])


        list_of_figure_paths.append(fig_path_psd[m_or_g])
        list_of_figure_paths_pie.append(fig_path_pie[m_or_g])
        list_of_fig_descriptions_pie.append(fig_desc_pie[m_or_g])

    list_of_figures += list_of_figures_pie
    list_of_figure_paths += list_of_figure_paths_pie
    list_of_fig_descriptions += list_of_fig_descriptions_pie

    # to remove None values in list:
    list_of_figures = [i for i in list_of_figures if i is not None]
    list_of_figure_paths = [i for i in list_of_figure_paths if i is not None]
    list_of_figure_descriptions = [i for i in list_of_fig_descriptions if i is not None]

    # make_PSD_report(sid=sid, list_of_figure_paths=list_of_figure_paths)
    # make_std_peak_report(sid=sid, what_data='psd', list_of_figure_paths=list_of_figure_paths, config=config)

    return list_of_figures, list_of_figure_paths, list_of_figure_descriptions

# In[56]:
# This command was used to convert notebook to this .py file:

# !jupyter nbconvert PSD_meg_qc.ipynb --to python


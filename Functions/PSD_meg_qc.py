#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import mne
import plotly.graph_objects as go
from scipy.integrate import simpson
from universal_plots import Plot_psd, plot_pie_chart_freq, QC_derivative, get_tit_and_unit
from scipy.signal import find_peaks, peak_widths
from universal_html_report import simple_metric_basic

# ISSUE IN /Volumes/M2_DATA/MEG_QC_stuff/data/from openneuro/ds004107/sub-mind004/ses-01/meg/sub-mind004_ses-01_task-auditory_meg.fif...
# COULDNT SPLIT  when filtered data - check with new psd version
# In[42]:

def Power_of_band(freqs: np.ndarray, f_low: float, f_high: float, psds: np.ndarray):

    """Calculates the area under the curve of one chosen band (e.g. alpha, beta, gamma, delta, ...) for mag or grad.
    (Named here as power, but in fact it s amplitude of the signal, since psds are turned into amplitudes already.)
    Adopted from: https://raphaelvallat.com/bandpower.html

    This function is called in Power_of_freq_meg
    
    Parameters
    ----------
    freqs : np.ndarray
        numpy array of frequencies.
    f_low : float
        minimal frequency of the chosend band, in Hz (For delta it would be: 0.5).
    f_high : float
        maximal frequency of the chosend band, in Hz (For delta it would be: 4)
    psds : np.ndarray
        numpy array of power spectrum dencities. Expects array of arrays: channels*psds. !
        Will not work properly if 1 dimentional array given. In this case do: np.array([your_1d_array])

    Returns
    -------
    bandpower_per_ch_list : list
        List of amplitudes of each band like: [abs_power_of_delta, abs_power_of_gamma, etc...] - in absolute values
    power_by_Nfreq_per_ch_list : list
        List of amplitudes of bands divided by the  number of frequencies in the band - to compare
        with RMSE later. Like: [power_of_delta/n_freqs, power_of_gamma/n_freqs, etc...]
    rel_bandpower_per_ch_list : list
        List of amplitudes of each band like: [rel_power_of_delta, rel_power_of_gamma, etc...] - in relative values:
        what percentage of the total power does this band take.
    total_power : float
        Total power of the signal.

    """

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

    return bandpower_per_ch_list, power_by_Nfreq_per_ch_list, rel_bandpower_per_ch_list, total_power


    
# In[53]:

def Power_of_freq_meg(ch_names: list, m_or_g: str, freqs: np.ndarray, psds: np.ndarray, mean_power_per_band_needed: bool, plotflag: bool):

    """
    - Power of frequencies calculation for all channels, 
    - If desired: creating a pie chart of mean power of every band over the entire data.

    Parameters
    ----------
    ch_names : list
        List of channel names
    m_or_g : str
        'mag' or 'grad' - to choose which channels to calculate power for.
    freqs : np.ndarray
        numpy array of frequencies for mag  or grad
    psds : np.ndarray
        numpy array of power spectrum dencities for mag or grad
    mean_power_per_band_needed : bool
        need to calculate mean band power in the ENTIRE signal (averaged over all channels) or not.
    plotflag : bool
        need to plot pie chart of mean_power_per_band_needed or not

    Returns
    -------
    psd_pie_derivative : QC_derivative object or empty list
        If plotflag is True, returns one QC_derivative object, which is a plotly piechart figure.
        If plotflag is False, returns empty list.
    dfs_with_name : list
        List of dataframes with power of each frequency band in each channel

    """
    
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
            psd_pie_derivative = plot_pie_chart_freq(mean_relative_freq=mean_relative, m_or_g=m_or_g, bands_names=bands_names, fig_tit = "Relative amplitude of each band: ", fig_name = 'PSD_Relative_band_amplitude_all_channels_')
        else:
            psd_pie_derivative = []

    
    return psd_pie_derivative, dfs_with_name


def split_blended_freqs_at_the_lowest_point(noisy_bands_indexes:list[list], one_psd:list, noisy_freqs_indexes:list):

    """If there are 2 bands that are blended together, split them at the lowest point between 2 central noise frequencies.
    
    Parameters
    ----------
    noisy_bands_indexes : list[list]
        list of lists with indexes of noisy bands. Indexes! not frequency bands themselves. Index is defined by fequency/freq_resolution.
    one_psd : list
        vector if psd values for 1 channel (or 1 average over all channels)
    noisy_freqs_indexes : list
        list of indexes of noisy frequencies. Indexes! not frequencies themselves. Index is defined by fequency/freq_resolution.

    Returns
    -------
    noisy_bands_final_indexes : list[list]
        list of lists with indexes of noisy bands After the split.
        Indexes! not frequency bands themselves. Index is defined by fequency/freq_resolution.
    split_indexes : list
        list of indexes at which the bands were split (used later for plotting only).
    
    """
    
    #check that noisy_bands_indexes dont contain floats and negative numbers:
    for i, _ in enumerate(noisy_bands_indexes):
        for j, _ in enumerate(noisy_bands_indexes[i]):
            if noisy_bands_indexes[i][j] % 1 == 0: #if the number can be divided by 1 without remainder - it is integer. works for both int and float
                noisy_bands_indexes[i][j] = int(noisy_bands_indexes[i][j])
            else:
                print('ERROR: float index in noisy_bands_indexes', noisy_bands_indexes[i][j])
            if noisy_bands_indexes[i][j]<0:
                print('ERROR: negative index in noisy_bands_indexes', noisy_bands_indexes[i][j])

    noisy_bands_final_indexes = noisy_bands_indexes.copy()
    split_indexes = []

    if len(noisy_bands_indexes)>1: #if there are at least 2 bands
        for i, _ in enumerate(noisy_bands_indexes[:-1]):
            #if bands overlap - SPLIT them:
            if noisy_bands_final_indexes[i+1][0]<=noisy_bands_final_indexes[i][1]: #if the end of the previous band is after the start of the current band
                
                split_ind=np.argmin(one_psd[noisy_freqs_indexes[i]:noisy_freqs_indexes[i+1]])
                split_ind=noisy_freqs_indexes[i]+split_ind
                #here need to sum them, because argmin above found the index counted from the start of the noisy_freqs_indexes[iter-1] band, not from the start of the freqs array
                #print('split at the lowest point between 2 peaks', split_point)

                noisy_bands_final_indexes[i][1]=split_ind #assign end of the previous band
                noisy_bands_final_indexes[i+1][0]=split_ind #assigne beginnning of the current band
                split_indexes.append(int(split_ind))

    #print('split_indexes', split_indexes, 'noisy_bands_final_indexes', noisy_bands_final_indexes)

    return noisy_bands_final_indexes, split_indexes


def cut_the_noise_from_psd(noisy_bands_indexes: list[list], freqs: list, one_psd: list, helper_plots: bool, ch_name: str ='', noisy_freqs_indexes: list =[], unit: str =''):

    """Cut the noise peaks out of PSD curve. By default, it is not used, but can be turned on.
    If turned on, in the next steps, the area under the curve will be calculated only for the cut out peaks.

    By default, the area under the curve is calculated under the whole peak, uncluding the 'main brain signal' psd area + peak area. 
    This is done, because in reality we can not define, which part of the 'noisy' frequency is signal and which is noise. 
    In case later, during preprocessing this noise will be filtered out, it will be done completely: both the peak and the main psd area.

    Process:
    1. Find the height of the noise peaks. For this take the average between the height of the start and end of this noise bend.
    2. Cut the noise peaks out of PSD curve at the found height.
    3. Baseline the peaks: all the peaks are brought to 0 level.

    Function also can prodece helper plot to demonstrate the process.

    Parameters
    ----------
    noisy_bands_indexes : list[list]
        list of lists with indexes of noisy bands. Indexes! Not frequency bands themselves. Index is defined by fequency/freq_resolution.
    freqs : list
        vector of frequencies
    one_psd : list
        vector if psd values for 1 channel (or 1 average over all channels)
    helper_plots : bool
        if True, helper plots will be produced
    ch_name : str, optional
        channel name, by default '', used for plot display
    noisy_freqs_indexes : list, optional
        list of indexes of noisy frequencies. Indexes! not frequencies themselves. Index is defined by fequency/freq_resolution., 
        by default [] because we might have no noisy frequencies at all. Used for plot display.
    unit : str, optional
        unit of the psd values, by default '', used for plot display

    Returns
    -------
    psd_only_peaks_baselined : list
        vector of psd values for 1 channel (or 1 average over all channels) with the noise peaks cut out and baselined to 0 level.
        Later used to calculate area under the curve for the noise peaks only.
    """



    #band height will be chosen as average between the height of the limits of this bend.
    peak_heights = []
    for band_indexes in noisy_bands_indexes:
        peak_heights.append(np.mean([one_psd[band_indexes[0]], one_psd[band_indexes[-1]]]))

    psd_only_signal=one_psd.copy()
    psd_only_peaks=one_psd.copy()
    psd_only_peaks[:]=None
    psd_only_peaks_baselined=one_psd.copy()
    psd_only_peaks_baselined[:]=0
    
    for fr_n, fr_b in enumerate(noisy_bands_indexes):
        #turn fr_b into a range from start to end of it:
        fr_b=[i for i in range(fr_b[0], fr_b[1]+1)]
        
        psd_only_signal[fr_b]=None #keep only main psd, remove noise bands, just for visual
        psd_only_peaks[fr_b]=one_psd[fr_b].copy() #keep only noise bands, remove psd, again for visual
        psd_only_peaks_baselined[fr_b]=one_psd[fr_b].copy()-[peak_heights[fr_n]]*len(psd_only_peaks[fr_b])
        #keep only noise bands and baseline them to 0 (remove the signal which is under the noise line)

        # clip the values to 0 if they are negative, they might appear in the beginning of psd curve
        psd_only_peaks_baselined=np.array(psd_only_peaks_baselined) 
        psd_only_peaks_baselined = np.clip(psd_only_peaks_baselined, 0, None) 

    #Plot psd before and after cutting the noise:
    if helper_plots is True:
        fig1 = plot_one_psd(ch_name, freqs, one_psd, noisy_freqs_indexes, noisy_bands_indexes, unit)
        fig1.update_layout(title=ch_name+' Original noncut PSD')
        fig2 = plot_one_psd(ch_name, freqs, psd_only_peaks, noisy_freqs_indexes, noisy_bands_indexes, unit)
        fig2.update_layout(title=ch_name+' PSD with noise peaks only')
        fig3 = plot_one_psd(ch_name, freqs, psd_only_signal, noisy_freqs_indexes, noisy_bands_indexes, unit)
        fig3.update_layout(title=ch_name+' PSD with signal only')
        fig4 = plot_one_psd(ch_name, freqs, psd_only_peaks_baselined, noisy_freqs_indexes, noisy_bands_indexes, unit)
        fig4.update_layout(title=ch_name+' PSD with noise peaks only, baselined to 0')

        #put all 4 figures in one figure as subplots:
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=2, subplot_titles=(ch_name+' Original noncut PSD', ch_name+' PSD with noise peaks only', ch_name+' PSD with signal only', ch_name+' PSD with noise peaks only, baselined to 0'))
        fig.add_trace(fig1.data[0], row=1, col=1)
        fig.add_trace(fig1.data[1], row=1, col=1)
        fig.add_trace(fig2.data[0], row=1, col=2)
        fig.add_trace(fig2.data[1], row=1, col=2)
        fig.add_trace(fig3.data[0], row=2, col=1)
        fig.add_trace(fig3.data[1], row=2, col=1)
        fig.add_trace(fig4.data[0], row=2, col=2)
        fig.add_trace(fig4.data[1], row=2, col=2)
        #add rectagles to every subplot:
        for i in range(len(noisy_bands_indexes)):
            fig.add_shape(type="rect", xref="x", yref="y", x0=freqs[noisy_bands_indexes[i][0]], y0=0, x1=freqs[noisy_bands_indexes[i][1]], y1=max(one_psd), line_color="LightSeaGreen", line_width=2, fillcolor="LightSeaGreen", opacity=0.3, layer="below", row=1, col=1)
            fig.add_shape(type="rect", xref="x", yref="y", x0=freqs[noisy_bands_indexes[i][0]], y0=0, x1=freqs[noisy_bands_indexes[i][1]], y1=max(one_psd), line_color="LightSeaGreen", line_width=2, fillcolor="LightSeaGreen", opacity=0.3, layer="below", row=1, col=2)
            fig.add_shape(type="rect", xref="x", yref="y", x0=freqs[noisy_bands_indexes[i][0]], y0=0, x1=freqs[noisy_bands_indexes[i][1]], y1=max(one_psd), line_color="LightSeaGreen", line_width=2, fillcolor="LightSeaGreen", opacity=0.3, layer="below", row=2, col=1)
            fig.add_shape(type="rect", xref="x", yref="y", x0=freqs[noisy_bands_indexes[i][0]], y0=0, x1=freqs[noisy_bands_indexes[i][1]], y1=max(one_psd), line_color="LightSeaGreen", line_width=2, fillcolor="LightSeaGreen", opacity=0.3, layer="below", row=2, col=2)

        fig.update_layout(height=800, width=1300, title_text=ch_name+' PSD before and after cutting the noise')

        fig.show()
        #or show each figure separately:
        # fig1.show()
        # fig2.show()
        # fig3.show()
        # fig4.show()

    return psd_only_peaks_baselined


def plot_one_psd(ch_name: str, freqs: list, one_psd: list, peak_indexes: list, noisy_freq_bands_indexes: list[list], unit: str, yaxis_log = True):
    """plot PSD for one channels or for the average over multiple channels with noise peaks and split points using plotly.
    
    Parameters
    ----------
    ch_name : str
        channel name like 'MEG1234' or just 'Average'
    freqs : list
        list of frequencies
    one_psd : list
        list of psd values for one channels or for the average over multiple channels
    peak_indexes : list
        list of indexes of the noise peaks in the psd
    noisy_freq_bands_indexes : list[list]
        list of lists of indexes of the noisy frequency bands in the psd. Indexes! Not frequency bands themselves. Index is defined by fequency/freq_resolution.
    unit : str
        unit of the psd values. For example 'T/Hz'
    yaxis_log : bool, optional
        if True, y axis will be log, by default True. If False, y axis will be linear.

    Returns
    -------
    fig
        plotly figure of the psd with noise peaks and bands around them.

    """

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=one_psd, name=ch_name+' psd'))
    fig.add_trace(go.Scatter(x=freqs[peak_indexes], y=one_psd[peak_indexes], mode='markers', name='peaks'))
    #plot split points as vertical lines and noise bands as red rectangles:

    noisy_freq_bands = [[freqs[noisy_freq_bands_indexes[i][0]], freqs[noisy_freq_bands_indexes[i][1]]] for i in range(len(noisy_freq_bands_indexes))]

    for fr_b in noisy_freq_bands:
        fig.add_vrect(x0=fr_b[0], x1=fr_b[-1], line_width=1, fillcolor="red", opacity=0.2, layer="below")
    
    fig.update_layout(title=ch_name+' PSD with noise peaks and split edges', xaxis_title='Frequency', yaxis_title='Amplitude ('+unit+')')
    if yaxis_log is True:
        fig.update_yaxes(type="log")
    
    return fig


def find_noisy_freq_bands_complex(ch_name: str, freqs: list, one_psd: list, helper_plots: bool, m_or_g: str, prominence_lvl_pos: int):

    """'
    Detect the frequency band around the noise peaks.
    Complex approach: This function is trying to detect the actual start and end of peaks.
    1. Bands around the noise frequencies are created based on detected peak_width.
    2. If the found bands overlap, they are cut at the lowest point between 2 neighbouring noise peaks pn PSD curve.

    This function is not used by default, becausesuch a complex approach, even though can accurately find start and end of the noise bands, 
    is not very robust. It can sometimes take too much of the area arouund the noise peak, leading to a large part of the signel folsely counted as noise.
    By default, the more simple approach is used. See find_noisy_freq_bands_simple() function.

    Parameters
    ----------
    ch_name : str
        channel name like 'MEG1234' or just 'Average'. For plotting purposes only.
    freqs : list
        list of frequencies
    one_psd : list
        list of psd values for one channels or for the average over multiple channels
    helper_plots : bool
        if True, helper plots will be shown
    m_or_g : str
        'mag' or 'grad' - for plotting purposes only - to get the unit of the psd values
    prominence_lvl_pos : int
        prominence level for peak detection. The higher the value, the more peaks will be detected. 

    Returns
    -------
    noisy_freqs : list
        list of noisy frequencies
    noisy_freqs_indexes : list
        list of indexes of noisy frequencies in the psd
    noisy_bands_final : list[list]
        list of lists of noisy frequency bands. Each list contains 2 values: start and end of the band.
    noisy_bands_final_indexes : list[list]
        list of lists of indexes of noisy frequency bands. Each list contains 2 values: start and end of the band.
    split_indexes : list
        list of indexes of the split points in the psd
    
     """

    _, unit = get_tit_and_unit(m_or_g, True)
    # Run peak detection on psd -> get number of noise freqs, define freq bands around them
     
    prominence_pos=(max(one_psd) - min(one_psd)) / prominence_lvl_pos
    noisy_freqs_indexes, _ = find_peaks(one_psd, prominence=prominence_pos)

    if noisy_freqs_indexes.size==0: #if no noise found

        if helper_plots is True: #visual
            _, unit = get_tit_and_unit(m_or_g, True)
            fig = plot_one_psd(ch_name, freqs, one_psd, [], [], unit)
            fig.show()

        return [], [], [], [], [], []


    noisy_freqs=freqs[noisy_freqs_indexes]

    # Make frequency bands around noise frequencies on base of the detected width of the peaks:
    _, _, left_ips, right_ips = peak_widths(one_psd, noisy_freqs_indexes, rel_height=1)

    noisy_bands_indexes=[]
    for ip_n, _ in enumerate(noisy_freqs_indexes):
        #+1 here because I  will use these values as range,and range in python is usually "up to the value but not including", this should fix it to the right rang
        noisy_bands_indexes.append([round(left_ips[ip_n]), round(right_ips[ip_n])+1])


    # Split the blended freqs at the lowest point between 2 peaks 
    noisy_bands_final_indexes, split_indexes = split_blended_freqs_at_the_lowest_point(noisy_bands_indexes, one_psd, noisy_freqs_indexes)
    #print(ch_name, 'LOWEST POINT ', 'noisy_bands_final_indexes: ', noisy_bands_final_indexes, 'split_indexes: ', split_indexes)

    if helper_plots is True: #visual of the split
        fig = plot_one_psd(ch_name, freqs, one_psd, noisy_freqs_indexes, noisy_bands_final_indexes, unit)
        fig.show()

    #Get actual freq bands from their indexes:
    noisy_bands_final=[]
    for fr_b in noisy_bands_final_indexes:
        noisy_bands_final.append([freqs[fr_b][0], freqs[fr_b][1]])

    return noisy_freqs, noisy_freqs_indexes, noisy_bands_final, noisy_bands_final_indexes, split_indexes


def find_noisy_freq_bands_simple(ch_name: str, freqs: list, one_psd: list, helper_plots: bool, m_or_g: str, prominence_lvl_pos: int, band_length: float):
    """
    Detect the frequency band around the noise peaks.
    Simple approach: used by default.
    1. Create frequency band around central noise frequency just by adding -x...+x Hz around.
    2. If the found bands overlap, they are cut at the lowest point between 2 neighbouring noise peaks pn PSD curve.

    Parameters
    ----------
    ch_name : str
        channel name like 'MEG1234' or just 'Average'. For plotting purposes only.
    freqs : list
        list of frequencies
    one_psd : list
        list of psd values for one channels or for the average over multiple channels
    helper_plots : bool
        if True, helper plots will be shown
    m_or_g : str
        'mag' or 'grad' - for plotting purposes only - to get the unit of the psd values
    prominence_lvl_pos : int
        prominence level for peak detection. The higher the value, the more peaks will be detected. 
    band_length : float
        length of the frequency band around the noise peak. The band will be created by adding -band_length/2...+band_length/2 Hz around the noise peak.

    Returns
    -------
    noisy_freqs : list
        list of noisy frequencies
    noisy_freqs_indexes : list
        list of indexes of noisy frequencies in the psd
    noisy_bands_final : list[list]
        list of lists of noisy frequency bands. Each list contains 2 values: start and end of the band.
    noisy_bands_final_indexes : list[list]
        list of lists of indexes of noisy frequency bands. Each list contains 2 values: start and end of the band.
    split_indexes : list
        list of indexes of the split points in the psd
    
    """

    prominence_pos=(max(one_psd) - min(one_psd)) / prominence_lvl_pos
    noisy_freqs_indexes, _ = find_peaks(one_psd, prominence=prominence_pos)

    if noisy_freqs_indexes.size==0:

        if helper_plots is True: #visual
            _, unit = get_tit_and_unit(m_or_g, True)
            fig = plot_one_psd(ch_name, freqs, one_psd, [], [], unit)
            fig.show()

        return [], [], [], [], []

    #make frequency bands around the central noise frequency (-2...+2 Hz band around the peak):
    freq_res = freqs[1] - freqs[0]
    noisy_bands_indexes=[]
    for i, _ in enumerate(noisy_freqs_indexes):
        noisy_bands_indexes.append([round(noisy_freqs_indexes[i] - band_length/freq_res), round(noisy_freqs_indexes[i] + band_length/freq_res)])
        #need to round the indexes. because freq_res has sometimes many digits after coma, like 0.506686867543 instead of 0.5, so the indexes might be floats.


    # It might happen that when the band was created around the noise frequency, it is outside of the freqs range.
    # For example noisy freq is 1Hz, band is -2..+2Hz, freq rage was 0.5...100Hz. 
    # In this case we need to set the first and last bands to the edge of the freq range:

    if noisy_bands_indexes[0][0]<freqs[0]/freq_res: #if the first band starts before the first freq in freqs:
        noisy_bands_indexes[0][0]=freqs[0]/freq_res
    if noisy_bands_indexes[-1][1]>freqs[-1]/freq_res: #if the last band ends after the last freq in freqs:
        noisy_bands_indexes[-1][1]=freqs[-1]/freq_res
    
    # Split the blended freqs if their bands cross:
    noisy_bands_final_indexes, split_indexes = split_blended_freqs_at_the_lowest_point(noisy_bands_indexes, one_psd, noisy_freqs_indexes)
    if helper_plots is True: #visual of the split
        _, unit = get_tit_and_unit(m_or_g, True)
        fig = plot_one_psd(ch_name, freqs, one_psd, noisy_freqs_indexes, noisy_bands_final_indexes, unit)
        fig.show()

    noisy_freqs = freqs[noisy_freqs_indexes]

    noisy_bands_final = [[freqs[noisy_bands_final_indexes[i][0]], freqs[noisy_bands_final_indexes[i][1]]] for i in range(len(noisy_bands_final_indexes))]

    return noisy_freqs, noisy_freqs_indexes, noisy_bands_final, noisy_bands_final_indexes, split_indexes


def find_number_and_ampl_of_noise_freqs(ch_name: str, freqs: list, one_psd: list, pie_plotflag: bool, helper_plots: bool, m_or_g: str, cut_noise_from_psd: bool, prominence_lvl_pos: int, simple_or_complex='simple'):

    """
    1. Calculate average psd curve over all channels
    2. Run peak detection on it -> get number of noise freqs. Create the bands around them. Split blended freqs.
    3*. Fit a curve to the general psd OR cut the noise peaks at the point they start and baseline them to 0. Optional. By default not used
    4. Calculate area under the curve for each noisy peak (amplitude of the noise)): 
        If 3* was done: area is limited to where noise band crosses the fitted curve. - count from there.
        If not (default): area is limited to the whole area under the noise band, including the psd of the signal.
    5. Calculate what part of the whole psd is the noise (noise amplitude) and what part is the signal (signal amplitude) + plot as pie chart

    Parameters
    ----------
    ch_name : str
        name of the channel or 'average'
    freqs : list
        list of frequencies
    one_psd : list
        list of psd values for one channel or average psd
    pie_plotflag : bool
        if True, plot the pie chart
    helper_plots : bool
        if True, plot the helper plots (will show the noise bands, how they are split and how the peaks are cut from the psd if this is activated).
    m_or_g : str
        'mag' or 'grad'
    cut_noise_from_psd : bool
        if True, cut the noise peaks at the point they start and baseline them to 0. Optional. By default not used
    prominence_lvl_pos : int
        prominence level for peak detection (central frequencies of noise bands). The higher the value, the more peaks will be detected. 
        prominence_lvl will be different for average psd and psd of 1 channel, because average has small peaks smoothed.
    simple_or_complex : str
        'simple' or 'complex' approach to create the bands around the noise peaks. Simple by default. See functions above for details.

    Returns
    -------
    noise_pie_derivative : list
        list with QC_derivative object containing the pie chart with the noise amplitude and signal amplitude
    noise_ampl : list
        list of noise amplitudes for each noisy frequency band
    noise_ampl_relative_to_signal : list
        list of noise amplitudes relative to the signal amplitude for each noisy frequency band
    noisy_freqs : list
        list of noisy frequencies
    
    
    """

    m_or_g_tit, unit = get_tit_and_unit(m_or_g, True)

    #Total amplitude of the signal together with noise:
    freq_res = freqs[1] - freqs[0]
    total_amplitude = simpson(one_psd, dx=freq_res) 

    if simple_or_complex == 'simple':
        noisy_freqs, noisy_freqs_indexes, noisy_bands_final, noisy_bands_indexes_final, split_indexes = find_noisy_freq_bands_simple(ch_name, freqs, one_psd, helper_plots, m_or_g, prominence_lvl_pos, band_length=1)
    elif simple_or_complex == 'complex':
        noisy_freqs, noisy_freqs_indexes, noisy_bands_final, noisy_bands_indexes_final, split_indexes = find_noisy_freq_bands_complex(ch_name, freqs, one_psd, helper_plots, m_or_g, prominence_lvl_pos)
    else:
        print('simple_or_complex should be either "simple" or "complex"')
        return

    #3*. Cut the noise peaks at the point they start and baseline them to 0.
    if cut_noise_from_psd is True:
        psd_noise_final = cut_the_noise_from_psd(noisy_bands_indexes_final, freqs, one_psd, helper_plots, ch_name, noisy_freqs_indexes, unit)
    else:
        psd_noise_final = one_psd


    #4. Calculate area under the curve for each noisy peak: 
    # if cut the noise -> area is limited to where amplitude crosses the fitted curve. - count from there to the peak amplitude.
    # if dont cut the noise -> area is calculated from 0 to the peak amplitude.
    

    noise_ampl=[]
    noise_ampl_relative_to_signal=[]

    if noisy_bands_final: #if not empty
        for band in noisy_bands_final:

            idx_band = np.logical_and(freqs >= band[0], freqs <= band[-1]) 
            # Find closest indices of band in frequency vector so idx_band is a list of indices frequencies that 
            # correspond to this band.

            # Compute the absolute power of the band by approximating the area under the curve:
            band_ampl = simpson(psd_noise_final[idx_band], dx=freq_res) #power of chosen band
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
        noise_pie_derivative = plot_pie_chart_freq(mean_relative_freq=Snr, m_or_g=m_or_g, bands_names=bands_legend, fig_tit = "Ratio of signal and noise in the data: ", fig_name = 'PSD_SNR_all_channels_')
        noise_pie_derivative.content.show()
    else:
        noise_pie_derivative = []

    return noise_pie_derivative, noise_ampl, noise_ampl_relative_to_signal, noisy_freqs


def make_dict_global_psd(noisy_freqs_global: list, noise_ampl_global: list, noise_ampl_relative_to_all_signal_global: list):

    """ Create a dictionary for the global part of psd simple metrics. Global: overall part of noise in the signal (all channels averaged).

    Parameters
    ----------
    noisy_freqs_global : list
        list of noisy frequencies
    noise_ampl_global : list
        list of noise amplitudes for each noisy frequency band
    noise_ampl_relative_to_all_signal_global : list
        list of noise amplitudes relative to the total signal amplitude for each noisy frequency band
    
    Returns
    -------
    dict_global : dict
        dictionary with the global part of psd simple metrics
    """
        
    noisy_freqs_dict={}
    for fr_n, fr in enumerate(noisy_freqs_global):
        noisy_freqs_dict[fr]={'noise_ampl_global': float(noise_ampl_global[fr_n]), 'percent_of_this_noise_ampl_relative_to_all_signal_global': round(float(noise_ampl_relative_to_all_signal_global[fr_n]*100), 2)}

    dict_global = {
        "noisy_frequencies_count: ": len(noisy_freqs_global),
        "details": noisy_freqs_dict}

    return dict_global


def make_dict_local_psd(noisy_freqs_local: dict, noise_ampl_local: dict, noise_ampl_relative_to_all_signal_local: dict, channels: list):

    """ Create a dictionary for the local part of psd simple metrics. Local: part of noise in the signal for each channel separately.
    
    Parameters
    ----------
    noisy_freqs_local : dict
        dictionary with noisy frequencies for each channel
    noise_ampl_local : dict
        dictionary with noise amplitudes for each noisy frequency band for each channel
    noise_ampl_relative_to_all_signal_local : dict
        dictionary with noise amplitudes relative to the total signal amplitude for each noisy frequency band for each channel
        
    Returns
    -------
    dict_local : dict
        dictionary with the local part of psd simple metrics
        
    """

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

    """ Create a dictionary for the psd simple metrics.

    Parameters
    ----------
    noise_ampl_global : dict
        dictionary with noise amplitudes for each noisy frequency band 
    noise_ampl_relative_to_all_signal_global : dict
        dictionary with noise amplitudes relative to the total signal amplitude for each noisy frequency band 
    noisy_freqs_global : dict
        dictionary with noisy frequencies
    noise_ampl_local : dict
        dictionary with noise amplitudes for each noisy frequency band for each channel
    noise_ampl_relative_to_all_signal_local : dict
        dictionary with noise amplitudes relative to the total signal amplitude for each noisy frequency band for each channel
    noisy_freqs_local : dict
        dictionary with noisy frequencies for each channel
    m_or_g_chosen : list
        list with chosen channel types: 'mag' or/and 'grad'

    Returns
    -------
    simple_metric : dict
        dictionary with the psd simple metrics

    """

    metric_global_name = 'PSD_global'
    metric_global_description = 'Noise frequencies detected globally (based on average over all channels in this data file). Details show each detected noisy frequency in Hz with info about its amplitude and this amplitude relative to the whole signal amplitude.'
    metric_local_name = 'PSD_local'
    metric_local_description = 'Noise frequencies detected locally (present only on individual channels). Details show each detected noisy frequency in Hz with info about its amplitude and this amplitude relative to the whole signal amplitude'

    metric_global_content={'mag': None, 'grad': None}
    metric_local_content={'mag': None, 'grad': None}

    for m_or_g in m_or_g_chosen:

        metric_global_content[m_or_g]=make_dict_global_psd(noisy_freqs_global[m_or_g], noise_ampl_global[m_or_g], noise_ampl_relative_to_all_signal_global[m_or_g])
        metric_local_content[m_or_g]=make_dict_local_psd(noisy_freqs_local[m_or_g], noise_ampl_local[m_or_g], noise_ampl_relative_to_all_signal_local[m_or_g], channels[m_or_g])
        
    simple_metric = simple_metric_basic(metric_global_name, metric_global_description, metric_global_content['mag'], metric_global_content['grad'], metric_local_name, metric_local_description, metric_local_content['mag'], metric_local_content['grad'], psd=True)

    return simple_metric


def get_nfft_nperseg(raw: mne.io.Raw, psd_step_size: float):
    """Get nfft and nperseg parameters for Welch psd function. 
    Allowes to always have the step size in psd which is chosen by the user. Recommended 0.5 Hz.
    
    Parameters
    ----------
    raw : mne.io.Raw
        raw data
    psd_step_size : float
        step size for PSD chosen by user, recommended 0.5 Hz
        
    Returns
    -------
    nfft : int
        Number of points for fft. Used in welch psd function from mne.
        The length of FFT used, must be >= n_per_seg (default: 256). The segments will be zero-padded if n_fft > n_per_seg. 
        If n_per_seg is None, n_fft must be <= number of time points in the data.
    nperseg : int
        Number of points for each segment. Used in welch psd function from mne.
        Length of each Welch segment (windowed with a Hamming window). Defaults to None, which sets n_per_seg equal to n_fft.

    """

    sfreq=raw.info['sfreq']
    nfft=int(sfreq/psd_step_size)
    nperseg=int(sfreq/psd_step_size)
    return nfft, nperseg

#%%
def PSD_meg_qc(psd_params: dict, channels:dict, raw: mne.io.Raw, m_or_g_chosen: list, helperplots: bool):
    
    """Main psd function. Calculates:
    - psd for each channel
    - amplitudes (area under the curve) of functionally distinct frequency bands, such as delta (0.5–4 Hz), 
        theta (4–8 Hz), alpha (8–12 Hz), beta (12–30 Hz), and gamma (30–100 Hz) for each channel + average power of band over all channels
    - average psd over all channels
    - noise frequencies for average psd + creates a band around them
    - noise frequencies for each channel + creates a band around them
    - noise amplitudes (area under the curve) for each noisy frequency band for average psd
    - noise amplitudes (area under the curve) for each noisy frequency band for each channel.

    Freq spectrum peaks we can often see:
    50, 100, 150 - powerline EU
    60, 120, 180 - powerline US
    6 - noise of shielding chambers 
    44 - meg noise
    17 - train station 
    10 - Secret :)
    1hz - highpass filter.
    flat spectrum is white noise process. Has same energy in every frequency (starts around 50Hz or even below)

    Parameters
    ----------
    psd_params : dict
        dictionary with psd parameters originating from config file
    channels : dict
        dictionary with channel names for each channel type: 'mag' or/and 'grad'
    raw : mne.io.Raw
        raw data
    m_or_g_chosen : list
        list with chosen channel types: 'mag' or/and 'grad'
    helperplots : bool
        if True, helperplots will be created

    Returns
    -------
    derivs_psd : list
        list with the psd derivatives as QC_derivative objects (figures)
    simple_metric : dict
        dictionary with the psd simple metrics
    noisy_freqs_global : dict
        dictionary with noisy frequencies for average psd

    """
    
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

    method = 'welch'
    nfft, nperseg = get_nfft_nperseg(raw, psd_params['psd_step_size'])

    for m_or_g in m_or_g_chosen:

        psds[m_or_g], freqs[m_or_g] = raw.compute_psd(method=method, fmin=psd_params['freq_min'], fmax=psd_params['freq_max'], picks=m_or_g, n_jobs=-1, n_fft=nfft, n_per_seg=nperseg).get_data(return_freqs=True)
        psds[m_or_g]=np.sqrt(psds[m_or_g]) # amplitude of the noise in this band. without sqrt it is power.

        psd_derivative=Plot_psd(m_or_g, freqs[m_or_g], psds[m_or_g], channels[m_or_g], method) 
        
        fig_power_with_name, dfs_with_name = Power_of_freq_meg(ch_names=channels[m_or_g], m_or_g = m_or_g, freqs = freqs[m_or_g], psds = psds[m_or_g], mean_power_per_band_needed = psd_params['mean_power_per_band_needed'], plotflag = True)

        #Calculate noise freqs globally: on the average psd curve over all channels together:
        avg_psd=np.mean(psds[m_or_g],axis=0) 
        noise_pie_derivative, noise_ampl_global[m_or_g], noise_ampl_relative_to_all_signal_global[m_or_g], noisy_freqs_global[m_or_g] = find_number_and_ampl_of_noise_freqs('Average', freqs[m_or_g], avg_psd, True, True, m_or_g, cut_noise_from_psd=False, prominence_lvl_pos=50, simple_or_complex='simple')

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

            _, noise_ampl_local_all_ch[ch], noise_ampl_relative_to_all_signal_local_all_ch[ch], noisy_freqs_local_all_ch[ch] = find_number_and_ampl_of_noise_freqs(ch, freqs[m_or_g], psds[m_or_g][ch_n,:], False, helper_plotflag, m_or_g, cut_noise_from_psd=False, prominence_lvl_pos=15, simple_or_complex='simple')
        
        noisy_freqs_local[m_or_g]=noisy_freqs_local_all_ch
        noise_ampl_local[m_or_g]=noise_ampl_local_all_ch
        noise_ampl_relative_to_all_signal_local[m_or_g]=noise_ampl_relative_to_all_signal_local_all_ch

        #collect all noise freqs from each channel, then find which freqs there are in total. Make a list for each freq: affected cannels, power of this freq in this channel, power of this freq relative to the main signal power in this channel


    # Make a simple metric for SNR:
    simple_metric=make_simple_metric_psd(noise_ampl_global, noise_ampl_relative_to_all_signal_global, noisy_freqs_global, noise_ampl_local, noise_ampl_relative_to_all_signal_local, noisy_freqs_local, m_or_g_chosen, freqs, channels)

    return derivs_psd, simple_metric, noisy_freqs_global

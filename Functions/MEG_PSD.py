#Load data, filter, make folders
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import mne
from mne.time_frequency import psd_welch #tfr_morlet, psd_multitaper


def Plot_periodogram(tit:str, freqs: np.ndarray, psds:np.ndarray, sid: str, mg_names: list):

    '''Plotting periodogram on the data.

    Args:
    tit (str): title, like "Magnetometers", or "Gradiometers", 
    sid (str): subject id number, like '1'
    freqs (np.ndarray): numpy array of frequencies after performing Welch (or other method) psd decomposition
    psds (np.ndarray): numpy array of psds after performing Welch (or other method) psd decomposition
    mg_names (list of tuples): channel name + its index

    Returns:
    fig (go.Figure): plottly figure
    fig_path (str): path where the figure is saved as html file
    '''

    unit='?'
    if tit=='Magnetometers':
        unit='T/Hz'
    elif tit=='Gradiometers':
        unit='T/m / Hz'
    else:
        print('Please check tit input. Has to be "Magnetometers" or "Gradiometers"')

    mg_only_names=[n[0] for n in mg_names]

    df_psds=pd.DataFrame(np.sqrt(psds.T), columns=mg_only_names)

    fig = go.Figure()

    for col in df_psds:
        fig.add_trace(go.Scatter(x=freqs, y=df_psds[col].values, name=df_psds[col].name));

    #fig.update_xaxes(type="log")
    #fig.update_yaxes(type="log")
    
    fig.update_layout(
    title={
    'text': "Welch's periodogram for all "+tit,
    'y':0.85,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'},
    yaxis_title="Amplitude, "+unit,
    yaxis = dict(
        showexponent = 'all',
        exponentformat = 'e'),
    xaxis_title="Frequency (Hz)")
    fig.update_traces(hovertemplate='Frequency: %{x} Hz<br>Amplitude: %{y: .2e} T/Hz')

    fig.show()
    
    fig_name='PSD_over_all_data_'+tit+'.html'
    fig_path='../derivatives/sub-'+sid+'/megqc/figures/'+fig_name
    fig.write_html(fig_path)

    return fig, fig_path


def Freq_Spectrum_meg(data: mne.io.Raw, mags_or_grads: str, plotflag: bool, sid:str, freq_min:float or None, freq_max:float or None, n_fft: int, n_per_seg: int or None, freq_tmin: float or None, freq_tmax: float or None, ch_names: list):

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
    mags_or_grads (str): which channel type to use
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

    if mags_or_grads == 'mags':
        picks = mne.pick_types(data.info, meg='mag', eeg=False, eog=False, stim=False)
        tit = 'Magnetometers'
    elif mags_or_grads == 'grads':
        picks = mne.pick_types(data.info, meg='grad', eeg=False, eog=False, stim=False)
        tit = 'Gradiometers'
    else:
        TypeError('Check channel type')

    psds, freqs = psd_welch(data, fmin=freq_min, fmax=freq_max, n_jobs=-1, picks=picks, n_fft=n_fft, n_per_seg=n_per_seg, tmin=freq_tmin, tmax=freq_tmax, verbose=False)
    if plotflag==True:
        fig, fig_path=Plot_periodogram(tit, freqs, psds, sid, ch_names) 
        return(freqs, psds, fig_path) 

    return(freqs, psds)
    

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


def plot_pie_chart_freq(mean_relative_freq: list, tit: str, sid: str):
    
    ''''Pie chart representation of relative power of each frequency band in given data - in the entire 
    signal of mags or of grads, not separated by individual channels.

    Args:
    mean_relative_freq (list): list of power of each band like: [rel_power_of_delta, rel_power_of_gamma, etc...] - in relative  
        (percentage) values: what percentage of the total power does this band take,
    tit (str): title, like "Magnetometers", or "Gradiometers", 
    sid (str): subject id number, like '1'.
    
    Returns:
    fig (go.Figure): plottly piechart figure
    fig_path (str): path where the figure is saved as html file
    '''

    #If mean relative percentages dont sum up into 100%, add the 'unknown' part.
    mean_relative_unknown=[v * 100 for v in mean_relative_freq]  #in percentage
    power_unknown_m=100-(sum(mean_relative_freq))*100
    if power_unknown_m>0:
        mean_relative_unknown.append(power_unknown_m)
        bands_names=['delta', 'theta', 'alpha', 'beta', 'gamma', 'unknown']
    else:
        bands_names=['delta', 'theta', 'alpha', 'beta', 'gamma']

    fig = go.Figure(data=[go.Pie(labels=bands_names, values=mean_relative_unknown)])
    fig.update_layout(
    title={
    'text': "Relative power of each band: "+tit,
    'y':0.85,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'})

    fig.show()

    fig_name='Relative_power_per_band_over_all_channels_'+tit+'.html'
    fig_path='../derivatives/sub-'+sid+'/megqc/figures/'+fig_name
    fig.write_html(fig_path)

    return fig, fig_path


def Power_of_freq_meg(ch_names: list, mags_or_grads: str, freqs: np.ndarray, psds: np.ndarray, mean_power_per_band_needed: bool, plotflag: bool, sid: str):

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
    renamed_df_power.to_csv('../derivatives/sub-'+sid+'/megqc/csv files/abs_power_'+mags_or_grads+'.csv')
    renamed_df_power_freq.to_csv('../derivatives/sub-'+sid+'/megqc/csv files/power_by_Nfreq_'+mags_or_grads+'.csv')
    renamed_df_rel_power.to_csv('../derivatives/sub-'+sid+'/megqc/csv files/relative_power_'+mags_or_grads+'.csv')


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

        if mags_or_grads == 'mags':
            tit='Magnetometers'
        elif mags_or_grads == 'grads':
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
            fig, fig_path = plot_pie_chart_freq(mean_relative_freq=mean_relative, tit=tit, sid=sid)
            return fig, fig_path




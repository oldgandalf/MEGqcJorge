import mne
import numpy as np
from universal_plots import QC_derivative
import plotly.graph_objects as go

def ECG_meg_qc(ecg_params: dict, raw: mne.io.Raw, m_or_g_chosen: list):
    """Main ECG function"""


    # picks_ECG = mne.pick_types(raw.info, ecg=True)
    # if picks_ECG.size == 0:
    #     print('No ECG channels found is this data set, cardio artifacts can not be detected. ECG data can be reconstructed on base of magnetometers, but this will not be accurate and is not recommended.')
    #     return None, None
    # else:
    #     ECG_channel_name=[]
    #     for i in range(0,len(picks_ECG)):
    #         ECG_channel_name.append(raw.info['chs'][picks_ECG[i]]['ch_name'])

    ecg_events, ch_ecg, average_pulse, ecg_data=mne.preprocessing.find_ecg_events(raw, return_ecg=True, verbose=False)

    print('ECG channel used to identify hearbeats: ', raw.info['chs'][ch_ecg]['ch_name'])
    print('Average pulse: ', round(average_pulse), ' per minute') 

    ecg_events_times  = (ecg_events[:, 0] - raw.first_samp) / raw.info['sfreq']

    #WHAT SHOULD WE SHOW? CAN PLOT THE ECG CHANNEL. OR ECG EVENTS ON TOP OF 1 OF THE CHANNELS DATA. OR ON EVERY CHANNELS DATA?

    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    avg_ecg_epochs = ecg_epochs.average().apply_baseline((-0.5, -0.2))
    # about baseline see here: https://mne.tools/stable/auto_tutorials/preprocessing/10_preprocessing_overview.html#sphx-glr-auto-tutorials-preprocessing-10-preprocessing-overview-py
    
    ecg_deriv = []

    for m_or_g  in m_or_g_chosen:
        fig_ecg = ecg_epochs.plot_image(combine='mean', picks = m_or_g[0:-1])[0] #plot averageg over ecg epochs artifact
        # [0] is to plot only 1 figure. the function by default is trying to plot both mags and grads, but here we want 
        # to do them saparetely depending on what was chosen for analysis
        ecg_deriv += [QC_derivative(fig_ecg, 'mean_ECG_epoch_'+m_or_g, None, 'matplotlib')]

        #averaging the ECG epochs together:
        avg_ecg_epochs = ecg_epochs.average().apply_baseline((-0.5, -0.2))
        fig_ecg_sensors = avg_ecg_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25], picks = m_or_g[0:-1])
        #plot average artifact with topomap
        ecg_deriv += [QC_derivative(fig_ecg_sensors, 'ECG_field_pattern_sensors_'+m_or_g, None, 'matplotlib')]

    # Need to output: channels contamminated with ecg artifacts. 

    return ecg_deriv, ecg_events_times

class Mean_artif_peak:
    """Average ecg epoch for a particular channel  with its main peak (location and magnitude),
    info if this magnitude is concidered as artifact or not."""

    def __init__(self, channel_or_epoch:str, mean_artifact_epoch:list, peak_loc:float, peak_magnitude:float, artif_over_threshold:bool):
        self.channel =  channel_or_epoch
        self.mean_artifact_epoch = mean_artifact_epoch
        self.peak_loc = peak_loc
        self.peak_magnitude = peak_magnitude
        self.artif_over_threshold = artif_over_threshold

    def __repr__(self):
        return 'Mean artifact peak on: ' + str(self.channel_or_epoch) + '\n - peak location inside artifact epoch: ' + str(self.peak_loc) + '\n - peak magnitude: ' + str(self.peak_magnitude) + '\n - artifact magnitude over threshold: ' + str(self.artif_over_threshold)+ '\n'


def plot_affected_channels(sfreq, tmin, tmax, affected_channels_epochs, artifact_lvl, fig_tit, ch_type):

    if ch_type=='mags':
        fig_ch_tit='Magnetometers'
        unit='Tesla'
    elif ch_type=='grads':
        fig_ch_tit='Gradiometers'
        unit='Tesla/meter'
    else:
        fig_ch_tit='?'
        unit='?unknown unit?'
        print('Please check ch_type input. Has to be "mags" or "grads"')

    fig = go.Figure()
    t = np.arange(tmin, tmax+1/sfreq, 1/sfreq)
    fig.add_trace(go.Scatter(x=t, y=[(artifact_lvl)]*len(t), name='mean ECG magnitude'))

    for ch in affected_channels_epochs:

        fig.add_trace(go.Scatter(x=t, y=abs(ch.mean_artifact_epoch), name=ch.channel))
        fig.add_trace(go.Scatter(x=[t[ch.peak_loc]], y=[ch.peak_magnitude], mode='markers', name='+peak'));

    fig.update_layout(
        xaxis_title='Time in seconds',
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'),
        yaxis_title='Mean artifact magnitude in '+unit,
        title={
            'text': fig_tit+fig_ch_tit,
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
            
    fig.show()

    return fig


def epochs_or_channels_over_limit(loop_over, thresh_lvl_mean, norm_lvl, list_ecg_epochs):

    ecg_peaks_on_channels=[]
    for ch_ind, ch in enumerate(loop_over):
        avg_ecg_epoch_data=list_ecg_epochs[ch_ind]
        thresh_mean=(max(abs(avg_ecg_epoch_data)) - min(abs(avg_ecg_epoch_data))) / thresh_lvl_mean
        mean_peak_locs, mean_peak_magnitudes = mne.preprocessing.peak_finder(abs(avg_ecg_epoch_data), extrema=1, verbose=False, thresh=thresh_mean) 
        biggest_peak_ind=np.argmax(mean_peak_magnitudes)
        ecg_peaks_on_channels.append(Mean_artif_peak(channel_or_epoch=ch, mean_artifact_epoch=avg_ecg_epoch_data, peak_loc=mean_peak_locs[biggest_peak_ind], peak_magnitude=mean_peak_magnitudes[biggest_peak_ind], artif_over_threshold=False))


    #2.
    #find mean ECG magnitude over all channels:
    mean_ecg_magnitude = np.mean([potentially_affected_channel.peak_magnitude for potentially_affected_channel in ecg_peaks_on_channels])

    affected_channels=[]
    not_affected_channels=[]
    artifact_lvl=mean_ecg_magnitude/norm_lvl
    for ch_ind, potentially_affected_channel in enumerate(ecg_peaks_on_channels):
        if potentially_affected_channel.peak_magnitude>artifact_lvl:
            potentially_affected_channel.artif_over_threshold=True
            affected_channels.append(potentially_affected_channel)
        else:
            not_affected_channels.append(potentially_affected_channel)

    return affected_channels, not_affected_channels, artifact_lvl




def find_ecg_affected_channels(raw: mne.io.Raw, channels:dict, m_or_g_chosen:list, norm_lvl: float, thresh_lvl_mean=1.3, tmin=-0.1, tmax=0.1, plotflag=True):

    '''
    1. Calculate average ECG epoch: 
    a) over all ecg epochs for each channel - to find contamminated channels 
    OR
    b) over all channels for each ecg epoch - to find strongest ecg epochs

    2.Set some threshold which defines a high amplitude of ECG event. All above this - counted as potential ECG peak.
    (Instead of comparing peak amplitudes could also calculate area under the curve. 
    But peak can be better because data can be so noisy for some channels, that area will be pretty large 
    even when the peak is not present.)
    If there are several peaks above the threshold found - find the biggest one and detect as ecg peak.

    (Peak is detected in a very short area of the ecg epoch: tmin=-0.1, tmax=0.1, instead of tmin=-0.5, tmax=0.5  
    which is default for ecg epoch detectin by mne.
    This is done to detect the central peak more precisely and skip all the non-ecg related fluctuations).

    3. Compare:
    a) found peaks will be compared across channels to decide which channels are affected the most:
    -Average the peak magnitude over all channels. 
    -Find all channels, where the magnitude is abover average by some (SET IT!) level.
    OR
    b) found peaks will be compared across ecg epochs to decide which epochs arestrong.

    Output:
    ecg_affected_channels: list of instances of Mean_artif_peak_on_channel
    2  figures: ecg affected + not affected channels OR epochs.

'''
    
    ecg_affected_channels={}
    ecg_not_affected_channels={}
    all_figs=[]
    for m_or_g in m_or_g_chosen:

        #1.:
        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, picks=channels[m_or_g], tmin=tmin, tmax=tmax)
        #HERE THINK IF THIS USE OF 'PICKS' IS OK and doesnt prevent ecg reconstruction from mags.
        #according to function description, it should not. parameter ch_name is in charge of what will be used for reconstruction. 
        # but still neefd to make sure! run some checks.

        #averaging the ECG epochs together:
        avg_ecg_epochs = ecg_epochs.average(picks=channels[m_or_g])#.apply_baseline((-0.5, -0.2))
        #avg_ecg_epochs is evoked:Evoked objects typically store EEG or MEG signals that have been averaged over multiple epochs.
        #The data in an Evoked object are stored in an array of shape (n_channels, n_times)
        

        avg_ecg_epoch_data_all=avg_ecg_epochs.data

        #2. and 3.:
        affected_channels, not_affected_channels, artifact_lvl = epochs_or_channels_over_limit(loop_over=channels[m_or_g], thresh_lvl_mean=thresh_lvl_mean, norm_lvl=norm_lvl, list_ecg_epochs=avg_ecg_epoch_data_all)

        ecg_affected_channels[m_or_g]=affected_channels
        ecg_not_affected_channels[m_or_g]=not_affected_channels

        if plotflag:

            fig_affected=plot_affected_channels(raw.info['sfreq'], tmin, tmax, affected_channels, artifact_lvl, 'Channels affected by ECG artifact: ', m_or_g)
            fig_not_affected=plot_affected_channels(raw.info['sfreq'], tmin, tmax,not_affected_channels, artifact_lvl, 'Channels not affected by ECG artifact: ', m_or_g)

            all_figs += [fig_affected, fig_not_affected]

    return ecg_affected_channels, all_figs


def find_ecg_affected_epochs(raw: mne.io.Raw, channels:dict, m_or_g_chosen:list, norm_lvl: float, thresh_lvl_mean=1.3, tmin=-0.1, tmax=0.1,plotflag=True):

    ecg_affected_epochs={}
    ecg_not_affected_epochs={}
    all_figs=[]

    for m_or_g in m_or_g_chosen:

        #1.:
        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, picks=channels[m_or_g], tmin=tmin, tmax=tmax)
        df_ecg_epochs = ecg_epochs.to_data_frame()

        #calculate mean of each time point over all channels. abs value of magnitude is taken.
        df_ecg_epochs['mean'] = df_ecg_epochs.iloc[:, 3:-1].abs().mean(axis=1)

        # Plot to check:
        # fig = go.Figure()
        # sfreq=raw.info['sfreq']
        # t = np.arange(tmin, tmax+1/sfreq, 1/sfreq)

        #collect the mean values of each echepoch into a list
        all_means_of_epochs = [None] * len(ecg_epochs) #preassign
        for ep in range(0,len(ecg_epochs)):
            df_one_ep=df_ecg_epochs.loc[df_ecg_epochs['epoch'] == ep]
            all_means_of_epochs[ep]=np.array(df_one_ep.loc[:,"mean"])

        #     if ep ==0:
        #         fig.add_trace(go.Scatter(x=t, y=all_means_of_epochs[ep], name='epoch '+str(ep)))
        # fig.show()

        epoch_numbers_list=list(range(0, len(ecg_epochs)))
        #2. and 3.:
        strong_ecg_epochs, weak_ecg_epochs, artifact_lvl = epochs_or_channels_over_limit(loop_over=epoch_numbers_list, thresh_lvl_mean=thresh_lvl_mean, norm_lvl=norm_lvl, list_ecg_epochs=all_means_of_epochs)
        ecg_affected_epochs[m_or_g]=strong_ecg_epochs
        ecg_not_affected_epochs[m_or_g]=weak_ecg_epochs

        if plotflag:

            fig_affected=plot_affected_channels(raw.info['sfreq'], tmin, tmax, strong_ecg_epochs, artifact_lvl, 'Strong ECG epochs: ', m_or_g)
            fig_not_affected=plot_affected_channels(raw.info['sfreq'], tmin, tmax, weak_ecg_epochs, artifact_lvl, 'Weak ECG epochs: ', m_or_g)

            all_figs += [fig_affected, fig_not_affected]
    
    return ecg_affected_epochs, all_figs
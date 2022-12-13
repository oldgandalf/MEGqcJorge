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

class Mean_artif_peak_on_channel:
    """Average ecg epoch for a particular channel  with its main peak (location and magnitude),
    info if this magnitude is concidered as artifact or not."""

    def __init__(self, channel:str, mean_artifact_epoch:list, peak_loc:float, peak_magnitude:float, artif_over_threshold:bool):
        self.channel =  channel
        self.mean_artifact_epoch = mean_artifact_epoch
        self.peak_loc = peak_loc
        self.peak_magnitude = peak_magnitude
        self.artif_over_threshold = artif_over_threshold

    def __repr__(self):
        return 'Mean artifact peak on channel: ' + str(self.channel) + '\n - peak location inside artifact epoch: ' + str(self.peak_loc) + '\n - peak magnitude: ' + str(self.peak_magnitude) + '\n - artifact magnitude over threshold: ' + str(self.artif_over_threshold)+ '\n'


def plot_affected_channels(sfreq, affected_channels, mean_ecg_magnitude, fig_tit):

    fig = go.Figure()
    t = np.arange(-0.1, 0.1, 1/sfreq)
    fig.add_trace(go.Scatter(x=t, y=[(mean_ecg_magnitude)]*len(t), name='mean ECG magnitude'))

    for ch in affected_channels:

        fig.add_trace(go.Scatter(x=t, y=abs(ch.mean_artifact_epoch), name=ch.channel))
        fig.add_trace(go.Scatter(x=[t[ch.peak_loc]], y=[ch.peak_magnitude], mode='markers', name='+peak'));


    fig.update_layout(
        xaxis_title='Time in seconds',
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'),
        yaxis_title='Mean artifact magnitude over epochs',
        title={
            'text': fig_tit,
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
            
    fig.show()

    return fig


def find_ecg_affected_channels(raw: mne.io.Raw, channels:dict, m_or_g_chosen:list, norm_lvl: float, thresh_lvl_mean=1.3, tmin=-0.1, tmax=0.1,plotflag=True):

    '''1. Calculate average ECG epoch over all ecg epochs for each channel. 
    Set some threshold which defines a high amplitude of ECG event. All above this - counted as potential ECG peak.
    (Instead of comparing peak amplitudes could also calculate area under the curve. 
    But peak can be better because data can be so noisy for some channels, that area will be pretty large 
    even when the peak is not present.)
    If there are severalpeaks above thetreshold found - find the biggest one and detect as ecg peak.

    (Peak is detected in a very short area of the ecg epoch: tmin=-0.1, tmax=0.1, instead of tmin=-0.5, tmax=0.5  
    which is default for ecg epoch detectin by mne.
    This is done to detect the central peak more precisely and skip all the non-ecg related fluctuations).

    2. After that, found peaks will be compared across channels to decide which channels are affected the most:
    -Average the peak magnitude over all channels. 
    -Find all channels, where the magnitude is abover average by some (SET IT!) level.

    Output:
    ecg_affected_channels: instance of Mean_artif_peak_on_channel
    2  figures: ecg affected + not affected  channels

'''
    #1.
    ecg_affected_channels={}
    ecg_not_affected_channels={}
    for m_or_g in m_or_g_chosen:

        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, tmin=tmin, tmax=tmax)

        #averaging the ECG epochs together:
        avg_ecg_epochs = ecg_epochs.average(picks=channels[m_or_g])#.apply_baseline((-0.5, -0.2))
        #avg_ecg_epochs is evoked:Evoked objects typically store EEG or MEG signals that have been averaged over multiple epochs.
        #The data in an Evoked object are stored in an array of shape (n_channels, n_times)
        
        ecg_peaks_on_channels=[]

        # fig = go.Figure()
        # t=np.arange(tmin, tmax, 1/raw.info['sfreq'])

        for ch_ind, ch in enumerate(channels[m_or_g]):
            avg_ecg_epoch_data=avg_ecg_epochs.data[ch_ind,:]
            thresh_mean=(max(abs(avg_ecg_epoch_data)) - min(abs(avg_ecg_epoch_data))) / thresh_lvl_mean
            mean_peak_locs, mean_peak_magnitudes = mne.preprocessing.peak_finder(abs(avg_ecg_epoch_data), extrema=1, verbose=False, thresh=thresh_mean) 
            biggest_peak_ind=np.argmax(mean_peak_magnitudes)
            ecg_peaks_on_channels.append(Mean_artif_peak_on_channel(channel=ch, mean_artifact_epoch=avg_ecg_epoch_data, peak_loc=mean_peak_locs[biggest_peak_ind], peak_magnitude=mean_peak_magnitudes[biggest_peak_ind], artif_over_threshold=False))


        #2.
        #find mean ECG magnitude over all channels:
        mean_ecg_magnitude = np.mean([potentially_affected_channel.peak_magnitude for potentially_affected_channel in ecg_peaks_on_channels])

        affected_channels=[]
        not_affected_channels=[]
        for ch_ind, potentially_affected_channel in enumerate(ecg_peaks_on_channels):
            if potentially_affected_channel.peak_magnitude>mean_ecg_magnitude/norm_lvl:
                potentially_affected_channel.artif_over_threshold=True
                affected_channels.append(potentially_affected_channel)
            else:
                not_affected_channels.append(potentially_affected_channel)
        
        ecg_affected_channels[m_or_g]=affected_channels
        ecg_not_affected_channels[m_or_g]=not_affected_channels

        if plotflag:
            fig_affected=plot_affected_channels(raw.info['sfreq'], affected_channels, mean_ecg_magnitude, 'Channels affected by ECG artifact')
            fig_not_affected=plot_affected_channels(raw.info['sfreq'], not_affected_channels, mean_ecg_magnitude, 'Channels not affected by ECG artifact')

    return ecg_affected_channels, fig_affected, fig_not_affected






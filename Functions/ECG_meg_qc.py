import mne
import numpy as np
from universal_plots import QC_derivative

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


def find_ecg_affected_channels(raw: mne.io.Raw, channels:dict, m_or_g_chosen:list, thresh_lvl_mean=1.3, plotflag=True):

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
    ECG affected channels+their ecg peaks(location inside an average ECG epoch, average magnitude).

'''
    #1.
    import plotly.graph_objects as go

    ecg_peaks_on_channels={}
    for m_or_g in m_or_g_chosen:

        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, tmin=-0.1, tmax=0.1)

        #averaging the ECG epochs together:
        avg_ecg_epochs = ecg_epochs.average(picks=channels[m_or_g])#.apply_baseline((-0.5, -0.2))
        #avg_ecg_epochs is evoked:Evoked objects typically store EEG or MEG signals that have been averaged over multiple epochs.
        #The data in an Evoked object are stored in an array of shape (n_channels, n_times)
        #print(avg_ecg_epochs.data[0,:]) #data of first channel,all time points
        t=np.arange(0, len(avg_ecg_epochs.data[0,:]), 1/raw.info['sfreq'])
        fig = go.Figure()
        
        affected_channels={}
        for ch_ind, ch in enumerate(channels[m_or_g]):
            avg_ecg_epoch_data=avg_ecg_epochs.data[ch_ind,:]
            thresh_mean=(max(abs(avg_ecg_epoch_data)) - min(abs(avg_ecg_epoch_data))) / thresh_lvl_mean

            #HERE INSTEAD OF RELATIVE THRESHOLD FOR EACH CHANNEL DECIDE HOW HIGH SHOULD TH ECG PEAK BE TO SAY THAT THERE IS ARTIFACT

            mean_peak_locs, mean_peak_magnitudes = mne.preprocessing.peak_finder(abs(avg_ecg_epoch_data), extrema=1, verbose=False, thresh=thresh_mean) 
            biggest_peak_ind=np.argmax(mean_peak_magnitudes)
            ecg_peaks_on_channels[ch]=[mean_peak_locs[biggest_peak_ind], mean_peak_magnitudes[biggest_peak_ind]] #[0].itemis used to extractfloat ot of arrayof arrays of 1 element.

            if ch_ind==0 or ch_ind==1 or ch_ind==2 or ch_ind==11 or ch_ind==18:
    
                fig.add_trace(go.Scatter(x=t, y=abs(avg_ecg_epoch_data), name=ch))
                fig.add_trace(go.Scatter(x=[t[mean_peak_locs[biggest_peak_ind]]], y=[mean_peak_magnitudes[biggest_peak_ind]], mode='markers', name='+peak'));

                fig.update_layout(
                yaxis = dict(
                    showexponent = 'all',
                    exponentformat = 'e'),
                title={
                    'text': 'Example of an averaged ECG epoch on a few channels',
                    'y':0.85,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
        
        ecg_peaks_on_channels[m_or_g]=affected_channels

            
        fig.show()

    #2.
    


        return ecg_peaks_on_channels




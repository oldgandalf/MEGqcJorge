import mne
from universal_plots import QC_derivative

def ECG_meg_qc(ecg_params: dict, raw: mne.io.Raw, m_or_g_chosen: list):
    """Main ECG function"""


    picks_ECG = mne.pick_types(raw.info, ecg=True)
    if picks_ECG.size == 0:
        print('No ECG channels found is this data set, cardio artifacts can not be detected. ECG data can be reconstructed on base of magnetometers, but this will not be accurate and is not recommended.')
        return None, None
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




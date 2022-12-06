import mne
from universal_plots import QC_derivative

def EOG_meg_qc(eog_params: dict, raw: mne.io.Raw, m_or_g_chosen: list):
    """Main EOG function"""

    # picks_EOG = mne.pick_types(raw.info, eog=True)
    # if picks_EOG.size == 0:
    #     print('No EOG channels found is this data set - EOG artifacts can not be detected.')
    #     return None, None

    # else:
    #     EOG_channel_name=[]
    #     for i in range(0,len(picks_EOG)):
    #         EOG_channel_name.append(raw.info['chs'][picks_EOG[i]]['ch_name'])
    #     print('EOG channels found: ', EOG_channel_name)
    # eog_events=mne.preprocessing.find_eog_events(raw, thresh=None, ch_name=EOG_channel_name)

    eog_events=mne.preprocessing.find_eog_events(raw, thresh=None, ch_name=None)
    # ch_name: This doesn’t have to be a channel of eog type; it could, for example, also be an ordinary 
    # EEG channel that was placed close to the eyes, like Fp1 or Fp2.
    # or just use none as channel, so the eog will be found automatically

    eog_events_times  = (eog_events[:, 0] - raw.first_samp) / raw.info['sfreq']

    #WHAT SHOULD WE SHOW? CAN PLOT THE EOG CHANNEL. OR EOG EVENTS ON TOP OF 1 OF THE CHANNELS DATA. OR ON EVERY CHANNELS DATA?

    eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2))
    eog_deriv = []

    for m_or_g  in m_or_g_chosen:
        fig_eog = eog_epochs.plot_image(combine='mean', picks = m_or_g[0:-1])[0]
        eog_deriv += [QC_derivative(fig_eog, 'mean_ECG_epoch_'+m_or_g, None, 'matplotlib')]

        #averaging the ECG epochs together:
        fig_eog_sensors = eog_epochs.average().plot_joint(picks = m_or_g[0:-1])
        eog_deriv += [QC_derivative(fig_eog_sensors, 'EOG_field_pattern_sensors_'+m_or_g, None, 'matplotlib')]

    return eog_deriv, eog_events_times




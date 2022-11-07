import mne

def ECG_meg_qc(config, raw: mne.io.Raw, m_or_g_chosen: list):
    """Main psd function"""

    #ecg_section = config['ECG']

    picks_ECG = mne.pick_types(raw.info, ecg=True)
    if picks_ECG.size == 0:
        print('No ECG channels found is this data set. ECG data can be reconstructed on base of magnetometers, but this will not be accurate and us not recommended.')
        return
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

    list_of_desc = []
    list_of_figs = []
    list_of_figs_ecg_sensors = []
    list_of_desc_fig_ecg_sensors = []

    for m_or_g  in m_or_g_chosen:
        list_of_figs.append(ecg_epochs.plot_image(combine='mean', picks = m_or_g[0:-1])[0])
        list_of_desc.append('mean_ECG_epoch_'+m_or_g)

        #averaging the ECG epochs together:
        avg_ecg_epochs = ecg_epochs.average().apply_baseline((-0.5, -0.2))

        list_of_figs_ecg_sensors.append(avg_ecg_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25], picks = m_or_g[0:-1]))
        list_of_desc_fig_ecg_sensors.append('ECG_field_pattern_sensors_'+m_or_g)

    list_of_figs += list_of_figs_ecg_sensors
    list_of_desc += list_of_desc_fig_ecg_sensors

    output_format = 'matplotlib'

    return list_of_figs, list_of_desc, output_format




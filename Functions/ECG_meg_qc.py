import mne
from universal_plots import add_output_format

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

    figs_with_names = []
    fig_path=None

    for m_or_g  in m_or_g_chosen:
        fig_ecg = ecg_epochs.plot_image(combine='mean', picks = m_or_g[0:-1])[0]
        fig_ecg_desc = 'mean_ECG_epoch_'+m_or_g
        figs_with_names.append((fig_ecg,fig_ecg_desc,fig_path))

        #averaging the ECG epochs together:
        avg_ecg_epochs = ecg_epochs.average().apply_baseline((-0.5, -0.2))

        fig_ecg_sensors = avg_ecg_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25], picks = m_or_g[0:-1])
        fig_ecg_sensors_desc = 'ECG_field_pattern_sensors_'+m_or_g
        figs_with_names.append((fig_ecg_sensors,fig_ecg_sensors_desc,fig_path))

    figs_with_name_and_format = add_output_format(figs_with_names, 'matplotlib')

    return figs_with_name_and_format




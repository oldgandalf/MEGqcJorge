import mne
from universal_plots import QC_derivative
from ECG_meg_qc import find_affected_channels, make_simple_metric_ECG_EOG

def EOG_meg_qc(eog_params: dict, raw: mne.io.Raw, channels, m_or_g_chosen: list):
    """Main EOG function"""

    eog_events=mne.preprocessing.find_eog_events(raw, thresh=None, ch_name=None)
    # ch_name: This doesn’t have to be a channel of eog type; it could, for example, also be an ordinary 
    # EEG channel that was placed close to the eyes, like Fp1 or Fp2.
    # or just use none as channel, so the eog will be found automatically

    eog_events_times  = (eog_events[:, 0] - raw.first_samp) / raw.info['sfreq']

    sfreq=raw.info['sfreq']
    tmin=eog_params['eog_epoch_tmin']
    tmax=eog_params['eog_epoch_tmax']
    norm_lvl=eog_params['norm_lvl']
    use_abs_of_all_data=eog_params['use_abs_of_all_data']


    eog_derivs = []
    all_eog_affected_channels={}
    top_eog_magnitudes={}
    top_10_eog_magnitudes={}
    simple_metric_EOG={}

    for m_or_g  in m_or_g_chosen:

        eog_epochs = mne.preprocessing.create_eog_epochs(raw, picks=channels[m_or_g], tmin=tmin, tmax=tmax)

        fig_eog = eog_epochs.plot_image(combine='mean', picks = m_or_g)[0]
        eog_derivs += [QC_derivative(fig_eog, 'mean_EOG_epoch_'+m_or_g, None, 'matplotlib')]

        #averaging the ECG epochs together:
        fig_eog_sensors = eog_epochs.average().plot_joint(picks = m_or_g)
        eog_derivs += [QC_derivative(fig_eog_sensors, 'EOG_field_pattern_sensors_'+m_or_g, None, 'matplotlib')]


        eog_affected_channels, fig_affected, fig_not_affected, fig_avg=find_affected_channels(eog_epochs, channels[m_or_g], m_or_g, norm_lvl, ecg_or_eog='EOG', thresh_lvl_peakfinder=5, tmin=tmin, tmax=tmax, plotflag=True, sfreq=sfreq, use_abs_of_all_data=use_abs_of_all_data)
        eog_derivs += [QC_derivative(fig_affected, 'EOG_affected_channels_'+m_or_g, None, 'plotly')]
        eog_derivs += [QC_derivative(fig_not_affected, 'EOG_not_affected_channels_'+m_or_g, None, 'plotly')]
        eog_derivs += [QC_derivative(fig_avg, 'overall_average_EOG_epoch_'+m_or_g, None, 'plotly')]
        all_eog_affected_channels[m_or_g]=eog_affected_channels

        simple_metric_EOG[m_or_g]=make_simple_metric_ECG_EOG(all_eog_affected_channels[m_or_g], m_or_g, 'EOG', channels[m_or_g])

    return eog_derivs, simple_metric_EOG, eog_events_times, all_eog_affected_channels




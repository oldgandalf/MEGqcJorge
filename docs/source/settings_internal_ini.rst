Internal Pipeline Settings
==========================

These are internal settings stored in the ``internal_settings.ini`` file. 
They have been experimentally found and should not be changed by a user.


[ECG]
-----
- **max_n_peaks_allowed_for_avg*** (int) - this is for the whole averaged over all channels ecg epoch, it should be much smoother - therefore less peaks are allowed.
- **ecg_epoch_tmin** (float) - time in seconds before the event. Unit: seconds. Default: -0.04 seconds. Dont set smaller than -0.03. 
- **ecg_epoch_tmax** (float) - time in seconds after the event. Unit: seconds. Default: 0.04 seconds. Dont set larger than 0.05.


All below parameters are for the mean_threshold method:

- **max_n_peaks_allowed_for_ch** (int) - this is for an individual ch, it can be more noisy, therefore more peaks are allowed. It also depends on the length of chosen window
- **timelimit_min=-0.02** (float) - time in seconds before the event. These define time window where the peak of ECG of EOG wave is typically located relative to the event found by MNE. MNE event is usually not accurate. Unit: seconds. Default: -0.02 seconds.
- **timelimit_max=0.012** (float) - time in seconds after the event. These define time window where the peak of ECG of EOG wave is typically located relative to the event found by MNE. MNE event is usually not accurate. Unit: seconds. Default: 0.012 seconds.
- **window_size_for_mean_threshold_method** (float) - this value will be taken before and after the t0_actual in detect_channels_above_norm(). It defines the time window in which the peak of artifact on the channel has to present to be counted as artifact peak and compared t the threshold. Unit: seconds. Default: 0.02 seconds.


For the 3 time windows above:

- ecg_epoch_tmin & ecg_epoch_tmax: simply how large the time window around ECG event we want to see. 
- timelimit_min & timelimit_max : used to detect the average over all artifact (based on all channels). It is used to corret for imperfect even detection by mne.
- window_size_for_mean_threshold_method: after the average is detected and new time0 is set as the peak of that average - the second time window defines where (relative to t0) the peak of every individual channel can be located to be compared to the threshold (threshold is average overall magntude*multiplier).


[EOG]
-----
- **max_n_peaks_allowed_for_avg*** (int) - this is for the whole averaged over all channels eog epoch, it should be much smoother - therefore less peaks are allowed.
- **eog_epoch_tmin** (float) - time in seconds before the event. Unit: seconds. Default: -0.04 seconds. Dont set smaller than -0.03. 
- **eog_epoch_tmax** (float) - time in seconds after the event. Unit: seconds. Default: 0.04 seconds. Dont set larger than 0.05.


All below parameters are for the mean_threshold method:

- **max_n_peaks_allowed_for_ch** (int) - this is for an individual ch, it can be more noisy, therefore more peaks are allowed. It also depends on the length of chosen window
- **timelimit_min=-0.02** (float) - time in seconds before the event. These define time window where the peak of ECG of EOG wave is typically located relative to the event found by MNE. MNE event is usually not accurate. Unit: seconds. Default: -0.02 seconds.
- **timelimit_max=0.012** (float) - time in seconds after the event. These define time window where the peak of ECG of EOG wave is typically located relative to the event found by MNE. MNE event is usually not accurate. Unit: seconds. Default: 0.012 seconds.
- **window_size_for_mean_threshold_method** (float) - this value will be taken before and after the t0_actual in detect_channels_above_norm(). It defines the time window in which the peak of artifact on the channel has to present to be counted as artifact peak and compared t the threshold. Unit: seconds. Default: 0.02 seconds.


For the 3 time windows above:

- eog_epoch_tmin & eog_epoch_tmax: simply how large the time window around ECG event we want to see. 
- timelimit_min & timelimit_max : used to detect the average over all artifact (based on all channels). It is used to corret for imperfect even detection by mne.
- window_size_for_mean_threshold_method: after the average is detected and new time0 is set as the peak of that average - the second time window defines where (relative to t0) the peak of every individual channel can be located to be compared to the threshold (threshold is average overall magntude*multiplier).



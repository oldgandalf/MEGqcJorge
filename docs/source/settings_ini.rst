Pipeline Settings
=================

The pipeline settings are stored in the ``settings.ini`` file. 

Settings are divided into sections, each corresponding to a different pipeline step.

Default settings [DEFAULT]
--------------------------
- **do_for** (str) : which channels to process. Enter 1 or both values separated by , . Default: *mag, grad*
- **data_directory** (str) : path to the data directory . Example: */Volumes/M2_DATA/MEG_QC_stuff/data/from openneuro/ds003483*
- **data_crop_tmin** (int) & data_crop_tmax** (int) : time in seconds. Settings for data crop. If no cropping needed, leave blank.

Filtering [Filtering]
---------------------
- **apply_filtering** (bool): if True, the data will be filtered. Default: *True*
- **l_freq** (int or float) : lower frequency for bandpass filter. Default: *0*
- **h_freq** (int or float) : higher frequency for bandpass filter. Default: *140*. Reason: output of PSD can be used for filtering the data before muscle artifact detection. Musce artifacts are usually around 110-140 Hz, so this setting allows to see if there are extra frequencies which would need to be filtered out
- **method** (str) : method of filtering. Default: *iir*. Or turn off filtering completely by setting apply_filtering = False. Parameters in this case dont matter.


Epoching [Epoching]
-------------------
- **event_dur** (float) : duration of the event in seconds. Default: *0.2*
- **epoch_tmin** (float) : time in seconds before the event. Default: *-0.2*
- **epoch_tmax** (float) : time in seconds after the event. Default: 1
- **stim_channel** (str) : leave empty if want it to be detected automatically or write explicitely. Default:  *STI101*. 

Standard deviation [STD]
------------------------
- **std_lvl** (int) : set like std_lvl = 1 or std_lvl = 3, etc. Defines how many std from the mean to use for the threshold. Default: *1*
- **allow_percent_noisy_flat_epochs** (int) : Defines how many percent of epochs can be noisy or flat. Higher than this  - epoch is marked as noisy/flat. Default: *70*
- **noisy_channel_multiplier** (float or int) : Multiplier to define noisy channel, if std of this channel for this epoch is over** (the mean std of this channel for all epochs together*multipliar), then this channel is noisy. Higher value - less channels are marked as noisy. Default: *1.2*
- **flat_multiplier** (float or int) : Multiplier to define flat channel, if std of this channel for this epoch is under** (the mean std of this channel for all epochs together*multipliar), then this channel is flat. Default: *0.5*

Power spectral density [PSD]
----------------------------
- **freq_min** (int or float) : lower frequency for PSD calculation. Default: *0.5*
- **freq_max** (int or float) : higher frequency for PSD calculation. Default: *140*. Reason: output of PSD can be used for filtering the data before muscle artifact detection. Musce artifacts are usually around 110-140 Hz, so this setting allows to see if there are extra frequencies which would need to be filtered out
- **psd_step_size** (float or int) : frequency resolution of the PSD. Default: *0.5*
- **use_logscale** (bool) : if True - will use logscale for the PSD plot. Default: True


Peak-to-peak amplitude manual [PTP_manual]
------------------------------------------
- **max_pair_dist_sec** (float) : will hard code it when decide on best value after trying out different data sets. might be different for mag and grad. Default: *20*
- **thresh_lvl** (int) : scaling factor for threshold. The higher this vaues is - the more peaks will be detected. Default: *10*
- **allow_percent_noisy_flat_epochs** (int) : defines how many percent of epochs can be noisy or flat. Over this number - epoch is marged as noisy/flat. Default: *70*
- **std_lvl** (int) : set like std_lvl = 1 or std_lvl = 3, etc. Defines how many std from the mean to use for the threshold. Default: *1*
- **noisy_channel_multiplier** (float or int) : multiplier to define noisy channel, if std of this channel for this epoch is over** (the mean std of this channel for all epochs together*multipliar), then this channel is noisy. Default: *1.2*
- **flat_multiplier** (float or int) : multiplier to define flat channel, if std of this channel for this epoch is under** (the mean std of this channel for all epochs together*multipliar), then this channel is flat. Default: *0.5*
- **ptp_top_limit & ptp_bottom_limit** (float or int) : these 2 are not used now. done in case we want to limit by exact number not by std level. 


Peak-to-peak amplitude auto (based on MNE annotatons) [PTP_auto]
----------------------------------------------------------------
- **peak_m** (float or int) : minimal PTP amplitude to count as peak for magnetometers. Default: *4e-14*
- **peak_g** (float or int) : minimal PTP amplitude to count as peak for gradiometers. Default: *4e-14*
- **flat_m** (float or int) : max PTP amplitude to count as flat for magnetometers. Default: *3e-14*
- **flat_g** (float or int) : max PTP amplitude to count as flat for gradiometers. Default: *3e-14*
- **bad_percent** (int) : percentage of the time a channel can be above or below thresholds. Below this percentage, Annotations are created. Above this percentage, the channel involved is return in bads. Note the returned bads are not automatically added to info['bads']. Default: *5*
- **min_duration** (float) : minimum duration** (sec) required by consecutives samples to be above peak or below flat thresholds to be considered. to consider as above or below threshold. For some systems, adjacent time samples with exactly the same value are not totally uncommon. Default: *0.002*


Heart beat artifacts [ECG]
--------------------------
- **drop_bad_ch** (bool) - if True - will drop the bad ECG channel from the data and attempt to reconstruct ECG data on base of magnetometers. If False - will not drop the bad ECG channel and will attempt to calculate ECG events on base of the bad ECG channel. Default: *True*
- **n_breaks_allowed_per_10min** (int) - number of breaks in ECG channel allowed per 10 minutes of recording. (This setting is for ECG channel only, not for any other channels Used to detect a noisy ECG channel). Default: *3*
- **allowed_range_of_peaks_stds** (float) - the allowed range of peaks in standard deviations. (This setting is for ECG channel only, not for any other channels Used to detect a noisy ECG channel). Defaault: *0.05* (experimentally chosen value). How the setting is used:
    - The channel data will be scaled from 0 to 1, so the setting is universal for all data sets.
    - The peaks will be detected on the scaled data
    - The average std of all peaks has to be within this allowed range, If it is higher - the channel has too high deviation in peaks height and is counted as noisy

- **ecg_epoch_tmin** (float) : time in seconds before the event. Recommended -0.04 seconds. Dont set smaller than -0.03. Default: *-0.04*
- **ecg_epoch_tmax** (float) : time in seconds after the event. Recommended 0.04 seconds. Dont set smaller than 0.03. Default: *0.04*
- **norm_lvl** (int) : The norm level is the scaling factor for the threshold. The mean artifact amplitude over all channels is multiplied by the norm_lvl to get the threshold. Default: *1*
- **flip_data** (bool) : if True, then the data will be flipped if some epochs are negative due to magnetic fields orintation. If False the data will not be flipped and results might be less accurate. Default: *True*

Eye movement artifacts [EOG]
----------------------------
- **n_breaks_allowed_per_10min** (int) - number of breaks in ECG channel allowed per 10 minutes of recording. (This setting is for EOG channel only, not for any other channels Used to detect a noisy EOG channel). Default: *3*
- **allowed_range_of_peaks_stds** (float) - the allowed range of peaks in standard deviations. (This setting is for EOG channel only, not for any other channels Used to detect a noisy EOG channel). Default: *0.15* (experimentally chosen value). How the setting is used:
    - The channel data will be scaled from 0 to 1, so the setting is universal for all data sets.
    - The peaks will be detected on the scaled data
    - The average std of all peaks has to be within this allowed range, If it is higher - the channel has too high deviation in peaks height and is counted as noisy

- **eog_epoch_tmin** (float) : time in seconds before the event. Default: *-0.2*
- **eog_epoch_tmax** (float) : time in seconds after the event. Default: *0.4*
- **norm_lvl** (int) : the norm level is the scaling factor for the threshold. The mean artifact amplitude over all channels is multiplied by the norm_lvl to get the threshold. Default: *1*
- **flip_data** (bool) : if True, then the data will be flipped if some epochs are negative due to magnetic fields orintation. If False the data will not be flipped and results might be less accurate. Default: *True*


Head_movement artifacts [Head_movement]
---------------------------------------
No available settings


Muscle artifacts [Muscle]
-------------------------
- **muscle_freqs** (2 ints or 2 float) : defines the frequency band for detecting muscle activity. Default: 110, 140
- **threshold_muscle** (int or float) : threshold for muscle detection. Zscores detected above this threshold will be considered as muscle artifacts. Default: *5, 10*
- **min_length_good** (int or float) : The shortest allowed duration of "good data"** (in seconds) between adjacent muscle annotations; shorter segments will be incorporated into the surrounding annotations. Default: *0.2*
- **min_distance_between_different_muscle_events** (int or float) : minimum distance between different muscle events in seconds. If events happen closer to each other they will all be counted as one event and the time will be assigned as the first peak. Default: *1*  

Difference between last 2 settings: **min_length_good** - used to detect ALL muscle events, **min_distance_between_different_muscle_events** - used to detect evets with z-score higher than the threshold on base of ALL muscle events



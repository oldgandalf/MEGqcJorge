Pipeline Settings
=================

The pipeline settings are stored in the ``settings.ini`` file. 

Settings are divided into sections, each corresponding to a different pipeline step.

[Default]
---------
- do_for: mag or/and grad - which channels to process - enter 1 or both values separated by ,
- data_directory: path to the data directory . Example: /Volumes/M2_DATA/MEG_QC_stuff/data/from openneuro/ds003483
- data_crop_tmin & data_crop_tmax: time in seconds. Setting for data crop. If no cropping needed, leave blank.

[Filtering]
-----------
- apply_filtering : True or False. If True, the data will be filtered.
- l_freq: int or float. Lower frequency for bandpass filter. Recommended 0.


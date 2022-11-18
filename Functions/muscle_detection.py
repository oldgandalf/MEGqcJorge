# -*- coding: utf-8 -*-
"""
MNE TUTORIAL: changes: using open neuro data here

.. _ex-muscle-artifacts:

=========================
Annotate muscle artifacts
=========================

Muscle contractions produce high frequency activity that can mask brain signal
of interest. Muscle artifacts can be produced when clenching the jaw,
swallowing, or twitching a cranial muscle. Muscle artifacts are most
noticeable in the range of 110-140 Hz.

This example uses :func:`~mne.preprocessing.annotate_muscle_zscore` to annotate
segments where muscle activity is likely present. This is done by band-pass
filtering the data in the 110-140 Hz range. Then, the envelope is taken using
the hilbert analytical signal to only consider the absolute amplitude and not
the phase of the high frequency signal. The envelope is z-scored and summed
across channels and divided by the square root of the number of channels.
Because muscle artifacts last several hundred milliseconds, a low-pass filter
is applied on the averaged z-scores at 4 Hz, to remove transient peaks.
Segments above a set threshold are annotated as ``BAD_muscle``. In addition,
the ``min_length_good`` parameter determines the cutoff for whether short
spans of "good data" in between muscle artifacts are included in the
surrounding "BAD" annotation.

"""
# Authors: Adonay Nunes <adonay.s.nunes@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt
import numpy as np
from mne.preprocessing import annotate_muscle_zscore

from data_load_and_folders import load_meg_data
duration=5 #in minutes
#n_events, df_epochs_mags, df_epochs_grads, epochs_mags, epochs_grads, mags, grads, filtered_d, filtered_d_resamp, raw_cropped, raw=initial_stuff(duration)


data_file = '/Volumes/M2_DATA/MEG_QC_stuff/data/from openneuro/ds003483/sub-009/ses-1/meg/sub-009_ses-1_task-deduction_run-1_meg.fif'
#data_file = '/Volumes/M2_DATA/MEG_QC_stuff/data/from openneuro/ds003352/sub-1/ses-01/meg/sub-1_ses-01_task-ColorSpirals_run-00_meg.fif'
#data_file = '/Volumes/M2_DATA/MEG_QC_stuff/data/from openneuro/ds003352/sub-4/ses-02/meg/sub-4_ses-02_task-ColorSpirals_run-01_meg.fif'
#data_file = '/Volumes/M2_DATA/MEG_QC_stuff/data/from openneuro/ds003392/sub-01/meg/sub-01_task-localizer_meg.fif'
#data_file = '/Volumes/M2_DATA/MEG_QC_stuff/data/from openneuro/ds003645/sub-002/meg/sub-002_task-FacePerception_run-1_meg.fif'
#data_file = '/Volumes/M2_DATA/MEG_QC_stuff/data/from openneuro/ds003694/sub-02/meg/sub-02_task-MEM_run-01_meg.fif'
#data_file = '/Volumes/M2_DATA/MEG_QC_stuff/data/from openneuro/ds004107/sub-mind002/ses-01/meg/sub-mind002_ses-01_task-auditory_meg.fif'
#data_file = '/Volumes/M2_DATA/MEG_QC_stuff/data/from openneuro/ds004276/sub-002/meg/sub-002_task-words_meg.fif'
#data_file = '/Volumes/M2_DATA/MEG_QC_stuff/data/from openneuro/ds003703/sub-a68d5xp5/meg/sub-a68d5xp5_task-listeningToSpeech_run-01_meg.fif'
#data_file = '/Volumes/M2_DATA/MEG_QC_stuff/data/from openneuro/ds003682/sub-001/ses-01/meg/sub-001_ses-01_task-AversiveLearningReplay_run-01_meg.fif'


raw, channels = load_meg_data(data_file)

#And resample it!

#crop the data to calculate faster
# raw_cropped = raw.copy()
# raw_cropped.crop(tmin=1000, tmax=None) 

raw.crop(tmin=200, tmax=None).load_data() 

# %%
# Notch filter the data:
#
# .. note::
#     If line noise is present, you should perform notch-filtering *before*
#     detecting muscle artifacts. See :ref:`tut-section-line-noise` for an
#     example.

raw.notch_filter([60, 120])

# %%

# The threshold is data dependent, check the optimal threshold by plotting
# ``scores_muscle``.
threshold_muscle = 5  # z-score
# Choose one channel type, if there are axial gradiometers and magnetometers,
# select magnetometers as they are more sensitive to muscle activity.
annot_muscle, scores_muscle = annotate_muscle_zscore(
    raw, ch_type="grad", threshold=threshold_muscle, min_length_good=0.2,
    filter_freq=[110, 140])

# %%
# Plot muscle z-scores across recording
# --------------------------------------------------------------------------

fig, ax = plt.subplots()
ax.plot(raw.times, scores_muscle)
ax.axhline(y=threshold_muscle, color='r')
ax.set(xlabel='time, (s)', ylabel='zscore', title='Muscle activity')
# %%
# View the annotations
# --------------------------------------------------------------------------
order = np.arange(144, 164)
raw.set_annotations(annot_muscle)
raw.plot(start=5, duration=20, order=order)

#%% . Set annotations and view:
order = np.arange(144, 164) #this is how we set which channels will be shown in the plot
raw.set_annotations(annot_muscle)
raw.plot(start=5, duration=10, order=order)


# %%

print(annot_muscle)

# %%

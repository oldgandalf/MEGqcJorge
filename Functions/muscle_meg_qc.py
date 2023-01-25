
# # Annotate muscle artifacts
# 
# Muscle contractions produce high frequency activity that can mask brain signal
# of interest. Muscle artifacts can be produced when clenching the jaw,
# swallowing, or twitching a cranial muscle. Muscle artifacts are most
# noticeable in the range of 110-140 Hz.
# 
# This example uses :func:`~mne.preprocessing.annotate_muscle_zscore` to annotate
# segments where muscle activity is likely present. This is done by band-pass
# filtering the data in the 110-140 Hz range. Then, the envelope is taken using
# the hilbert analytical signal to only consider the absolute amplitude and not
# the phase of the high frequency signal. The envelope is z-scored and summed
# across channels and divided by the square root of the number of channels.
# Because muscle artifacts last several hundred milliseconds, a low-pass filter
# is applied on the averaged z-scores at 4 Hz, to remove transient peaks.
# Segments above a set threshold are annotated as ``BAD_muscle``. In addition,
# the ``min_length_good`` parameter determines the cutoff for whether short
# spans of "good data" in between muscle artifacts are included in the
# surrounding "BAD" annotation.
# 


import mne
mne.viz.set_browser_backend('matplotlib')
import plotly.graph_objects as go
from scipy.signal import find_peaks
import numpy as np
from mne.preprocessing import annotate_muscle_zscore
from universal_plots import QC_derivative

def MUSCLE_meg_qc(raw, interactive_matplot=False):

    #ADD checks:
    # Are there mags?
    # Do we even wanan try with grads?
    # Do we want to notch filter? Check first on psd if there is powerline peak and at which freq.
    # add several z-score options. or make it as input param??
    # add legend for threshold line

    muscle_derivs=[]

    # Notch filter the data:
    # 
    # If line noise is present, you should perform notch-filtering *before*
    #     detecting muscle artifacts. See `tut-section-line-noise` for an example.

    raw.load_data() 
    raw.notch_filter([60, 120])

    # The threshold is data dependent, check the optimal threshold by plotting
    # ``scores_muscle``.
    threshold_muscle = 10  # z-score
    # Choose one channel type, if there are axial gradiometers and magnetometers,
    # select magnetometers as they are more sensitive to muscle activity.
    annot_muscle, scores_muscle = annotate_muscle_zscore(
        raw, ch_type="mag", threshold=threshold_muscle, min_length_good=0.2,
        filter_freq=[110, 140])


    # ## Plot muscle z-scores across recording
    # 
    peak_locs_pos, _ = find_peaks(scores_muscle, height=threshold_muscle, distance=raw.info['sfreq']*5)

    muscle_times = raw.times[peak_locs_pos]
    muscle_magnitudes=scores_muscle[peak_locs_pos]

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=raw.times, y=scores_muscle, mode='lines', name='muscle scores'))
    fig.add_trace(go.Scatter(x=muscle_times, y=muscle_magnitudes, mode='markers', name='muscle events'))
    fig.update_layout(title='Muscle activity', xaxis_title='time, (s)', yaxis_title='zscore')
    fig.add_shape(type="line", x0=0, y0=threshold_muscle, x1=raw.times[-1], y1=threshold_muscle, line=dict(color="Red", width=2), name='threshold')
    fig.show()

    muscle_derivs += [QC_derivative(fig, 'muscle_z_scores_over_time', None, 'plotly')]

    # ## View the annotations (interactive_matplot)
    if interactive_matplot is True:
        order = np.arange(144, 164)
        raw.set_annotations(annot_muscle)
        fig2=raw.plot(start=5, duration=20, order=order)
        #Change settings to show all channels!

    simple_metric=make_simple_metric_muscle(muscle_times, muscle_magnitudes, threshold_muscle)

    return muscle_derivs, simple_metric

def make_simple_metric_muscle(muscle_times, muscle_magnitudes, threshold_muscle):

    # Make 1 digit metrics:
    # Dict: z score 5
    # zscore 10.
    # num of events.
    # in brackets time of the events. (or in nested dict).

    simple_metric={}

    z_scores_dict={}
    z_score_disct_details={}
    z_score_disct_details['muscle event times'] = muscle_times.tolist()
    z_score_disct_details['muscle event zscore'] = muscle_magnitudes.tolist()
    z_scores_dict[threshold_muscle] = ['Number of events', len(muscle_times), z_score_disct_details]
    simple_metric['Muscle artifact events at different z score thresholds'] = z_scores_dict

    return simple_metric

    # %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import mne
from mne.preprocessing import annotate_movement, compute_average_dev_head_t
import time
from universal_plots import QC_derivative

mne.viz.set_browser_backend('matplotlib')

def HEAD_movement_meg_qc(raw, extra_visual=True):

    # 1. Main part:
    head_derivs = []
    head_not_calculated=False

    try: 
        #for Neuromag use (3 steps):
        chpi_freqs, ch_idx, chpi_codes = mne.chpi.get_chpi_info(info=raw.info)
        #We can use mne.chpi.get_chpi_info to retrieve the coil frequencies, 
        # the index of the channel indicating when which coil was switched on, 
        # and the respective “event codes” associated with each coil’s activity.
        # Output:
        # - The frequency used for each individual cHPI coil.
        # - The index of the STIM channel containing information about when which cHPI coils were switched on.
        # - The values coding for the “on” state of each individual cHPI coil.

        print(f'cHPI coil frequencies extracted from raw: {chpi_freqs} Hz')


        #Estimating continuous head position
        print('Start Computing HPI amplitudes and locations...')
        start_time = time.time()
        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
        chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
        print("Finished. --- Execution %s seconds ---" % (time.time() - start_time))

    except:
        print('Neuromag appriach to compute Head positions failed. Trying CTF approach...')
        try:
            #for CTF use:
            chpi_locs = mne.chpi.extract_chpi_locs_ctf(raw)
        except:
            print('Also CTF appriach to compute Head positions failed. Trying KIT approach...')
            try:
                #for KIT use:
                chpi_locs = mne.chpi.extract_chpi_locs_kit(raw)
            except:
                print('Also KIT appriach to compute Head positions failed. Head positions can not be computed')
                return head_derivs, True

    # Next steps - for all systems:
    print('Start computing head positions...')
    start_time = time.time()
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs)
    print("Finished. --- Execution %s seconds ---" % (time.time() - start_time))

    fig0 = mne.viz.plot_head_positions(head_pos, mode='traces')

    head_derivs += [QC_derivative(fig0, 'Head_position_rotation', None, 'matplotlib')]

    # 2. Optional visual part:
    if extra_visual is True:
        original_head_dev_t = mne.transforms.invert_transform(
            raw.info['dev_head_t'])
        average_head_dev_t = mne.transforms.invert_transform(
            compute_average_dev_head_t(raw, head_pos))
        fig1 = mne.viz.plot_head_positions(head_pos)
        for ax, val, val_ori in zip(fig1.axes[::2], average_head_dev_t['trans'][:3, 3],
                            original_head_dev_t['trans'][:3, 3]):
            print('val:', val, 'val_ori:', val_ori)
            ax.axhline(1000 * val, color='r')
            ax.axhline(1000 * val_ori, color='g')
        # The green horizontal lines represent the original head position, whereas the
        # red lines are the new head position averaged over all the time points.

        fig1.show()
        head_derivs += [QC_derivative(fig1, 'Head_position_rotation_average', None, 'matplotlib')]

        # 3. Plot raw data with annotated head movement:
        mean_distance_limit = 0.0015  # in meters
        annotation_movement, hpi_disp = annotate_movement(
            raw, head_pos, mean_distance_limit=mean_distance_limit)
        raw.set_annotations(annotation_movement)
        fig2=raw.plot(n_channels=100, duration=20)
        head_derivs += [QC_derivative(fig2, 'Head_position_annot', None, 'matplotlib')]

    return head_derivs, head_not_calculated
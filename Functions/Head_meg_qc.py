import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import mne
from mne.preprocessing import annotate_movement, compute_average_dev_head_t
import time
from universal_plots import QC_derivative

mne.viz.set_browser_backend('matplotlib')


def compute_head_pos_std_and_max_rotation_movement(head_pos):

    #head positions as data frame just for visualization and check:
    df_head_pos = pd.DataFrame(head_pos, columns = ['t', 'q1', 'q2', 'q3', 'x', 'y', 'z', 'gof', 'err', 'v']) #..., goodness of fit, error, velocity

    #get the head position in xyz coordinates:
    head_pos_transposed=head_pos.transpose()
    xyz_coords=np.array([[x, y, z] for x, y, z in zip(head_pos_transposed[4], head_pos_transposed[5], head_pos_transposed[6])])

    # Calculate the maximum movement in 3 directions:
    max_movement_x = (np.max(xyz_coords[:,0])-np.min(xyz_coords[:,0]))
    max_movement_y = (np.max(xyz_coords[:,1])-np.min(xyz_coords[:,1]))
    max_movement_z = (np.max(xyz_coords[:,2])-np.min(xyz_coords[:,2]))

    # Calculate the maximum rotation in 3 directions:
    rotation_coords=np.array([[q1, q2, q3] for q1, q2, q3 in zip(head_pos_transposed[1], head_pos_transposed[2], head_pos_transposed[3])])
    max_rotation_q1 = (np.max(rotation_coords[:,0])-np.min(rotation_coords[:,0]))
    max_rotation_q2 = (np.max(rotation_coords[:,1])-np.min(rotation_coords[:,1]))
    max_rotation_q3 = (np.max(rotation_coords[:,2])-np.min(rotation_coords[:,2]))
    #max_rotation_q1 = (df_head_pos['q1'].max()-df_head_pos['q1'].min()) #or like this using dataframes


    # Calculate the standard deviation of the movement of the head over time:
    # 1. Calculate the distances between each consecutive pair of coordinates. Like x2-x1, y2-y1, z2-z1
    # Use Pythagorean theorem: the distance between two points (x1, y1, z1) and (x2, y2, z2) in 3D space 
    # is the square root of (x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2.
    # 2. Then calculate the standard deviation of the distances: σ = √(Σ(x_i - mean)^2 / n)

    # 1. Calculate the distances between each consecutive pair of coordinates
    distances = np.sqrt(np.sum((xyz_coords[1:] - xyz_coords[:-1])**2, axis=1))

    # 2. Calculate the standard deviation
    std_head_pos = np.std(distances)

    return std_head_pos, (max_movement_x, max_movement_y, max_movement_z), (max_rotation_q1, max_rotation_q2, max_rotation_q3), df_head_pos

def make_simple_metric_head(std_head_pos, max_movement_xyz, max_rotation_q):
    '''Make simple metric for head positions'''

    simple_metric={}
    simple_metric_details={}

    simple_metric_details['Maximum movement in x direction in mm'] = max_movement_xyz[0]*1000
    simple_metric_details['Maximum movement in y direction in mm'] = max_movement_xyz[1]*1000
    simple_metric_details['Maximum movement in z direction in mm'] = max_movement_xyz[2]*1000
    simple_metric_details['Maximum rotation in q1 direction in quat'] = max_rotation_q[0]
    simple_metric_details['Maximum rotation in q2 direction in quat'] = max_rotation_q[1]
    simple_metric_details['Maximum rotation in q3 direction in quat'] = max_rotation_q[2]

    simple_metric['STD of the movement of the head over time'] = std_head_pos
    simple_metric['Details']=simple_metric_details
    
    return simple_metric

def HEAD_movement_meg_qc(raw, plot_with_lines=True, plot_annotations=False):

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
    if plot_with_lines is True:
        original_head_dev_t = mne.transforms.invert_transform(
            raw.info['dev_head_t'])
        average_head_dev_t = mne.transforms.invert_transform(
            compute_average_dev_head_t(raw, head_pos))
        fig1 = mne.viz.plot_head_positions(head_pos)
        for ax, val, val_ori in zip(fig1.axes[::2], average_head_dev_t['trans'][:3, 3],
                            original_head_dev_t['trans'][:3, 3]):
            ax.axhline(1000*val, color='r')
            ax.axhline(1000*val_ori, color='g')
            #print('val', val, 'val_ori', val_ori)
        # The green horizontal lines represent the original head position, whereas the
        # Red lines are the new head position averaged over all the time points.

        head_derivs += [QC_derivative(fig1, 'Head_position_rotation_average', None, 'matplotlib', description_for_user = 'The green horizontal lines - original head position. Red lines - the new head position averaged over all the time points.')]
    
    if plot_annotations is True:
        # 3. Plot raw data with annotated head movement:
        mean_distance_limit = 0.0015  # in meters
        annotation_movement, hpi_disp = annotate_movement(
            raw, head_pos, mean_distance_limit=mean_distance_limit)
        raw.set_annotations(annotation_movement)
        fig2=raw.plot(n_channels=100, duration=20)
        head_derivs += [QC_derivative(fig2, 'Head_position_annot', None, 'matplotlib')]


    # 4. Calculate the standard deviation of the movement of the head over time:
    std_head_pos, max_movement_xyz, max_rotation_q, df_head_pos = compute_head_pos_std_and_max_rotation_movement(head_pos)


    print('Std of head positions in mm: ', std_head_pos*1000)
    print('Max movement (x, y, z) in mm: ', [m*1000 for m in max_movement_xyz])
    print('Max rotation (q1, q2, q3) in quat: ', max_rotation_q)

    # 5. Make a simple metric:
    simple_metrics_head = make_simple_metric_head(std_head_pos, max_movement_xyz, max_rotation_q)
    
    return head_derivs, simple_metrics_head, head_not_calculated, df_head_pos



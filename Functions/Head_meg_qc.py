import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    q1q2q3_coords_quads=np.array([[q1, q2, q3] for q1, q2, q3 in zip(head_pos_transposed[1], head_pos_transposed[2], head_pos_transposed[3])])

    #Translate rotations into degrees: (360/2pi)*value 
    q1q2q3_coords=360/(2*np.pi)*q1q2q3_coords_quads

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
    distances_xyz = np.sqrt(np.sum((xyz_coords[1:] - xyz_coords[:-1])**2, axis=1))
    distances_q = np.sqrt(np.sum((q1q2q3_coords[1:] - q1q2q3_coords[:-1])**2, axis=1))

    # 2. Calculate the standard deviation
    std_head_pos = np.std(distances_xyz)
    std_head_rotations = np.std(distances_q)

    return std_head_pos, std_head_rotations, (max_movement_x, max_movement_y, max_movement_z), (max_rotation_q1, max_rotation_q2, max_rotation_q3), df_head_pos, head_pos


def make_simple_metric_head(std_head_pos,std_head_rotations, max_movement_xyz, max_rotation_q):
    '''Make simple metric for head positions'''

    simple_metric={}
    simple_metric_details={}

    simple_metric_details['Maximum movement in x direction in mm'] = max_movement_xyz[0]*1000
    simple_metric_details['Maximum movement in y direction in mm'] = max_movement_xyz[1]*1000
    simple_metric_details['Maximum movement in z direction in mm'] = max_movement_xyz[2]*1000
    simple_metric_details['Maximum rotation in q1 direction in degrees'] = max_rotation_q[0]
    simple_metric_details['Maximum rotation in q2 direction in degrees'] = max_rotation_q[1]
    simple_metric_details['Maximum rotation in q3 direction in degrees'] = max_rotation_q[2]

    simple_metric['STD of the movement of the head over time: '] = std_head_pos
    simple_metric['STD of the rotation of the head over time'] = std_head_rotations
    simple_metric['Details']=simple_metric_details
    
    return simple_metric


def make_head_pos_plot(raw, head_pos):

    ''' Plot positions and rotations of the head'''

    head_derivs = []

    original_head_dev_t = mne.transforms.invert_transform(
        raw.info['dev_head_t'])
    average_head_dev_t = mne.transforms.invert_transform(
        compute_average_dev_head_t(raw, head_pos))

    #plot using MNE:
    fig1 = mne.viz.plot_head_positions(head_pos, mode='traces')
    #fig1 = mne.viz.plot_head_positions(head_pos_degrees)
    for ax, val, val_ori in zip(fig1.axes[::2], average_head_dev_t['trans'][:3, 3],
                        original_head_dev_t['trans'][:3, 3]):
        ax.axhline(1000*val, color='r')
        ax.axhline(1000*val_ori, color='g')
        #print('___MEG QC___: ', 'val', val, 'val_ori', val_ori)
    # The green horizontal lines represent the original head position, whereas the
    # Red lines are the new head position averaged over all the time points.

    head_derivs += [QC_derivative(fig1, 'Head_position_rotation_average', 'matplotlib', description_for_user = 'The green horizontal lines - original head position. Red lines - the new head position averaged over all the time points.')]


    #plot head_pos using PLOTLY:

    # First, for each head position subtract the first point from all the other points to make it always deviate from 0:
    head_pos_baselined=head_pos.copy()
    #head_pos_baselined=head_pos_degrees.copy()
    for i, pos in enumerate(head_pos_baselined.T[1:7]):
        pos -= pos[0]
        head_pos_baselined.T[i]=pos

    t = head_pos.T[0]

    average_head_pos=average_head_dev_t['trans'][:3, 3]
    original_head_pos=original_head_dev_t['trans'][:3, 3]

    fig1p = make_subplots(rows=3, cols=2, subplot_titles=("Position (mm)", "Rotation (quats)"))

    # head_pos ndarray of shape (n_pos, 10): [t, q1, q2, q3, x, y, z, gof, err, v]
    # https://mne.tools/stable/generated/mne.chpi.compute_head_pos.html
    indexes=[4, 5, 6, 1, 2,3]
    names=['x', 'y', 'z', 'q1', 'q2', 'q3']
    for counter in [0, 1, 2]:
        position=1000*-head_pos.T[indexes][counter]
        #position=1000*-head_pos_baselined.T[indexes][counter]
        name_pos=names[counter]
        fig1p.add_trace(go.Scatter(x=t, y=position, mode='lines', name=name_pos), row=counter+1, col=1)
        fig1p.update_yaxes(title_text=name_pos, row=counter+1, col=1)
        #print('name', name_pos, 'position', position)
        rotation=head_pos.T[indexes][counter+3]
        #rotation=head_pos_baselined.T[indexes][counter+3]
        name_rot=names[counter+3]
        fig1p.add_trace(go.Scatter(x=t, y=rotation, mode='lines', name=name_rot), row=counter+1, col=2)
        fig1p.update_yaxes(title_text=name_rot, row=counter+1, col=2)
        #print('name', name_rot, 'rotation', rotation)

        # fig1p.add_hline(y=1000*average_head_pos[counter], line_dash="dash", line_color="red", row=counter+1, col=1)
        # fig1p.add_hline(y=1000*original_head_pos[counter], line_dash="dash", line_color="green", row=counter+1, col=1)

    fig1p.update_xaxes(title_text='Time (s)', row=3, col=1)
    fig1p.update_xaxes(title_text='Time (s)', row=3, col=2)
    fig1p.show()
    head_derivs += [QC_derivative(fig1p, 'Head_position_rotation_average_plotly', 'plotly', description_for_user = 'The green horizontal lines - original head position. Red lines - the new head position averaged over all the time points.')]

    return head_derivs, head_pos_baselined


def make_head_annots_plot(raw, head_pos):
    '''Plot raw data with annotated head movement:'''

    head_derivs = []

    mean_distance_limit = 0.0015  # in meters
    annotation_movement, hpi_disp = annotate_movement(
        raw, head_pos, mean_distance_limit=mean_distance_limit)
    raw.set_annotations(annotation_movement)
    fig2=raw.plot(n_channels=100, duration=20)
    head_derivs += [QC_derivative(fig2, 'Head_position_annot', 'matplotlib')]

    return head_derivs

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

        print('___MEG QC___: ', f'cHPI coil frequencies extracted from raw: {chpi_freqs} Hz')


        #Estimating continuous head position
        print('___MEG QC___: ', 'Start Computing cHPI amplitudes and locations...')
        start_time = time.time()
        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
        chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
        print('___MEG QC___: ', "Finished. --- Execution %s seconds ---" % (time.time() - start_time))
        #print('___MEG QC___: ', 'chpi_locs:', chpi_locs)

    except:
        print('___MEG QC___: ', 'Neuromag appriach to compute Head positions failed. Trying CTF approach...')
        try:
            #for CTF use:
            chpi_locs = mne.chpi.extract_chpi_locs_ctf(raw)
        except:
            print('___MEG QC___: ', 'Also CTF appriach to compute Head positions failed. Trying KIT approach...')
            try:
                #for KIT use:
                chpi_locs = mne.chpi.extract_chpi_locs_kit(raw)
            except:
                print('___MEG QC___: ', 'Also KIT appriach to compute Head positions failed. Head positions can not be computed')
                return head_derivs, {}, True, []

    # Next steps - for all systems:
    print('___MEG QC___: ', 'Start computing head positions...')
    start_time = time.time()
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs)
    print('___MEG QC___: ', "Finished computing head positions. --- Execution %s seconds ---" % (time.time() - start_time))
    #print('___MEG QC___: ', 'Head positions:', head_pos)


    # check if head positions are computed successfully:
    if head_pos.size == 0:
        print('___MEG QC___: ', 'Head positions were not computed successfully.')
        return head_derivs, {}, True, []

    # translate rotation columns [1:4] in head_pos.T into degrees: (360/2pi)*value: 
    # (we assume they are in radients. But in the plot it says they are in quats! 
    # see: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)

    head_pos_degrees=head_pos.T.copy()
    for q in range(1,4):
        head_pos_degrees[q]=360/(2*np.pi)*head_pos_degrees[q]
    head_pos_degrees=head_pos_degrees.transpose()


    # 2. Optional visual part:
    if plot_with_lines is True:
        head_pos_derivs, head_pos_baselined = make_head_pos_plot(raw, head_pos)
    else:
        head_pos_derivs = []

    if plot_annotations is True:
        plot_annot_derivs = make_head_annots_plot(raw, head_pos)
    else:
        plot_annot_derivs = []

    head_derivs += head_pos_derivs + plot_annot_derivs

    # 4. Calculate the standard deviation of the movement of the head over time:
    std_head_pos, std_head_rotations, max_movement_xyz, max_rotation_q, df_head_pos, head_pos = compute_head_pos_std_and_max_rotation_movement(head_pos)


    print('___MEG QC___: ', 'Std of head positions in mm: ', std_head_pos*1000)
    print('___MEG QC___: ', 'Std of head rotations in quat: ', std_head_rotations)
    print('___MEG QC___: ', 'Max movement (x, y, z) in mm: ', [m*1000 for m in max_movement_xyz])
    print('___MEG QC___: ', 'Max rotation (q1, q2, q3) in quat: ', max_rotation_q)

    # 5. Make a simple metric:
    simple_metrics_head = make_simple_metric_head(std_head_pos, std_head_rotations, max_movement_xyz, max_rotation_q)
    
    return head_derivs, simple_metrics_head, head_not_calculated, df_head_pos, head_pos



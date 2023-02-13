import mne
import numpy as np
from universal_plots import QC_derivative, get_tit_and_unit
from universal_html_report import simple_metric_basic
import plotly.graph_objects as go
from scipy.signal import find_peaks


class Mean_artifact_with_peak:
    """Contains average ecg epoch for a particular channel,
    calculates its main peak (location and magnitude),
    info if this magnitude is concidered as artifact or not."""

    def __init__(self, name: str, mean_artifact_epoch:list, peak_loc=None, peak_magnitude=None, r_wave_shape:bool=None, artif_over_threshold:bool=None, main_peak_loc: int=None, main_peak_magnitude: float=None):
        self.name =  name
        self.mean_artifact_epoch = mean_artifact_epoch
        self.peak_loc = peak_loc
        self.peak_magnitude = peak_magnitude
        self.r_wave_shape =  r_wave_shape
        self.artif_over_threshold = artif_over_threshold
        self.main_peak_loc = main_peak_loc
        self.main_peak_magnitude = main_peak_magnitude

    def __repr__(self):
        return 'Mean artifact peak on: ' + str(self.name) + '\n - peak location inside artifact epoch: ' + str(self.peak_loc) + '\n - peak magnitude: ' + str(self.peak_magnitude) +'\n - main_peak_loc: '+ str(self.main_peak_loc) +'\n - main_peak_magnitude: '+str(self.main_peak_magnitude)+'\n r_wave_shape: '+ str(self.r_wave_shape) + '\n - artifact magnitude over threshold: ' + str(self.artif_over_threshold)+ '\n'
    
    def find_peaks_and_detect_Rwave(self, max_n_peaks_allowed, thresh_lvl_peakfinder=None):
        
        peak_locs_pos, peak_locs_neg, peak_magnitudes_pos, peak_magnitudes_neg = find_epoch_peaks(ch_data=self.mean_artifact_epoch, thresh_lvl_peakfinder=thresh_lvl_peakfinder)
        
        self.peak_loc=np.concatenate((peak_locs_pos, peak_locs_neg), axis=None)

        if np.size(self.peak_loc)==0: #no peaks found - set peaks as just max of the whole epoch
            self.peak_loc=np.array([np.argmax(np.abs(self.mean_artifact_epoch))])
            self.r_wave_shape=False
        elif 1<=len(self.peak_loc)<=max_n_peaks_allowed:
            self.r_wave_shape=True
        elif len(self.peak_loc)>max_n_peaks_allowed:
            self.r_wave_shape=False
        else:
            self.r_wave_shape=False
            print('___MEG QC___: ', self.name + ': no expected artifact wave shape, check the reason!')

        self.peak_magnitude=np.array(self.mean_artifact_epoch[self.peak_loc])

        peak_locs=np.concatenate((peak_locs_pos, peak_locs_neg), axis=None)
        peak_magnitudes=np.concatenate((peak_magnitudes_pos, peak_magnitudes_neg), axis=None)

        return peak_locs, peak_magnitudes, peak_locs_pos, peak_locs_neg, peak_magnitudes_pos, peak_magnitudes_neg


    def plot_epoch_and_peak(self, fig, t, fig_tit, ch_type):

        fig_ch_tit, unit = get_tit_and_unit(ch_type)

        fig.add_trace(go.Scatter(x=np.array(t), y=np.array(self.mean_artifact_epoch), name=self.name))
        fig.add_trace(go.Scatter(x=np.array(t[self.peak_loc]), y=self.peak_magnitude, mode='markers', name='peak: '+self.name));

        fig.update_layout(
            xaxis_title='Time in seconds',
            yaxis = dict(
                showexponent = 'all',
                exponentformat = 'e'),
            yaxis_title='Mean artifact magnitude in '+unit,
            title={
                'text': fig_tit+fig_ch_tit,
                'y':0.85,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        return fig

    def find_largest_peak_in_timewindow(self, t, timelimit_min, timelimit_max):

        #find the highest peak inside the timelimit_min and timelimit_max:

        if self.peak_loc is None:
            self.main_peak_magnitude=None
            self.main_peak_loc=None
            return None, None

        self.main_peak_magnitude = -100
        for peak_loc in self.peak_loc:
            if timelimit_min<t[peak_loc]<timelimit_max: #if peak is inside the timelimit_min and timelimit_max was found:
                if self.mean_artifact_epoch[peak_loc] > self.main_peak_magnitude: #if this peak is higher than the previous one:
                    self.main_peak_magnitude=self.mean_artifact_epoch[peak_loc]
                    self.main_peak_loc=peak_loc 
  
        if self.main_peak_magnitude == -100: #if no peak was found inside the timelimit_min and timelimit_max:
            self.main_peak_magnitude=None
            self.main_peak_loc=None

        return self.main_peak_loc, self.main_peak_magnitude


def detect_channels_above_norm(norm_lvl, list_mean_ecg_epochs, mean_ecg_magnitude_peak, t, t0_actual, ecg_or_eog):


    if ecg_or_eog=='ECG':
        window_size=0.02
    elif ecg_or_eog=='EOG':
        window_size=0.1
    else:
        print('___MEG QC___: ', 'ecg_or_eog should be either ECG or EOG')

    timelimit_min=-window_size+t0_actual
    timelimit_max=window_size+t0_actual


    #Find the channels which got peaks over this mean:
    affected_channels=[]
    not_affected_channels=[]
    artifact_lvl=mean_ecg_magnitude_peak/norm_lvl #data over this level will be counted as artifact contaminated
    for potentially_affected_channel in list_mean_ecg_epochs:
        #if np.max(np.abs(potentially_affected_channel.peak_magnitude))>abs(artifact_lvl) and potentially_affected_channel.r_wave_shape is True:


        #find the highest peak inside the timelimit_min and timelimit_max:
        main_peak_loc, main_peak_magnitude = potentially_affected_channel.find_largest_peak_in_timewindow(t, timelimit_min, timelimit_max)

        print('___MEG QC___: ', potentially_affected_channel.name, ' Main Peak magn: ', potentially_affected_channel.main_peak_magnitude, ', Main peak loc ', potentially_affected_channel.main_peak_loc, ' Rwave: ', potentially_affected_channel.r_wave_shape)
        
        if main_peak_magnitude is not None: #if there is a peak in time window of artifact - check if it s high enough and has right shape
            if main_peak_magnitude>abs(artifact_lvl) and potentially_affected_channel.r_wave_shape is True:
                potentially_affected_channel.artif_over_threshold=True
                affected_channels.append(potentially_affected_channel)
            else:
                not_affected_channels.append(potentially_affected_channel)
                print('___MEG QC___: ', potentially_affected_channel.name, ' Peak magn over th: ', potentially_affected_channel.main_peak_magnitude>abs(artifact_lvl), ', in the time window: ', potentially_affected_channel.main_peak_loc, ' Rwave: ', potentially_affected_channel.r_wave_shape)
        else:
            not_affected_channels.append(potentially_affected_channel)
            print('___MEG QC___: ', potentially_affected_channel.name, ' Peak magn over th: NO PEAK in time window')

    return affected_channels, not_affected_channels, artifact_lvl


def plot_affected_channels(ecg_affected_channels, artifact_lvl, t, ch_type: str, fig_tit, use_abs_of_all_data):

    fig_ch_tit, unit = get_tit_and_unit(ch_type)

    fig=go.Figure()

    for ch in ecg_affected_channels:
        fig=ch.plot_epoch_and_peak(fig, t, 'Channels affected by ECG artifact: ', ch_type)

    fig.add_trace(go.Scatter(x=t, y=[(artifact_lvl)]*len(t), name='Thres=mean_peak/norm_lvl'))

    if use_abs_of_all_data == 'False':
        fig.add_trace(go.Scatter(x=t, y=[(-artifact_lvl)]*len(t), name='-Thres=mean_peak/norm_lvl'))

    fig.update_layout(
        xaxis_title='Time in seconds',
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'),
        yaxis_title='Mean artifact magnitude in '+unit,
        title={
            'text': fig_tit+str(len(ecg_affected_channels))+' '+fig_ch_tit,
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})


    fig.show()

    return fig


def find_epoch_peaks(ch_data, thresh_lvl_peakfinder):

    thresh_mean=(max(ch_data) - min(ch_data)) / thresh_lvl_peakfinder
    peak_locs_pos, _ = find_peaks(ch_data, prominence=thresh_mean)
    peak_locs_neg, _ = find_peaks(-ch_data, prominence=thresh_mean)

    try:
        peak_magnitudes_pos=ch_data[peak_locs_pos]
    except:
        peak_magnitudes_pos=np.empty(0)

    try:
        peak_magnitudes_neg=ch_data[peak_locs_neg]
    except:
        peak_magnitudes_neg=np.empty(0)

    return peak_locs_pos, peak_locs_neg, peak_magnitudes_pos, peak_magnitudes_neg



def flip_channels(avg_ecg_epoch_data_nonflipped, channels, max_n_peaks_allowed, thresh_lvl_peakfinder, t0_estimated_ind_start, t0_estimated_ind_end, t0_estimated_ind, t):

    '''4. flip all channels with negative peak around estimated t0.'''

    ecg_epoch_per_ch_only_data=np.empty_like(avg_ecg_epoch_data_nonflipped)
    ecg_epoch_per_ch=[]

    for i, ch_data in enumerate(avg_ecg_epoch_data_nonflipped): 
        ecg_epoch_nonflipped = Mean_artifact_with_peak(name=channels[i], mean_artifact_epoch=ch_data)
        peak_locs, peak_magnitudes, _, _, _, _ = ecg_epoch_nonflipped.find_peaks_and_detect_Rwave(max_n_peaks_allowed, thresh_lvl_peakfinder)
        print('___MEG QC___: ', channels[i], ' peak_locs:', peak_locs)

        #find peak_locs which is located the closest to t0_estimated_ind:
        if peak_locs.size>0:
            peak_loc_closest_to_t0=peak_locs[np.argmin(np.abs(peak_locs-t0_estimated_ind))]

        #if peak_loc_closest_t0 is negative and is located in the estimated time window of the wave - flip the data:
        if (ch_data[peak_loc_closest_to_t0]<0) & (peak_loc_closest_to_t0>t0_estimated_ind_start) & (peak_loc_closest_to_t0<t0_estimated_ind_end):
            ecg_epoch_per_ch_only_data[i]=-ch_data
            peak_magnitudes=-peak_magnitudes
            #print('___MEG QC___: ', channels[i]+' was flipped: peak_loc_near_t0: ', peak_loc_closest_to_t0, t[peak_loc_closest_to_t0], ', peak_magn:', ch_data[peak_loc_closest_to_t0], ', t0_estimated_ind_start: ', t0_estimated_ind_start, t[t0_estimated_ind_start], 't0_estimated_ind_end: ', t0_estimated_ind_end, t[t0_estimated_ind_end])
        else:
            ecg_epoch_per_ch_only_data[i]=ch_data
            #print('___MEG QC___: ', channels[i]+' was NOT flipped: peak_loc_near_t0: ', peak_loc_closest_to_t0, t[peak_loc_closest_to_t0], ', peak_magn:', ch_data[peak_loc_closest_to_t0], ', t0_estimated_ind_start: ', t0_estimated_ind_start, t[t0_estimated_ind_start], 't0_estimated_ind_end: ', t0_estimated_ind_end, t[t0_estimated_ind_end])

        ecg_epoch_per_ch.append(Mean_artifact_with_peak(name=channels[i], mean_artifact_epoch=ecg_epoch_per_ch_only_data[i], peak_loc=peak_locs, peak_magnitude=peak_magnitudes, r_wave_shape=ecg_epoch_nonflipped.r_wave_shape))

    return ecg_epoch_per_ch, ecg_epoch_per_ch_only_data


def estimate_t0(ecg_or_eog: str, avg_ecg_epoch_data_nonflipped: list, t: np.ndarray):
    
    ''' 
    1. find peaks on all channels it time frame around -0.02<t[peak_loc]<0.012 (here R wave is typica;ly dettected by mne - for ecg, for eog it is -0.1<t[peak_loc]<0.2)
    2. take 5 channels with most prominent peak 
    3. find estimated average t0 for all 5 channels, because t0 of event which mne estimated is often not accurate.'''
    

    if ecg_or_eog=='ECG':
        timelimit_min=-0.02
        timelimit_max=0.012
        window_size=0.02
    elif ecg_or_eog=='EOG':
        timelimit_min=-0.1
        timelimit_max=0.2
        window_size=0.1

        #these define different windows: 
        # - timelimit is where the peak of the wave is normally located counted from event time defined by mne. 
        #       It is a larger window, it is used to estimate t0, more accurately than mne does (based on 5 most promiment channels).
        # - window_size - where the peak of the wave must be located, counted from already estimated t0. It is a smaller window.
    else:
        print('___MEG QC___: ', 'Choose ecg_or_eog input correctly!')


    #find indexes of t where t is between timelimit_min and timelimit_max (limits where R wave typically is detected by mne):
    t_event_ind=np.argwhere((t>timelimit_min) & (t<timelimit_max))

    # cut the data of each channel to the time interval where R wave is expected to be:
    avg_ecg_epoch_data_nonflipped_limited_to_event=avg_ecg_epoch_data_nonflipped[:,t_event_ind[0][0]:t_event_ind[-1][0]]

    #find 5 channels with max values in the time interval where R wave is expected to be:
    max_values=np.max(np.abs(avg_ecg_epoch_data_nonflipped_limited_to_event), axis=1)
    max_values_ind=np.argsort(max_values)[::-1]
    max_values_ind=max_values_ind[:5]

    # find the index of max value for each of these 5 channels:
    max_values_ind_in_avg_ecg_epoch_data_nonflipped=np.argmax(np.abs(avg_ecg_epoch_data_nonflipped_limited_to_event[max_values_ind]), axis=1)
    
    #find average index of max value for these 5 channels th then derive t0_estimated:
    t0_estimated_average=int(np.round(np.mean(max_values_ind_in_avg_ecg_epoch_data_nonflipped)))
    #limited to event means that the index is limited to the time interval where R wave is expected to be.
    #Now need to get back to actual time interval of the whole epoch:

    #find t0_estimated to use as the point where peak of each ch data should be:
    t0_estimated_ind=t_event_ind[0][0]+t0_estimated_average #sum because time window was cut from the beginning of the epoch previously
    t0_estimated=t[t0_estimated_ind]

    # window of 0.015 or 0.05s around t0_estimated where the peak on different channels should be detected:
    t0_estimated_ind_start=np.argwhere(t==round(t0_estimated-window_size, 3))[0][0] 
    t0_estimated_ind_end=np.argwhere(t==round(t0_estimated+window_size, 3))[0][0]
    #yes you have to round it here because the numbers stored in in memery like 0.010000003 even when it looks like 0.01, hence np.where cant find the target float in t vector


    #another way without round would be to find the closest index of t to t0_estimated-0.015:
    #t0_estimated_ind_start=np.argwhere(t==np.min(t[t<t0_estimated-window_size]))[0][0]
    # find the closest index of t to t0_estimated+0.015:
    #t0_estimated_ind_end=np.argwhere(t==np.min(t[t>t0_estimated+window_size]))[0][0]

    print('___MEG QC___: ', t0_estimated_ind, '-t0_estimated_ind, ', t0_estimated, '-t0_estimated,     ', t0_estimated_ind_start, '-t0_estimated_ind_start, ', t0_estimated_ind_end, '-t0_estimated_ind_end')

    
    return t0_estimated, t0_estimated_ind, t0_estimated_ind_start, t0_estimated_ind_end




def find_affected_channels(ecg_epochs: mne.Epochs, channels: list, m_or_g: str, norm_lvl: float, ecg_or_eog: str, thresh_lvl_peakfinder: float, sfreq:float, tmin: float, tmax: float, plotflag=True, use_abs_of_all_data=False):

    '''
    1. Calculate average ECG epoch: 
    a) over all ecg epochs for each channel - to find contamminated channels (this func)
    OR
    b) over all channels for each ecg epoch - to find strongest ecg epochs (next func)

    2.Set some threshold which defines a high amplitude of ECG event. All above this - counted as potential ECG peak.
    (Instead of comparing peak amplitudes could also calculate area under the curve. 
    But peak can be better because data can be so noisy for some channels, that area will be pretty large 
    even when the peak is not present.)
    If there are several peaks above the threshold found - find the biggest one and detect as ecg peak.

    3. Compare:
    a) found peaks will be compared across channels to decide which channels are affected the most:
    -Average the peak magnitude over all channels. 
    -Find all channels, where the magnitude is abover average by some level.
    OR
    b) found peaks will be compared across ecg epochs to decide which epochs arestrong.

    Output:
    ecg_affected_channels: list of instances of Mean_artif_peak_on_channel
    2  figures: ecg affected + not affected channels OR epochs.

'''

    if  ecg_or_eog=='ECG':
        max_n_peaks_allowed_per_ms=8 
    elif ecg_or_eog=='EOG':
        max_n_peaks_allowed_per_ms=5
    else:
        print('___MEG QC___: ', 'Choose ecg_or_eog input correctly!')

    max_n_peaks_allowed=round(((abs(tmin)+abs(tmax))/0.1)*max_n_peaks_allowed_per_ms)
    print('___MEG QC___: ', 'max_n_peaks_allowed: '+str(max_n_peaks_allowed))

    t = np.round(np.arange(tmin, tmax+1/sfreq, 1/sfreq), 3) #yes, you need to round


    #1.:
    #averaging the ECG epochs together:
    avg_ecg_epochs = ecg_epochs.average(picks=channels)#.apply_baseline((-0.5, -0.2))
    #avg_ecg_epochs is evoked:Evoked objects typically store EEG or MEG signals that have been averaged over multiple epochs.
    #The data in an Evoked object are stored in an array of shape (n_channels, n_times)

    ecg_epoch_per_ch=[]
    if use_abs_of_all_data == 'True':
        ecg_epoch_per_ch_only_data=np.abs(avg_ecg_epochs.data)
        for i, ch_data in enumerate(ecg_epoch_per_ch_only_data):
            ecg_epoch_per_ch.append(Mean_artifact_with_peak(name=channels[i], mean_artifact_epoch=ch_data))
            ecg_epoch_per_ch[i].find_peaks_and_detect_Rwave(max_n_peaks_allowed, thresh_lvl_peakfinder)

    elif use_abs_of_all_data == 'False':
        ecg_epoch_per_ch_only_data=avg_ecg_epochs.data
        for i, ch_data in enumerate(ecg_epoch_per_ch_only_data):
            ecg_epoch_per_ch.append(Mean_artifact_with_peak(name=channels[i], mean_artifact_epoch=ch_data))
            ecg_epoch_per_ch[i].find_peaks_and_detect_Rwave(max_n_peaks_allowed, thresh_lvl_peakfinder)

    elif use_abs_of_all_data == 'flip':

        # New ecg flip approach:

        # 1. find peaks on all channels it time frame around -0.02<t[peak_loc]<0.012 (here R wave is typica;ly dettected by mne - for ecg, for eog it is -0.1<t[peak_loc]<0.2)
        # 2. take 5 channels with most prominent peak 
        # 3. find estimated average t0 for all 5 channels, because t0 of event which mne estimated is often not accurate
        # 4. flip all channels with negative peak around estimated t0 


        avg_ecg_epoch_data_nonflipped=avg_ecg_epochs.data

        _, t0_estimated_ind, t0_estimated_ind_start, t0_estimated_ind_end = estimate_t0(ecg_or_eog, avg_ecg_epoch_data_nonflipped, t)
        ecg_epoch_per_ch, ecg_epoch_per_ch_only_data = flip_channels(avg_ecg_epoch_data_nonflipped, channels, max_n_peaks_allowed, thresh_lvl_peakfinder, t0_estimated_ind_start, t0_estimated_ind_end, t0_estimated_ind, t)

      
    else:
        print('___MEG QC___: ', 'Wrong set variable: use_abs_of_all_data=', use_abs_of_all_data)


    # Find affected channels after flipping:
    # 5. calculate average of all channels
    # 6. find peak on average channel and set it as actual t0
    # 7. affected channels will be the ones which have peak amplitude over average in limits of -0.05:0.05s from actual t0 '''


    avg_ecg_overall=np.mean(ecg_epoch_per_ch_only_data, axis=0) 
    # will show if there is ecg artifact present  on average. should have ecg shape if yes. 
    # otherwise - it was not picked up/reconstructed correctly

    avg_ecg_overall_obj=Mean_artifact_with_peak(name='Mean_'+ecg_or_eog+'_overall',mean_artifact_epoch=avg_ecg_overall)
    avg_ecg_overall_obj.find_peaks_and_detect_Rwave(max_n_peaks_allowed, thresh_lvl_peakfinder)
    mean_ecg_magnitude_peak=np.max(avg_ecg_overall_obj.peak_magnitude)
    mean_ecg_loc_peak = avg_ecg_overall_obj.peak_loc[np.argmax(avg_ecg_overall_obj.peak_magnitude)]
    
    #set t0_actual as the time of the peak of the average ecg artifact:
    t0_actual=t[mean_ecg_loc_peak]

    if avg_ecg_overall_obj.r_wave_shape is True:
        print('___MEG QC___: ', "GOOD " +ecg_or_eog+ " average.")
    else:
        print('___MEG QC___: ', "BAD " +ecg_or_eog+ " average - no typical ECG peak.")


    if plotflag is True:
        fig_avg = go.Figure()
        avg_ecg_overall_obj.plot_epoch_and_peak(fig_avg, t, 'Mean '+ecg_or_eog+' artifact over all data: ', m_or_g)
        fig_avg.show()

    #2. and 3.:
    ecg_affected_channels, ecg_not_affected_channels, artifact_lvl = detect_channels_above_norm(norm_lvl=norm_lvl, list_mean_ecg_epochs=ecg_epoch_per_ch, mean_ecg_magnitude_peak=mean_ecg_magnitude_peak, t=t, t0_actual=t0_actual, ecg_or_eog=ecg_or_eog)

    if plotflag is True:
        fig_affected = plot_affected_channels(ecg_affected_channels, artifact_lvl, t, ch_type=m_or_g, fig_tit=ecg_or_eog+' affected channels: ', use_abs_of_all_data=use_abs_of_all_data)
        fig_not_affected = plot_affected_channels(ecg_not_affected_channels, artifact_lvl, t, ch_type=m_or_g, fig_tit=ecg_or_eog+' not affected channels: ', use_abs_of_all_data=use_abs_of_all_data)

    if avg_ecg_overall_obj.r_wave_shape is False and (not ecg_not_affected_channels or len(ecg_not_affected_channels)/len(channels)<0.2):
        print('___MEG QC___: ', 'Something went wrong! The overall average ' +ecg_or_eog+ ' is  bad, but all  channels are affected by ' +ecg_or_eog+ ' artifact.')

    return ecg_affected_channels, fig_affected, fig_not_affected, fig_avg



#%%
def make_dict_global_ECG_EOG(all_affected_channels, channels):
    ''' Make simple metric for ECG/EOG artifacts. '''

    if not all_affected_channels:
        number_of_affected_ch = 0
        percent_of_affected_ch = 0
        affected_chs = None
        top_10_magnitudes = None
    else:
        number_of_affected_ch = len(all_affected_channels)
        percent_of_affected_ch = round(len(all_affected_channels)/len(channels)*100, 1)
        affected_chs = {ch.name: ch.main_peak_magnitude for ch in all_affected_channels}
        # CHECK HERE! MAX MAGNITUDE MIGHT BE NOT THE ONE I NEED. NEED THE ONE WHICH IS IN TIME WINDOW OF INTEREST. 
        # Create in the class a method that returns the peak magnitude in the time window of interest.

        #get top 10 magnitudes: 
        # sort all_affected_channels by main_peak_magnitude:
        all_affected_channels_sorted = sorted(all_affected_channels, key=lambda ch: ch.main_peak_magnitude, reverse=True)
        #make a dictionary of top 10 channels with highest peak magnitude:
        top_10_magnitudes = {ch_peak.name: max(ch_peak.peak_magnitude) for ch_peak in all_affected_channels_sorted[0:10]}

    metric_global_content = {
        'number_of_affected_ch': number_of_affected_ch,
        'percent_of_affected_ch': percent_of_affected_ch, 
        'Details':  affected_chs,
        'Top_10_affected_chs': top_10_magnitudes}

    return metric_global_content


def make_simple_metric_ECG_EOG(all_affected_channels, m_or_g_chosen, ecg_or_eog, channels):
    """ Make simple metric for ECG/EOG artifacts. """

    metric_global_name = 'All_'+ecg_or_eog+'_affected_channels'
    metric_global_description = 'Affected channels are the channels with average (over '+ecg_or_eog+' epochs of this channel)' +ecg_or_eog+ ' artifact above the threshold. Threshld is defined as average '+ecg_or_eog+' artifact peak magnitude over al channels * norm_lvl. norm_lvl is defined in the config file. Metrci also provides a list of 10 most strongly affected channels + their artfact peaks magnitdes.'

    metric_global_content={'mag': None, 'grad': None}
    for m_or_g in m_or_g_chosen:
        metric_global_content[m_or_g]= make_dict_global_ECG_EOG(all_affected_channels[m_or_g], channels[m_or_g])

    simple_metric = simple_metric_basic(metric_global_name, metric_global_description, metric_global_content['mag'], metric_global_content['grad'], display_only_global=True)

    return simple_metric

#%%
def ECG_meg_qc(ecg_params: dict, raw: mne.io.Raw, channels, m_or_g_chosen: list):
    """Main ECG function"""

    ecg_events, ch_ecg, average_pulse, ecg_data=mne.preprocessing.find_ecg_events(raw, return_ecg=True, verbose=False)

    if ch_ecg:
        print('___MEG QC___: ', 'ECG channel used to identify hearbeats: ', raw.info['chs'][ch_ecg]['ch_name'])
    else:
        print('___MEG QC___: ', 'No ECG channel found. The signal is reconstructed based  of magnetometers data.')
    print('___MEG QC___: ', 'Average pulse: ', round(average_pulse), ' per minute') 

    ecg_events_times  = (ecg_events[:, 0] - raw.first_samp) / raw.info['sfreq']
    
    sfreq=raw.info['sfreq']
    tmin=ecg_params['ecg_epoch_tmin']
    tmax=ecg_params['ecg_epoch_tmax']
    norm_lvl=ecg_params['norm_lvl']
    use_abs_of_all_data=ecg_params['use_abs_of_all_data']
    
    ecg_derivs = []
    all_ecg_affected_channels={}

    for m_or_g  in m_or_g_chosen:

        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, picks=channels[m_or_g], tmin=tmin, tmax=tmax)

        fig_ecg = ecg_epochs.plot_image(combine='mean', picks = m_or_g)[0] #plot averageg over ecg epochs artifact
        # [0] is to plot only 1 figure. the function by default is trying to plot both mag and grad, but here we want 
        # to do them saparetely depending on what was chosen for analysis
        ecg_derivs += [QC_derivative(fig_ecg, 'mean_ECG_epoch_'+m_or_g, 'matplotlib')]
        fig_ecg.show()

        #averaging the ECG epochs together:
        avg_ecg_epochs = ecg_epochs.average() #.apply_baseline((-0.5, -0.2))
        # about baseline see here: https://mne.tools/stable/auto_tutorials/preprocessing/10_preprocessing_overview.html#sphx-glr-auto-tutorials-preprocessing-10-preprocessing-overview-py
    
        fig_ecg_sensors = avg_ecg_epochs.plot_joint(times=[tmin-tmin/100, tmin/2, 0, tmax/2, tmax-tmax/100], picks = m_or_g)
        # tmin+tmin/10 and tmax-tmax/10 is done because mne sometimes has a plotting issue, probably connected tosamplig rate: 
        # for example tmin is  set to -0.05 to 0.02, but it  can only plot between -0.0496 and 0.02.

        #plot average artifact with topomap
        ecg_derivs += [QC_derivative(fig_ecg_sensors, 'ECG_field_pattern_sensors_'+m_or_g, 'matplotlib')]
        fig_ecg_sensors.show()

        ecg_affected_channels, fig_affected, fig_not_affected, fig_avg=find_affected_channels(ecg_epochs, channels[m_or_g], m_or_g, norm_lvl, ecg_or_eog='ECG', thresh_lvl_peakfinder=6, tmin=tmin, tmax=tmax, plotflag=True, sfreq=sfreq, use_abs_of_all_data=use_abs_of_all_data)
        ecg_derivs += [QC_derivative(fig_affected, 'ECG_affected_channels_'+m_or_g, 'plotly')]
        ecg_derivs += [QC_derivative(fig_not_affected, 'ECG_not_affected_channels_'+m_or_g, 'plotly')]
        ecg_derivs += [QC_derivative(fig_avg, 'overall_average_ECG_epoch_'+m_or_g, 'plotly')]
        all_ecg_affected_channels[m_or_g]=ecg_affected_channels

    simple_metric_ECG = make_simple_metric_ECG_EOG(all_ecg_affected_channels, m_or_g_chosen, 'ECG', channels)

    return ecg_derivs, simple_metric_ECG, ecg_events_times, all_ecg_affected_channels


#%%
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

    for m_or_g  in m_or_g_chosen:

        eog_epochs = mne.preprocessing.create_eog_epochs(raw, picks=channels[m_or_g], tmin=tmin, tmax=tmax)

        fig_eog = eog_epochs.plot_image(combine='mean', picks = m_or_g)[0]
        eog_derivs += [QC_derivative(fig_eog, 'mean_EOG_epoch_'+m_or_g, 'matplotlib')]

        #averaging the ECG epochs together:
        fig_eog_sensors = eog_epochs.average().plot_joint(picks = m_or_g)
        eog_derivs += [QC_derivative(fig_eog_sensors, 'EOG_field_pattern_sensors_'+m_or_g, 'matplotlib')]

        eog_affected_channels, fig_affected, fig_not_affected, fig_avg=find_affected_channels(eog_epochs, channels[m_or_g], m_or_g, norm_lvl, ecg_or_eog='EOG', thresh_lvl_peakfinder=2, tmin=tmin, tmax=tmax, plotflag=True, sfreq=sfreq, use_abs_of_all_data=use_abs_of_all_data)
        eog_derivs += [QC_derivative(fig_affected, 'EOG_affected_channels_'+m_or_g, 'plotly')]
        eog_derivs += [QC_derivative(fig_not_affected, 'EOG_not_affected_channels_'+m_or_g, 'plotly')]
        eog_derivs += [QC_derivative(fig_avg, 'overall_average_EOG_epoch_'+m_or_g, 'plotly')]
        all_eog_affected_channels[m_or_g]=eog_affected_channels

    simple_metric_EOG=make_simple_metric_ECG_EOG(all_eog_affected_channels, m_or_g_chosen, 'EOG', channels)

    return eog_derivs, simple_metric_EOG, eog_events_times, all_eog_affected_channels

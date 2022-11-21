# This version of main allows to only choose mags, grads or both for 
# the entire pipeline in the beginning. Not for separate QC measures

#%%

import os
import mne
import configparser
import ancpbids
from ancpbids import BIDSLayout
from ancpbids import load_dataset
import mpld3

from data_load_and_folders import Epoch_meg
from RMSE_meq_qc import RMSE_meg_qc
from PSD_meg_qc import PSD_meg_qc
from Peaks_manual_meg_qc import PP_manual_meg_qc
from Peaks_auto_meg_qc import PP_auto_meg_qc
from ECG_meg_qc import ECG_meg_qc
from EOG_meg_qc import EOG_meg_qc
from universal_html_report import make_joined_report
from universal_plots import QC_derivative


#%%

def get_all_config_params(config_file_name: str):
    '''Parse all the parameters from config and put into a python dictionary 
    divided by sections. Parsing approach can be changed here, which 
    will not affect working of other fucntions.
    
    Return:
    all_qc_params: dictionary of dictionaries, where each one refers to 
    a QC pipeline section and contains corresponding parameters.
    '''
    
    all_qc_params = {}

    config = configparser.ConfigParser()
    config.read(config_file_name)

    default_section = config['DEFAULT']

    m_or_g_chosen = default_section['do_for'] 
    m_or_g_chosen = m_or_g_chosen.replace(" ", "")
    m_or_g_chosen = m_or_g_chosen.split(",")

    dataset_path = default_section['data_directory']
    tmin = default_section['data_crop_tmin']
    tmax = default_section['data_crop_tmax']

    if not tmin: 
        tmin = 0
    else:
        tmin=float(tmin)
    if not tmax: 
        tmax = None
    else:
        tmax=float(tmax)

    default_params = dict({
        'm_or_g_chosen': m_or_g_chosen, 
        'dataset_path': dataset_path,
        'crop_tmin': tmin,
        'crop_tmax': tmax})
    all_qc_params['default'] = default_params

    filtering_section = config['Filtering']
    if filtering_section['apply_filtering'] is True:
        filtering_params = dict({
            'l_freq': filtering_section.getfloat('l_freq'),
            'h_freq': filtering_section.getfloat('h_freq'),
            'method': filtering_section['method']})
        all_qc_params['Filtering'] = filtering_params
    else: 
        all_qc_params['Filtering'] = 'Not apply'


    epoching_section = config['Epoching']
    stim_channel = epoching_section['stim_channel'] 
    stim_channel = stim_channel.replace(" ", "")
    stim_channel = stim_channel.split(",")
    if stim_channel==['']:
        stim_channel=None

    epoching_params = dict({
    'event_dur': epoching_section.getfloat('event_dur'),
    'epoch_tmin': epoching_section.getfloat('epoch_tmin'),
    'epoch_tmax': epoching_section.getfloat('epoch_tmax'),
    'stim_channel': stim_channel})

    all_qc_params['Epoching'] = epoching_params

    return all_qc_params


def initial_processing(default_settings: dict, filtering_settings: dict, epoching_params:dict, data_file: str):

    '''Here all the initial actions need to work with MEG data are done: 
    - load fif file and convert into raw,
    - create folders in BIDS compliant format,
    - crop the data if needed,
    - filter and downsample the data,
    - epoch the data.

    Args:
    config: config file like settings.ini
    data_file (str): path to the data file

    Returns: 
    dict_of_dfs_epoch (dict with 2 pd. Dataframe): 2 data frames containing data for all epochs for mags and grads
    epochs_mg (dict with 2 mne. Epochs): 2 epoch objects for mags and  grads
    channels (dict): mags and grads channels names
    raw_bandpass(mne.raw): data only filtered, cropped (*)
    raw_bandpass_resamp(mne.raw): data filtered and resampled, cropped (*)
    raw_cropped(mne.io.Raw): data in raw format, cropped, not filtered, not resampled (*)
    raw(mne.io.Raw): original data in raw format, not cropped, not filtered, not resampled.
    (*): if duration was set to None - the data will not be cropped and these outputs 
    will return what is stated, but in origibal duration.
    '''

    active_shielding_used = False
    try:
        raw = mne.io.read_raw_fif(data_file, on_split_missing='ignore')
    except: 
        raw = mne.io.read_raw_fif(data_file, allow_maxshield=True, on_split_missing='ignore')
        active_shielding_used = True

    mag_ch_names = raw.copy().pick_types(meg='mag').ch_names if 'mag' in raw else None
    grad_ch_names = raw.copy().pick_types(meg='grad').ch_names if 'grad' in raw else None
    channels = {'mags': mag_ch_names, 'grads': grad_ch_names}

    #crop the data to calculate faster:
    tmax=default_settings['crop_tmax']
    if tmax is None: 
        tmax = raw.times[-1] 
    raw_cropped = raw.copy().crop(tmin=default_settings['crop_tmin'], tmax=tmax)

    #Data filtering:
    raw_filtered = raw_cropped.copy()
    if filtering_settings != 'Not apply':
        raw_cropped.load_data(verbose=True) #Data has to be loaded into mememory before filetering:
        raw_filtered = raw_cropped.copy()
        raw_filtered.filter(l_freq=filtering_settings['lfreq'], h_freq=filtering_settings['h_freq'], picks='meg', method=filtering_settings['method'], iir_params=None)
        
        #And downsample:
        raw_filtered_resampled = raw_filtered.copy().resample(sfreq=filtering_settings['h_freq']*5)
        #frequency to resample is 5 times higher than the maximum chosen frequency of the function
    else:
        raw_filtered_resampled = raw_filtered.copy()
        #OR maybe we dont need these 2 copies of data at all? Think how to get rid of them, 
        # because they are used later. Referencing might mess up things, check that.
    

    #Apply epoching: USE NON RESAMPLED DATA. Or should we resample after epoching? 
    # Since sampling freq is 1kHz and resampling is 500Hz, it s not that much of a win...

    dict_of_dfs_epoch, epochs_mg = Epoch_meg(epoching_params, data=raw_filtered)

    return dict_of_dfs_epoch, epochs_mg, channels, raw_filtered, raw_filtered_resampled, raw_cropped, raw, active_shielding_used



def sanity_check(m_or_g_chosen, channels):
    '''Check if the channels which the user gave in config file to analize actually present in the data set'''

    if m_or_g_chosen != ['mags'] and m_or_g_chosen != ['grads'] and m_or_g_chosen != ['mags', 'grads']:
        m_or_g_chosen = []
    if channels['mags'] is None and 'mags' in m_or_g_chosen:
        print('There are no magnetometers in this data set: check parameter do_for in config file. Analysis will be done only for gradiometers.')
        m_or_g_chosen.remove('mags')
    elif channels['grads'] is None and 'grads' in m_or_g_chosen:
        print('There are no gradiometers in this data set: check parameter do_for in config file. Analysis will be done only for magnetometers.')
        m_or_g_chosen.remove('grads')
    elif channels['mags'] is None and channels['grads'] is None:
        print ('There are no magnetometers or gradiometers in this data set. Analysis will not be done.')
        m_or_g_chosen = []
    return m_or_g_chosen


#%%
def make_derivative_meg_qc(config_file_name):

    """Main function of MEG QC:
    - Parse parameters from config
    - Get the data .fif file for each subject
    - Run whole analysis for every subject, every fif
    - Make and save derivatives (html figures, csvs, html reports)"""

    all_qc_params = get_all_config_params(config_file_name)

    m_or_g_chosen = sanity_check(m_or_g_chosen=all_qc_params['default']['m_or_g_chosen'], channels=channels)
    if len(m_or_g_chosen) == 0: 
        raise ValueError('No channels to analyze. Check presence of mags and grads in your data set and parameter do_for in settings.')

    dataset_path = all_qc_params['default']['dataset_path']
    layout = BIDSLayout(dataset_path)
    schema = layout.schema

    #create derivative folder first!
    if os.path.isdir(dataset_path+'/derivatives')==False: 
            os.mkdir(dataset_path+'/derivatives')

    derivative = layout.dataset.create_derivative(name="Meg_QC")
    derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"

    list_of_subs = layout.get_subjects()
    if not list_of_subs:
        print('No subjects found. Check your data set and directory path.')
        return

    for sid in [list_of_subs[0]]: #RUN OVER JUST 1 SUBJ

        subject_folder = derivative.create_folder(type_=schema.Subject, name='sub-'+sid)

        list_of_fifs = layout.get(suffix='meg', extension='.fif', return_type='filename', subj=sid)
        #Devide here fifs by task, ses , run

        dataset_ancp_loaded = ancpbids.load_dataset(dataset_path)
        list_of_sub_jsons = dataset_ancp_loaded.query(sub=sid, suffix='meg', extension='.fif')

        for fif_ind,data_file in enumerate(list_of_fifs): #RUN OVER JUST 1 FIF because is not divided by tasks yet..

            dict_of_dfs_epoch, epochs_mg, channels, raw_filtered, raw_filtered_resampled, raw_cropped, raw, active_shielding_used = initial_processing(default_settings=all_qc_params['default'], filtering_settings=all_qc_params['Filtering'], epoching_params=all_qc_params['Epoching'], data_file=data_file)
            
            rmse_derivs, psd_derivs, pp_manual_derivs, ptp_auto_derivs, ecg_derivs, eog_derivs = [],[],[],[],[], []
            
            rmse_derivs, big_rmse_with_value_all_data, small_rmse_with_value_all_data = RMSE_meg_qc(config, channels, dict_of_dfs_epoch, raw_filtered_resampled, m_or_g_chosen)

            # psd_derivs = PSD_meg_qc(config, channels, raw_filtered_resampled, m_or_g_chosen)

            # pp_manual_derivs = PP_manual_meg_qc(config, channels, dict_of_dfs_epoch, raw_filtered_resampled, m_or_g_chosen)

            # ptp_auto_derivs, bad_channels = PP_auto_meg_qc(config, channels, raw_filtered_resampled, m_or_g_chosen)

            # ecg_derivs, ecg_events_times = ECG_meg_qc(config, raw, m_or_g_chosen)

            eog_derivs, eog_events_times = EOG_meg_qc(config, raw, m_or_g_chosen)

            # HEAD_movements_meg_qc()

            # MUSCLE_meg_qc()


            # Make strings to add to html report:
            shielding_str, channels_skipped_str, epoching_skipped_str, no_ecg_str, no_eog_str = '', '', '', '', ''

            if active_shielding_used is True: 
                shielding_str=''' <p>This file contains Internal Active Shielding data. Quality measurements calculated on this data should not be compared to the measuremnts calculated on the data without active shileding, since in the current case invironmental noise reduction was already partially performed by shileding, which normally should not be done before assesing the quality.</p><br></br>'''
            
            if 'mags' not in m_or_g_chosen:
                channels_skipped_str = ''' <p>This data set contains no magnetometers or they were not chosen for analysis. Quality measurements were performed only on gradiometers.</p><br></br>'''
            elif 'grads' not in m_or_g_chosen:
                channels_skipped_str = ''' <p>This data set contains no gradiometers or they were not chosen for analysis. Quality measurements were performed only on magnetometers.</p><br></br>'''

            if dict_of_dfs_epoch['mags'] is None and dict_of_dfs_epoch['grads'] is None:
                epoching_skipped_str = ''' <p>No epoching could be done in this data set: no events found. Quality measurement were only performed on the entire time series. If this was not expected, try: 1) checking the presence of stimulus channel in the data set, 2) setting stimulus channel explicitly in config file, 3) setting different event duration in config file.</p><br></br>'''

            if ecg_derivs is None:
                no_ecg_str = 'No ECG channels found is this data set, cardio artifacts can not be detected. ECG data can be reconstructed on base of magnetometers, but this will not be accurate and is not recommended.'
                ecg_derivs = []

            if eog_derivs is None:
                no_eog_str = 'No EOG channels found is this data set - EOG artifacts can not be detected.'
                eog_derivs = []

            QC_derivs={
            'Standart deviation of the data':rmse_derivs, 
            'Frequency spectrum': psd_derivs, 
            'Peak-to-Peak manual': pp_manual_derivs, 
            'Peak-to-Peak auto from MNE': ptp_auto_derivs, 
            'ECG': ecg_derivs, 
            'EOG': eog_derivs,
            'Head movement artifacts': [],
            'Muscle artifacts': []}

            report_html_string = make_joined_report(QC_derivs, shielding_str, channels_skipped_str, epoching_skipped_str, no_ecg_str, no_eog_str)
            QC_derivs['Report']= [QC_derivative(report_html_string, 'REPORT', None, 'report')]

            for section in QC_derivs.values():
                if section: #if there are any derivs calculated in this section:
                    for deriv in section:

                        meg_artifact = subject_folder.create_artifact(raw=list_of_sub_jsons[fif_ind]) #shell. empty derivative
                        meg_artifact.add_entity('desc', deriv.description) #file name
                        #meg_artifact.add_entity('task', task_label)
                        meg_artifact.suffix = 'meg'
                        meg_artifact.extension = '.html'

                        if deriv.content_type == 'matplotlib':
                            #mpld3.save_html(list_of_figures[i], list_of_fig_descriptions[i]+'.html')
                            meg_artifact.content = lambda file_path, cont=deriv.content: mpld3.save_html(cont, file_path)
                        elif deriv.content_type == 'plotly':
                            meg_artifact.content = lambda file_path, cont=deriv.content: cont.write_html(file_path)
                        elif deriv.content_type == 'df':
                            meg_artifact.extension = '.csv'
                            meg_artifact.content = lambda file_path, cont=deriv.content: cont.to_csv(file_path)
                        elif deriv.content_type == 'report':
                            def html_writer(file_path):
                                with open(file_path, "w") as file:
                                    file.write(deriv.content)
                                #'with'command doesnt work in lambda
                            meg_artifact.content = html_writer # function pointer instead of lambda
                        #problem with lambda explained:
                        #https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
        
    layout.write_derivative(derivative) #maybe put inside the loop if can't have so much in memory?

    return raw

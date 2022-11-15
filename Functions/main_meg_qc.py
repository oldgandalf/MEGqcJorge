# This version of main allows to only choose mags, grads or both for 
# the entire pipeline in the beginning. Not for separate QC measures

#%%

import mne
import configparser
import ancpbids
from ancpbids import BIDSLayout
import mpld3

from data_load_and_folders import load_meg_data, make_folders_meg, Epoch_meg
from RMSE_meq_qc import RMSE_meg_qc
from PSD_meg_qc import PSD_meg_qc
from Peaks_manual_meg_qc import PP_manual_meg_qc
from Peaks_auto_meg_qc import PP_auto_meg_qc
from ECG_meg_qc import ECG_meg_qc
from EOG_meg_qc import EOG_meg_qc
from universal_html_report import keep_fig_derivs, make_joined_report
from universal_plots import QC_derivative


#%%
def initial_stuff(config: dict, data_file: str):

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

    raw = mne.io.read_raw_fif(data_file, allow_maxshield=True, on_split_missing='ignore')


    mag_ch_names = raw.copy().pick_types(meg='mag').ch_names if 'mag' in raw else None
    grad_ch_names = raw.copy().pick_types(meg='grad').ch_names if 'grad' in raw else None
    channels = {'mags': mag_ch_names, 'grads': grad_ch_names}

    default_section = config['DEFAULT']

    #crop the data to calculate faster:
    tmin = default_section['data_crop_tmin']
    tmax = default_section['data_crop_tmax']

    if not tmin: 
        tmin = 0
    else:
        tmin=float(tmin)
    if not tmax: 
        tmax = raw.times[-1] 
    else:
        tmax=float(tmax)

    raw_cropped = raw.copy()
    raw_cropped.crop(tmin, tmax)

    #Data filtering:
    filtering_section = config['Filter_and_resample']
    if filtering_section['apply_filtering'] is True:
        l_freq = filtering_section.getfloat('l_freq') 
        h_freq = filtering_section.getfloat('h_freq') 
        method = filtering_section['method']
    
        raw_cropped.load_data(verbose=True) #Data has to be loaded into mememory before filetering:
        raw_filtered = raw_cropped.copy()
        raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, picks='meg', method=method, iir_params=None)

        #And downsample:
        raw_filtered_resampled=raw_filtered.copy()
        raw_filtered_resampled.resample(sfreq=h_freq*5)
        #frequency to resample is 5 times higher than the maximum chosen frequency of the function

    else:
        raw_filtered = raw_cropped.copy()
        raw_filtered_resampled=raw_filtered.copy()
        #OR maybe we dont need these 2 copies of data at all? Think how to get rid of them, 
        # because they are used later. Referencing might mess up things, check that.


    #Apply epoching: USE NON RESAMPLED DATA. Or should we resample after epoching? 
    # Since sampling freq is 1kHz and resampling is 500Hz, it s not that much of a win...

    dict_of_dfs_epoch, epochs_mg = Epoch_meg(config, raw_filtered)

    return dict_of_dfs_epoch, epochs_mg, channels, raw_filtered, raw_filtered_resampled, raw_cropped, raw



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

    """Main function of MEG QC.
    - Parse parameters from config
    - Get the data .fif file for each subject
    - Run whole analysis for every subject, every fif
    - Make and save derivatives (html figures, csvs, html reports, etc...)"""

    config = configparser.ConfigParser()
    config.read(config_file_name)

    default_section = config['DEFAULT']

    m_or_g_chosen = default_section['do_for'] 
    m_or_g_chosen = m_or_g_chosen.replace(" ", "")
    m_or_g_chosen = m_or_g_chosen.split(",")
    #m_or_g_chosen = select_m_or_g(default_section)

    dataset_path = default_section['data_directory']

    layout = BIDSLayout(dataset_path)
    schema = layout.schema

    #create derivative folder first!
    derivative = layout.dataset.create_derivative(name="Meg_QC")
    derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"

    list_of_subs = layout.get_subjects()
    #print(list_of_subs)

    for sid in [list_of_subs[0]]: #RUN OVER JUST 1 SUBJ
    #for sid in list_of_subs: 

        subject_folder = derivative.create_folder(type_=schema.Subject, name='sub-'+sid)

        list_of_fifs = layout.get(suffix='meg', extension='.fif', return_type='filename', subj=sid)
        #Devide here fifs by task, ses , run

        for data_file in [list_of_fifs[0]]: #RUN OVER JUST 1 FIF because is not divided by tasks yet..
            dict_of_dfs_epoch, epochs_mg, channels, raw_filtered, raw_filtered_resampled, raw_cropped, raw = initial_stuff(config, data_file)

            m_or_g_chosen = sanity_check(m_or_g_chosen, channels)
            if len(m_or_g_chosen) == 0: 
                raise ValueError('No channels to analyze. Check presence of mags and grads in your data set and parameter do_for in settings.')

            
            rmse_derivs, psd_derivs, pp_manual_derivs, ptp_auto_derivs, ecg_derivs, eog_derivs = [],[],[],[],[], []
            
            #rmse_derivs = RMSE_meg_qc(sid, config, channels, dict_of_dfs_epoch, raw_filtered_resampled, m_or_g_chosen)

            psd_derivs = PSD_meg_qc(sid, config, channels, raw_filtered_resampled, m_or_g_chosen)

            #pp_manual_derivs = PP_manual_meg_qc(sid, config, channels, dict_of_dfs_epoch, raw_filtered_resampled, m_or_g_chosen)

            #ptp_auto_derivs, bad_channels = PP_auto_meg_qc(sid, config, channels, raw_filtered_resampled, m_or_g_chosen)

            #ecg_derivs = ECG_meg_qc(config, raw, m_or_g_chosen)

            eog_derivs = EOG_meg_qc(config, raw, m_or_g_chosen)

            # HEAD_movements_meg_qc()

            # MUSCLE_meg_qc()

            all_derivs = rmse_derivs + psd_derivs + pp_manual_derivs + ptp_auto_derivs + ecg_derivs + eog_derivs
            # all_derivs: list of QC_derivative objects - output of every funct also has this format

            #put all figure derivatives into dectionary, each with own section to later use in making report with sections
            # derivs_list=[rmse_derivs, psd_derivs, pp_manual_derivs, ecg_derivs, eog_derivs]
            # derivs_names_list=['rmse', 'psd', 'pp_manual', 'ecg', 'eog']
            # derivs_per_section = {}
            # for d_ind, d in enumerate(derivs_list):
            #     all_fig_derivs = keep_fig_derivs(d)
            #     derivs_per_section[derivs_names_list[d_ind]]=all_fig_derivs

            # html_string='''<html>
            #     <head>
            #     <title>HTML File</title>
            #     </head>
            #     <body>
            #     <h1>stuff</h1>
            #     <p>Example stuff</p>
            #     </body>
            #     </html>'''
            # all_derivs=[(html_string, 'stuff_report', None, 'report')]

            if all_derivs:

                report_html_string = make_joined_report(rmse_derivs, psd_derivs, pp_manual_derivs, ptp_auto_derivs, ecg_derivs, eog_derivs)
                all_derivs += [QC_derivative(report_html_string, 'REPORT', None, 'report')]

                # print('HERE!!')
                # print(all_derivs[1])
                # print(all_derivs[2])
                # print(all_derivs[3])

                for i in range(0, len(all_derivs)):
                    meg_artifact = subject_folder.create_artifact() #shell. empty derivative
                    meg_artifact.add_entity('desc', all_derivs[i].description) #file name
                    #meg_artifact.add_entity('task', task_label)
                    meg_artifact.suffix = 'meg'
                    meg_artifact.extension = '.html'

                    if all_derivs[i].content_type == 'matplotlib':
                        #mpld3.save_html(list_of_figures[i], list_of_fig_descriptions[i]+'.html')
                        meg_artifact.content = lambda file_path, cont=all_derivs[i].content: mpld3.save_html(cont, file_path)
                    elif all_derivs[i].content_type == 'plotly':
                        meg_artifact.content = lambda file_path, cont=all_derivs[i].content: cont.write_html(file_path)
                    elif all_derivs[i].content_type == 'df':
                        meg_artifact.extension = '.csv'
                        meg_artifact.content = lambda file_path, cont=all_derivs[i].content: cont.to_csv(file_path)
                    elif all_derivs[i].content_type == 'report':
                        def html_writer(file_path):
                            with open(file_path, "w") as file:
                                file.write(all_derivs[i].content)
                            #'with'command doesnt work in lambda
                        meg_artifact.content = html_writer # function pointer instead of lambda
                    #problem with lambda explained:
                    #https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
        
    layout.write_derivative(derivative) #maybe put inside the loop if can't have so much in memory?

    return 

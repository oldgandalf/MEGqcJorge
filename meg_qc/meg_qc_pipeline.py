import os
import ancpbids
import time
import json

import sys

# Needed to import the modules without specifying the full path, for command line and jupyter notebook
sys.path.append('./meg_qc/source/')

# relative path for `make html` (docs)
sys.path.append('../meg_qc/source/')

# relative path for `make html` (docs) run from https://readthedocs.org/
# every time rst file is nested insd of another, need to add one more path level here:
sys.path.append('../../meg_qc/source/')
sys.path.append('../../../meg_qc/source/')
sys.path.append('../../../../meg_qc/source/')


from meg_qc.source.initial_meg_qc import get_all_config_params, sanity_check, initial_processing
from meg_qc.source.STD_meg_qc import STD_meg_qc
from meg_qc.source.PSD_meg_qc import PSD_meg_qc
from meg_qc.source.Peaks_manual_meg_qc import PP_manual_meg_qc
from meg_qc.source.Peaks_auto_meg_qc import PP_auto_meg_qc
from meg_qc.source.ECG_EOG_meg_qc import ECG_meg_qc, EOG_meg_qc
from meg_qc.source.Head_meg_qc import HEAD_movement_meg_qc
from meg_qc.source.muscle_meg_qc import MUSCLE_meg_qc
from meg_qc.source.universal_html_report import make_joined_report, make_joined_report_mne
from meg_qc.source.universal_plots import QC_derivative

def make_derivative_meg_qc(config_file_path):

    """ 
    Main function of MEG QC:
    
    * Parse parameters from config
    * Get the data .fif file for each subject using ancpbids
    * Run initial processing (filtering, epoching, resampling)
    * Run whole QC analysis for every subject, every fif
    * Save derivatives (csvs, html reports) into the file system using ancpbids.
    
    Parameters
    ----------
    config_file_path : str
        Path the config file with all the parameters for the QC analysis and data directory path.
        
    Returns
    -------
        raw : mne.io.Raw
            The raw MEG data.

    """

    all_qc_params = get_all_config_params(config_file_path)

    if all_qc_params is None:
        return

    dataset_path = all_qc_params['default']['dataset_path']

    try:
        dataset = ancpbids.load_dataset(dataset_path)
        schema = dataset.get_schema()
    except:
        print('___MEG QC___: ', 'No data found in the given directory path! \nCheck directory path in config file and presence of data on your device.')
        return

    #create derivatives folder first:
    if os.path.isdir(dataset_path+'/derivatives')==False: 
            os.mkdir(dataset_path+'/derivatives')

    derivative = dataset.create_derivative(name="Meg_QC")
    derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"


    # schema = dataset.get_schema()
    # artifacts = filter(lambda m: isinstance(m, schema.Artifact), query(folder, scope=scope))

    # print('___MEG QC___: ', schema)
    # print('___MEG QC___: ', "\n")
    # print('___MEG QC___: ', schema.Artifact)

    # print('___MEG QC___: ', dataset.files)
    # print('___MEG QC___: ', dataset.folders)
    # print('___MEG QC___: ', dataset.derivatives)
    # print('___MEG QC___: ', dataset.items())
    # print('___MEG QC___: ', dataset.keys())
    # print('___MEG QC___: ', dataset.code)
    # print('___MEG QC___: ', dataset.name)

    #return

    # entities = dataset.query_entities()
    # print('___MEG QC___: ', 'entities', entities)

    # list_of_subs = list(entities["sub"])
    list_of_subs = sorted(list(dataset.query_entities()["sub"]))
    print('___MEG QC___: ', 'list_of_subs', list_of_subs)

    if not list_of_subs:
        print('___MEG QC___: ', 'No subjects found. Check your data set and directory path in config.')
        return

    #list_of_subs = ['009', '012', '019', '020', '021', '022', '023', '024', '025'] #especia;ly 23 in ds 83! There doesnt detect all the ecg peaks and says bad ch, but it s good.
    for sid in list_of_subs[0:20]: 
        print('___MEG QC___: ', 'Dataset: ', dataset_path)
        print('___MEG QC___: ', 'Take SID: ', sid)
        
        subject_folder = derivative.create_folder(type_=schema.Subject, name='sub-'+sid)

        list_of_fifs = dataset.query(suffix='meg', extension='.fif', return_type='filename', subj=sid)

        list_of_sub_jsons = dataset.query(sub=sid, suffix='meg', extension='.fif')

        for fif_ind, data_file in enumerate(list_of_fifs[0:2]): #RUN OVER JUST 1 fif to save time

            # Preassign strings with notes for the user to add to html report (in case some QC analysis was skipped):
            shielding_str, m_or_g_skipped_str, epoching_str, ecg_str, eog_str, head_str, muscle_str, pp_manual_str, pp_auto_str, std_str, psd_str = '', '', '', '', '', '', '', '', '', '', ''
 
            print('___MEG QC___: ', 'Starting initial processing...')
            start_time = time.time()
            print('___MEG QC___: ', 'data_file', data_file)
            dict_epochs_mg, channels, raw_cropped_filtered, raw_cropped_filtered_resampled, raw_cropped, raw, shielding_str, epoching_str = initial_processing(default_settings=all_qc_params['default'], filtering_settings=all_qc_params['Filtering'], epoching_params=all_qc_params['Epoching'], data_file=data_file)
                
            m_or_g_chosen = sanity_check(m_or_g_chosen=all_qc_params['default']['m_or_g_chosen'], channels=channels)
            if len(m_or_g_chosen) == 0: 
                m_or_g_skipped_str = '''No channels to analyze. Check presence of mag and grad in your data set and parameter do_for in settings.'''
                raise ValueError(m_or_g_skipped_str)
               
            if 'mag' not in m_or_g_chosen:
                m_or_g_skipped_str = ''' <p>This data set contains no magnetometers or they were not chosen for analysis. Quality measurements were performed only on gradiometers.</p><br></br>'''
            if 'grad' not in m_or_g_chosen:
                m_or_g_skipped_str = ''' <p>This data set contains no gradiometers or they were not chosen for analysis. Quality measurements were performed only on magnetometers.</p><br></br>'''

            print('___MEG QC___: ', "Finished initial processing. --- Execution %s seconds ---" % (time.time() - start_time))

            # QC measurements:
            std_derivs, psd_derivs, pp_manual_derivs, pp_auto_derivs, ecg_derivs, eog_derivs, head_derivs, muscle_derivs = [],[],[],[],[], [],  [], []
            simple_metrics_psd, simple_metrics_std, simple_metrics_pp_manual, simple_metrics_pp_auto, simple_metrics_ecg, simple_metrics_eog, simple_metrics_head, simple_metrics_muscle = [],[],[],[],[],[], [], []
            df_head_pos, head_pos, noisy_freqs_global = [], [], []
            scores_muscle_all1, scores_muscle_all2, scores_muscle_all3 = [], [], []
            # powerline predefined for the the muscle artif function. If powerline noise is present - need to notch filter it first.
            # For this either need to run psd first, or just guess which powerline freq to use based on the country of the data collection.
            # USA: 60, Europe 50. NOT save to assume powerline noise in every data set. Some really dont have it.


            print('___MEG QC___: ', 'Starting STD...')
            start_time = time.time()
            std_derivs, simple_metrics_std, std_str = STD_meg_qc(all_qc_params['STD'], channels, dict_epochs_mg, raw_cropped_filtered_resampled, m_or_g_chosen)
            print('___MEG QC___: ', "Finished STD. --- Execution %s seconds ---" % (time.time() - start_time))
 
            print('___MEG QC___: ', 'Starting PSD...')
            start_time = time.time()
            psd_derivs, simple_metrics_psd, psd_str, noisy_freqs_global = PSD_meg_qc(all_qc_params['PSD'], channels, raw_cropped_filtered, m_or_g_chosen, helperplots=True)
            print('___MEG QC___: ', "Finished PSD. --- Execution %s seconds ---" % (time.time() - start_time))

            print('___MEG QC___: ', 'Starting Peak-to-Peak manual...')
            start_time = time.time()
            pp_manual_derivs, simple_metrics_pp_manual, pp_manual_str = PP_manual_meg_qc(all_qc_params['PTP_manual'], channels, dict_epochs_mg, raw_cropped_filtered_resampled, m_or_g_chosen)
            print('___MEG QC___: ', "Finished Peak-to-Peak manual. --- Execution %s seconds ---" % (time.time() - start_time))

            print('___MEG QC___: ', 'Starting Peak-to-Peak auto...')
            start_time = time.time()
            pp_auto_derivs, bad_channels, pp_auto_str = PP_auto_meg_qc(all_qc_params['PTP_auto'], channels, raw_cropped_filtered_resampled, m_or_g_chosen)
            print('___MEG QC___: ', "Finished Peak-to-Peak auto. --- Execution %s seconds ---" % (time.time() - start_time))

            print('___MEG QC___: ', 'Starting ECG...')
            start_time = time.time()
            ecg_derivs, simple_metrics_ecg, ecg_str = ECG_meg_qc(all_qc_params['ECG'], raw_cropped, channels,  m_or_g_chosen)
            print('___MEG QC___: ', "Finished ECG. --- Execution %s seconds ---" % (time.time() - start_time))

            print('___MEG QC___: ', 'Starting EOG...')
            start_time = time.time()
            eog_derivs, simple_metrics_eog, eog_str = EOG_meg_qc(all_qc_params['EOG'], raw_cropped, channels,  m_or_g_chosen)
            print('___MEG QC___: ', "Finished EOG. --- Execution %s seconds ---" % (time.time() - start_time))

            print('___MEG QC___: ', 'Starting Head movement calculation...')
            head_derivs, simple_metrics_head, head_str, df_head_pos, head_pos = HEAD_movement_meg_qc(raw_cropped, plot_with_lines=True, plot_annotations=False)
            print('___MEG QC___: ', "Finished Head movement calculation. --- Execution %s seconds ---" % (time.time() - start_time))

            print('___MEG QC___: ', 'Starting Muscle artifacts calculation...')
            #use the same form of raw as in the PSD func! Because psd func calculates first if there are powerline noise freqs.
            
            #noisy_freqs_global = {'mag': [50], 'grad': [50]}
            raw_cropped_filtered1 = raw_cropped_filtered.copy()
            raw_cropped_filtered2 = raw_cropped_filtered.copy()
            raw_cropped_filtered3 = raw_cropped_filtered.copy()
            muscle_derivs, simple_metrics_muscle, muscle_str, scores_muscle_all1 = MUSCLE_meg_qc(all_qc_params['Muscle'], raw_cropped_filtered1, noisy_freqs_global, m_or_g_chosen, interactive_matplot=False, attach_dummy = False, cut_dummy = False)
            muscle_derivs, simple_metrics_muscle, muscle_str, scores_muscle_all2 = MUSCLE_meg_qc(all_qc_params['Muscle'], raw_cropped_filtered2, noisy_freqs_global, m_or_g_chosen, interactive_matplot=False, attach_dummy = True, cut_dummy = False)
            muscle_derivs, simple_metrics_muscle, muscle_str, scores_muscle_all3 = MUSCLE_meg_qc(all_qc_params['Muscle'], raw_cropped_filtered3, noisy_freqs_global, m_or_g_chosen, interactive_matplot=False, attach_dummy = True, cut_dummy = True)
            print('___MEG QC___: ', "Finished Muscle artifacts calculation. --- Execution %s seconds ---" % (time.time() - start_time))

            


            report_strings = {
            'STD': std_str,
            'PSD': psd_str,
            'PTP_MANUAL': pp_manual_str,
            'PTP_AUTO': pp_auto_str,
            'ECG': ecg_str,
            'EOG': eog_str,
            'HEAD': head_str,
            'MUSCLE': muscle_str,
            'M_OR_G_SKIPPED': m_or_g_skipped_str,
            'EPOCHING': epoching_str,
            'SHIELDING': shielding_str,}

            QC_derivs={
            'Standard deviation of the data': std_derivs, 
            'Frequency spectrum': psd_derivs, 
            'Peak-to-Peak manual': pp_manual_derivs, 
            'Peak-to-Peak auto from MNE': pp_auto_derivs, 
            'ECG': ecg_derivs, 
            'EOG': eog_derivs,
            'Head movement artifacts': head_derivs,
            'Muscle artifacts': muscle_derivs}

            QC_simple={
            'STD': simple_metrics_std, 
            'PSD': simple_metrics_psd,
            'PTP_MANUAL': simple_metrics_pp_manual, 
            'PTP_AUTO': simple_metrics_pp_auto,
            'ECG': simple_metrics_ecg, 
            'EOG': simple_metrics_eog,
            'HEAD': simple_metrics_head,
            'MUSCLE': simple_metrics_muscle}  


            # Testing errors:
            # for key, value in QC_simple.items():
            #     if isinstance(value, set):
            #         print(f"Set type of data found in key '{key}'")
            

            #Make report and add to QC_derivs:
            # report_html_string = make_joined_report(QC_derivs, shielding_str, m_or_g_skipped_str, epoching_skipped_str, no_ecg_str, no_eog_str, no_head_pos_str, muscle_str)
            # QC_derivs['Report']= [QC_derivative(report_html_string, 'REPORT', 'report')]

            report_html_string = make_joined_report_mne(raw, QC_derivs, report_strings)
            QC_derivs['Report MNE']= [QC_derivative(report_html_string, 'REPORT MNE', 'report mne')]

            #Collect all simple metrics into a dictionary and add to QC_derivs:
            QC_derivs['Simple_metrics']=[QC_derivative(QC_simple, 'Simple_metrics', 'json')]


            # d=0

            #if there are any derivs calculated in this section:
            for section in (section for section in QC_derivs.values() if section):
                # loop over section where deriv.content_type is not 'matplotlib' or 'plotly' or 'report'
                for deriv in (deriv for deriv in section if deriv.content_type != 'matplotlib' and deriv.content_type != 'plotly' and deriv.content_type != 'report'):
                    
                    # d=d+1
                    # print('___MEG QC___: ', 'writing deriv: ', d)
                    # print('___MEG QC___: ', deriv)

                    # if deriv.content_type == 'matplotlib':
                    #     continue
                    #     meg_artifact.extension = '.png'
                    #     meg_artifact.content = lambda file_path, cont=deriv.content: cont.savefig(file_path) 

                    # elif deriv.content_type == 'plotly':
                    #     continue
                    #     meg_artifact.content = lambda file_path, cont=deriv.content: cont.write_html(file_path)

                    # elif deriv.content_type == 'report':
                    #     def html_writer(file_path, cont=deriv.content):
                    #         with open(file_path, "w") as file:
                    #             file.write(cont)
                    #         #'with'command doesnt work in lambda
                    #     meg_artifact.content = html_writer # function pointer instead of lambda

                    meg_artifact = subject_folder.create_artifact(raw=list_of_sub_jsons[fif_ind]) #shell. empty derivative
                    meg_artifact.add_entity('desc', deriv.name) #file name
                    meg_artifact.suffix = 'meg'
                    meg_artifact.extension = '.html'

                    if deriv.content_type == 'df':
                        meg_artifact.extension = '.csv'
                        meg_artifact.content = lambda file_path, cont=deriv.content: cont.to_csv(file_path)

                    elif deriv.content_type == 'report mne':
                        meg_artifact.content = lambda file_path, cont=deriv.content: cont.save(file_path, overwrite=True, open_browser=False)

                    elif deriv.content_type == 'json':
                        meg_artifact.extension = '.json'
                        def json_writer(file_path, cont=deriv.content):
                            with open(file_path, "w") as file_wrapper:
                                json.dump(cont, file_wrapper, indent=4)
                        meg_artifact.content = json_writer 

                        # with open('derivs.json', 'w') as file_wrapper:
                        #     json.dump(metric, file_wrapper, indent=4)

                    else:
                        print('___MEG QC___: ', meg_artifact.name)
                        meg_artifact.content = 'dummy text'
                        meg_artifact.extension = '.txt'
                    # problem with lambda explained:
                    # https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result


    ancpbids.write_derivative(dataset, derivative) 

    return raw, raw_cropped_filtered_resampled, QC_derivs, QC_simple, df_head_pos, head_pos, scores_muscle_all1, scores_muscle_all2, scores_muscle_all3


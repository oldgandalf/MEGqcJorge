import os
import ancpbids
import time
import json
import sys

# Needed to import the modules without specifying the full path, for command line and jupyter notebook
sys.path.append('./')
sys.path.append('./meg_qc/calculation/')

# relative path for `make html` (docs)
sys.path.append('../meg_qc/calculation/')

# relative path for `make html` (docs) run from https://readthedocs.org/
# every time rst file is nested insd of another, need to add one more path level here:
sys.path.append('../../meg_qc/calculation/')
sys.path.append('../../../meg_qc/calculation/')
sys.path.append('../../../../meg_qc/calculation/')


from meg_qc.calculation.initial_meg_qc import get_all_config_params, initial_processing, get_internal_config_params
from meg_qc.plotting.universal_html_report import make_joined_report, make_joined_report_mne
from meg_qc.plotting.universal_plots import QC_derivative

from meg_qc.calculation.metrics.STD_meg_qc import STD_meg_qc
from meg_qc.calculation.metrics.PSD_meg_qc import PSD_meg_qc
from meg_qc.calculation.metrics.Peaks_manual_meg_qc import PP_manual_meg_qc
from meg_qc.calculation.metrics.Peaks_auto_meg_qc import PP_auto_meg_qc
from meg_qc.calculation.metrics.ECG_EOG_meg_qc import ECG_meg_qc, EOG_meg_qc
from meg_qc.calculation.metrics.Head_meg_qc import HEAD_movement_meg_qc
from meg_qc.calculation.metrics.muscle_meg_qc import MUSCLE_meg_qc


def make_derivative_meg_qc(config_file_path,internal_config_file_path):

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
    internal_config_file_path : str
        Path the config file with all the parameters for the QC analysis preset - not to be changed by the user.
        
    Returns
    -------
        raw : mne.io.Raw
            The raw MEG data.

    """

    all_qc_params = get_all_config_params(config_file_path)
    internal_qc_params = get_internal_config_params(internal_config_file_path)

    if all_qc_params is None:
        return

    ds_paths = all_qc_params['default']['dataset_path']


    for dataset_path in ds_paths: #run over several data sets

        try:
            dataset = ancpbids.load_dataset(dataset_path)
            schema = dataset.get_schema()
        except:
            print('___MEGqc___: ', 'No data found in the given directory path! \nCheck directory path in config file and presence of data on your device.')
            return

        #create derivatives folder first:
        if os.path.isdir(dataset_path+'/derivatives')==False: 
                os.mkdir(dataset_path+'/derivatives')

        derivative = dataset.create_derivative(name="Meg_QC")
        derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"


        # print('_____derivs here___')
        # print(type(dataset.derivatives))
        # deriv = dataset.derivatives.folders['Meg_QC'].files
        # print(deriv)

        # schema = dataset.get_schema()
        # artifacts = filter(lambda m: isinstance(m, schema.Artifact), query(folder, scope=scope))

        # print('___MEGqc___: ', schema)
        # print('___MEGqc___: ', "\n")
        # print('___MEGqc___: ', schema.Artifact)

        # print('___MEGqc___: ', dataset.files)
        # print('___MEGqc___: ', dataset.folders)
        # print('___MEGqc___: ', dataset.derivatives)
        # print('___MEGqc___: ', dataset.items())
        # print('___MEGqc___: ', dataset.keys())
        # print('___MEGqc___: ', dataset.code)
        # print('___MEGqc___: ', dataset.name)

        #return

        # entities = dataset.query_entities()
        # print('___MEGqc___: ', 'entities', entities)

        # list_of_subs = list(entities["sub"])
        if all_qc_params['default']['subjects'][0] != 'all':
            list_of_subs = all_qc_params['default']['subjects']
        elif all_qc_params['default']['subjects'][0] == 'all':
            list_of_subs = sorted(list(dataset.query_entities()["subject"]))
            print('___MEGqc___: ', 'list_of_subs', list_of_subs)
            if not list_of_subs:
                print('___MEGqc___: ', 'No subjects found by ANCP BIDS. Check your data set and directory path in config.')
                return
        else:
            print('___MEGqc___: ', 'Something went wrong with the subjects list. Check parameter "subjects" in config file or simply set it to "all".')
            return

        avg_ecg=[]
        avg_eog=[]

        print('___MEGqc___: ', 'TOTAL subs', len(list_of_subs))
        print('___MEGqc___: ', 'EMPTY room?', sorted(list_of_subs)[0], sorted(list_of_subs)[-1])
        #list_of_subs = ['009', '012', '019', '020', '021', '022', '023', '024', '025'] #especially 23 in ds 83! There doesnt detect all the ecg peaks and says bad ch, but it s good.
        
        raw=None #preassign in case no calculation will be successful
        for sid in list_of_subs[4:5]: #[0:4]: 
            print('___MEGqc___: ', 'Dataset: ', dataset_path)
            print('___MEGqc___: ', 'Take SID: ', sid)
            
            subject_folder = derivative.create_folder(type_=schema.Subject, name='sub-'+sid)

            list_of_fifs = sorted(list(dataset.query(suffix='meg', extension='.fif', return_type='filename', subj=sid)))
            print('___MEGqc___: ', 'list_of_fifs', list_of_fifs)
            print('___MEGqc___: ', 'TOTAL fifs: ', len(list_of_fifs))


            # GET all derivs!
            # derivs_list = sorted(list(dataset.query(suffix='meg', extension='.tsv', return_type='filename', subj=sid, scope='derivatives')))
            # print('___MEGqc___: ', 'derivs_list', derivs_list)

            # entities = dataset.query_entities()
            # print('___MEGqc___: ', 'entities', entities)


            list_of_sub_jsons = dataset.query(sub=sid, suffix='meg', extension='.fif')

            print('___MEGqc___: ', 'list_of_sub_jsons', list_of_sub_jsons)

            #list_of_fifs = list_of_fifs[0:1] #DELETE THIS LINE WHEN DONE TESTING

            counter = 0

            for fif_ind, data_file in enumerate(list_of_fifs): 
                print('___MEGqc___: ', 'Take fif: ', data_file)

                if 'acq-crosstalk' in data_file:
                    print('___MEGqc___: ', 'Skipping crosstalk file ', data_file)
                    #read about crosstalk files here: https://bids-specification.readthedocs.io/en/stable/appendices/meg-file-formats.html
                    continue

                # Preassign strings with notes for the user to add to html report (in case some QC analysis was skipped):
                shielding_str, m_or_g_skipped_str, epoching_str, ecg_str, eog_str, head_str, muscle_str, pp_manual_str, pp_auto_str, std_str, psd_str = '', '', '', '', '', '', '', '', '', '', ''
    
                print('___MEGqc___: ', 'Starting initial processing...')
                start_time = time.time()

                try:
                    dict_epochs_mg, chs_by_lobe, channels, raw_cropped_filtered, raw_cropped_filtered_resampled, raw_cropped, raw, shielding_str, epoching_str, sensors_derivs, time_series_derivs, time_series_str, m_or_g_chosen, m_or_g_skipped_str, lobes_color_coding_str, clicking_str, resample_str = initial_processing(default_settings=all_qc_params['default'], filtering_settings=all_qc_params['Filtering'], epoching_params=all_qc_params['Epoching'], data_file=data_file)
                except:
                    print('___MEGqc___: ', 'Could not process file ', data_file, '. Skipping it.')
                    #in case some file can not be processed, the pipeline will continue. To figure out the issue, run the file separately: raw=mne.io.read_raw_fif('.../filepath/...fif')
                    continue
                
                print('___MEGqc___: ', "Finished initial processing. --- Execution %s seconds ---" % (time.time() - start_time))

                # QC measurements:

                #predefine in case some metrics are not calculated:
                noisy_freqs_global = None #if we run PSD, this will be properly defined. It is used as an input for Muscle and is supposed to represent powerline noise.
                std_derivs, psd_derivs, pp_manual_derivs, pp_auto_derivs, ecg_derivs, eog_derivs, head_derivs, muscle_derivs = [],[],[],[],[], [],  [], []
                simple_metrics_psd, simple_metrics_std, simple_metrics_pp_manual, simple_metrics_pp_auto, simple_metrics_ecg, simple_metrics_eog, simple_metrics_head, simple_metrics_muscle = [],[],[],[],[],[], [], []
                df_head_pos, head_pos = [], []
                scores_muscle_all1, scores_muscle_all2, scores_muscle_all3 = [], [], []
                raw1, raw2, raw3 = [], [], []

                if all_qc_params['default']['run_STD'] is True:
                    print('___MEGqc___: ', 'Starting STD...')
                    start_time = time.time()
                    std_derivs, simple_metrics_std, std_str = STD_meg_qc(all_qc_params['STD'], channels, chs_by_lobe, dict_epochs_mg, raw_cropped_filtered_resampled, m_or_g_chosen)
                    print('___MEGqc___: ', "Finished STD. --- Execution %s seconds ---" % (time.time() - start_time))
    
                if all_qc_params['default']['run_PSD'] is True:
                    print('___MEGqc___: ', 'Starting PSD...')
                    start_time = time.time()
                    psd_derivs, simple_metrics_psd, psd_str, noisy_freqs_global = PSD_meg_qc(all_qc_params['PSD'], channels, chs_by_lobe , raw_cropped_filtered, m_or_g_chosen, helper_plots=False)
                    print('___MEGqc___: ', "Finished PSD. --- Execution %s seconds ---" % (time.time() - start_time))

                if all_qc_params['default']['run_PTP_manual'] is True:
                    print('___MEGqc___: ', 'Starting Peak-to-Peak manual...')
                    start_time = time.time()
                    pp_manual_derivs, simple_metrics_pp_manual, pp_manual_str = PP_manual_meg_qc(all_qc_params['PTP_manual'], channels, chs_by_lobe, dict_epochs_mg, raw_cropped_filtered_resampled, m_or_g_chosen)
                    print('___MEGqc___: ', "Finished Peak-to-Peak manual. --- Execution %s seconds ---" % (time.time() - start_time))

                if all_qc_params['default']['run_PTP_auto_mne'] is True:
                    print('___MEGqc___: ', 'Starting Peak-to-Peak auto...')
                    start_time = time.time()
                    pp_auto_derivs, bad_channels, pp_auto_str = PP_auto_meg_qc(all_qc_params['PTP_auto'], channels, raw_cropped_filtered_resampled, m_or_g_chosen)
                    print('___MEGqc___: ', "Finished Peak-to-Peak auto. --- Execution %s seconds ---" % (time.time() - start_time))

                if all_qc_params['default']['run_ECG'] is True:
                    print('___MEGqc___: ', 'Starting ECG...')
                    start_time = time.time()
                    ecg_derivs, simple_metrics_ecg, ecg_str, avg_objects_ecg = ECG_meg_qc(all_qc_params['ECG'], internal_qc_params['ECG'], raw_cropped, channels, chs_by_lobe, m_or_g_chosen)
                    print('___MEGqc___: ', "Finished ECG. --- Execution %s seconds ---" % (time.time() - start_time))

                    avg_ecg += avg_objects_ecg

                if all_qc_params['default']['run_EOG'] is True:
                    print('___MEGqc___: ', 'Starting EOG...')
                    start_time = time.time()
                    eog_derivs, simple_metrics_eog, eog_str, avg_objects_eog = EOG_meg_qc(all_qc_params['EOG'], internal_qc_params['EOG'], raw_cropped, channels, chs_by_lobe, m_or_g_chosen)
                    print('___MEGqc___: ', "Finished EOG. --- Execution %s seconds ---" % (time.time() - start_time))

                    avg_eog += avg_objects_eog

                if all_qc_params['default']['run_Head'] is True:
                    print('___MEGqc___: ', 'Starting Head movement calculation...')
                    head_derivs, simple_metrics_head, head_str, df_head_pos, head_pos = HEAD_movement_meg_qc(raw_cropped, plot_annotations=False)
                    print('___MEGqc___: ', "Finished Head movement calculation. --- Execution %s seconds ---" % (time.time() - start_time))

                if all_qc_params['default']['run_Muscle'] is True:
                    print('___MEGqc___: ', 'Starting Muscle artifacts calculation...')
                    muscle_derivs, simple_metrics_muscle, muscle_str, scores_muscle_all3, raw3 = MUSCLE_meg_qc(all_qc_params['Muscle'], all_qc_params['PSD'], raw_cropped_filtered, noisy_freqs_global, m_or_g_chosen, attach_dummy = True, cut_dummy = True)
                    print('___MEGqc___: ', "Finished Muscle artifacts calculation. --- Execution %s seconds ---" % (time.time() - start_time))

                
                report_strings = {
                'INITIAL_INFO': m_or_g_skipped_str+resample_str+epoching_str+shielding_str+lobes_color_coding_str+clicking_str,
                'TIME_SERIES': time_series_str,
                'STD': std_str,
                'PSD': psd_str,
                'PTP_MANUAL': pp_manual_str,
                'PTP_AUTO': pp_auto_str,
                'ECG': ecg_str,
                'EOG': eog_str,
                'HEAD': head_str,
                'MUSCLE': muscle_str}

                # Save report strings as json to read it back in when plotting:
                report_str_derivs=[QC_derivative(report_strings, 'ReportStrings', 'json')]
                

                QC_derivs={
                'MEG data quality analysis report': [],
                'Report_strings': report_str_derivs,
                'Interactive time series': time_series_derivs,
                'Sensors locations': sensors_derivs,
                'Standard deviation of the data': std_derivs, 
                'Frequency spectrum': psd_derivs, 
                'Peak-to-Peak manual': pp_manual_derivs, 
                'Peak-to-Peak auto from MNE': pp_auto_derivs, 
                'ECG': ecg_derivs, 
                'EOG': eog_derivs,
                'Head movement artifacts': head_derivs,
                'High frequency (Muscle) artifacts': muscle_derivs}

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
                

                #Make report and add to QC_derivs (old report version):
                # report_html_string = make_joined_report(QC_derivs, shielding_str, m_or_g_skipped_str, epoching_skipped_str, no_ecg_str, no_eog_str, no_head_pos_str, muscle_str)
                # QC_derivs['Report']= [QC_derivative(report_html_string, 'Report', 'report')]

                report_html_string = make_joined_report_mne(raw, QC_derivs, report_strings, default_settings=all_qc_params['default'])
                QC_derivs['Report_MNE']= [QC_derivative(report_html_string, 'REPORT', 'report mne')]

                #Collect all simple metrics into a dictionary and add to QC_derivs:
                QC_derivs['Simple_metrics']=[QC_derivative(QC_simple, 'SimpleMetrics', 'json')]

                #if there are any derivs calculated in this section:
                for section in (section for section in QC_derivs.values() if section):
                    # loop over section where deriv.content_type is not 'matplotlib' or 'plotly' or 'report'
                    for deriv in (deriv for deriv in section if deriv.content_type != 'matplotlib' and deriv.content_type != 'plotly' and deriv.content_type != 'report'):
                        
                        # print('___MEGqc___: ', 'writing deriv: ', d)
                        # print('___MEGqc___: ', deriv)

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

                        counter +=1
                        print('___MEGqc___: ', 'counter of subject_folder.create_artifact', counter)

                        meg_artifact.add_entity('desc', deriv.name) #file name
                        meg_artifact.suffix = 'meg'
                        meg_artifact.extension = '.html'

                        if deriv.content_type == 'df':
                            meg_artifact.extension = '.tsv'
                            meg_artifact.content = lambda file_path, cont=deriv.content: cont.to_csv(file_path, sep='\t')

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
                            print('___MEGqc___: ', meg_artifact.name)
                            meg_artifact.content = 'dummy text'
                            meg_artifact.extension = '.txt'
                        # problem with lambda explained:
                        # https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result


        ancpbids.write_derivative(dataset, derivative) 

        if raw is None:
            print('___MEGqc___: ', 'No data files could be processed.')
            #return [None]*14
            return

    #for now will return raw, etc for the very last data set and fif file. In final version shoud not return anything
    # return raw, raw_cropped_filtered_resampled, QC_derivs, QC_simple, df_head_pos, head_pos, scores_muscle_all1, scores_muscle_all2, scores_muscle_all3, raw1, raw2, raw3, avg_ecg, avg_eog
    

    # for_report = {'ch_chosen': m_or_g_chosen}
    # # save as json:
    # with open('for_report.json', 'w') as file_wrapper:
    #     json.dump(for_report, file_wrapper, indent=4)

    return 

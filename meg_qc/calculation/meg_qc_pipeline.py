import os
import ancpbids
from ancpbids.query import query_entities
import time
import json
import sys
import mne
import shutil
from typing import List, Union

# Needed to import the modules without specifying the full path, for command line and jupyter notebook
sys.path.append(os.path.join('.'))
sys.path.append(os.path.join('.', 'meg_qc', 'calculation'))

# relative path for `make html` (docs)
sys.path.append(os.path.join('..', 'meg_qc', 'calculation'))

# relative path for `make html` (docs) run from https://readthedocs.org/
# every time rst file is nested inside of another, need to add one more path level here:
sys.path.append(os.path.join('..', '..', 'meg_qc', 'calculation'))
sys.path.append(os.path.join('..', '..', '..', 'meg_qc', 'calculation'))
sys.path.append(os.path.join('..', '..', '..', '..', 'meg_qc', 'calculation'))


from meg_qc.calculation.initial_meg_qc import get_all_config_params, initial_processing, get_internal_config_params
# from meg_qc.plotting.universal_html_report import make_joined_report, make_joined_report_mne
from meg_qc.plotting.universal_plots import QC_derivative

from meg_qc.calculation.metrics.STD_meg_qc import STD_meg_qc
from meg_qc.calculation.metrics.PSD_meg_qc import PSD_meg_qc
from meg_qc.calculation.metrics.Peaks_manual_meg_qc import PP_manual_meg_qc
from meg_qc.calculation.metrics.Peaks_auto_meg_qc import PP_auto_meg_qc
from meg_qc.calculation.metrics.ECG_EOG_meg_qc import ECG_meg_qc, EOG_meg_qc
from meg_qc.calculation.metrics.Head_meg_qc import HEAD_movement_meg_qc
from meg_qc.calculation.metrics.muscle_meg_qc import MUSCLE_meg_qc

def ctf_workaround(dataset, sid):

    artifacts = dataset.query(suffix="meg", return_type="object", subj=sid, scope='raw')
    # convert to folders of found files
    folders = map(lambda a: a.get_parent().get_absolute_path(), artifacts)
    # remove duplicates
    folders = set(folders)
    # convert to liust before filtering
    folders = list(folders)

    # filter for folders which end with ".ds" (including os specific path separator)
    # folders = list(filter(lambda f: f.endswith(f"{os.sep}.ds"), folders))

    # Filter for folders which end with ".ds"
    filtered_folders = [f for f in folders if f.endswith('.ds')]

    return sorted(filtered_folders)


def get_files_list(sid: str, dataset_path: str, dataset):

    """
    Different ways for fif, ctf, etc...
    Using ancpbids to get the list of files for each subject in ds.

    Parameters
    ----------
    sid : str
        Subject ID to get the files for.
    dataset_path : str
        Path to the BIDS-conform data set to run the QC on.
    dataset : ancpbids.Dataset
        Dataset object to work with.
    

    Returns
    -------
    list_of_files : list
        List of paths to the .fif files for each subject.
    entities_per_file : list
        List of entities for each file in list_of_files.
    """

    has_fif = False
    has_ctf = False


    for root, dirs, files in os.walk(dataset_path):

        # Exclude the 'derivatives' folder. 
        # Because we will later save ds info as derivative with extension .fif
        # so if we work on this ds again it might see a ctf ds as fif.
        dirs[:] = [d for d in dirs if d != 'derivatives']

        # Check for .fif files
        if any(file.endswith('.fif') for file in files):
            has_fif = True
        
        # Check for folders ending with .ds
        if any(dir.endswith('.ds') for dir in dirs):
            has_ctf = True

        # If both are found, no need to continue walking
        if has_fif and has_ctf:
            raise ValueError('Both fif and ctf files found in the dataset. Can not define how to read the ds.')


    if has_fif:
        list_of_files = sorted(list(dataset.query(suffix='meg', extension='.fif', return_type='filename', subj=sid, scope='raw')))
        
        entities_per_file = dataset.query(subj=sid, suffix='meg', extension='.fif', scope='raw')
        # sort list_of_sub_jsons by name key to get same order as list_of_files
        entities_per_file = sorted(entities_per_file, key=lambda k: k['name'])

    elif has_ctf:
        list_of_files = ctf_workaround(dataset, sid)
        entities_per_file = dataset.query(subj=sid, suffix='meg', extension='.res4', scope='raw')

        # entities_per_file is a list of Artifact objects of ancpbids created from raw files. (fif for fif files and res4 for ctf files)
        # TODO: this assumes every .ds directory has a single corresponding .res4 file. 
        # Is it always so?
        # Used because I cant get entities_per_file from .ds folders, ancpbids doesnt support folder query.
        # But we need entities_per_file to pass into subject_folder.create_artifact(), 
        # so that it can add automatically all the entities to the new derivative on base of entities from raw file.
    
        
        # sort list_of_sub_jsons by name key to get same order as list_of_files
        entities_per_file = sorted(entities_per_file, key=lambda k: k['name'])

    else:
        list_of_files = []
        raise ValueError('No fif or ctf files found in the dataset.')
    
    # Find if we have crosstalk in list of files and entities_per_file, give notification that they will be skipped:
    #read about crosstalk files here: https://bids-specification.readthedocs.io/en/stable/appendices/meg-file-formats.html
    crosstalk_files = [f for f in list_of_files if 'crosstalk' in f]
    if crosstalk_files:
        print('___MEGqc___: ', 'Crosstalk files found in the list of files. They will be skipped.')

    list_of_files = [f for f in list_of_files if 'crosstalk' not in f]
    entities_per_file = [e for e in entities_per_file if 'crosstalk' not in e['name']]

    # Check if the names in list_of_files and entities_per_file are the same:
    for i in range(len(list_of_files)):
        file_name_in_path = os.path.basename(list_of_files[i]).split('_meg.')[0]
        file_name_in_obj = entities_per_file[i]['name'].split('_meg.')[0]

        if file_name_in_obj not in file_name_in_path:
            raise ValueError('Different names in list_of_files and entities_per_file')

    # we can also check that final file of path in list of files is same as name in jsons

    return list_of_files, entities_per_file
    

def create_config_artifact(derivative, config_file_path: str, f_name_to_save: str, all_taken_raw_files: List[str]):

    """
    Save the config file used for this run as a derivative.

    Note: it is important the config and json to it have the exact same name except the extention!
    The code relies on it later on in add_raw_to_config_json() function.


    Parameters
    ----------
    derivative : ancpbids.Derivative
        Derivative object to save the config file.
    config_file_path : str
        Path to the config file used for this ds conversion
    f_name_to_save : str
        Name of the config file to save.
    all_taken_raw_files : list
        List of all the raw files processed in this run, for this ds.
    
    """

    #get current time stamp for config file

    timestamp = time.strftime("Date%Y%m%dTime%H%M%S")

    f_name_to_save = f_name_to_save + str(timestamp)

    config_folder = derivative.create_folder(name='config')
    config_artifact = config_folder.create_artifact()

    config_artifact.content = lambda file_path, cont = config_file_path: shutil.copy(cont, file_path)
    config_artifact.add_entity('desc', f_name_to_save) #file name
    config_artifact.suffix = 'meg'
    config_artifact.extension = '.ini'

    #Create a seconf json file with config name as key and all taken raw files as value
    # and prepare it to be save as derivative

    config_json = {f_name_to_save: all_taken_raw_files}

    config_json_artifact = config_folder.create_artifact()
    config_json_artifact.content = lambda file_path, cont = config_json: json.dump(cont, open(file_path, 'w'), indent=4)
    config_json_artifact.add_entity('desc', f_name_to_save) #file name
    config_json_artifact.suffix = 'meg'
    config_json_artifact.extension = '.json'

    return


def add_raw_to_config_json(derivative, reuse_config_file_path: str, all_taken_raw_files: List[str]):

    """
    Add the list of all taken raw files to the existing list of used settings in the config file.

    Expects that the config file .ini and the .json file (with the same name) are already saved as derivatives.

    To get corresponding json here use the easy way: 
    just exchange ini to json in reuse file path (not using ANCPbids for it).
    The 'proper' way would be to:
    - query the desc entitiy of the reused config file
    - get the json file with the same desc entity
    This way will still assume that desc are exactly the same, so we use the easy way without ANCPbids d-tour.

    Parameters
    ----------
    derivative : ancpbids.Derivative
        Derivative object to save the config file.
    reuse_config_file_path : str
        Path to the config file used for this ds conversion before.
    all_taken_raw_files : list
        List of all the raw files processed in this run, for this ds.
    
    """

    #exchange ini to json:
    json_for_reused_config = reuse_config_file_path.replace('.ini', '.json')

    #check if the json file exists:
    if not os.path.isfile(json_for_reused_config):
        print('___MEGqc___: ', 'No json file found for the config file used before. Can not add the new raw files to it.')
        return

    print('___MEGqc___: ', 'json_for_reused_config', json_for_reused_config)

    try:
        with open(json_for_reused_config, 'r') as file:
            config_json = json.load(file)
    except json.JSONDecodeError as e:
        with open(json_for_reused_config, 'r') as file:
            content = file.read()
        print(f"Error decoding JSON: {e}")
        print(f"File content:\n{content}")
        # Handle the error appropriately, e.g., set config_json to an empty dict or raise an error
        config_json = {}
        return

    # from file name get desc entity to use it as a key in the json file: 
    # after desc- and before the underscores:
    file_name = os.path.basename(reuse_config_file_path).split('.')[0]
    config_desc = file_name.split('desc-')[1].split('_')[0]

    # get what files already were in the config file
    list_of_files = config_json[config_desc]

    #update the list with new files:
    list_of_files += all_taken_raw_files

    #sort and remove duplicates:
    list_of_files = sorted(list(set(list_of_files)))

    #overwrite the old json (premake ancp bids artifact):
    config_json = {config_desc: list_of_files}


    config_folder = derivative.create_folder(name='config')
    #TODO: we dont need to create config folder again, already got it, how to get it?

    config_json_artifact = config_folder.create_artifact()
    config_json_artifact.content = lambda file_path, cont = config_json: json.dump(cont, open(file_path, 'w'), indent=4)
    config_json_artifact.add_entity('desc', config_desc) #file name
    config_json_artifact.suffix = 'meg'
    config_json_artifact.extension = '.json'

    return


def check_ds_paths(ds_paths: Union[List[str], str]):

    """
    Check if the given paths to the data sets exist.
    
    Parameters
    ----------
    ds_paths : list or str
        List of paths to the BIDS-conform data sets to run the QC on.
        
    Returns
    -------
    ds_paths : list
        List of paths to the BIDS-conform data sets to run the QC on.
    """

    #has to be a list, even if there is just one path:
    if isinstance(ds_paths, str):
        ds_paths = [ds_paths]
    
    #make sure all directories in the list exist:
    for ds_path in ds_paths:
        if not os.path.isdir(ds_path):
            raise ValueError(f'Given path to the dataset does not exist. Path: {ds_path}')
        
    return ds_paths

def check_config_saved_ask_user(dataset):

    """
    Check if there is already config file used for this ds:
    If yes - ask the user if he wants to use it again. If not - use default one.
    When no config found or user doesnt want to reuse - will return None.
    otherwise will return the path to one config file used for this ds before to reuse now.

    Parameters
    ----------
    dataset : ancpbids.Dataset
        Dataset object to work with.

    Returns
    -------
    config_file_path : str
        Path to the config file used for this ds conversion.
    """

    # if os.path.isfile(os.path.join(derivatives_path, 'config', 'UsedSettings.ini')):
    #     print('___MEGqc___: ', 'There is already a config file used for this data set. Do you want to use it again?')
    #     #ask user if he wants to use the same config file again

    entities = query_entities(dataset, scope='derivatives')

    #print('___MEGqc___: ', 'entities', entities)

    # search if there is already a derivative with 'UsedSettings' in the name
    # if yes - ask the user if he wants to use it again. If not - use default one.
    used_settings_entity_list = []
    for key, entity_set in entities.items():
        if key == 'description':
            for ent in entity_set:
                if 'usedsettings' in ent.lower():
                    used_settings_entity_list.append(ent)

    used_setting_file_list = []
    for used_settings_entity in used_settings_entity_list:
        
        used_setting_file_list += sorted(list(dataset.query(suffix='meg', extension='.ini', desc = used_settings_entity, return_type='filename', scope='derivatives')))

    reuse_config_file_path = None

    # Ask the user if he wants to use any of existing config files:
    if used_setting_file_list:
        print('___MEGqc___: ', 'There are already config files used for this data set. Do you want to use any of them again?')
        print('___MEGqc___: ', 'List of the config files previously used for this data set:')
        for i, file in enumerate(used_setting_file_list):
            print('___MEGqc___: ', i, file)

        user_input = input('___MEGqc___: Enter the number of the config file you want to use, or press Enter to use the default one: ')
        if user_input:
            reuse_config_file_path = used_setting_file_list[int(user_input)]
        else:
            print('___MEGqc___: ', 'You chose to use the default config file.')
            

    return reuse_config_file_path

def check_sub_list(sub_list: Union[List[str], str], dataset):

    """
    Check if the given subjects are in the data set.
    
    Parameters
    ----------
    sub_list : list or str
        List of subjects to run the QC on.
    dataset : ancpbids.Dataset
        Dataset object to work with.
        
    Returns
    -------
    sub_list : list
        Updated list of subjects to run the QC on.
        
    """

    available_subs = sorted(list(dataset.query_entities(scope='raw')['subject']))
    if sub_list == 'all':
        sub_list = available_subs
    elif isinstance(sub_list, str) and sub_list != 'all':
        sub_list = [sub_list]
        #check if this sub is available:
        if sub_list[0] not in available_subs:
            print('___MEGqc___: ', 'The subject you want to run the QC on is not in your data set. Check the subject ID.')
            return
    elif isinstance(sub_list, list):
        #if they are given as str - IDs:
        if all(isinstance(sub, str) for sub in sub_list):
            sub_list_missing = [sub for sub in sub_list if sub not in available_subs]
            sub_list = [sub for sub in sub_list if sub in available_subs]
            if sub_list_missing:
                print('___MEGqc___: ', 'Could NOT find these subs in your data set. Check the subject IDs:', sub_list_missing)
                print('___MEGqc___: ', 'Requested subjects found in your data set:', sub_list, 'Only these subjects will be processed.')
            
        #if they are given as int - indexes:
        elif all(isinstance(sub, int) for sub in sub_list):
            sub_list = [available_subs[i] for i in sub_list]


    print('___MEGqc___: ', 'sub_list to process: ', sub_list)


    return sub_list

def make_derivative_meg_qc(default_config_file_path: str, internal_config_file_path: str, ds_paths: Union[List[str], str], sub_list: Union[List[str], str] = 'all'):

    """ 
    Main function of MEG QC:
    
    * Parse parameters from config: user config + internal config
    * Get the data .fif file for each subject using ancpbids
    * Run initial processing (filtering, epoching, resampling)
    * Run whole QC analysis for every subject, every fif (only chosen metrics from config)
    * Save derivatives (csvs, html reports) into the file system using ancpbids.
    
    Parameters
    ----------
    default_config_file_path : str
        Path the config file with all the parameters for the QC analysis - default.
        later the function will ask the user if he wants to use the same config file again or use another one.
    internal_config_file_path : str
        Path the config file with all the parameters for the QC analysis preset - not to be changed by the user.
    ds_paths : list or str
        List of paths to the BIDS-conform data sets to run the QC on. Has to be list even if there is just one path.
    sub_list : list or str
        List of subjects to run the QC on. Can be 'all' or 1 subj like '009' or list of several subjects like ['009', '012'].
    """

    ds_paths = check_ds_paths(ds_paths)

    internal_qc_params = get_internal_config_params(internal_config_file_path) 
    # assume these are not user changable by user, so apply without asking to all data sets.

    for dataset_path in ds_paths: #run over several data sets

        print('___MEGqc___: ', 'DS path:', dataset_path)

        dataset = ancpbids.load_dataset(dataset_path)
        schema = dataset.get_schema()


        #create derivatives folder first:
        derivatives_path = os.path.join(dataset_path, 'derivatives')
        if not os.path.isdir(derivatives_path):
            os.mkdir(derivatives_path)

        derivative = dataset.create_derivative(name="Meg_QC")
        derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"

        # Check if there is already config file used for this ds:
        reuse_config_file_path = check_config_saved_ask_user(dataset) # will give None if no config file was used before
        if reuse_config_file_path:
            config_file_path = reuse_config_file_path
        else:
            config_file_path = default_config_file_path
        print('___MEGqc___: ', 'Using config file: ', config_file_path)

        # Get all the parameters from the config file:
        all_qc_params = get_all_config_params(config_file_path)

        if all_qc_params is None:
            return

        #entities = dataset.query_entities(dataset_path)
        #entities = query_entities(dataset, scope='raw')

        # print('_____BIDS data info___')
        # print(schema)
        # print(dataset)
        # print(type(dataset.derivatives))

        # print('___MEGqc___: ', schema)
        # print('___MEGqc___: ', schema.Artifact)

        # print('___MEGqc___: ', dataset.files)
        # print('___MEGqc___: ', dataset.folders)
        # print('___MEGqc___: ', dataset.derivatives)
        # print('___MEGqc___: ', dataset.items())
        # print('___MEGqc___: ', dataset.keys())
        # print('___MEGqc___: ', dataset.code)
        # print('___MEGqc___: ', dataset.name)

        # print('______')

        # entities = query_entities(dataset)
        # print('___MEGqc___: ', 'entities', entities)


        sub_list = check_sub_list(sub_list, dataset)

        avg_ecg=[]
        avg_eog=[]
        
        raw=None #preassign in case no calculation will be successful

        all_taken_raw_files = []

        for sub in sub_list: #[0:4]:
    
            print('___MEGqc___: ', 'Take SUB: ', sub)
            
            calculation_folder = derivative.create_folder(name='calculation')
            subject_folder = calculation_folder.create_folder(type_=schema.Subject, name='sub-'+sub)

            list_of_files, entities_per_file = get_files_list(sub, dataset_path, dataset)

            if not list_of_files:
                print('___MEGqc___: ', 'No files to work on. Check that given subjects are present in your data set.')
                return

            print('___MEGqc___: ', 'list_of_files to process:', list_of_files)
            print('___MEGqc___: ', 'entities_per_file to process', entities_per_file)
            print('___MEGqc___: ', 'TOTAL files to process: ', len(list_of_files))

            all_taken_raw_files += [os.path.basename(f) for f in list_of_files]

            # GET all derivs!
            # derivs_list = sorted(list(dataset.query(suffix='meg', extension='.tsv', return_type='filename', subj=sid, scope='derivatives')))
            # print('___MEGqc___: ', 'derivs_list', derivs_list)

            # entities = dataset.query_entities()
            # print('___MEGqc___: ', 'entities', entities)


            #TODO; check here that order is really the same as in list_of_fifs
            #same as list_of_fifs, but return type is not filename, but dict


            counter = 0

            for file_ind, data_file in enumerate(list_of_files): #[0:1]: #run over several data files

                print('___MEGqc___: ', 'Processing file: ', data_file)

                # Preassign strings with notes for the user to add to html report (in case some QC analysis was skipped):
                shielding_str, m_or_g_skipped_str, epoching_str, ecg_str, eog_str, head_str, muscle_str, pp_manual_str, pp_auto_str, std_str, psd_str = '', '', '', '', '', '', '', '', '', '', ''
    
                print('___MEGqc___: ', 'Starting initial processing...')
                start_time = time.time()

                meg_system, dict_epochs_mg, chs_by_lobe, channels, raw_cropped_filtered, raw_cropped_filtered_resampled, raw_cropped, raw, info_derivs, stim_deriv, shielding_str, epoching_str, sensors_derivs, m_or_g_chosen, m_or_g_skipped_str, lobes_color_coding_str, resample_str = initial_processing(default_settings=all_qc_params['default'], filtering_settings=all_qc_params['Filtering'], epoching_params=all_qc_params['Epoching'], file_path=data_file)
                
                # Commented out this, because it would cover the actual error while allowing to continue processing.
                # I wanna see the actual error. Often it happens while reading raw and says: 
                # file '...' does not start with a file id tag
                
                # try:
                #     dict_epochs_mg, chs_by_lobe, channels, raw_cropped_filtered, raw_cropped_filtered_resampled, raw_cropped, raw, shielding_str, epoching_str, sensors_derivs, m_or_g_chosen, m_or_g_skipped_str, lobes_color_coding_str, resample_str = initial_processing(default_settings=all_qc_params['default'], filtering_settings=all_qc_params['Filtering'], epoching_params=all_qc_params['Epoching'], file_path=data_file)
                # except:
                #     print('___MEGqc___: ', 'Could not process file ', data_file, '. Skipping it.')
                #     #in case some file can not be processed, the pipeline will continue. To figure out the issue, run the file separately: raw=mne.io.read_raw_fif('.../filepath/...fif')
                #     continue
                
                print('___MEGqc___: ', "Finished initial processing. --- Execution %s seconds ---" % (time.time() - start_time))

                # QC measurements:

                #predefine in case some metrics are not calculated:
                noisy_freqs_global = None #if we run PSD, this will be properly defined. It is used as an input for Muscle and is supposed to represent powerline noise.
                std_derivs, psd_derivs, pp_manual_derivs, pp_auto_derivs, ecg_derivs, eog_derivs, head_derivs, muscle_derivs = [],[],[],[],[], [],  [], []
                simple_metrics_psd, simple_metrics_std, simple_metrics_pp_manual, simple_metrics_pp_auto, simple_metrics_ecg, simple_metrics_eog, simple_metrics_head, simple_metrics_muscle = [],[],[],[],[],[], [], []


                if all_qc_params['default']['run_STD'] is True:
                    print('___MEGqc___: ', 'Starting STD...')
                    start_time = time.time()
                    std_derivs, simple_metrics_std, std_str = STD_meg_qc(all_qc_params['STD'], channels, chs_by_lobe, dict_epochs_mg, raw_cropped_filtered_resampled, m_or_g_chosen)
                    print('___MEGqc___: ', "Finished STD. --- Execution %s seconds ---" % (time.time() - start_time))
    
                if all_qc_params['default']['run_PSD'] is True:
                    print('___MEGqc___: ', 'Starting PSD...')
                    start_time = time.time()
                    psd_derivs, simple_metrics_psd, psd_str, noisy_freqs_global = PSD_meg_qc(all_qc_params['PSD'], internal_qc_params['PSD'], channels, chs_by_lobe , raw_cropped_filtered, m_or_g_chosen, helper_plots=False)
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
                    head_derivs, simple_metrics_head, head_str, df_head_pos, head_pos = HEAD_movement_meg_qc(raw_cropped)
                    print('___MEGqc___: ', "Finished Head movement calculation. --- Execution %s seconds ---" % (time.time() - start_time))

                if all_qc_params['default']['run_Muscle'] is True:
                    print('___MEGqc___: ', 'Starting Muscle artifacts calculation...')
                    muscle_derivs, simple_metrics_muscle, muscle_str, scores_muscle_all3, raw3 = MUSCLE_meg_qc(all_qc_params['Muscle'], all_qc_params['PSD'], internal_qc_params['PSD'], channels, raw_cropped_filtered, noisy_freqs_global, m_or_g_chosen, attach_dummy = True, cut_dummy = True)
                    print('___MEGqc___: ', "Finished Muscle artifacts calculation. --- Execution %s seconds ---" % (time.time() - start_time))

                
                report_strings = {
                'INITIAL_INFO': m_or_g_skipped_str+resample_str+epoching_str+shielding_str+lobes_color_coding_str,
                'STD': std_str,
                'PSD': psd_str,
                'PTP_MANUAL': pp_manual_str,
                'PTP_AUTO': pp_auto_str,
                'ECG': ecg_str,
                'EOG': eog_str,
                'HEAD': head_str,
                'MUSCLE': muscle_str,
                'STIMULUS': 'If the data was cropped for this calculation, the stimulus data is also cropped.'}

                # Save report strings as json to read it back in when plotting:
                report_str_derivs=[QC_derivative(report_strings, 'ReportStrings', 'json')]
                

                QC_derivs={
                'Raw info': info_derivs,
                'Stimulus channels': stim_deriv,
                'Report_strings': report_str_derivs,
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

                #Collect all simple metrics into a dictionary and add to QC_derivs:
                QC_derivs['Simple_metrics']=[QC_derivative(QC_simple, 'SimpleMetrics', 'json')]

                #if there are any derivs calculated in this section:
                for section in (section for section in QC_derivs.values() if section):
                    # loop over section where deriv.content_type is not 'matplotlib' or 'plotly' or 'report'
                    for deriv in (deriv for deriv in section if deriv.content_type != 'matplotlib' and deriv.content_type != 'plotly' and deriv.content_type != 'report'):

                        # This is how you would save matplotlib, plotly and reports separately with ancpbids:

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

                        meg_artifact = subject_folder.create_artifact(raw=entities_per_file[file_ind]) #shell. empty derivative

                        counter +=1
                        print('___MEGqc___: ', 'counter of subject_folder.create_artifact', counter)

                        meg_artifact.add_entity('desc', deriv.name) #file name
                        meg_artifact.suffix = 'meg'
                        meg_artifact.extension = '.html'

                        if deriv.content_type == 'df':
                            meg_artifact.extension = '.tsv'
                            meg_artifact.content = lambda file_path, cont=deriv.content: cont.to_csv(file_path, sep='\t')

                        elif deriv.content_type == 'json':
                            meg_artifact.extension = '.json'
                            def json_writer(file_path, cont=deriv.content):
                                with open(file_path, "w") as file_wrapper:
                                    json.dump(cont, file_wrapper, indent=4)
                            meg_artifact.content = json_writer # function pointer instead of lambda

                        elif deriv.content_type == 'info':
                            meg_artifact.extension = '.fif'
                            meg_artifact.content = lambda file_path, cont=deriv.content: mne.io.write_info(file_path, cont)

                        else:
                            print('___MEGqc___: ', meg_artifact.name)
                            meg_artifact.content = 'dummy text'
                            meg_artifact.extension = '.txt'
                        # problem with lambda explained:
                        # https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result


        #Save config file used for this run as a derivative:
        if reuse_config_file_path is None:
            # if no config file was used before, save the one used now
            create_config_artifact(derivative, config_file_path, 'UsedSettings', all_taken_raw_files)
        else:
            #otherwise - dont save config again, but add list of all taken raw files to the existing list of used settings:
            add_raw_to_config_json(derivative, reuse_config_file_path, all_taken_raw_files)


        ancpbids.write_derivative(dataset, derivative) 

        if raw is None:
            print('___MEGqc___: ', 'No data files could be processed.')
            return

    return 

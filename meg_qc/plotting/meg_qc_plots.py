import sys
import os
import ancpbids
import json
from prompt_toolkit.shortcuts import checkboxlist_dialog
from prompt_toolkit.styles import Style
from collections import defaultdict
import re

# Get the absolute path of the parent directory of the current script
parent_dir = os.path.dirname(os.getcwd())
gradparent_dir = os.path.dirname(parent_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)
sys.path.append(gradparent_dir)

from meg_qc.plotting.universal_plots import *
from meg_qc.plotting.universal_html_report import make_joined_report_mne

# IMPORTANT: keep this order of imports, first need to add parent dir to sys.path, then import from it.

# ____________________________

# How plotting in MEGqc works:
# During calculation save in the right folders the csvs with data for plotting
# During plotting step - read the csvs (find using ancpbids), plot them, save them as htmls in the right folders.


def modify_entity_name(entities):

    """
    Modify the names of the entities to be comply with ancpbids.

    Parameters
    ----------
    entities : dict
        A dictionary of entities and their subcategories.
    
    Returns
    -------
    dict
        A dictionary of entities and their subcategories with modified names
    """

    #old_new_categories = {'description': 'METRIC', 'subject': 'SUBJECT', 'session': 'SESSION', 'task': 'TASK', 'run': 'RUN'}

    old_new_categories = {'description': 'METRIC'}

    categories_copy = entities.copy()

    for category, subcategories in categories_copy.items():
        # Convert the set of subcategories to a sorted list
        sorted_subcategories = sorted(subcategories, key=str)
        # If the category is in old_new_categories, replace it with the new category
        if category in old_new_categories: 
            #This here is to replace the old names with new like desc -> METRIC
            #Normally we d use it only for the METRIC, but left this way in case the principle will extend to other categories
            #see old_new_categories above.

            new_category = old_new_categories[category]
            entities[new_category] = entities.pop(category)
            # Replace the original set of subcategories with the modified list
            sorted_subcategories.insert(0, '_ALL_'+new_category+'S_')
            entities[new_category] = sorted_subcategories
        else: #if we dont want to rename categories
            sorted_subcategories.insert(0, '_ALL_'+category+'s_')
            entities[category] = sorted_subcategories

    #From METRIC remove whatever is not metric. 
    #Cos METRIC is originally a desc entity which can contain just anything:
            
    if 'METRIC' in entities:
        entities['METRIC'] = [x for x in entities['METRIC'] if x in ['_ALL_METRICS_', 'STDs', 'PSDs', 'PtPsManual', 'PtPsAuto', 'ECGs', 'EOGs', 'Head', 'Muscle']]

    return entities


def selector(entities):

    """
    Creates a in-terminal visual selector for the user to choose the entities and settings for plotting.

    Loop over categories (keys)
    for every key use a subfunction that will create a selector for the subcategories.

    Parameters
    ----------
    entities : dict
        A dictionary of entities and their subcategories.

    Returns
    -------
    selected_entities : dict
        A dictionary of selected entities.
    plot_settings : dict
        A dictionary of selected settings for plotting.

    """

    # SELECT ENTITIES and SETTINGS
    # Define the categories and subcategories
    categories = modify_entity_name(entities)
    categories['m_or_g'] = ['_ALL_', 'mag', 'grad']

    selected = {}
    # Create a list of values with category titles
    for key, values in categories.items():
        subcategory, quit_selector = select_subcategory(categories[key], key)

        if quit_selector: # if user clicked cancel - stop:
            print('___MEGqc___: You clicked cancel. Please start over.')
            return None, None
        
        selected[key] = subcategory


    #Check 1) if nothing was chosen, 2) if ALL was chosen
    for key, values in selected.items():

        if not selected[key]: # if nothing was chosen:
            title = 'You did not choose the '+key+'. Please try again:'
            subcategory, quit_selector = select_subcategory(categories[key], key, title)
            if not subcategory: # if nothing was chosen again - stop:
                print('___MEGqc___: You still  did not choose the '+key+'. Please start over.')
                return None, None
            
        else:
            for item in values:
                if 'ALL' in item.upper():
                    all_selected = [str(category) for category in categories[key] if 'ALL' not in str(category).upper()]
                    selected[key] = all_selected #everything

    #Separate into selected_entities and plot_settings:
        selected_entities, plot_settings = {}, {}
        for key, values in selected.items():
            if key != 'm_or_g':
                selected_entities[key] = values
            elif key == 'm_or_g':
                plot_settings[key] = values
            else:
                print('___MEGqc__: wow, weird key in selector()! check it.')

    return selected_entities, plot_settings


def select_subcategory(subcategories, category_title, title="What would you like to plot? Click to select."):

    """
    Create a checkbox list dialog for the user to select subcategories.
    Example:
    sub: 009, 012, 013

    Parameters
    ----------
    subcategories : list
        A list of subcategories, such as: sub, ses, task, run, metric, mag/grad.
    category_title : str
        The title of the category.
    title : str
        The title of the checkbox list dialog, for visual.

    Returns
    -------
    results : list
        A list of selected subcategories.
    quit_selector : bool
        A boolean indicating whether the user clicked Cancel.

    """

    quit_selector = False

    # Create a list of values with category titles
    values = []
    for items in subcategories:
        values.append((str(items), str(items)))

        # Each tuple represents a checkbox item and should contain two elements:
        # A string that will be returned when the checkbox is selected.
        # A string that will be displayed as the label of the checkbox.

    results = checkboxlist_dialog(
        title=title,
        text=category_title,
        values=values,
        style=Style.from_dict({
            'dialog': 'bg:#cdbbb3',
            'button': 'bg:#bf99a4',
            'checkbox': '#e8612c',
            'dialog.body': 'bg:#a9cfd0',
            'dialog shadow': 'bg:#c98982',
            'frame.label': '#fcaca3',
            'dialog.body label': '#fd8bb6',
        })
    ).run()

    # Set quit_selector to True if the user clicked Cancel (results is None)
    quit_selector = results is None

    return results, quit_selector


def get_ds_entities(dataset_path: str):

    """
    Get the entities of the dataset using ancpbids, only get derivatove entities, not all raw data.

    Parameters
    ----------
    dataset_path : str
        The path to the dataset.
    
    Returns
    -------
    entities : dict
        A dictionary of entities and their subcategories.

    """

    try:
        dataset = ancpbids.load_dataset(dataset_path)

    except:
        print('___MEGqc___: ', 'No data found in the given directory path! \nCheck directory path in config file and presence of data on your device.')
        return

    #create derivatives folder first:
    if os.path.isdir(dataset_path+'/derivatives')==False: 
            os.mkdir(dataset_path+'/derivatives')

    derivative = dataset.create_derivative(name="Meg_QC")
    derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"

    entities = dataset.query_entities(scope='derivatives')
    #we only get entities of calculated derivatives here, not entire raw ds.

    print('___MEGqc___: ', 'Entities found in the dataset: ', entities)
    
    return entities


def csv_to_html_report(metric: str, tsv_paths: list, report_str_path: str, plot_settings):

    """
    Create an HTML report from the CSV files.

    Parameters
    ----------
    metric : str
        The metric to be plotted.
    tsv_paths : list
        A list of paths to the CSV files.
    report_str_path : str
        The path to the JSON file containing the report strings.
    plot_settings : dict
        A dictionary of selected settings for plotting.
    
    Returns
    -------
    report_html_string : str
        The HTML report as a string.
    
    """

    m_or_g_chosen = plot_settings['m_or_g'] 

    raw = [] # TODO: if empty - we cant print raw information. 
    # Or we need to save info from it somewhere separately and export as csv/jspn and then read back in.

    time_series_derivs, sensors_derivs, ptp_manual_derivs, pp_auto_derivs, ecg_derivs, eog_derivs, std_derivs, psd_derivs, muscle_derivs, head_derivs = [], [], [], [], [], [], [], [], [], []
    #TODO: think about it! the order goes by tsv files so it s messed uo cos ecg channels comes seond!
    
    for tsv_path in tsv_paths: #if we got several tsvs for same metric, like for PSD:

        if 'STD' in metric.upper():

            fig_std_epoch0 = []
            fig_std_epoch1 = []

            std_derivs += plot_sensors_3d_csv(tsv_path)
        
            for m_or_g in m_or_g_chosen:

                fig_all_time = boxplot_all_time_csv(tsv_path, ch_type=m_or_g, what_data='stds')
                fig_std_epoch0 = boxplot_epoched_xaxis_channels_csv(tsv_path, ch_type=m_or_g, what_data='stds')
                fig_std_epoch1 = boxplot_epoched_xaxis_epochs_csv(tsv_path, ch_type=m_or_g, what_data='stds')

                std_derivs += fig_all_time + fig_std_epoch0 + fig_std_epoch1

        if 'PTP' in metric.upper():

            fig_ptp_epoch0 = []
            fig_ptp_epoch1 = []

            ptp_manual_derivs += plot_sensors_3d_csv(tsv_path)
        
            for m_or_g in m_or_g_chosen:

                fig_all_time = boxplot_all_time_csv(tsv_path, ch_type=m_or_g, what_data='peaks')
                fig_ptp_epoch0 = boxplot_epoched_xaxis_channels_csv(tsv_path, ch_type=m_or_g, what_data='peaks')
                fig_ptp_epoch1 = boxplot_epoched_xaxis_epochs_csv(tsv_path, ch_type=m_or_g, what_data='peaks')

                ptp_manual_derivs += fig_all_time + fig_ptp_epoch0 + fig_ptp_epoch1

        elif 'PSD' in metric.upper():

            method = 'welch' #is also hard coded in PSD_meg_qc() for now

            psd_derivs += plot_sensors_3d_csv(tsv_path)

            for m_or_g in m_or_g_chosen:

                psd_derivs += Plot_psd_csv(m_or_g, tsv_path, method)

                psd_derivs += plot_pie_chart_freq_csv(tsv_path, m_or_g=m_or_g, noise_or_waves = 'noise')

                psd_derivs += plot_pie_chart_freq_csv(tsv_path, m_or_g=m_or_g, noise_or_waves = 'waves')

        elif 'ECG' in metric.upper():

            ecg_derivs += plot_sensors_3d_csv(tsv_path)

            ecg_derivs += plot_ECG_EOG_channel_csv(tsv_path)

            ecg_derivs += plot_mean_rwave_csv(tsv_path, 'ECG')

            #TODO: add ch description like here? export it as separate report strings?
            #noisy_ch_derivs += [QC_derivative(fig, bad_ecg_eog[ecg_ch]+' '+ecg_ch, 'plotly', description_for_user = ecg_ch+' is '+ bad_ecg_eog[ecg_ch]+ ': 1) peaks have similar amplitude: '+str(ecg_eval[0])+', 2) tolerable number of breaks: '+str(ecg_eval[1])+', 3) tolerable number of bursts: '+str(ecg_eval[2]))]

            for m_or_g in m_or_g_chosen:
                ecg_derivs += plot_artif_per_ch_correlated_lobes_csv(tsv_path, m_or_g, 'ECG', flip_data=False)
                #ecg_derivs += plot_correlation_csv(tsv_path, 'ECG', m_or_g)

        elif 'EOG' in metric.upper():

            eog_derivs += plot_sensors_3d_csv(tsv_path)

            eog_derivs += plot_ECG_EOG_channel_csv(tsv_path)

            eog_derivs += plot_mean_rwave_csv(tsv_path, 'EOG')
                
            for m_or_g in m_or_g_chosen:
                eog_derivs += plot_artif_per_ch_correlated_lobes_csv(tsv_path, m_or_g, 'EOG', flip_data=False)
                #eog_derivs += plot_correlation_csv(tsv_path, 'EOG', m_or_g)

            
        elif 'MUSCLE' in metric.upper():

            if 'mag' in m_or_g_chosen:
                m_or_g_decided=['mag']
            elif 'grad' in m_or_g_chosen and 'mag' not in m_or_g_chosen:
                m_or_g_decided=['grad']
            else:
                print('___MEGqc___: ', 'No magnetometers or gradiometers found in data. Artifact detection skipped.')


            muscle_derivs +=  plot_muscle_csv(tsv_path, m_or_g_decided[0])

            
        elif 'HEAD' in metric.upper():
                
            head_pos_derivs, _ = make_head_pos_plot_csv(tsv_path)
            # head_pos_derivs2 = make_head_pos_plot_mne(raw, head_pos, verbose_plots=verbose_plots)
            # head_pos_derivs += head_pos_derivs2
            head_derivs += head_pos_derivs

    QC_derivs={
    'Time_series': time_series_derivs,
    'Sensors': sensors_derivs,
    'STD': std_derivs, 
    'PSD': psd_derivs, 
    'PtP_manual': ptp_manual_derivs, 
    'PtP_auto': pp_auto_derivs, 
    'ECG': ecg_derivs, 
    'EOG': eog_derivs,
    'Head': head_derivs,
    'Muscle': muscle_derivs,
    'Report_MNE': []}


    # print('____Number of figs in QC_derivs[STD]: ', len(QC_derivs['STD']))

    # # Filter out empty figures from QC_derivs:
    # for metric, fig_list in QC_derivs.items():
    #     if fig_list:
    #         QC_derivs[metric] = [
    #             fig_deriv for fig_deriv in fig_list 
    #             if fig_deriv.content.data and any(
    #                 (hasattr(trace, 'x') and hasattr(trace, 'y') and len(trace.x) > 0 and len(trace.y) > 0) or
    #                 (trace.type == 'pie' and hasattr(trace, 'labels') and hasattr(trace, 'values') and len(trace.labels) > 0 and len(trace.values) > 0)
    #                 for trace in fig_deriv.content.data
    #             )
    #         ]


    # print('____Number of figs in QC_derivs[STD] NEW: ', len(QC_derivs['STD']))

    #Sort all based on fig_order of QC_derivative:
    #(To plot them in correct order in the report)
    for metric, values in QC_derivs.items():
        if values:
            QC_derivs[metric] = sorted(values, key=lambda x: x.fig_order)


    if not report_str_path: #if no report strings were saved. happens when mags/grads didnt run to make tsvs.
        report_strings = {
        'INITIAL_INFO': '',
        'TIME_SERIES': '',
        'STD': '',
        'PSD': '',
        'PTP_MANUAL': '',
        'PTP_AUTO': '',
        'ECG': '',
        'EOG': '',
        'HEAD': '',
        'MUSCLE': ''}
    else:
        with open(report_str_path) as json_file:
            report_strings = json.load(json_file)


    report_html_string = make_joined_report_mne(raw, QC_derivs, report_strings, [])

    return report_html_string 



# Function to create a key from the object excluding the 'desc' attribute
def create_key_from_obj(obj):
    # Remove the 'desc' part from the name
    name_without_desc = re.sub(r'_desc-[^_]+', '', obj.name)
    return (name_without_desc, obj.extension, obj.suffix)

def combine_tsvs_dict(tsvs_by_metric: dict):

    """
    For every metric, if we got same raw entitites, we can combine dwerivatives for the same raw into a list.
    Since we collected entities not from raw but from derivatives, we need to remove the desc part from the name.
    After that we combine files with the same 'name' in entity_val objects in 1 list:

    Parameters
    ----------
    tsvs_by_metric : dict
        A dictionary of metrics and their corresponding TSV files.
    
    Returns
    -------
    combined_tsvs_by_metric : dict
        A dictionary of metrics and their corresponding TSV files combined by raw entity

    """

    combined_tsvs_by_metric = {}

    for metric, obj_dict in tsvs_by_metric.items():
        combined_dict = defaultdict(list)
        
        for obj, tsv_path in obj_dict.items():
            key = create_key_from_obj(obj)
            combined_dict[key].extend(tsv_path)
        
        # Convert keys back to original objects
        final_dict = {}
        for key, paths in combined_dict.items():
            # Find the first object with the same key
            for obj in obj_dict.keys():
                if create_key_from_obj(obj) == key:
                    final_dict[obj] = paths
                    break
    
        combined_tsvs_by_metric[metric] = final_dict

    return combined_tsvs_by_metric

def make_plots_meg_qc(ds_paths: list):

    """
    Create plots for the MEG QC pipeline.

    Parameters
    ----------
    ds_paths : list
        A list of paths to the datasets.
    
    Returns
    -------
    tsvs_to_plot : dict
        A dictionary of metrics and their corresponding TSV files.
    
    """

    for dataset_path in ds_paths: #run over several data sets #TODO: do we even need several??

        try:
            dataset = ancpbids.load_dataset(dataset_path)
            schema = dataset.get_schema()
        except:
            print('___MEGqc___: ', 'No data found in the given directory path! \nCheck directory path in config file and presence of data on your device.')
            return

        derivative = dataset.create_derivative(name="Meg_QC")
        derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"

        entities = get_ds_entities(dataset_path) #get entities of the dataset using ancpbids

        chosen_entities, plot_settings = selector(entities)
        if not chosen_entities:
            return

        # chosen_entities = {'subject': ['009'], 'session': ['1'], 'task': ['deduction', 'induction'], 'run': ['1'], 'METRIC': ['ECGs', 'Muscle']}
        # uncomment for debugging, so no need to start selector every time
        
        print('___MEGqc___: CHOSEN entities to plot: ', chosen_entities)
        print('___MEGqc___: CHOSEN settings: ', plot_settings)

        for sub in chosen_entities['subject']:

            subject_folder = derivative.create_folder(type_=schema.Subject, name='sub-'+sub)
            reports_folder = subject_folder.create_folder(name='reports')

            try:
                report_str_path = sorted(list(dataset.query(suffix='meg', extension='.json', return_type='filename', subj=sub, ses = chosen_entities['session'], task = chosen_entities['task'], run = chosen_entities['run'], desc = 'ReportStrings', scope='derivatives')))[0]
            except:
                report_str_path = '' #in case none was created yet
                print('___MEGqc___: No report strings were created for sub ', sub)

            tsvs_to_plot = {}
            entities_per_file = {}

            for metric in chosen_entities['METRIC']:
                # Creating the full list of files for each combination
                additional_str = None  # or additional_str = 'your_string'
                desc = metric + additional_str if additional_str else metric
                

                # We call query with entities that always must present + entities that might present, might not:
                # This is how the call would look if we had all entities:
                # tsv_path = sorted(list(dataset.query(suffix='meg', extension='.tsv', return_type='filename', subj=sub, ses = chosen_entities['session'], task = chosen_entities['task'], run = chosen_entities['run'], desc = desc, scope='derivatives')))

                entities = {
                    'subj': sub,
                    'suffix': 'meg',
                    'extension': 'tsv',
                    'return_type': 'filename',
                    'desc': desc,
                    'scope': 'derivatives',
                }

                if 'session' in chosen_entities and chosen_entities['session']:
                    entities['session'] = chosen_entities['session']

                if 'task' in chosen_entities and chosen_entities['task']:
                    entities['task'] = chosen_entities['task']

                if 'run' in chosen_entities and chosen_entities['run']:
                    entities['run'] = chosen_entities['run']


                if metric == 'PSDs':
                    descriptions = ['PSDs', 'PSDnoiseMag', 'PSDnoiseGrad', 'PSDwavesMag', 'PSDwavesGrad']
                elif metric == 'ECGs':
                    descriptions = ['ECGchannel', 'ECGs']
                elif metric == 'EOGs':
                    descriptions = ['EOGchannel', 'EOGs']
                else:
                    descriptions = [metric]

                # Query tsv derivs and get the tsv paths:
                tsv_path = []
                for desc in descriptions:
                    entities['desc'] = desc
                    tsv_path += list(dataset.query(**entities))

                tsvs_to_plot[metric] = sorted(tsv_path)


                #Query same tsv derivs and get the tsv entities to later use them to save report with same entities:
                entities = copy.deepcopy(entities)
                entities['return_type'] = 'object'
                #this time we need to return objects, not file paths, rest is same.
                entities_obj = []
                for desc in descriptions:
                    entities['desc'] = desc
                    
                    entities_obj += list(dataset.query(**entities))
                    entities_obj = sorted(entities_obj, key=lambda k: k['name'])

                entities_per_file[metric] = entities_obj


            # 1. Check that we got same entities_per_file and tsvs_to_plot:
            
            # 2. we can have several tsvs for one metric with same raw entities, 
            # all these tsvs have to be added to one report later.
            # so we create a dict: {metric: {entities: [tsv1, tsv2, tsv3]}}

            print('___MEGqc___: ', 'entities_per_file', entities_per_file)
            print('___MEGqc___: ', 'tsvs_to_plot', tsvs_to_plot)

            tsvs_by_metric = {}
            for (tsv_metric, tsv_paths), (entity_metric, entity_vals) in zip(tsvs_to_plot.items(), entities_per_file.items()):

                # Here start part 1:
                if len(tsv_paths) != len(entity_vals):
                    raise ValueError('Different number of tsvs and entities for metric: ', tsv_metric)
                
                for tsv_path, entity_val in zip(tsv_paths, entity_vals):
                #check that every metric_value is same as file_value:
                    file_name_in_path = os.path.basename(tsv_path).split('_meg.')[0]
                    file_name_in_obj = entity_val['name'].split('_meg.')[0]

                    if file_name_in_obj not in file_name_in_path:
                        raise ValueError('Different names in tsvs_to_plot and entities_per_file')

                    # Here start part 2:
                    # Initialize the dictionary for the metric if it doesn't exist

                    
                    #this is the collection of entities belonging to the same raw file disregarding the desc part 
                    # (desc appears from derivatives, but we care about the basic raw entitites).
                    #from entity_val name remove the description part:


                    if tsv_metric not in tsvs_by_metric:
                        tsvs_by_metric[tsv_metric] = {}

                    # Initialize the list for the entity if it doesn't exist
                    if entity_val not in tsvs_by_metric[tsv_metric]:
                        tsvs_by_metric[tsv_metric][entity_val] = []

                    # Append the tsv_path to the list
                    tsvs_by_metric[tsv_metric][entity_val].append(tsv_path)

            tsvs_by_metric = combine_tsvs_dict(tsvs_by_metric)

            # We can loop over the dict and create the derivatives: all tsvs for 1 metric used to create 1 report
            # Then save report with the same entities from original tsv derivatives


            for metric, vals in tsvs_by_metric.items():

                for entity_val, tsv_paths in vals.items():

                    # Now prepare the derivative to be written:
                    meg_artifact = reports_folder.create_artifact(raw=entity_val) 
                    # create artifact, take entities from entities of the previously calculated tsv derivative

                    meg_artifact.add_entity('desc', metric) #file name
                    meg_artifact.suffix = 'meg'
                    meg_artifact.extension = '.html'

                    deriv = csv_to_html_report(metric, tsv_paths, report_str_path, plot_settings)

                    #define method how the derivative will be written to file system:
                    meg_artifact.content = lambda file_path, cont=deriv: cont.save(file_path, overwrite=True, open_browser=False)
        
                    
    ancpbids.write_derivative(dataset, derivative) 

    return tsvs_to_plot



# ____________________________
# RUN IT:
#tsvs_to_plot = make_plots_meg_qc(ds_paths=['/Volumes/M2_DATA/MEG_QC_stuff/data/openneuro/ds003483'])
tsvs_to_plot = make_plots_meg_qc(ds_paths=['/Users/jenya/Local Storage/Job Uni Rieger lab/data/ds83'])
#tsvs_to_plot = make_plots_meg_qc(ds_paths=['/Volumes/SSD_DATA/camcan'])
#tsvs_to_plot = make_plots_meg_qc(ds_paths=['/Volumes/SSD_DATA/MEG_QC_stuff/data/CTF/ds000246'])
#tsvs_to_plot = make_plots_meg_qc(ds_paths=['/Volumes/SSD_DATA/MEG_QC_stuff/data/openneuro/ds000117'])
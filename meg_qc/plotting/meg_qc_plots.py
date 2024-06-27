import sys
import os
import ancpbids
import json
import re

from prompt_toolkit.shortcuts import checkboxlist_dialog
from prompt_toolkit.styles import Style

# Get the absolute path of the parent directory of the current script
parent_dir = os.path.dirname(os.getcwd())
gradparent_dir = os.path.dirname(parent_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)
sys.path.append(gradparent_dir)

# from meg_qc.source.universal_plots import QC_derivative, boxplot_all_time_csv, boxplot_epoched_xaxis_channels_csv, boxplot_epoched_xaxis_epochs_csv, Plot_psd_csv, plot_artif_per_ch_correlated_lobes_csv, plot_correlation_csv, plot_muscle_csv, make_head_pos_plot_csv
# from meg_qc.source.universal_html_report import make_joined_report, make_joined_report_mne

# from meg_qc.plotting.universal_plots import QC_derivative, boxplot_all_time_csv, boxplot_epoched_xaxis_channels_csv, boxplot_epoched_xaxis_epochs_csv, Plot_psd_csv, plot_artif_per_ch_correlated_lobes_csv, plot_correlation_csv, plot_muscle_csv, make_head_pos_plot_csv, plot_sensors_3d_csv, plot_pie_chart_freq_csv, plot_ECG_EOG_channel
from meg_qc.plotting.universal_plots import *
from meg_qc.plotting.universal_html_report import make_joined_report, make_joined_report_mne


# # Needed to import the modules without specifying the full path, for command line and jupyter notebook
# sys.path.append('./')
# sys.path.append('./meg_qc/source/')

# # relative path for `make html` (docs)
# sys.path.append('../meg_qc/source/')

# # relative path for `make html` (docs) run from https://readthedocs.org/
# # every time rst file is nested insd of another, need to add one more path level here:
# sys.path.append('../../meg_qc/source/')
# sys.path.append('../../../meg_qc/source/')
# sys.path.append('../../../../meg_qc/source/')


# What we want: 
# save in the right folders the csvs - to do in pipeline as qc derivative
# get it from the folders fro plotting - over ancp bids again??
# plot only whst is requested by user: separate config + if condition like in pipeline?
# do we save them as derivatives to write over abcp bids as report as before?


def modify_entity_name(entities):

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

def selector_one_window(entities):

    ''' Old version where everything is done in 1 window'''

    # Define the categories and subcategories
    categories = modify_entity_name(entities)

    # Create a list of values with category titles
    values = []
    for category, items in categories.items():
        values.append((f'== {category} ==', f'== {category} =='))
        for item in items:
            values.append((str(item), str(item)))

    results = checkboxlist_dialog(
        title="Select metrics to plot:",
        text="Select subcategories:",
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

    # Ignore the category titles
    selected_subcategories = [result for result in results if not result.startswith('== ')]

    print('___MEGqc___: You selected:', selected_subcategories)

    return selected_subcategories


def selector(entities):

    '''
    Loop over categories (keys)
    for every key use a subfunction that will create a selector for the subcategories.
    '''

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


def get_ds_entities(ds_paths):

    for dataset_path in ds_paths: #run over several data sets

        try:
            dataset = ancpbids.load_dataset(dataset_path)

            #schema = dataset.get_schema() #Remove?

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
    
    return entities


def csv_to_html_report(metric: str, tsv_paths: list, report_str_path: str, plot_settings):

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

                std_derivs += [fig_all_time] + [fig_std_epoch0] + [fig_std_epoch1] 

        if 'PTP' in metric.upper():

            fig_ptp_epoch0 = []
            fig_ptp_epoch1 = []

            ptp_manual_derivs += plot_sensors_3d_csv(tsv_path)
        
            for m_or_g in m_or_g_chosen:

                fig_all_time = boxplot_all_time_csv(tsv_path, ch_type=m_or_g, what_data='peaks')
                fig_ptp_epoch0 = boxplot_epoched_xaxis_channels_csv(tsv_path, ch_type=m_or_g, what_data='peaks')
                fig_ptp_epoch1 = boxplot_epoched_xaxis_epochs_csv(tsv_path, ch_type=m_or_g, what_data='peaks')

                ptp_manual_derivs += [fig_all_time] + [fig_ptp_epoch0] + [fig_ptp_epoch1] 

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


def make_plots_meg_qc(ds_paths):

    for dataset_path in ds_paths: #run over several data sets
        dataset = ancpbids.load_dataset(dataset_path)
        schema = dataset.get_schema()

        derivative = dataset.create_derivative(name="Meg_QC")
        derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"

        entities = get_ds_entities(ds_paths) 

        print('_____All entities of ds derivs:', entities)

        chosen_entities, plot_settings = selector(entities)
        if not chosen_entities:
            return

        #chosen_entities = {'subject': ['009'], 'session': ['1'], 'task': ['deduction', 'induction'], 'run': ['1'], 'METRIC': ['ECGs', 'Muscle']}
        
        print('___MEGqc___: CHOSEN entities to plot: ', chosen_entities)
        print('___MEGqc___: CHOSEN settings: ', plot_settings)

        for sub in chosen_entities['subject']:

            subject_folder = derivative.create_folder(type_=schema.Subject, name='sub-'+sub)
            reports_folder = subject_folder.create_folder(name='reports')
            list_of_sub_jsons = dataset.query(sub=sub, suffix='meg', extension='.fif')

            try:
                report_str_path = sorted(list(dataset.query(suffix='meg', extension='.json', return_type='filename', subj=sub, ses = chosen_entities['session'], task = chosen_entities['task'], run = chosen_entities['run'], desc = 'ReportStrings', scope='derivatives')))[0]
            except:
                report_str_path = '' #in case none was created yet
                print('___MEGqc___: No report strings were created for sub ', sub)

            tsvs_to_plot = {}

            for metric in chosen_entities['METRIC']:
                # Creating the full list of files for each combination
                additional_str = None  # or additional_str = 'your_string'
                desc = metric + additional_str if additional_str else metric
                

                # We call query with entities that always must present + entities that might present, might not:
                #This is how the call would look if we had all entities:
                #tsv_path = sorted(list(dataset.query(suffix='meg', extension='.tsv', return_type='filename', subj=sub, ses = chosen_entities['session'], task = chosen_entities['task'], run = chosen_entities['run'], desc = desc, scope='derivatives')))

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

                #Now call query and get the tsvs:
                tsv_path = []
                for desc in descriptions:
                    entities['desc'] = desc
                    tsv_path += sorted(list(dataset.query(**entities)))

                tsvs_to_plot[metric] = tsv_path

            # tsvs_to_plot is a dictionary with metrics as keys and lists of tsv paths as values
            # it contains ALL tsv files that have been created for CHOSEN in selector sub, ses, task, run and metrics.

            print('___MEGqc___: list_of_sub_jsons', list_of_sub_jsons)
            print('___MEGqc___: tsvs_to_plot', tsvs_to_plot)

            #Next, we need to create a report of the metrcis and save it with the right bids entities. 
            #Problem is, we cant just parce entities from tsv and put them in report name. 
            # We need to create a report on base of raw file: meg_artifact = reports_folder.create_artifact(raw=sub_json)
            #so we need to match the entities of the raw file with the entities of the tsv files. 
            #and for each raw file create a report with all tsv files that match the entities of the raw file.


            for sub_json in list_of_sub_jsons:
                #Loop over sub jsons - meaning over separate fif (raw) files belonging to the same subject:

                for metric in tsvs_to_plot:
                #Loop over calculated metrics:
                    
                    tsv_paths_for_one_metric_one_raw = []

                    for tsv_path in tsvs_to_plot[metric]:


                        #get the last part of the path containig the file name:
                        file_name = tsv_path.split('/')[-1]

                        #get the part of the file name that is the same as the raw file name, 
                        #so everything before '_desc', will contain all entities:
                        # (only derivatives have _desc in their name, raw should not):
                        tsv_bids_name = file_name.split('_desc')[0]

                        #take everything in sub_json['name'] before '_meg.fif', it will contain all entities:
                        raw_bids_name = sub_json['name'].split('_meg.fif')[0]

                        #if the raw file name and the tsv file name match - we found the right tsv file for this raw file
                        # Now we can create a derivative on base of this TSV and save it in connection with the right raw file:
                        if raw_bids_name == tsv_bids_name:

                            tsv_paths_for_one_metric_one_raw += [tsv_path]
                            #collect all tsvs for the same metric in one list 
                            #to later add them all to the same report for this metric
                        else:
                            #skip to next tsv file:
                            continue


                    # Now prepare the derivative to be written:
                    meg_artifact = reports_folder.create_artifact(raw=sub_json)

                    meg_artifact.add_entity('desc', metric) #file name
                    meg_artifact.suffix = 'meg'
                    meg_artifact.extension = '.html'


                    deriv = csv_to_html_report(metric, tsv_paths_for_one_metric_one_raw, report_str_path, plot_settings)

                    #define method how the derivative will be written to file system:
                    meg_artifact.content = lambda file_path, cont=deriv: cont.save(file_path, overwrite=True, open_browser=False)
                    
                        
    ancpbids.write_derivative(dataset, derivative) 

    return tsvs_to_plot


# RUN IT:
#tsvs_to_plot = make_plots_meg_qc(ds_paths=['/Volumes/M2_DATA/MEG_QC_stuff/data/openneuro/ds003483'])
tsvs_to_plot = make_plots_meg_qc(ds_paths=['/Users/jenya/Local Storage/Job Uni Rieger lab/data/ds83'])
#tsvs_to_plot = make_plots_meg_qc(ds_paths=['/Volumes/SSD_DATA/camcan'])


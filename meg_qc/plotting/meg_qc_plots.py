import sys
import os
import ancpbids
import json

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

    #old_new_categories = {'desc': 'METRIC', 'sub': 'SUBJECT', 'ses': 'SESSION', 'task': 'TASK', 'run': 'RUN'}

    old_new_categories = {'desc': 'METRIC'}

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
    categories['verbose_plots'] = ['True', 'False']

    selected = {}
    # Create a list of values with category titles
    for key, values in categories.items():
        subcategory = select_subcategory(categories[key], key)
        selected[key] = subcategory


    #Check 1) if nothing was chosen, 2) if ALL was chosen
    for key, values in selected.items():

        if not selected[key]: # if nothing was chosen:
            title = 'You did not choose the '+key+'. Please try again:'
            subcategory = select_subcategory(categories[key], key, title)
            if not subcategory: # if nothing was chosen again - stop:
                print('___MEGqc___: You still  did not choose the '+key+'. Please start over.')
                return None
            
        else:
            for item in values:
                if 'ALL' in item.upper():
                    all_selected = [str(category) for category in categories[key] if 'ALL' not in str(category).upper()]
                    selected[key] = all_selected #everything

    #Separate into selected_entities and plot_settings:
        selected_entities, plot_settings = {}, {}
        for key, values in selected.items():
            if key != 'verbose_plots' and key != 'm_or_g':
                selected_entities[key] = values
            elif key == 'verbose_plots' or key == 'm_or_g':
                plot_settings[key] = values
            else:
                print('___MEGqc__: wow, weird key in selector()! check it.')

    return selected_entities, plot_settings


def select_subcategory(subcategories, category_title, title="What would you like to plot? Click to select."):

    # Create a list of values with category titles
    values = []
    for items in subcategories:
        values.append((str(items), str(items)))

        # Each tuple represents a checkbox item and should contain two elements:
        # A string that will be returned when the checkbox is selected.
        # A string that will be displayed as the label of the checkbox.

    if category_title =='verbose_plots':
        title = 'Do you want to see plots while running the script? (True) Or only after inside the report? (False)'

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

    return results


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


        entities = dataset.query_entities()
        print('___MEGqc___: ', 'entities', entities)
    
    return entities


def csv_to_html_report(metric: str, tsv_paths: list, report_str_path: str, plot_settings):

    m_or_g_chosen = plot_settings['m_or_g'] 
    verbose_plots = bool(plot_settings['verbose_plots'][0]=='True')

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

                fig_all_time = boxplot_all_time_csv(tsv_path, ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)
                fig_std_epoch0 = boxplot_epoched_xaxis_channels_csv(tsv_path, ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)
                fig_std_epoch1 = boxplot_epoched_xaxis_epochs_csv(tsv_path, ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)

                #older versions, no color coding:
                #fig_std_epoch1 += [boxplot_epochs(df_mg=df_std, ch_type=m_or_g, what_data='stds', x_axis_boxes='channels', verbose_plots=verbose_plots)] #old version
                #fig_std_epoch2 += [boxplot_epochs(df_mg=df_std, ch_type=m_or_g, what_data='stds', x_axis_boxes='epochs', verbose_plots=verbose_plots)]

                std_derivs += [fig_all_time] + [fig_std_epoch0] + [fig_std_epoch1] 

        if 'PTP' in metric.upper():

            fig_ptp_epoch0 = []
            fig_ptp_epoch1 = []

            ptp_manual_derivs += plot_sensors_3d_csv(tsv_path)
        
            for m_or_g in m_or_g_chosen:

                fig_all_time = boxplot_all_time_csv(tsv_path, ch_type=m_or_g, what_data='peaks', verbose_plots=verbose_plots)
                fig_ptp_epoch0 = boxplot_epoched_xaxis_channels_csv(tsv_path, ch_type=m_or_g, what_data='peaks', verbose_plots=verbose_plots)
                fig_ptp_epoch1 = boxplot_epoched_xaxis_epochs_csv(tsv_path, ch_type=m_or_g, what_data='peaks', verbose_plots=verbose_plots)

                #older versions, no color coding:
                #fig_std_epoch1 += [boxplot_epochs(df_mg=df_std, ch_type=m_or_g, what_data='stds', x_axis_boxes='channels', verbose_plots=verbose_plots)] #old version
                #fig_std_epoch2 += [boxplot_epochs(df_mg=df_std, ch_type=m_or_g, what_data='stds', x_axis_boxes='epochs', verbose_plots=verbose_plots)]

                ptp_manual_derivs += [fig_all_time] + [fig_ptp_epoch0] + [fig_ptp_epoch1] 

        elif 'PSD' in metric.upper():

            method = 'welch' #is also hard coded in PSD_meg_qc() for now

            psd_derivs += plot_sensors_3d_csv(tsv_path)

            for m_or_g in m_or_g_chosen:

                psd_derivs += Plot_psd_csv(m_or_g, tsv_path, method, verbose_plots)

                psd_derivs += plot_pie_chart_freq_csv(tsv_path, m_or_g=m_or_g, noise_or_waves = 'noise', verbose_plots=verbose_plots)

                psd_derivs += plot_pie_chart_freq_csv(tsv_path, m_or_g=m_or_g, noise_or_waves = 'waves', verbose_plots=verbose_plots)

        elif 'ECG' in metric.upper():

            ecg_derivs += plot_sensors_3d_csv(tsv_path)

            ecg_derivs += plot_ECG_EOG_channel_csv(tsv_path, verbose_plots)

            ecg_derivs += plot_mean_ecg_ch_data_csv(tsv_path, 'ECG', verbose_plots)

            #TODO: add ch description like here? export it as separate report strings?
            #noisy_ch_derivs += [QC_derivative(fig, bad_ecg_eog[ecg_ch]+' '+ecg_ch, 'plotly', description_for_user = ecg_ch+' is '+ bad_ecg_eog[ecg_ch]+ ': 1) peaks have similar amplitude: '+str(ecg_eval[0])+', 2) tolerable number of breaks: '+str(ecg_eval[1])+', 3) tolerable number of bursts: '+str(ecg_eval[2]))]

            for m_or_g in m_or_g_chosen:
                ecg_derivs += plot_artif_per_ch_correlated_lobes_csv(tsv_path, m_or_g, 'ECG', flip_data=False, verbose_plots=verbose_plots)
                ecg_derivs += plot_correlation_csv(tsv_path, 'ECG', m_or_g, verbose_plots=verbose_plots)

        elif 'EOG' in metric.upper():

            eog_derivs += plot_sensors_3d_csv(tsv_path)
                
            for m_or_g in m_or_g_chosen:
                eog_derivs += plot_artif_per_ch_correlated_lobes_csv(tsv_path, m_or_g, 'EOG', flip_data=False, verbose_plots=verbose_plots)
                eog_derivs += plot_correlation_csv(tsv_path, 'EOG', m_or_g, verbose_plots=verbose_plots)

            
        elif 'MUSCLE' in metric.upper():

            if 'mag' in m_or_g_chosen:
                m_or_g_decided=['mag']
            elif 'grad' in m_or_g_chosen and 'mag' not in m_or_g_chosen:
                m_or_g_decided=['grad']
            else:
                print('___MEGqc___: ', 'No magnetometers or gradiometers found in data. Artifact detection skipped.')


            muscle_derivs +=  plot_muscle_csv(tsv_path, m_or_g_decided[0], verbose_plots = verbose_plots)

            
        elif 'HEAD' in metric.upper():
                
            head_pos_derivs, _ = make_head_pos_plot_csv(tsv_path, verbose_plots=verbose_plots)
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

    for metric, values in QC_derivs.items():
        if values and metric != 'Sensors':
            QC_derivs['Report_MNE'] += [QC_derivative(report_html_string, 'REPORT_'+metric, 'report mne')]

    return QC_derivs


def make_plots_meg_qc(ds_paths):

    for dataset_path in ds_paths[0:1]: #run over several data sets
        dataset = ancpbids.load_dataset(dataset_path)
        schema = dataset.get_schema()

        derivative = dataset.create_derivative(name="Meg_QC")
        derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"

        entities = get_ds_entities(ds_paths) 

        chosen_entities, plot_settings = selector(entities)

        #chosen_entities = {'sub': ['009'], 'ses': ['1'], 'task': ['deduction', 'induction'], 'run': ['1'], 'METRIC': ['ECGs', 'Muscle']}
        
        print('___MEGqc___: CHOSEN entities to plot: ', chosen_entities)
        print('___MEGqc___: CHOSEN settings: ', plot_settings)

        for sub in chosen_entities['sub']:

            print('_____sub____', sub)

            subject_folder = derivative.create_folder(type_=schema.Subject, name='sub-'+sub)
            list_of_sub_jsons = dataset.query(sub=sub, suffix='meg', extension='.fif')

            print('______list_of_sub_jsons')
            print(list_of_sub_jsons)

            try:
                report_str_path = sorted(list(dataset.query(suffix='meg', extension='.json', return_type='filename', subj=sub, ses = chosen_entities['ses'], task = chosen_entities['task'], run = chosen_entities['run'], desc = 'ReportStrings', scope='derivatives')))[0]
            except:
                report_str_path = '' #in case none was created yet
                print('___MEGqc___: No report strings were created for sub ', sub)

            tsvs_to_plot = {}
            for metric in chosen_entities['METRIC']:
                # Creating the full list of files for each combination
                additional_str = None  # or additional_str = 'your_string'
                desc = metric + additional_str if additional_str else metric
                
                tsv_path = sorted(list(dataset.query(suffix='meg', extension='.tsv', return_type='filename', subj=sub, ses = chosen_entities['ses'], task = chosen_entities['task'], run = chosen_entities['run'], desc = desc, scope='derivatives')))

                if metric == 'PSDs':
                    tsv_path += sorted(list(dataset.query(suffix='meg', extension='.tsv', return_type='filename', subj=sub, ses = chosen_entities['ses'], task = chosen_entities['task'], run = chosen_entities['run'], desc = 'PSDnoiseMag', scope='derivatives')))
                    tsv_path += sorted(list(dataset.query(suffix='meg', extension='.tsv', return_type='filename', subj=sub, ses = chosen_entities['ses'], task = chosen_entities['task'], run = chosen_entities['run'], desc = 'PSDnoiseGrad', scope='derivatives')))
                    tsv_path += sorted(list(dataset.query(suffix='meg', extension='.tsv', return_type='filename', subj=sub, ses = chosen_entities['ses'], task = chosen_entities['task'], run = chosen_entities['run'], desc = 'PSDwavesMag', scope='derivatives')))
                    tsv_path += sorted(list(dataset.query(suffix='meg', extension='.tsv', return_type='filename', subj=sub, ses = chosen_entities['ses'], task = chosen_entities['task'], run = chosen_entities['run'], desc = 'PSDwavesGrad', scope='derivatives')))
                
                if metric == 'ECGs':
                    tsv_path += sorted(list(dataset.query(suffix='meg', extension='.tsv', return_type='filename', subj=sub, ses = chosen_entities['ses'], task = chosen_entities['task'], run = chosen_entities['run'], desc = 'ECGchannel', scope='derivatives')))

                if metric == 'EOGs':
                    tsv_path += sorted(list(dataset.query(suffix='meg', extension='.tsv', return_type='filename', subj=sub, ses = chosen_entities['ses'], task = chosen_entities['task'], run = chosen_entities['run'], desc = 'EOGchannel', scope='derivatives')))

                tsvs_to_plot[metric] = tsv_path

            print('___MEGqc___: TSVs to plot: ', tsvs_to_plot)

            counter = 0
            for metric, tsv_paths in tsvs_to_plot.items():
                #for n_tsv, tsv_path in enumerate(files):

                meg_artifact = subject_folder.create_artifact(raw=list_of_sub_jsons[counter]) #shell. empty derivative
                meg_artifact.add_entity('desc', metric) #file name
                meg_artifact.suffix = 'meg'
                meg_artifact.extension = '.html'

                # Here convert csv into figure and into html report:
                deriv = csv_to_html_report(metric, tsv_paths, report_str_path, plot_settings)

                meg_artifact.content = lambda file_path, cont=deriv['Report_MNE'][0].content: cont.save(file_path, overwrite=True, open_browser=False)
                counter += 1

    ancpbids.write_derivative(dataset, derivative) 

    return tsvs_to_plot


# RUN IT:
#tsvs_to_plot = make_plots_meg_qc(ds_paths=['/Volumes/M2_DATA/MEG_QC_stuff/data/openneuro/ds003483'])
tsvs_to_plot = make_plots_meg_qc(ds_paths=['/Users/jenya/Local Storage/Job Uni Rieger lab/data/ds83'])


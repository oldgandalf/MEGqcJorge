import configparser
import sys
import os
import ancpbids
import itertools

from prompt_toolkit.shortcuts import checkboxlist_dialog
from prompt_toolkit.styles import Style


# Needed to import the modules without specifying the full path, for command line and jupyter notebook
sys.path.append('./')
sys.path.append('./meg_qc/source/')

# relative path for `make html` (docs)
sys.path.append('../meg_qc/source/')

# relative path for `make html` (docs) run from https://readthedocs.org/
# every time rst file is nested insd of another, need to add one more path level here:
sys.path.append('../../meg_qc/source/')
sys.path.append('../../../meg_qc/source/')
sys.path.append('../../../../meg_qc/source/')


# What we want: 
# save in the right folders the csvs - to do in pipeline as qc derivative
# get it from the folders fro plotting - over ancp bids again??
# plot only whst is requested by user: separate config + if condition like in pipeline?
# do we save them as derivatives to write over abcp bids as report as before?


def get_plot_config_params(config_plot_file_name: str):

    """
    Parse all the parameters from config and put into a python dictionary 
    divided by sections. Parsing approach can be changed here, which 
    will not affect working of other fucntions.
    

    Parameters
    ----------
    config_file_name: str
        The name of the config file.

    Returns
    -------
    all_qc_params: dict
        A dictionary with all the parameters from the config file.

    """
    
    plot_params = {}

    config = configparser.ConfigParser()
    config.read(config_plot_file_name)

    default_section = config['DEFAULT']

    m_or_g_chosen = default_section['do_for'] 
    m_or_g_chosen = [chosen.strip() for chosen in m_or_g_chosen.split(",")]
    if 'mag' not in m_or_g_chosen and 'grad' not in m_or_g_chosen:
        print('___MEG QC___: ', 'No channels to analyze. Check parameter do_for in config file.')
        return None

    # subjects = default_section['subjects']
    # subjects = [sub.strip() for sub in subjects.split(",")]

    plot_sensors = default_section.getboolean('plot_sensors')
    plot_STD = default_section.getboolean('STD')
    plot_PSD = default_section.getboolean('PSD')
    plot_PTP_manual = default_section.getboolean('PTP_manual')
    plot_PTP_auto_mne = default_section.getboolean('PTP_auto_mne')
    plot_ECG = default_section.getboolean('ECG')
    plot_EOG = default_section.getboolean('EOG')
    plot_Head = default_section.getboolean('Head')
    plot_Muscle = default_section.getboolean('Muscle')

    ds_paths = default_section['data_directory']
    ds_paths = [path.strip() for path in ds_paths.split(",")]
    if len(ds_paths) < 1:
        print('___MEG QC___: ', 'No datasets to analyze. Check parameter data_directory in config file. Data path can not contain spaces! You can replace them with underscores or remove completely.')
        return None

    try:

        default_params = dict({
            'm_or_g_chosen': m_or_g_chosen, 
            'subjects': [],
            'plot_sensors': plot_sensors,
            'plot_STD': plot_STD,
            'plot_PSD': plot_PSD,
            'plot_PTP_manual': plot_PTP_manual,
            'plot_PTP_auto_mne': plot_PTP_auto_mne,
            'plot_ECG': plot_ECG,
            'plot_EOG': plot_EOG,
            'plot_Head': plot_Head,
            'plot_Muscle': plot_Muscle,
            'dataset_path': ds_paths,
            'plot_mne_butterfly': default_section.getboolean('plot_mne_butterfly'),
            'plot_interactive_time_series': default_section.getboolean('plot_interactive_time_series'),
            'plot_interactive_time_series_average': default_section.getboolean('plot_interactive_time_series_average'),
            'verbose_plots': default_section.getboolean('verbose_plots')})
        plot_params['default'] = default_params

    except:
        print('___MEG QC___: ', 'Invalid setting in config file! Please check instructions for each setting. \nGeneral directions: \nDon`t write any parameter as None. Don`t use quotes.\nLeaving blank is only allowed for parameters: \n- stim_channel, \n- data_crop_tmin, data_crop_tmax, \n- freq_min and freq_max in Filtering section, \n- all parameters of Filtering section if apply_filtering is set to False.')
        return None

    return plot_params

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
        entities['METRIC'] = [x for x in entities['METRIC'] if x in ['_ALL_METRICS_', 'STDs', 'PSDs', 'PtPmanual', 'PtPauto', 'ECGs', 'EOGs', 'Head', 'Muscle']]

    return entities

def selector_old(entities):

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

    print('You selected:', selected_subcategories)

    return selected_subcategories


def selector(entities):

    '''
    Loop over categories (keys)
    for every key use a subfunction that will create a selector for the subcategories.
    '''

    # Define the categories and subcategories
    categories = modify_entity_name(entities)

    selected = {}
    # Create a list of values with category titles
    for key, items in categories.items():
        subcategory = select_subcategory(categories[key], key)
        selected[key] = subcategory


    #Check 1) if nothing was chosen, 2) if ALL was chosen
    for key, items in selected.items():

        if not selected[key]: # if nothing was chosen:
            title = 'You did not choose the '+key+'. Please try again:'
            subcategory = select_subcategory(categories[key], key, title)
            if not subcategory: # if nothing was chosen again - stop:
                print('You still  did not choose the '+key+'. Please start over.')
                return None
            
        else:
            for item in items:
                if 'ALL' in item.upper():
                    all_selected = [str(category) for category in categories[key] if 'ALL' not in str(category).upper()]
                    selected[key] = all_selected #everything

    return selected

def select_subcategory(subcategories, category_title, title="What would you like to plot?"):

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

    return results



def make_plots_meg_qc(config_plot_file_path):

    plot_params = get_plot_config_params(config_plot_file_path)

    #derivs_path  = '/Volumes/M2_DATA/'

    if plot_params is None:
        return


    ds_paths = plot_params['default']['dataset_path']
    for dataset_path in ds_paths: #run over several data sets

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


        entities = dataset.query_entities()
        print('___MEG QC___: ', 'entities', entities)


        # SELECTOR:
        # get all entities
        # create selector for each subject, metric, run/task

        chosen_entities = selector(entities)


        # list_of_subs = list(entities["sub"])
        if plot_params['default']['subjects'][0] != 'all':
            list_of_subs = plot_params['default']['subjects']
        elif plot_params['default']['subjects'][0] == 'all':
            list_of_subs = sorted(list(dataset.query_entities()["sub"]))
            print('___MEG QC___: ', 'list_of_subs', list_of_subs)
            if not list_of_subs:
                print('___MEG QC___: ', 'No subjects found by ANCP BIDS. Check your data set and directory path in config.')
                return
        else:
            print('___MEG QC___: ', 'Something went wrong with the subjects list. Check parameter "subjects" in config file or simply set it to "all".')
            return


        for sid in list_of_subs[0:1]: #[0:4]: 
            print('___MEG QC___: ', 'Dataset: ', dataset_path)
            print('___MEG QC___: ', 'Take SID: ', sid)
            
            subject_folder = derivative.create_folder(type_=schema.Subject, name='sub-'+sid)
            list_of_sub_jsons = dataset.query(sub=sid, suffix='meg', extension='.fif')

            # GET all derivs!
            derivs_list = sorted(list(dataset.query(suffix='meg', extension='.tsv', return_type='filename', subj=sid, scope='derivatives')))
            print('___MEG QC___: ', 'derivs_list', derivs_list)

            for fif_ind, data_file in enumerate(derivs_list): 
                print('___MEG QC___: ', 'Take deriv: ', data_file)

                #here goes the actual code for plotting:
                
        return
    

def plot_metrics(config_plot_file_name: str, chosen_entities):

    plot_params = get_plot_config_params(config_plot_file_path)

    #derivs_path  = '/Volumes/M2_DATA/'

    if plot_params is None:
        return


    ds_paths = plot_params['default']['dataset_path']
    for dataset_path in ds_paths: #run over several data sets

        try:
            dataset = ancpbids.load_dataset(dataset_path)
        except:
            print('___MEG QC___: ', 'No data found in the given directory path! \nCheck directory path in config file and presence of data on your device.')
            return
        
    #create derivatives folder first:
        if os.path.isdir(dataset_path+'/derivatives')==False: 
                os.mkdir(dataset_path+'/derivatives')

        derivative = dataset.create_derivative(name="Meg_QC")
        derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"

    list_of_subs = ['009'] #TODO: get real list from chosen entities


    for sid in list_of_subs[0:1]: #[0:4]: 
        print('___MEG QC___: ', 'Dataset: ', dataset_path)
        print('___MEG QC___: ', 'Take SID: ', sid)
        
        subject_folder = derivative.create_folder(type_=schema.Subject, name='sub-'+sid)
        list_of_sub_jsons = dataset.query(sub=sid, suffix='meg', extension='.fif')

        # GET all derivs!
        derivs_list = sorted(list(dataset.query(suffix='meg', extension='.tsv', return_type='filename', subj=sid, scope='derivatives')))
        print('___MEG QC___: ', 'derivs_list', derivs_list)

        for fif_ind, data_file in enumerate(derivs_list): 
            print('___MEG QC___: ', 'Take deriv: ', data_file)

            #here goes the actual code for plotting:
                
        return
    


def get_all_entities(config_plot_file_path):

    plot_params = get_plot_config_params(config_plot_file_path)

    #derivs_path  = '/Volumes/M2_DATA/'

    if plot_params is None:
        return


    ds_paths = plot_params['default']['dataset_path']
    for dataset_path in ds_paths: #run over several data sets

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


        entities = dataset.query_entities()
        print('___MEG QC___: ', 'entities', entities)
    
    return entities

def stuff(config_plot_file_path):

    plot_params = get_plot_config_params(config_plot_file_path)
    ds_paths = plot_params['default']['dataset_path']
    for dataset_path in ds_paths[0:1]: #run over several data sets
        dataset = ancpbids.load_dataset(dataset_path)

        entities = get_all_entities(config_plot_file_path) 

        chosen_entities = selector(entities)
        print('CHOSEN entities to plot: ', chosen_entities)

        # Creating the full list of files for each combination
        files_to_plot = sorted(list(dataset.query(suffix='meg', extension='.tsv', return_type='filename', subj=chosen_entities['sub'], ses = chosen_entities['ses'], task = chosen_entities['task'], run = chosen_entities['run'], desc = chosen_entities['METRIC'], scope='derivatives')))


        for f in files_to_plot:
            print('File to plot: ', f)

    return files_to_plot


def herewego(plot_params):

    verbose_plots = plot_params['default']['verbose_plots']
    m_or_g_chosen = plot_params['default']['m_or_g_chosen']

    std_derivs, psd_derivs, pp_manual_derivs, pp_auto_derivs, ecg_derivs, eog_derivs, head_derivs, muscle_derivs, sensors_derivs, time_series_derivs = [],[],[],[],[], [],  [], [], [], []

    # sensors:

    if plot_params['default']['plot_sensors'] is True:
        sensors_csv_path = derivs_path+'Sensors.tsv'

        sensors_derivs = plot_sensors_3d_csv(sensors_csv_path)

    # STD

    if plot_params['default']['plot_STD'] is True:

        f_path = derivs_path+'STDs_by_lobe.tsv'
        fig_std_epoch0 = []
        fig_std_epoch1 = []
    
        for m_or_g in m_or_g_chosen:

            std_derivs += [boxplot_all_time_csv(f_path, ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)]

            # fig_std_epoch0 += [boxplot_epoched_xaxis_channels(chs_by_lobe_copy[m_or_g], df_std, ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)]
            fig_std_epoch0 += [boxplot_epoched_xaxis_channels_csv(f_path, ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)]

            fig_std_epoch1 += [boxplot_epoched_xaxis_epochs_csv(f_path, ch_type=m_or_g, what_data='stds', verbose_plots=verbose_plots)]

        std_derivs += fig_std_epoch0+fig_std_epoch1 


    # PtP
        
    if plot_params['default']['plot_PTP_manual'] is True:
        f_path = derivs_path+'PtPs_by_lobe.tsv'



    # PSD
    # TODO: also add pie psd plot
        
    if plot_params['default']['plot_PSD'] is True:

        for m_or_g in m_or_g_chosen:

            method = 'welch' #is also hard coded in PSD_meg_qc() for now
            f_path = derivs_path+'PSDs_by_lobe.tsv'

            psd_plot_derivative=Plot_psd_csv(m_or_g, f_path, method, verbose_plots)

            psd_derivs += [psd_plot_derivative]


    # ECG
    
    if plot_params['default']['plot_ECG'] is True:
            
        f_path = derivs_path+'ECGs_by_lobe.tsv'
            
        for m_or_g in m_or_g_chosen:
            affected_derivs = plot_artif_per_ch_correlated_lobes_csv(f_path, m_or_g, 'ECG', flip_data=False, verbose_plots=verbose_plots)
            correlation_derivs = plot_correlation_csv(f_path, 'ECG', m_or_g, verbose_plots=verbose_plots)

        ecg_derivs += affected_derivs + correlation_derivs

    # EOG

    if plot_params['default']['plot_EOG'] is True:
         
        f_path = derivs_path+'EOGs_by_lobe.tsv'
            
        for m_or_g in m_or_g_chosen:
            affected_derivs = plot_artif_per_ch_correlated_lobes_csv(f_path, m_or_g, 'EOG', flip_data=False, verbose_plots=verbose_plots)
            correlation_derivs = plot_correlation_csv(f_path, 'EOG', m_or_g, verbose_plots=verbose_plots)

        eog_derivs += affected_derivs + correlation_derivs 

    # Muscle
        
    if plot_params['default']['plot_Muscle'] is True:

        f_path = derivs_path+'muscle.tsv'

        if 'mag' in m_or_g_chosen:
                m_or_g_decided=['mag']
        elif 'grad' in m_or_g_chosen and 'mag' not in m_or_g_chosen:
                m_or_g_decided=['grad']
        else:
                print('___MEG QC___: ', 'No magnetometers or gradiometers found in data. Artifact detection skipped.')


        muscle_derivs =  plot_muscle_csv(f_path, m_or_g_decided[0], verbose_plots = verbose_plots)

    # Head
        
    if plot_params['default']['plot_Head'] is True:

        f_path = derivs_path+'Head.tsv'
            
        head_pos_derivs, _ = make_head_pos_plot_csv(f_path, verbose_plots=verbose_plots)
        # head_pos_derivs2 = make_head_pos_plot_mne(raw, head_pos, verbose_plots=verbose_plots)
        # head_pos_derivs += head_pos_derivs2
        head_derivs += head_pos_derivs

    QC_derivs={
        'Time_series': time_series_derivs,
        'Sensors': sensors_derivs,
        'STD': std_derivs, 
        'PSD': psd_derivs, 
        'PtP_manual': pp_manual_derivs, 
        'PtP_auto': pp_auto_derivs, 
        'ECG': ecg_derivs, 
        'EOG': eog_derivs,
        'Head': head_derivs,
        'Muscle': muscle_derivs}
    

    # TODO: if we make a report based on mne - we do need the raw data file.
    # if we do just simple html, dont need raw, but then we dont have the fif file info
    # What to do? Every time find a raw connected to the data? (open it from fif)

    report_html_string = make_joined_report_mne(raw, QC_derivs, report_strings = [], default_setting = [])

    for metric, values in QC_derivs.items():
        if values and metric != 'Sensors':
            QC_derivs['Report_MNE'] += [QC_derivative(report_html_string, 'REPORT_'+metric, 'report mne')]

    return QC_derivs
import configparser
import sys

from meg_qc.source.universal_plots import boxplot_all_time_csv, boxplot_epoched_xaxis_channels_csv, boxplot_epoched_xaxis_epochs_csv, Plot_psd_csv, make_head_pos_plot_csv, make_head_pos_plot_mne, plot_muscle_csv
from meg_qc.source.plots_ecg_eog import plot_artif_per_ch_correlated_lobes_csv, plot_correlation_csv


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

    subjects = default_section['subjects']
    subjects = [sub.strip() for sub in subjects.split(",")]

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

    tmin = default_section['data_crop_tmin']
    tmax = default_section['data_crop_tmax']
    try:
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
            'subjects': subjects,
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
            'verbose_plots': default_section.getboolean('verbose_plots'),
            'crop_tmin': tmin,
            'crop_tmax': tmax})
        plot_params['default'] = default_params

    except:
        print('___MEG QC___: ', 'Invalid setting in config file! Please check instructions for each setting. \nGeneral directions: \nDon`t write any parameter as None. Don`t use quotes.\nLeaving blank is only allowed for parameters: \n- stim_channel, \n- data_crop_tmin, data_crop_tmax, \n- freq_min and freq_max in Filtering section, \n- all parameters of Filtering section if apply_filtering is set to False.')
        return None

    return plot_params



def make_plots_meg_qc(config_plot_file_path):

    plot_params = get_plot_config_params(config_plot_file_path)

    derivs_path  = '/Volumes/M2_DATA/'
    verbose_plots = plot_params['default']['verbose_plots']
    m_or_g_chosen = plot_params['default']['m_or_g_chosen']

    std_derivs, psd_derivs, pp_manual_derivs, pp_auto_derivs, ecg_derivs, eog_derivs, head_derivs, muscle_derivs, sensors_derivs, time_series_derivs = [],[],[],[],[], [],  [], [], [], []

    # STD

    if plot_params['default']['plot_STD'] is True:

        f_path = derivs_path+'STDs_by_lobe.csv'
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
        f_path = derivs_path+'PtPs_by_lobe.csv'



    # PSD
    # TODO: also add pie psd plot
        
    if plot_params['default']['plot_PSD'] is True:

        for m_or_g in m_or_g_chosen:

            method = 'welch' #is also hard coded in PSD_meg_qc() for now
            f_path = derivs_path+'PSDs_by_lobe.csv'

            psd_plot_derivative=Plot_psd_csv(m_or_g, f_path, method, verbose_plots)

            psd_derivs += [psd_plot_derivative]


    # ECG
    
    if plot_params['default']['plot_ECG'] is True:
            
        f_path = derivs_path+'ECGs_by_lobe.csv'
            
        for m_or_g in m_or_g_chosen:
            affected_derivs = plot_artif_per_ch_correlated_lobes_csv(f_path, m_or_g, 'ECG', flip_data=False, verbose_plots=verbose_plots)
            correlation_derivs = plot_correlation_csv(f_path, 'ECG', m_or_g, verbose_plots=verbose_plots)

        ecg_derivs += affected_derivs + correlation_derivs

    # EOG

    if plot_params['default']['plot_EOG'] is True:
         
        f_path = derivs_path+'EOGs_by_lobe.csv'
            
        for m_or_g in m_or_g_chosen:
            affected_derivs = plot_artif_per_ch_correlated_lobes_csv(f_path, m_or_g, 'EOG', flip_data=False, verbose_plots=verbose_plots)
            correlation_derivs = plot_correlation_csv(f_path, 'EOG', m_or_g, verbose_plots=verbose_plots)

        eog_derivs += affected_derivs + correlation_derivs 

    # Muscle
        
    if plot_params['default']['plot_Muscle'] is True:

        f_path = derivs_path+'muscle.csv'

        if 'mag' in m_or_g_chosen:
                m_or_g_decided=['mag']
        elif 'grad' in m_or_g_chosen and 'mag' not in m_or_g_chosen:
                m_or_g_decided=['grad']
        else:
                print('___MEG QC___: ', 'No magnetometers or gradiometers found in data. Artifact detection skipped.')


        muscle_derivs =  plot_muscle_csv(f_path, m_or_g_decided[0], verbose_plots = True)

    # Head
        
    if plot_params['default']['plot_Head'] is True:

        f_path = derivs_path+'Head.csv'
            
        head_pos_derivs, _ = make_head_pos_plot_csv(f_path, verbose_plots=verbose_plots)
        # head_pos_derivs2 = make_head_pos_plot_mne(raw, head_pos, verbose_plots=verbose_plots)
        # head_pos_derivs += head_pos_derivs2
        head_derivs += head_pos_derivs

    QC_derivs={
        'MEG data quality analysis report': [],
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
    
    return QC_derivs
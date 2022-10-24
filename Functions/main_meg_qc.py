# This version of main allows to only choose mags, grads or both for 
# the entire pipeline in the beginning. Not for separate QC measures

#%%

import mne
import configparser
from PSD_meg_qc import PSD_QC 
from RMSE_meq_qc import MEG_QC_rmse
import ancpbids

from data_load_and_folders import load_meg_data, make_folders_meg, Epoch_meg

#%%
def initial_stuff(config, data_file):

    '''Here all the initial actions need to work with MEG data are done: 
    - load fif file and convert into raw,
    - create folders in BIDS compliant format,
    - crop the data if needed,
    - filter and downsample the data,
    - epoch the data.

    Args:
    sid (str): subject id

    Returns: 
    n_events (int): number of events(=number of epochs)
    df_epochs_mags (pd. Dataframe): data frame containing data for all epochs for mags 
    df_epochs_grads (pd. Dataframe): data frame containing data for all epochs for grads 
    epochs_mags (mne. Epochs): epochs as mne data structure for magnetometers
    epochs_grads (mne. Epochs): epochs as mne data structure for gradiometers 
    mags (list of tuples): magnetometer channel name + its index
    grads (list of tuples): gradiometer channel name + its index
    raw_bandpass(mne.raw): data only filtered, cropped (*)
    raw_bandpass_resamp(mne.raw): data filtered and resampled, cropped (*)
    raw_cropped(mne.io.Raw): data in raw format, cropped, not filtered, not resampled (*)
    raw(mne.io.Raw): original data in raw format, not cropped, not filtered, not resampled.
    (*): if duration was set to None - the data will not be cropped and these outputs 
    will return what is stated, but in origibal duration.
    '''

    # config = configparser.ConfigParser()
    # config.read('settings.ini')
    
    # default_section = config['DEFAULT']
    # dataset_path = default_section[""]
    # from ancpbids import BIDSLayout
    # layout = BIDSLayout(dataset_path)

    # list_of_fifs = layout.get(suffix='meg', extension='.fif', return_type='filename')

    # data file list of fifs[i]
 
    # data_file = default_section['data_file']

    
    raw, mags, grads=load_meg_data(data_file)

    #Create folders:
    #make_folders_meg(sid)

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
        raw_bandpass = raw_cropped.copy()
        raw_bandpass.filter(l_freq=l_freq, h_freq=h_freq, picks='meg', method=method, iir_params=None)

        #And downsample:
        raw_bandpass_resamp=raw_bandpass.copy()
        raw_bandpass_resamp.resample(sfreq=h_freq*5)
        #frequency to resample is 5 times higher than the maximum chosen frequency of the function

    else:
        raw_bandpass = raw_cropped.copy()
        raw_bandpass_resamp=raw_bandpass.copy()
        #OR maybe we dont need these 2 copies of data at all? Think how to get rid of them, 
        # because they are used later. Referencing might mess up things, check that.


    #Apply epoching: USE NON RESAMPLED DATA. Or should we resample after epoching? 
    # Since sampling freq is 1kHz and resampling is 500Hz, it s not that much of a win...

    epoching_section = config['Epoching']
    event_dur = epoching_section.getfloat('event_dur') 
    epoch_tmin = epoching_section.getfloat('epoch_tmin') 
    epoch_tmax = epoching_section.getfloat('epoch_tmax') 
    stim_channel = default_section['stim_channel'] 

    if len(stim_channel) == 0:
        picks_stim = mne.pick_types(raw.info, stim=True)
        stim_channel = []
        for ch in picks_stim:
            stim_channel.append(raw.info['chs'][ch]['ch_name'])
    
    n_events, df_epochs_mags, df_epochs_grads, epochs_mags, epochs_grads=Epoch_meg(data=raw_bandpass, 
        stim_channel=stim_channel, event_dur=event_dur, epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax)

    channels = {'mags': mags, 'grads': grads}

    print('HERE IN INIT', channels)

    df_epochs = {
    'grads': df_epochs_grads,
    'mags': df_epochs_mags}

    epochs_mg = {
    'grads': epochs_grads,
    'mags': epochs_mags}

    return df_epochs, epochs_mg, channels, raw_bandpass, raw_bandpass_resamp, raw_cropped, raw

#%%
# TRY:
# data_file = '../data/sub_HT05ND16/210811/mikado-1.fif/'
# config = configparser.ConfigParser()
# config.read('settings.ini')

# df_epochs, epochs_mg, channels, raw_bandpass, raw_bandpass_resamp, raw_cropped, raw = initial_stuff(config, data_file)

# print(df_epochs['mags']['epoch'].nunique())

#%%
def select_m_or_g(section: configparser.SectionProxy):
    """get do_for selection for given config: is the calculation of this particilatr quality measure done for mags, grads or both"""

    do_for = section['do_for']

    if do_for == 'mags':
        return ['mags']
    elif do_for == 'grads':
        return ['grads']
    elif do_for == 'both':
        return ['mags', 'grads']



#%%
def save_derivative_html(dataset_path, list_of_subs):

    config = configparser.ConfigParser()
    config.read('settings.ini')

    default_section = config['DEFAULT']
    m_or_g_chosen = select_m_or_g(default_section)

    if m_or_g_chosen != ['mags'] and m_or_g_chosen != ['grads'] and m_or_g_chosen != ['mags', 'grads']:
        raise ValueError('Type of channels to analise has to be chosen in setting.ini. Use "mags", "grads" or "both" as parameter of do_for. Otherwise the analysis can not be done.')

    dataset_path = default_section['data_directory']

    from ancpbids import BIDSLayout
    layout = BIDSLayout(dataset_path)
    schema = layout.schema

    #create derivative folder first!
    derivative = layout.dataset.create_derivative(name="Meg_QC")
    derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"

    list_of_subs = layout.get_subjects()
    #print(list_of_subs)

    for sid in [list_of_subs[0]]: #RUN OVER JUST 1 SUBJ
        subject = derivative.create_folder(type_=schema.Subject, name='sub-'+sid)

        list_of_fifs = layout.get(suffix='meg', extension='.fif', return_type='filename', subj=sid)
        #Devide here fifs by task, ses , run

        for data_file in list_of_fifs: 
            df_epochs, epochs_mg, channels, raw_bandpass, raw_bandpass_resamp, raw_cropped, raw = initial_stuff(config, data_file)

            _, list_of_figures = MEG_QC_rmse(sid, config, channels, df_epochs, raw_bandpass_resamp, m_or_g_chosen)

            # _, list_of_figures, list_of_fig_descriptions = PSD_QC(sid, config, channels, raw_bandpass_resamp, m_or_g_chosen)

            # MEG_peaks_manual()

            # MEG_peaks_auto()

            # MEG_EOG()

            # MEG_ECG()

            # MEG_head_movements()

            # MEG_muscle()

            list_of_fig_descriptions = ['some_fig1', 'some_fig2', 'some_fig3', 'some_fig4']

            for deriv_n, _ in enumerate(list_of_figures):
                meg_artifact = subject.create_artifact() #shell. empty derivative
                meg_artifact.add_entity('desc', list_of_fig_descriptions[deriv_n]) #file name
                # HERE ADD FILE DESCRIPTION: GET IT FROM THE FUNCTION WHICH CREATED IT LIKE PSD OF MAGNETOMETERS, ETC..
                #meg_artifact.add_entity('task', task_label)
                meg_artifact.suffix = 'meg'
                meg_artifact.extension = ".html"
                meg_artifact.content = lambda file_path: list_of_figures[deriv_n].write_html(file_path)
    
    layout.write_derivative(derivative) #maybe put intide the loop if cant have so much in memory?



#%%
config = configparser.ConfigParser()
config.read('settings.ini')

direct = config['DEFAULT']['data_directory']
dataset_path = ancpbids.utils.fetch_dataset(direct)

from ancpbids import BIDSLayout
layout = BIDSLayout(dataset_path)

# list_of_fifs = layout.get(suffix='meg', extension='.fif', return_type='filename')

list_of_subs = layout.get_subjects()

# list_of_entities = layout.get_entities()
# print(list_of_entities)

save_derivative_html(dataset_path, list_of_subs)


#%%
# Try on one subject:
subj = '001'
data_file0 = layout.get(suffix='meg', extension='.fif', return_type='filename', subj=subj)[0]
print('Try now ', data_file0)

n_events, df_epochs_mags, df_epochs_grads, epochs_channels_mags, epochs_channels_grads, channels, filtered_d, filtered_d_resamp, raw_cropped, raw = initial_stuff(config, data_file=data_file0)

print('NOW START THE QC stuff)')

#list_of_figures_PSD, list_of_fig_descriptions_PSD = MEG_QC_measures(subj, config, n_events, df_epochs_mags, df_epochs_grads, epochs_channels_mags, epochs_channels_grads, channels, filtered_d, filtered_d_resamp, raw_cropped, raw)

#%%

from PSD_meg_qc import Freq_Spectrum_meg

print(len(channels['mags']))
freqs, psds, fig_path_psd, fig_psd, fig_desc = Freq_Spectrum_meg(data=filtered_d_resamp, m_or_g = 'mags', sid=subj, freq_min=0.5, freq_max=100, n_fft=1000, n_per_seg=1000, freq_tmin=None, freq_tmax=None, ch_names=channels['mags'])

#%%   Run the pipleine over subjects
# We actually cant loop just over sids, cos each need a new data file. Add more dats files in config or?

# for sid in sid_list:
#     list_of_figure_paths_RMSE, list_of_figures_RMSE = MEG_QC_measures(sid, config)

#print(list_of_figure_paths_RMSE)

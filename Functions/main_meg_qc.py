# This version of main allows to only choose mags, grads or both for 
# the entire pipeline in the beginning. Not for separate QC measures

#%%
import mne
import configparser
from PSD_meg_qc import PSD_QC 
from RMSE_meq_qc import MEG_QC_rmse
import ancpbids


#%%
def initial_stuff(sid: str):

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

    config = configparser.ConfigParser()
    config.read('settings.ini')

    default_section = config['DEFAULT']
    data_file = default_section['data_file']

    from data_load_and_folders import load_meg_data, make_folders_meg, Epoch_meg

    raw, mags, grads=load_meg_data(data_file)

    #Create folders:
    make_folders_meg(sid)

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

    return n_events, df_epochs_mags, df_epochs_grads, epochs_mags, epochs_grads, channels, raw_bandpass, raw_bandpass_resamp, raw_cropped, raw


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
def MEG_QC_measures(sid, config):

    """This function will call all the MEG QC functions."""

    n_events, df_epochs_mags, df_epochs_grads, epochs_channels_mags, epochs_channels_grads, channels, filtered_d, filtered_d_resamp, raw_cropped, raw = initial_stuff(sid)

    m_or_g_title = {
        'grads': 'Gradiometers',
        'mags': 'Magnetometers'
    }
    df_epochs = {
        'grads': df_epochs_grads,
        'mags': df_epochs_mags
    }
    epochs_channels = {
        'grads': epochs_channels_grads,
        'mags': epochs_channels_mags
    }

    default_section = config['DEFAULT']
    m_or_g_chosen = select_m_or_g(default_section)

    if m_or_g_chosen != ['mags'] and m_or_g_chosen != ['grads'] and m_or_g_chosen != ['mags', 'grads']:
        raise ValueError('Type of channels to analise has to be chosen in setting.ini. Use "mags", "grads" or "both" as parameter of do_for. Otherwise the analysis can not be done.')

    
    list_of_figure_paths_RMSE, list_of_figures_RMSE = MEG_QC_rmse(sid, config, channels, m_or_g_title, df_epochs, filtered_d_resamp, n_events, m_or_g_chosen)

    # list_of_figure_paths_PSD = PSD_QC(sid, channels, filtered_d_resamp, m_or_g_chosen, config)

    # MEG_peaks_manual()

    # MEG_peaks_auto()

    # MEG_EOG()

    # MEG_ECG()

    # MEG_head_movements()

    # MEG_muscle()

    return list_of_figure_paths_RMSE, list_of_figures_RMSE


import ancpbids
#from ancpbidsapps.app import App

def save_figs_html(dataset_path, list_of_figures, sid_list):

    layout = ancpbids.BIDSLayout(dataset_path)
    schema = layout.schema

    derivative = layout.dataset.create_derivative(name="Megqc_measurements")
    derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"

    for midx, model in enumerate(sid_list):
        subject = derivative.create_folder(type_=schema.Subject, name=sid_list[midx])

        # create the HTML figure
        # model.fit(imgs, events, confounds) #DO I NEED TO CREATE SMTH HERE?

        meg_artifact = subject.create_artifact()
        meg_artifact.add_entity('desc', "qc_measurements")
        #meg_artifact.add_entity('task', task_label)
        meg_artifact.suffix = 'rmse'
        meg_artifact.extension = ".html"
        meg_artifact.content = lambda file_path: figr.write_html(file_path)
    



def create_all_reports():

    return


#%%
config = configparser.ConfigParser()
config.read('settings.ini')
sids = config['DEFAULT']['sid']
sid_list = list(sids.split(','))

direct = config['DEFAULT']['data_directory']
dataset_path = ancpbids.utils.fetch_dataset(direct)

from ancpbids import BIDSLayout
layout = BIDSLayout(dataset_path)

list_of_fifs = layout.get(suffix='meg', extension='.fif', return_type='filename')

#print(list_of_fifs)


#%%   Run the pipleine over subjects
# We actually cant loop just over sids, cos each need a new data file. Add more dats files in config or?

# for sid in sid_list:
#     list_of_figure_paths_RMSE, list_of_figures_RMSE = MEG_QC_measures(sid, config)

#print(list_of_figure_paths_RMSE)

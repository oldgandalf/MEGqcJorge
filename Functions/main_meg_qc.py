# This version of main allows to only choose mags, grads or both for 
# the entire pipeline in the beginning. Not for separate QC measures

#%%

import mne
import configparser
import ancpbids
from ancpbids import BIDSLayout

from data_load_and_folders import load_meg_data, make_folders_meg, Epoch_meg
from RMSE_meq_qc import MEG_QC_rmse
from PSD_meg_qc import PSD_QC 


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

    raw = mne.io.read_raw_fif(data_file)

    mag_ch_names = raw.copy().pick_types(meg='mag').ch_names if 'mag' in raw else None
    grad_ch_names = raw.copy().pick_types(meg='grad').ch_names if 'grad' in raw else None
    channels = {'mags': mag_ch_names, 'grads': grad_ch_names}

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
    
    df_epochs_mags, df_epochs_grads, epochs_mags, epochs_grads=Epoch_meg(data=raw_bandpass, 
        stim_channel=stim_channel, event_dur=event_dur, epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax)

    df_epochs = {
    'grads': df_epochs_grads,
    'mags': df_epochs_mags}

    epochs_mg = {
    'grads': epochs_grads,
    'mags': epochs_mags}

    return df_epochs, epochs_mg, channels, raw_bandpass, raw_bandpass_resamp, raw_cropped, raw


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


def sanity_check(m_or_g_chosen, channels):
    if m_or_g_chosen != ['mags'] and m_or_g_chosen != ['grads'] and m_or_g_chosen != ['mags', 'grads']:
        m_or_g_chosen = []
    if channels['mags'] is None and 'mags' in m_or_g_chosen:
        print('There are no magnetometers in this data set: check parameter do_for in config file. Analysis will be done only for gradiometers.')
        m_or_g_chosen.remove('mags')
    elif channels['grads'] is None and 'grads' in m_or_g_chosen:
        print('There are no gradiometers in this data set: check parameter do_for in config file. Analysis will be done only for magnetometers.')
        m_or_g_chosen.remove('grads')
    elif channels['mags'] is None and channels['grads'] is None:
        print ('There are no magnetometers or gradiometers in this data set. Analysis will not be done.')
        m_or_g_chosen = []
    return m_or_g_chosen


#%%
def make_derivative_html():

    config = configparser.ConfigParser()
    config.read('settings.ini')

    default_section = config['DEFAULT']
    m_or_g_chosen = select_m_or_g(default_section)

    dataset_path = default_section['data_directory']

    layout = BIDSLayout(dataset_path)
    schema = layout.schema

    #create derivative folder first!
    derivative = layout.dataset.create_derivative(name="Meg_QC")
    derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"

    list_of_subs = layout.get_subjects()
    #print(list_of_subs)

    for sid in [list_of_subs[0]]: #RUN OVER JUST 1 SUBJ
    #for sid in list_of_subs: 

        subject_folder = derivative.create_folder(type_=schema.Subject, name='sub-'+sid)

        list_of_fifs = layout.get(suffix='meg', extension='.fif', return_type='filename', subj=sid)
        #Devide here fifs by task, ses , run

        for data_file in list_of_fifs: 
            df_epochs, epochs_mg, channels, raw_bandpass, raw_bandpass_resamp, raw_cropped, raw = initial_stuff(config, data_file)

            m_or_g_chosen = sanity_check(m_or_g_chosen, channels)
            if len(m_or_g_chosen) == 0: 
                raise ValueError('No channels to analyze. Check presence of mags and grads in your data set and parameter do_for in settings.')

            # list_of_figures, _, list_of_fig_descriptions = MEG_QC_rmse(sid, config, channels, df_epochs, raw_bandpass_resamp, m_or_g_chosen)

            list_of_figures, _, list_of_fig_descriptions = PSD_QC(sid, config, channels, raw_bandpass_resamp, m_or_g_chosen)

            # MEG_peaks_manual()

            # MEG_peaks_auto()

            # MEG_EOG()

            # MEG_ECG()

            # MEG_head_movements()

            # MEG_muscle()

            for i in range(0, len(list_of_figures)):
                meg_artifact = subject_folder.create_artifact() #shell. empty derivative
                meg_artifact.add_entity('desc', list_of_fig_descriptions[i]) #file name
                #meg_artifact.add_entity('task', task_label)
                meg_artifact.suffix = 'meg'
                meg_artifact.extension = '.html'

                print('FIGURE!', list_of_figures[i])
                meg_artifact.content = lambda file_path: list_of_figures[i].write_html(file_path)
    
                layout.write_derivative(derivative) #maybe put istide the loop if cant have so much in memory?


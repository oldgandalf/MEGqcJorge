
#%%
from math import sqrt
import os
import numpy as np
import mne

#sample_data_folder = mne.datasets.sample.data_path()
kath_raw_file2 = "/Users/jenya/Documents/Oldenburg and university/Job Uni Rieger lab/Katharinas_Data/sub_HT05ND16/210811/mikado-1.fif"
#kath_raw_file = os.path.join('sub_HT05ND16', '210811', 'mikado-1.fif')
##print(kath_raw_file)
print(kath_raw_file2)                                   
raw = mne.io.read_raw_fif(kath_raw_file2)
#raw.crop(0, 60).load_data()  # just use a fraction of data for speed here


print(raw)
print(raw.info)

#%%
raw.plot(block=True, duration=5, n_channels=30)

#%%
original_raw = raw.copy()
#raw.apply_hilbert()
#print(f'original data type was {original_raw.get_data().dtype}, after '
#      f'apply_hilbert the data type changed to {raw.get_data().dtype}.')

print(f'Original data had {original_raw.info["nchan"]} channels.')

#To pick eg channels:
#raw.pick('eeg')  # selects only the EEG channels
#print(f'after picking, it has {raw.info["nchan"]} channels.')

# %%
events = mne.find_events(original_raw) #, stim_channel='STI 101')
print(events[:5])  # show the first 5

#014 and 101 are not the right stim channels. which one? gives traceback


# %%
picks = mne.pick_types(original_raw.info, meg=True, exclude='bads')  
#t_idx = original_raw.time_as_index([10., 20.])  
#data, times = original_raw[picks, t_idx[0]:t_idx[1]]  
data, times = original_raw[picks, :]  

#%%
#Extract data from 1 channel:
#print(data[0])
chan1=data[0]

#print(len(data[0]))
# %%
# STD or MRSE of each channel and get rid of channels which got too high.
#So: iterate through all meg channels. calc MRSE of each. calc meand and std of all stds.
# find highes? look at it then decide if it s good channel to exclude.

std_data=np.std(data, axis=1)
print(std_data)

print(np.mean(std_data))

print(max(std_data))

print(min(std_data))
# %%
largest_std= np.where(std_data == max(std_data))
print(largest_std[0])

# %%

layout = mne.channels.read_layout('Vectorview-mag')
layout.plot()
raw.plot_psd_topo(tmax=30., fmin=5., fmax=60., n_fft=1024, layout=layout)

# %%

#picks2 = mne.pick_channels_regexp(raw.ch_names, regexp='EEG 05.')
#raw.plot(order=picks, n_channels=len(picks))

raw.plot(['MEG 050', 'MEG 051'])
# %%

original_bads = deepcopy(raw.info['bads'])
raw.info['bads'].append('EEG 050')               # add a single channel
raw.info['bads'].extend(['EEG 051', 'EEG 052'])  # add a list of channels
bad_chan = raw.info['bads'].pop(-1)  # remove the last entry in the list
raw.info['bads'] = original_bads     # change the whole list at once

#%% Look what channels (projectors) we got:
ssp_projectors = original_raw.info['projs']
original_raw.del_proj()


# %%
ecg_epochs = mne.preprocessing.create_ecg_epochs(original_raw)

#%%
ecg_epochs.plot_image(combine='mean', show=True)
# %%

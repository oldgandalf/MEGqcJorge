#%% Imports:
from math import sqrt
import os
import numpy as np
import mne
import matplotlib #doesnt it import through mne already?
from copy import deepcopy

#Open data:
#sample_data_folder = mne.datasets.sample.data_path()
#kath_raw_file2 = "/Users/jenya/Documents/Oldenburg and university/Job Uni Rieger lab/Katharinas_Data/sub_HT05ND16/210811/mikado-1.fif"
kath_raw_file = os.path.join('Katharinas_Data','sub_HT05ND16', '210811', 'mikado-1.fif')
print(kath_raw_file)
#print(kath_raw_file2)                                   
raw = mne.io.read_raw_fif(kath_raw_file)
#raw.crop(0, 60).load_data()  # just use a fraction of data for speed here

#Print info about the data:
print(raw)
print(raw.info)

#%% Plot 5 sec of the first 30 channels.
# HOW TO PLOT PARTICULAR RANGE OF CHANNELS?
# HOW DO I KNOW WHAT CHANNELS ARE THERE AT ALL? WHICH CHANNEL TYPES HAVE WHAT NUMBER?
# DO THEY KEEP ORDER OR ARE THEY RANDOM AS TUPLES?
raw.plot(block=True, duration=5, n_channels=30)

#%% copy into separate data variable for further manipulations, to not danage the original data:
original_raw = raw.copy()
#raw.apply_hilbert()
#print(f'original data type was {original_raw.get_data().dtype}, after '
#      f'apply_hilbert the data type changed to {raw.get_data().dtype}.')

print(f'Original data had {original_raw.info["nchan"]} channels.')


# %% Pick particular channel types to calculater RMSE or STD of them.
# WHICH channels do we exactly calculate? theer are 2 types of meg channels 
# (MAGNITOMETERS AND GRADIOMETERS). which?
# HOW TO DESTINGUISH THESE 2 TYPES, WHAT ARE THEIR NAMES TO CALL IN CODE?
picks = mne.pick_types(original_raw.info, meg=True, exclude='bads')  
data, times = original_raw[picks, :]  
#data now contains all meg channels, it s 2d.

# %% STD or MRSE of each channel and get rid of channels which got it too high.
# find highest? look at it then decide if it s a channel to exclude?

#STD:
std_data=np.std(data, axis=1) #calculate std of all channels (along second dimantion)
#print(std_data)

print(np.mean(std_data)) #average std
print(max(std_data)) #highest std
print(min(std_data)) #lowest std.

#%% RMSE:
# https://stackoverflow.com/questions/17197492/is-there-a-library-function-for-root-mean-square-error-rmse-in-python

from sklearn.metrics import mean_squared_error

y_actual=data
y_predicted=data.mean(axis=1)
#yeah i know i dont need to rename it, just easier for me this way

error_vec = np.zeros(len(y_predicted))

for i in range(len(y_predicted)):
    #print(i)
    #print(y_actual[i, :])
    #print(y_predicted[i])
    y_predicted_vec=np.ones(len(y_actual[0]))*y_predicted[i]
    error_vec[i] = mean_squared_error(y_actual[i, :], y_predicted_vec, squared=False)


# %% Pick channel with largest STD (RMSE)?
# HOW DO WE DECIDE WHICH CHANNELS TO PICK? ANY PARTICULSR LIMIT?
largest_std= np.where(std_data == max(std_data))
print(largest_std[0])

#%% This is how we plot particular channels. but how to plot range? 
# how do I know which channel names belong to the bads i just found?
raw.plot(['MEG 050', 'MEG 051'])

# %% ADD CHANNELS TO BADS. OR DO WE FILER THEM OR...?


# First - how do i now recognize which of my picked bad channels have what meg names? do they keep order?


original_bads = deepcopy(raw.info['bads'])
raw.info['bads'].append('EEG 050')               # add a single channel
raw.info['bads'].extend(['EEG 051', 'EEG 052'])  # add a list of channels
bad_chan = raw.info['bads'].pop(-1)  # remove the last entry in the list
raw.info['bads'] = original_bads     # change the whole list at once

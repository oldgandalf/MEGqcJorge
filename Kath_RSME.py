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
raw

#%% How to look up info and other usefull things. 
# CAN SKIP THIS WHOLE CELL, DOESNT AFFECT FURTHER STUFF

#name of particular channel:
raw.info['chs'][14]['ch_name']
#See all channel names:
print(raw.info['ch_names'])
# see all avalinle info keys
raw.info.keys()

#See unit of channels number 15: (cos indexing from 0)
raw.info['chs'][14]['unit']

# Plot 5 sec of the first 30 channels.
raw.plot(block=True, duration=5, n_channels=30)
# HOW TO PLOT PARTICULAR RANGE OF CHANNELS?

# copy into separate data variable for further manipulations, to not danage the original data:
original_raw = raw.copy()

print(f'Original data had {original_raw.info["nchan"]} channels.')


#%% Find magnetometers and gradiometers:
# unit t - magenetometer. unit M_T - gradiometer. (in this set name will end with 1 for magnet, 
# with 2and3 for grad.)

mags = []
grads=[]

for i, chs in enumerate(raw.info['chs']):

    if str(chs['unit']).endswith('UNIT_T)'):
        mags.append((chs['ch_name'], i))
    elif str(chs['unit']).endswith('UNIT_T_M)'):
        grads.append((chs['ch_name'], i))

print('Magnetometers: ', mags)
print('Gradiometers: ', grads)

#%%other way (shorter):
mags = [(chs['ch_name'], i) for i, chs in enumerate(raw.info['chs']) if str(chs['unit']).endswith('UNIT_T)')]
print('Magnetometers: ', mags)

grads = [(chs['ch_name'], i) for i, chs in enumerate(raw.info['chs']) if str(chs['unit']).endswith('UNIT_T_M)')]
print('Gradiometers: ', grads)


# %% Pick all meg channels type to calculater RMSE or STD of them.
# Commented as this might be usefull later, but not at the moment.
#This method doesnt separate magnetometers and gradiometers, just meg channels in total.

#picks = mne.pick_types(original_raw.info, meg=True, exclude='bads')  
#data, times = original_raw[picks, :]  

#data now contains all meg channels (magnetometers_gradiometers mixed), it s 2d.
#times is just time vector, same for any tiype of channel.

#%% Make a vector of indices of all magnetormeters and all gradiomerters 
# - their indexes in original data.
selected_mags = [item[1] for item in mags]
selected_grads = [item[1] for item in grads]
data_mags, times = raw[selected_mags, :]  
data_grads, times = raw[selected_grads, :]  

# %% Calculate STD or RMSE of each channel

#Time how long it takes to calculate STD or RMSE:
import time
t0_std = time.time()

#STD:
std_mags=np.std(data_mags, axis=1) #calculate std of all magnetometers (along second dimantion)
std_grads=np.std(data_grads, axis=1) #calculate std of all gradiometers (along second dimantion)

t1_std = time.time()
total_time_std = t1_std-t0_std

print('Mean of magnetometers data: ', np.mean(std_mags)) #average std
print('Max of magnetometers data: ',max(std_mags)) #highest std
print('Min of magnetometers data: ',min(std_mags)) #lowest std.
print('Mean of gradiometers data: ', np.mean(std_grads)) #average std
print('Max of gradiometers data: ',max(std_grads)) #highest std
print('Min of gradiometers data: ',min(std_grads)) #lowest std.

#%% RMSE:
# https://stackoverflow.com/questions/17197492/is-there-a-library-function-for-root-mean-square-error-rmse-in-python

t0_rmse = time.time()

from sklearn.metrics import mean_squared_error

#Magnitometers:
y_actual_mags=data_mags
y_predicted_mags=data_mags.mean(axis=1)
#yeah i know i dont need to rename it, just easier for me this way to deal with RMSE concept

error_vec_mags = np.zeros(len(y_predicted_mags))

for i in range(len(y_predicted_mags)):
    #print(i)
    #print(y_actual[i, :])
    #print(y_predicted[i])
    y_predicted_vec_mags=np.ones(len(y_actual_mags[0]))*y_predicted_mags[i]
    error_vec_mags[i] = mean_squared_error(y_actual_mags[i, :], y_predicted_vec_mags, squared=False)

#Gradiometers:
y_actual_grads=data_grads
y_predicted_grads=data_grads.mean(axis=1)
#yeah i know i dont need to rename it, just easier for me this way to deal with RMSE concept

error_vec_grads = np.zeros(len(y_predicted_grads))

for i in range(len(y_predicted_grads)):
    #print(i)
    #print(y_actual[i, :])
    #print(y_predicted[i])
    y_predicted_vec_grads=np.ones(len(y_actual_grads[0]))*y_predicted_grads[i]
    error_vec_grads[i] = mean_squared_error(y_actual_grads[i, :], y_predicted_vec_grads, squared=False)

t1_rmse = time.time()
total_time_rmse = t1_rmse-t0_rmse

print('Time to calculate std: ', total_time_std)
print('Time to calculate rmse: ', total_time_rmse)

#STD CALCULATION IS MUCH LESS COdE BUT TAKES USUALLY LONGER THAN RMSE

# %% Pick channel with largest STD (RMSE)?
# HOW DO WE DECIDE WHICH CHANNELS TO PICK? ANY PARTICULSR LIMIT?
# If not - just display here channels with largest STD for user to decide
largest_std_mags= np.where(std_mags == max(std_mags))
largest_std_grads= np.where(std_grads == max(std_grads))

mag_channel_largest_std=mags[largest_std_mags[0][0]]
grad_channel_largest_std=grads[largest_std_grads[0][0]]
print('Magnetometer with largest STD: ', mag_channel_largest_std[0])
print('Gradiometer with largest STD: ', grad_channel_largest_std[0])

#%% Now want to see these 2 channels. 
#chans = ['MEG2311', 'MEG1542']
noisy_chans = [mag_channel_largest_std[0], grad_channel_largest_std[0]]
chan_idxs = [raw.ch_names.index(ch) for ch in noisy_chans]
#original_raw.plot(order=chan_idxs, start=12, duration=4)
raw.plot(order=chan_idxs, start=12, duration=4) #plot here only a part of channel.

# %% ADD CHANNELS TO BADS. 
# In case we want to. This is unchanged example from tutorial, dont know if we need to do that:

original_bads = deepcopy(raw.info['bads'])
raw.info['bads'].append('EEG 050')               # add a single channel
raw.info['bads'].extend(['EEG 051', 'EEG 052'])  # add a list of channels
bad_chan = raw.info['bads'].pop(-1)  # remove the last entry in the list
raw.info['bads'] = original_bads     # change the whole list at once


#%% CELL DOESNT WORK BECAUSE EPOCHING DOESNT WORK. HOW TO FIX?
# 
# Now detect events to then epoch data - from here:
#https://mne.tools/stable/auto_tutorials/intro/10_overview.html#sphx-glr-auto-tutorials-intro-10-overview-py

events = mne.find_events(raw, stim_channel='STI101')
events = mne.find_events(raw, stim_channel='STI101', min_duration=1/raw.info['sfreq']) 
#STI101 is stim data in this file. might be different name in another!

#HM.. doesnt work. says smth is wrong with event duration. tried to fix, but no so far..

print(events[:5])  # show the first 5 events

#create even dictionary if needed:
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'smiley': 5, 'buttonpress': 32}

#This here would allow to reject particular epochs in data. Values are copid from tutorial.
#We would need to first epoch our own data, calcukate std over different epochs and channels 
# and then decide whic values to put here.

reject_criteria = dict(mag=4000e-15,     # 4000 fT
                       grad=4000e-13,    # 4000 fT/cm
                       eeg=150e-6,       # 150 µV
                       eog=250e-6)       # 250 µV


epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.5,
                    reject=reject_criteria, preload=True)

#once these work - look for further steps in link above.                   

# %%

#%%
from math import sqrt
import os
import numpy as np
import mne
import matplotlib

#sample_data_folder = mne.datasets.sample.data_path()
#kath_raw_file2 = "/Users/jenya/Documents/Oldenburg and university/Job Uni Rieger lab/Katharinas_Data/sub_HT05ND16/210811/mikado-1.fif"
kath_raw_file = os.path.join('Katharinas_Data','sub_HT05ND16', '210811', 'mikado-1.fif')
##print(kath_raw_file)
print(kath_raw_file)                                   
raw = mne.io.read_raw_fif(kath_raw_file)
#raw.crop(0, 60).load_data()  # just use a fraction of data for speed here


print(raw)
print(raw.info)

#%%
#raw.plot(block=True, duration=5, n_channels=30)

#%%
original_raw = raw.copy()
#raw.apply_hilbert()
#print(f'original data type was {original_raw.get_data().dtype}, after '
#      f'apply_hilbert the data type changed to {raw.get_data().dtype}.')

#print(f'Original data had {original_raw.info["nchan"]} channels.')

#%% Look what channels (projectors) we got:
ssp_projectors = original_raw.info['projs']
original_raw.del_proj()


# %%
#WE GOT NOW SEPARATE ECG CHANNEL IN THIS SET. IT RECONSTRUCTS FROM MAGNETOMETERS

ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
ecg_epochs.plot_image(combine='mean')
#%%
#ecg_epochs.plot_image(combine='mean', show=True)

#why no plotting ah??
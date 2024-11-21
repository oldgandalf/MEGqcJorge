import ancpbids
import os 
import sys
import subprocess
import pandas as pd

# We load the dataset using the BIDSlayout
#dataset_path= '/archive/Evgeniia_data/camcan_meg/camcan1409/cc700/meg/pipeline/release005/BIDSsep/passive'
dataset_path= '/data/areer/MEG_QC_stuff/data/openneuro/ds003483'
dataset = ancpbids.load_dataset(dataset_path)

print('loading dataset completed successfully')

list_of_PSDtables = dataset.query(suffix='meg', desc='STDs', scope="derivatives", return_type='filename')
print('list of PSD tables has been created')

df = pd.DataFrame(list_of_PSDtables)
df.to_csv('PSDtable_list_test.csv', index=False)

for elem in list_of_PSDtables:
    dftemp = pd.read_csv(elem, sep='\t')
    print(dftemp[['STD all']].mean())

    

#create a list of all csv, which contain STD information

import sys
import os
from meg_qc.calculation.meg_qc_pipeline import make_derivative_meg_qc
import argparse

#dataset_path_parser = argparse.ArgumentParser(description= "parser for BIDS dataset path")
#dataset_path_parser.add_argument("inputdata", type=str)
def hello_world():
    print("Hello World")
#args = dataset_path_parser.parse_args()
# Get the absolute path of the parent directory of the current script
parent_dir = os.path.dirname(os.getcwd())
print(parent_dir)

sys.path.append(parent_dir)

config_file_path = parent_dir+'/meg_qc/settings/settings.ini' 
internal_config_file_path=parent_dir+'/meg_qc/settings/settings_internal.ini' # internal settings in in
#raw, raw_cropped_filtered_resampled, QC_derivs, QC_simple, df_head_pos, head_pos, scores_muscle_all1, scores_muscle_all2, scores_muscle_all3, raw1, raw2, raw3, avg_ecg, avg_eog = make_derivative_meg_qc(config_file_path, internal_config_file_path)

#data_directory = '/archive/Evgeniia_data/camcan_meg/camcan1409/cc700/meg/pipeline/release005/BIDSsep/smt'
data_directory = '/data/areer/MEG_QC_stuff/data/openneuro/ds003483'
#data_directory = args.inputdata

make_derivative_meg_qc(config_file_path, internal_config_file_path, data_directory)


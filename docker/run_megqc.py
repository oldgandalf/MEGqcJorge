import sys
import os
from meg_qc.calculation.meg_qc_pipeline import make_derivative_meg_qc

config_file_path = '/home/areer1/Projects/MEGqc_myfork/MEGqc/meg_qc/settings/settings.ini' 
internal_config_file_path='/home/areer1/Projects/MEGqc_myfork/MEGqc/meg_qc/settings/settings_internal.ini' # internal settings in in
#raw, raw_cropped_filtered_resampled, QC_derivs, QC_simple, df_head_pos, head_pos, scores_muscle_all1, scores_muscle_all2, scores_muscle_all3, raw1, raw2, raw3, avg_ecg, avg_eog = make_derivative_meg_qc(config_file_path, internal_config_file_path)

make_derivative_meg_qc(config_file_path, internal_config_file_path)


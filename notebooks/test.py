import sys
import os

# Get the absolute path of the parent directory of the current script
parent_dir = os.path.dirname(os.getcwd())

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from meg_qc.calculation.meg_qc_pipeline import make_derivative_meg_qc

config_file_path = parent_dir+'/meg_qc/settings/settings.ini' 
internal_config_file_path=parent_dir+'/meg_qc/settings/settings_internal.ini' # internal settings in in

make_derivative_meg_qc(config_file_path, internal_config_file_path)
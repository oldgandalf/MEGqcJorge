
from meg_qc.calculation.meg_qc_pipeline import make_derivative_meg_qc

config_file_path = '/Users/jenya/Local Storage/Job Uni Rieger lab/MEG QC code/meg_qc/settings/settings.ini' 
internal_config_file_path='/Users/jenya/Local Storage/Job Uni Rieger lab/MEG QC code/meg_qc/settings/settings_internal.ini' # internal settings in in
make_derivative_meg_qc(config_file_path, internal_config_file_path)
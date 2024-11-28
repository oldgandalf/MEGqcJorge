import argparse
import os
import sys
from meg_qc.calculation.meg_qc_pipeline import make_derivative_meg_qc
from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc

def hello_world():
    dataset_path_parser = argparse.ArgumentParser(description= "parser for string to print")
    dataset_path_parser.add_argument("--inputstring", type=str, required=True, help="path to the root of your BIDS MEG dataset")
    args=dataset_path_parser.parse_args()
    print(args.inputstring)


def run_megqc():
    dataset_path_parser = argparse.ArgumentParser(description= "parser for MEGqc: --inputdata(mandatory) path/to/your/BIDSds --config path/to/config  if None default parameters are used)")
    dataset_path_parser.add_argument("--inputdata", type=str, required=True, help="path to the root of your BIDS MEG dataset")
    dataset_path_parser.add_argument("--config", type=str, required=False, help="path to config file")
    args=dataset_path_parser.parse_args()

    path_to_megqc_installation= os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))

    #parent_dir = os.path.dirname(os.getcwd())
    #print(parent_dir)
    print(path_to_megqc_installation)

    data_directory = args.inputdata
    print(data_directory)

    if args.config == None:
        config_file_path = path_to_megqc_installation + '/settings/settings.ini' 
    else:
        config_file_path = args.config

    
    internal_config_file_path=path_to_megqc_installation + '/settings/settings_internal.ini'

    make_derivative_meg_qc(config_file_path, internal_config_file_path, data_directory)

    user_input = input('Do you want to run the MEGqc plotting module on the MEGqc results? (y/n): ').lower().strip() == 'y'

    if user_input == True:
        make_plots_meg_qc(data_directory)
    else:
        print('MEGqc results can be found in' + data_directory +'/derivatives/MEGqc/calculation')


def get_config():
    import shutil

    path_to_megqc_installation= os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
    config_file_path =path_to_megqc_installation +'/settings/settings.ini'
    
    shutil.copy(config_file_path, destination_directory)



def get_plots():




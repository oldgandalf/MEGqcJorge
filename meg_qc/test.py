import argparse
import os
import sys
import shutil

def hello_world():
    dataset_path_parser = argparse.ArgumentParser(description= "parser for string to print")
    dataset_path_parser.add_argument("--inputstring", type=str, required=True, help="path to the root of your BIDS MEG dataset")
    args=dataset_path_parser.parse_args()
    print(args.inputstring)


def run_megqc():
    from meg_qc.calculation.meg_qc_pipeline import make_derivative_meg_qc
    from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc

    dataset_path_parser = argparse.ArgumentParser(description= "parser for MEGqc: --inputdata(mandatory) path/to/your/BIDSds --config path/to/config  if None default parameters are used)")
    dataset_path_parser.add_argument("--inputdata", type=str, required=True, help="path to the root of your BIDS MEG dataset")
    dataset_path_parser.add_argument("--config", type=str, required=False, help="path to config file")
    args=dataset_path_parser.parse_args()


    # get the path of the currently executed file (file is on the base level of the package). 
    # From there we know that the default settings and internal settings will be stored in settings/...
    path_to_megqc_installation= os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
    relative_path_to_internal_config = "settings/settings_internal.ini"
    relative_path_to_config = "settings/settings.ini"
    # Normalize both relative paths (optional but we stick to it for consistency)
    relative_path_to_internal_config = os.path.normpath(relative_path_to_internal_config)
    relative_path_to_config = os.path.normpath(relative_path_to_config)
    #join path 
    internal_config_file_path=os.path.join(path_to_megqc_installation,relative_path_to_internal_config)

    #parent_dir = os.path.dirname(os.getcwd())
    #print(parent_dir)
    print(path_to_megqc_installation)

    data_directory = args.inputdata
    print(data_directory)

    if args.config == None:
        url_megqc_book = 'https://aaronreer.github.io/docker_workshop_setup/settings_explanation.html'
        text = 'The settings explanation section of our MEGqc User Jupyterbook'

        print('You called the MEGqc pipeline without the optional \n \n --config <path/to/custom/config>  argument. \n \n MEGqc will proceed with the default parameter settings. Detailed information on the user parameters in MEGqc and their default values can be found in here: \n \n')
        print(f"\033]8;;{url_megqc_book}\033\\{text}\033]8;;\033\\")
        print("\n \n")
        user_input = input('Do you want to proceed with the default settings? (y/n): ').lower().strip() == 'y' 
        if user_input == True:
            config_file_path = os.path.join(path_to_megqc_installation,relative_path_to_config)
        else:
            print("Use the \n \n get-megqc-config --target_directory <path/to/your/target/directory> \n \n 2command line prompt. This will copy the config file to a target destination on your machine.YOu can edit this file, e.g adjust all user parameters to your needs, and run the pipeline command again \n run-megqc \n with the \n --config parameter \n providing a path to your customized config file") 

    else:
        config_file_path = args.config


    make_derivative_meg_qc(config_file_path, internal_config_file_path, data_directory)

    print('MEGqc has completed the calculation of metrics. Results can be found in' + data_directory +'/derivatives/MEGqc/calculation')

    user_input = input('Do you want to run the MEGqc plotting module on the MEGqc results? (y/n): ').lower().strip() == 'y'

    if user_input == True:
        make_plots_meg_qc(data_directory)
        return
    else:
        return

def get_config():
    
    target_directory_parser = argparse.ArgumentParser(description= "parser for MEGqc get_config: --target_directory(mandatory) path/to/directory/you/want/the/config/to/be/stored)")
    target_directory_parser.add_argument("--target_directory", type=str, required=True, help="path to the root of your BIDS MEG dataset")
    args=target_directory_parser.parse_args()
    destination_directory = args.target_directory + '/settings.ini'
    print(destination_directory)

    path_to_megqc_installation= os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
    print(path_to_megqc_installation)
    config_file_path =path_to_megqc_installation +'/settings/settings.ini'
    print(config_file_path)
    
    shutil.copy(config_file_path, destination_directory)
    print('The config file has been copied to '+ destination_directory)

    return



def get_plots():
    from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc

    dataset_path_parser = argparse.ArgumentParser(description= "parser for MEGqc: --inputdata(mandatory) path/to/your/BIDSds)")
    dataset_path_parser.add_argument("--inputdata", type=str, required=True, help="path to the root of your BIDS MEG dataset")
    args=dataset_path_parser.parse_args()
    data_directory = args.inputdata

    make_plots_meg_qc(data_directory)
    return




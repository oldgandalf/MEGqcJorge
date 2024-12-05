import os
import argparse
import shutil

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

get_config()

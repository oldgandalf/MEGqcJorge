import argparse
import os
import sys
from meg_qc.calculation.meg_qc_pipeline import make_derivative_meg_qc

def hello_world():
    dataset_path_parser = argparse.ArgumentParser(description= "parser for string to print")
    dataset_path_parser.add_argument("--inputstring", type=str, required=True, help="path to the root of your BIDS MEG dataset")
    args=dataset_path_parser.parse_args()
    print(args.inputstring)


def run_megqc():
    dataset_path_parser = argparse.ArgumentParser(description= "parser for BIDS dataset path")
    dataset_path_parser.add_argument("--inputdata", type=str, required=True, help="path to the root of your BIDS MEG dataset")
    args=dataset_path_parser.parse_args()

    path_to_entrypoint= os.path.abspath(__file__)

    parent_dir = os.path.dirname(os.getcwd())
    print(parent_dir)
    print(path_to_entrypoint)

    data_directory = args.inputdata
    print(data_directory)

    config_file_path = '/home/areer1/Projects/MEGqc_myfork/MEGqc/meg_qc/settings/settings.ini' 
    internal_config_file_path='/home/areer1/Projects/MEGqc_myfork/MEGqc/meg_qc/settings/settings_internal.ini'

    make_derivative_meg_qc(config_file_path, internal_config_file_path, data_directory)

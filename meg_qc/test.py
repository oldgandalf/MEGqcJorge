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

    parent_dir = os.path.dirname(os.getcwd())
    print(parent_dir)

    data_directory = args.inputdata
    print(data_directory)

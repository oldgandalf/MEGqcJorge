import sys
sys.path.append('./')
from meg_qc.calculating.meg_qc_pipeline import make_derivative_meg_qc

#Add initial setup here (offer to install dependencies, etc.)

def main():

    print("\nWelcome to MEG QC\n")

    # Ask user in terminal for path to config file, set default to 'meg_qc/settings.ini'
    config_file_path = input("Please enter the path to the config file (hit enter for default: 'meg_qc/settings.ini'): ")
    if config_file_path == '':
        config_file_path = 'meg_qc/settings.ini'

    # Print config file path and ask to continue, default to Yes
    print("Config file path: " + config_file_path)
    continue_ = input("Continue? (Y/n): ")
    if continue_ == 'n':
        return
    
    print("\n\n Running MEG QC...\n")

    make_derivative_meg_qc(config_file_path, internal_config_file_path='meg_qc/settings_internal.ini')

    #note: even though there are no outputs, it will for now still try to output a bunch of plotly and mne figures and open all of them in the browser. 
    #This will later be suppressed, but so far I need for running the pipeline in the notebook and seeing the figures there.
    #to suppress ECG/EOG figure from mne go to ECG module and uncomment the line matplotlib.use('Agg') at the top of the file.
    #to suppress all plotly figure - comment all the fig.show() lines in the plotly modules

if __name__ == "__main__":
    main()
    
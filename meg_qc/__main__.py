from meg_qc.meg_qc_pipeline import make_derivative_meg_qc

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

    raw, raw_cropped_filtered_resampled, QC_derivs, QC_simple, df_head_pos, head_pos = make_derivative_meg_qc(config_file_path)

if __name__ == "__main__":
    main()
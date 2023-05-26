""""""""""""""""""
API documentation
""""""""""""""""""

As the result of analysis will be produced (for each data file (.fif)):

- html report for all metrics 
- csv file with the results of the analysis for some of metrics
- machine readable json file with the results of the analysis for all metrics

In the html report:

- all the plots produced by MEQ-QC are interactive, they can be scrolled through and enlarged. 
- a few plots from MNE (in ECG and EOG sections) are not interactive.

UML diagrams presenting the flow of the analysis for each module are available here:
https://github.com/ANCPLabOldenburg/MEG-QC-code/tree/main/diagrams

.. toctree::
   :maxdepth: 3
   :caption: Modules:

   settings_ini
   main
   initial_processing
   std
   psd
   ptp_manual
   ptp_auto
   ecg_eog 
   muscle
   head 
   universal_plots 
   html_report
   
   


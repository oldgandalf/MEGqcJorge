from universal_plots import QC_derivative, get_tit_and_unit
import mne

def make_html_section(derivs_section, section_name, no_ecg_str, no_eog_str, no_head_pos_str, muscle_grad_str):

    """
    Create 1 section of html report. 1 section describes 1 metric like "ECG" or "EOG", "Head position" or "Muscle"...
    Functions does:
    - Add section title
    - Add user notification if needed (for example: head positions not calculated)
    - Loop over list of derivs belonging to 1 section, keep only figures
    - Put figures one after another with description under. Description should be set inside of the QC_derivative object.

    Parameters
    ----------
    derivs_section : list
        A list of QC_derivative objects belonging to 1 section.
    section_name : str
        The name of the section like "ECG" or "EOG", "Head position" or "Muscle"...
    no_ecg_str : str
        The user notification: if ECG was not calculated
    no_eog_str : str
        The user notification: if EOG was not calculated
    no_head_pos_str : str
        The user notification: if head positions were not calculated
    muscle_grad_str : str
        The user notification: if muscle was not calculated or which channel type was used for muscle calculation

    Returns
    -------
    html_section_str : str
        The html string of 1 section of the report.
    """

    fig_derivs_section = keep_fig_derivs(derivs_section)
    
    if 'ECG' in section_name:
        all_section_content='''<p>'''+no_ecg_str+'''</p>'''
    elif 'EOG' in section_name:
        all_section_content='''<p>'''+no_eog_str+'''</p>'''
    elif 'Head' in section_name:
        all_section_content='''<p>'''+no_head_pos_str+'''</p>'''
    elif 'Muscle' in section_name:
        all_section_content='''<p>'''+muscle_grad_str+'''</p>'''
    elif derivs_section and not fig_derivs_section:
        all_section_content='''<p>This measurement has no figures. Please see csv files.</p>'''
    else:
        all_section_content='''<p>''''''</p>'''

    if fig_derivs_section:
        for f in range(0, len(fig_derivs_section)):
            all_section_content += fig_derivs_section[f].convert_fig_to_html_add_description()

    html_section_str='''
        <!-- *** Section *** --->
        <center>
        <h2>'''+section_name+'''</h2>
        ''' + all_section_content+'''
        <br></br>
        <br></br>
        </center>'''

    # The way to get figures if need to open them from saved files:
    # figures = {}
    # figures_report= {}
    # for x in range(0, len(fig_derivs_section)):
    #     with open(fig_derivs_section[x], 'r') as figures["f{0}".format(x)]:
    #         figures_report["f{0}".format(x)] = figures["f{0}".format(x)].read()

    return html_section_str


def keep_fig_derivs(derivs_section:list[QC_derivative]):

    '''Loop over list of derivs belonging to 1 section, keep only figures to add to report.
    
    Parameters
    ----------
    derivs_section : list
        A list of QC_derivative objects belonging to 1 section.
        
    Returns
    -------
    fig_derivs_section : list
        A list of QC_derivative objects belonging to 1 section with only figures.'''
    
    fig_derivs_section=[]
    for d in derivs_section:
        if d.content_type == 'plotly' or d.content_type == 'matplotlib':
            fig_derivs_section.append(d)

    return fig_derivs_section


def make_joined_report(sections: dict, shielding_str: str, m_or_g_skipped_str: str, epoching_skipped_str: str, no_ecg_str: str, no_eog_str: str, no_head_pos_str: str, muscle_grad_str: str):

    '''
    Create report as html string with all sections. Currently make_joined_report_for_mne is used.

    Parameters
    ----------
    sections : dict
        A dictionary with section names as keys and lists of QC_derivative objects as values.
    shielding_str : str
        The user notification: if active shielding was used during data acquisition.
    m_or_g_skipped_str : str
        The user notification: if 'mags' or 'grads' were skipped during data analysis, becase they dont present in data or not chosen by usert.
    epoching_skipped_str : str
        The user notification: if epoching was skipped during data analysis, becase no events were found.
    no_ecg_str : str
        The user notification: if ECG was not calculated
    no_eog_str : str
        The user notification: if EOG was not calculated
    no_head_pos_str : str
        The user notification: if head positions were not calculated
    muscle_grad_str : str
        The user notification: if muscle was not calculated or which channel type was used for muscle calculation

    Returns
    -------
    html_string : str
        The html whole string of the report.
    
    '''


    header_html_string = '''
    <!doctype html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>MEG QC report</title>
            <style>body{ margin:0 100;}</style>
        </head>
        
        <body style="font-family: Arial">
            <center>
            <h1>MEG data quality analysis report</h1>
            <br></br>
            '''+shielding_str+m_or_g_skipped_str+epoching_skipped_str

    main_html_string = ''
    for key in sections:

        html_section_str = make_html_section(derivs_section = sections[key], section_name = key, no_ecg_str=no_ecg_str, no_eog_str=no_eog_str, no_head_pos_str=no_head_pos_str, muscle_grad_str=muscle_grad_str)

        # if sections[key]:
        #     html_section_str = make_html_section(derivs_section = sections[key], section_title = key, no_ecg_str=no_ecg_str, no_eog_str=no_eog_str)
        #     #html_section_str = make_html_section(figures_report["f{0}".format(x)], section_titles[x])
        # else:
        #     html_section_str = '''
        #     <!-- *** Section *** --->
        #     <h2>'''+key+'''</h2>
        #     <p>This measurement could not be calculated because.....</p>
        #     <br></br>
        #     <br></br>'''

        main_html_string += html_section_str


    end_string = '''
                     </center>
            </body>
        </html>'''


    html_string = header_html_string + main_html_string + end_string

    return html_string


def make_joined_report_for_mne(raw, sections:dict, shielding_str: str, m_or_g_skipped_str: str, epoching_skipped_str: str, no_ecg_str: str, no_eog_str: str, no_head_pos_str: str, muscle_grad_str: str):

    '''
    Create report as html string with all sections and embed the sections into MNE report object.

    Parameters
    ----------
    sections : dict
        A dictionary with section names as keys and lists of QC_derivative objects as values.
    shielding_str : str
        The user notification: if active shielding was used during data acquisition.
    m_or_g_skipped_str : str
        The user notification: if 'mags' or 'grads' were skipped during data analysis, becase they dont present in data or not chosen by usert.
    epoching_skipped_str : str
        The user notification: if epoching was skipped during data analysis, becase no events were found.
    no_ecg_str : str
        The user notification: if ECG was not calculated
    no_eog_str : str
        The user notification: if EOG was not calculated
    no_head_pos_str : str
        The user notification: if head positions were not calculated
    muscle_grad_str : str
        The user notification: if muscle was not calculated or which channel type was used for muscle calculation

    Returns
    -------
    report : mne.Report
        The MNE report object with all sections.
    
    '''

    report = mne.Report(title='& MEG QC Report')
    # This method also accepts a path, e.g., raw=raw_path
    report.add_raw(raw=raw, title='Raw', psd=False)  # omit PSD plot

    header_html_string = '''
    <!doctype html>
        <body style="font-family: Arial">
            <center>
            <h1>MEG data quality analysis report</h1>
            <br></br>
            '''+shielding_str+m_or_g_skipped_str+epoching_skipped_str+'''
            </center>
        </body>'''

    report.add_html(header_html_string, title='MEG QC report')

    
    for key in sections:
        if key != 'Report':
            html_section_str = make_html_section(derivs_section = sections[key], section_name = key, no_ecg_str=no_ecg_str, no_eog_str=no_eog_str, no_head_pos_str=no_head_pos_str, muscle_grad_str=muscle_grad_str)
            report.add_html(html_section_str, title=key)

    return report


def simple_metric_basic(metric_global_name: str, metric_global_description: str, metric_global_content_mag: dict, metric_global_content_grad: dict, metric_local_name: str =None, metric_local_description: str =None, metric_local_content_mag: dict =None, metric_local_content_grad: dict =None, display_only_global: bool =False, psd: bool=False):
    
    '''Basic structure of simple metric for all measurements.
    
    Parameters
    ----------
    metric_global_name : str
        Name of the global metric.
    metric_global_description : str
        Description of the global metric.
    metric_global_content_mag : dict
        Content of the global metric for the magnitometers as a dictionary.
        Content is created inside of the module for corresponding measurement.
    metric_global_content_grad : dict
        Content of the global metric for the gradiometers as a dictionary.
        Content is created inside of the module for corresponding measurement.
    metric_local_name : str, optional
        Name of the local metric, by default None (in case of no local metric is calculated)
    metric_local_description : str, optional
        Description of the local metric, by default None (in case of no local metric is calculated)
    metric_local_content_mag : dict, optional 
        Content of the local metric for the magnitometers as a dictionary, by default None (in case of no local metric is calculated)
        Content is created inside of the module for corresponding measurement.
    metric_local_content_grad : dict, optional
        Content of the local metric for the gradiometers as a dictionary, by default None (in case of no local metric is calculated)
        Content is created inside of the module for corresponding measurement.
    display_only_global : bool, optional
        If True, only global metric is displayed, by default False
        This parameter is set to True in case we dont need to display any info about local metric at all. For example for muscle artifacts.
        In case we want to display some notification about local metric, but not the actual metric (for example it failed to calculate for a reason), 
        this parameter is set to False and metric_local_description should contain that notification and metric_local_name - the name of missing local metric.
    psd : bool, optional
        If True, the metric is done for PSD and the units are changed accordingly, by default False

    Returns
    -------
    simple_metric : dict
        Dictionary with the whole simple metric to be converted into json in main script.
        '''
    
    _, unit_mag = get_tit_and_unit('mag', psd=psd)
    _, unit_grad = get_tit_and_unit('grad', psd=psd)

    if display_only_global is False:
       m_local = {metric_local_name: {
            "description": metric_local_description,
            "mag": metric_local_content_mag,
            "grad": metric_local_content_grad}}
    else:
        m_local = {}

    simple_metric={
        'measurement_unit_mag': unit_mag,
        'measurement_unit_grad': unit_grad,
        metric_global_name: {
            'description': metric_global_description,
            "mag": metric_global_content_mag,
            "grad": metric_global_content_grad}}

    #merge local and global metrics:
    simple_metric.update(m_local)

    return simple_metric
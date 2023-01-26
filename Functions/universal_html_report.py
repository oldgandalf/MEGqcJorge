from universal_plots import QC_derivative
import mne

# def add_fig_to_html_section(figure_report):

#     figure_with_descr_str = '''
#             <br></br>
#             ''' + figure_report + '''
#             <p>graph description...</p>'''

#     return figure_with_descr_str


# def make_html_section(section_name, section_figures):

#     section_headers={
#         'STD': 'Standart deviation of the data',
#         'PSD': 'Frequency spectrum',
#         'PTP': 'Peak-to-peak amplitudes',
#         'ECG': 'ECG artifacts',
#         'EOG': 'EOG artifacts'}
    

#     section_html = '''
#     <!-- *** Section *** --->
#     <h2>'''+section_headers[section_name]+'''</h2>
#     '''

#     for x in range(0, len(section_figures)):
#         figure_with_descr_str = add_fig_to_html_section(figures_report["f{0}".format(x)])

#         section_html += figure_with_descr_str

#     return section_html



def make_html_section(derivs_section, section_title, no_ecg_str, no_eog_str, no_head_pos_str, muscle_grad_str):

    """
    - Add section title
    - Loop over list of derivs belonging to 1 section, keep only figures
    - Put them one after another with description under."""

    all_section_content=''
    fig_derivs_section = keep_fig_derivs(derivs_section)
    
    if not fig_derivs_section and 'ECG' in section_title:
        all_section_content='''<p>'''+no_ecg_str+'''</p>'''
    elif not fig_derivs_section and 'EOG' in section_title:
        all_section_content='''<p>'''+no_eog_str+'''</p>'''
    elif not fig_derivs_section and 'Head' in section_title:
        all_section_content='''<p>'''+no_head_pos_str+'''</p>'''
    elif 'Muscle' in section_title:
        all_section_content='''<p>'''+muscle_grad_str+'''</p>'''
        for f in range(0, len(fig_derivs_section)):
            all_section_content += fig_derivs_section[f].convert_fig_to_html_add_description()
    elif derivs_section and not fig_derivs_section and 'EOG' not in section_title and 'ECG' not in section_title:
        all_section_content='''<p>This measurement has no figures. Please see csv files.</p>'''
    elif not derivs_section and not fig_derivs_section and 'EOG' not in section_title and 'ECG' not in section_title:
        all_section_content='''<p>This measurement was not calculated because...</p>'''
    else:
        for f in range(0, len(fig_derivs_section)):
            all_section_content += fig_derivs_section[f].convert_fig_to_html_add_description()

    html_section_str='''
        <!-- *** Section *** --->
        <center>
        <h2>'''+section_title+'''</h2>
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
    
    fig_derivs_section=[]
    for d in derivs_section:
        if d.content_type == 'plotly' or d.content_type == 'matplotlib':
            fig_derivs_section.append(d)

    return fig_derivs_section


def make_joined_report(sections:dict, shielding_str, channels_skipped_str, epoching_skipped_str, no_ecg_str, no_eog_str, no_head_pos_str, muscle_grad_str):

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
            '''+shielding_str+channels_skipped_str+epoching_skipped_str

    main_html_string = ''
    for key in sections:

        html_section_str = make_html_section(derivs_section = sections[key], section_title = key, no_ecg_str=no_ecg_str, no_eog_str=no_eog_str, no_head_pos_str=no_head_pos_str, muscle_grad_str=muscle_grad_str)

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


def make_joined_report_for_mne(raw, sections:dict, shielding_str, channels_skipped_str, epoching_skipped_str, no_ecg_str, no_eog_str, no_head_pos_str, muscle_grad_str):

    report = mne.Report(title='& MEG QC Report')
    # This method also accepts a path, e.g., raw=raw_path
    report.add_raw(raw=raw, title='Raw', psd=False)  # omit PSD plot

    header_html_string = '''
    <!doctype html>
        <body style="font-family: Arial">
            <center>
            <h1>MEG data quality analysis report</h1>
            <br></br>
            '''+shielding_str+channels_skipped_str+epoching_skipped_str+'''
            </center>
        </body>'''

    report.add_html(header_html_string, title='MEG QC report')

    
    for key in sections:
        if key != 'Report':
            html_section_str = make_html_section(derivs_section = sections[key], section_title = key, no_ecg_str=no_ecg_str, no_eog_str=no_eog_str, no_head_pos_str=no_head_pos_str, muscle_grad_str=muscle_grad_str)
            report.add_html(html_section_str, title=key)

    return report


def make_std_peak_report(sid: str, what_data: str, list_of_figure_paths: list, config):

    '''Create an html report with figures

    Args: 
    sid (str): subject id number, like '1'.
    what_data (str): 'stds' or 'peaks'
    list_of_figure_paths (list): list of paths to html extracted figues. 
        they will be plotted in report in the order they are in this list

    Returns:
    the report itsels save on the local machine as htms file (in derivatives folder -> reports)
    '''

    figures = {}
    figures_report= {}
    for x in range(0, len(list_of_figure_paths)):
        with open(list_of_figure_paths[x], 'r') as figures["f{0}".format(x)]:
            figures_report["f{0}".format(x)] = figures["f{0}".format(x)].read()

    # The code above is doing the same as this here, but allowes to create variable on the loop, 
    # when we dont know how many figeres we will have:
    # with open(list_of_figure_paths[0], 'r') as f1m:
    #     fig1m = f1m.read()

    if what_data == 'peaks':
        heading_measurement = 'Peak-to-peak amplitudes'
    elif what_data == 'stds':
        heading_measurement = 'Standard deviation'
    elif what_data == 'psd':
        heading_measurement = 'Frequency spectrum'


    header_html_string = '''
    <!doctype html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>MEG QC: '''+heading_measurement+''' report</title>
            <style>body{ margin:0 100;}</style>
        </head>
        
        <body style="font-family: Arial">
            <center>
            <h1>MEG data quality analysis report</h1>
            <br></br>'''

    main_html_string = ''
    for x in range(0, len(list_of_figure_paths)):
        new_html_string = make_html_section(figures_report["f{0}".format(x)])
        main_html_string +=new_html_string


    end_string = '''
                     </center>
            </body>
        </html>'''


    html_string = header_html_string + main_html_string + end_string

    with open('../derivatives/sub-'+sid+'/megqc/reports/report_'+what_data+'.html', 'w', encoding = 'utf8') as html_report:
        html_report.write(html_string)





def make_PSD_report(sid: str, list_of_figure_paths: list):

    # for fig_n in range(len(list_of_figure_paths)):

    #     with open(list_of_figure_paths[fig_n], 'r') as f[fig_n]:
    #     fig1m = f[fig_n].read()

    with open(list_of_figure_paths[0], 'r') as f1m:
        fig1m = f1m.read()

    with open(list_of_figure_paths[1], 'r') as f1g:
        fig1g = f1g.read()
        
    with open(list_of_figure_paths[2], 'r') as f2m:
        fig2m = f2m.read()

    with open(list_of_figure_paths[3], 'r') as f2g:
        fig2g = f2g.read()
        
    html_string = '''
    <!doctype html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>MEG QC: Frequency spectrum Report</title>
            <style>body{ margin:0 100;}</style>
        </head>
        
        <body style="font-family: Arial">
            <center>
            <h1>MEG data quality analysis report</h1>
            <br></br>
            <!-- *** Section 1 *** --->
            <h2>Frequency spectrum per channel</h2>
            ''' + fig1m + '''
            <p>graph description...</p>

            <br></br>
            ''' + fig1g + '''
            <p>graph description...</p>
            
            <!-- *** Section 2 *** --->
            <br></br>
            <br></br>
            <br></br>
            <h2>Relative power of each band over all channels</h2>
            ''' + fig2m + '''
            <p>graph description...</p>
            <br></br>
            ''' + fig2g + '''
            <p>graph description...</p>
            </center>
        
        </body>
    </html>'''

    with open('../derivatives/sub-'+sid+'/megqc/reports/report_PSD.html', 'w', encoding = 'utf8') as f:
        f.write(html_string)

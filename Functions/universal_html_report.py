from universal_plots import QC_derivative

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



def make_html_section(derivs_section, section_title):

    """
    - Add section title
    - Loop over list of derivs belonging to 1 section, keep only figures
    - Put them one after another with description under."""

    fig_derivs_section = keep_fig_derivs(derivs_section)

    # figures = {}
    # figures_report= {}
    # for x in range(0, len(fig_derivs_section)):
    #     with open(fig_derivs_section[x], 'r') as figures["f{0}".format(x)]:
    #         figures_report["f{0}".format(x)] = figures["f{0}".format(x)].read()

    all_figs_of_section=''
    for f in range(0, len(fig_derivs_section)):
        all_figs_of_section += fig_derivs_section[f].convert_fig_to_html_add_description()


    html_section_str='''
            <!-- *** Section *** --->
            <h2>'''+section_title+'''</h2>
            ''' + all_figs_of_section+'''
            <br></br>
            <br></br>'''

    return html_section_str


def keep_fig_derivs(derivs_section:list[QC_derivative]):
    
    fig_derivs_section=[]
    for d in derivs_section:
        if d.content_type == 'plotly' or d.content_type == 'matplotlib':
            fig_derivs_section.append(d)

    return fig_derivs_section


def make_joined_report(active_shielding_used: bool, sections:dict, m_or_g_chosen: list, dict_of_dfs_epoch):

    shielding_str, channels_skipped_str, epoching_skipped_str = '', '', ''

    if active_shielding_used is True: 
        shielding_str=''' <p>This file contains Internal Active Shielding data. Quality measurements calculated on this data should not be compared to the measuremnts calculated on the data without active shileding, since in the current case invironmental noise reduction was already partially performed by shileding, which normally should not be done before assesing the quality.</p><br></br>'''

    if 'mags' not in m_or_g_chosen:
        channels_skipped_str = ''' <p>This data set contains no magnetometers or they were not chosen for analysis. Quality measurements were performed only on gradiometers.</p><br></br>'''
    elif 'grads' not in m_or_g_chosen:
        channels_skipped_str = ''' <p>This data set contains no gradiometers or they were not chosen for analysis. Quality measurements were performed only on magnetometers.</p><br></br>'''

    if dict_of_dfs_epoch['mags'] is None and dict_of_dfs_epoch['grads'] is None:
        epoching_skipped_str = ''' <p>No epoching could be done in this data set: no events found. Quality measurement were only performed on the entire time series. If this was not expected, try: 1) checking the presence of stimulus channel in the data set, 2) setting stimulus channel explicitly in config file, 3) setting different event duration in config file.</p><br></br>'''


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

        if sections[key]:
            html_section_str = make_html_section(derivs_section = sections[key], section_title = key)
            #html_section_str = make_html_section(figures_report["f{0}".format(x)], section_titles[x])
        else:
            html_section_str = '''
            <!-- *** Section *** --->
            <h2>'''+key+'''</h2>
            <p>This measurement could not be calculated because.....</p>
            <br></br>
            <br></br>'''

        main_html_string += html_section_str


    end_string = '''
                     </center>
            </body>
        </html>'''


    html_string = header_html_string + main_html_string + end_string

    return html_string



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

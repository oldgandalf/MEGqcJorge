
def make_std_peak_report(sid: str, what_data: str, list_of_figure_paths: list, config):

    '''Create an html report with figures
    Example: https://towardsdatascience.com/automated-interactive-reports-with-plotly-and-python-88dbe3aae5

    NEED TO TURN THIS INTO UNIVERSAL REPORT FOR ALL DATA TYPES AND DIFFERENT NUMBER OF FIGURES.
    WHAT TO WRITE AS DESCRIPTION FOR FIGURES?

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

    # The code aboveb is doing the same as this here, but allowes to create variable on the loop, 
    # when we dont know how many figeres we will have:
    # with open(list_of_figure_paths[0], 'r') as f1m:
    #     fig1m = f1m.read()

    if what_data == 'peaks':
        heading_measurement = 'Peak-to-peak amplitudes'
    elif what_data == 'stds':
        heading_measurement = 'Standart deviation'


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
    
    def make_html_section(heading_measurement, figure_report, epoched: bool, m_or_g: str):
        
        if epoched is True:
            heading_epoch = 'over_epochs'
        elif epoched is False:
            heading_epoch = 'over_whole data'

        single_html_string='''
                <!-- *** Section *** --->
                <h2>'''+heading_measurement+heading_epoch+'''</h2>
                ''' + figure_report + '''
                <p>graph description...</p>'''
        return single_html_string

    default_section = config['DEFAULT']

    #case1: 1 plot: mag OR grad, no epoching
    single_html_string = make_html_section(heading_measurement, figure_report=figures_report['f0'])

    #case2: 2 plots: mag OR grad, with epoching
    if (default_section['do_for'] == 'mags' or default_section['do_for'] == 'grads') and default_section['stim_channel'] == '*':
        extra_html_string = make_html_section(heading_measurement, figure_report=figures_report['f1'], epoched = True)

    #case3: 2 plots: mag AND grad, no epoching
    elif (default_section['do_for'] == 'mags' and default_section['do_for'] == 'grads') and default_section['stim_channel'] == '':
        extra_html_string1 = make_html_section(heading_measurement, figure_report=figures_report['f1'], epoched=False)
    
    #case4: 4 plots: mag AND grad, with epoching
    elif (default_section['do_for'] == 'mags' and default_section['do_for'] == 'grads') and default_section['stim_channel'] == '*':
        extra_html_string1 = make_html_section(heading_measurement, figure_report=figures_report['f1'], epoched=True)
        extra_html_string2 = make_html_section(heading_measurement, figure_report=figures_report['f2'], epoched=True)
        extra_html_string3 = make_html_section(heading_measurement, figure_report=figures_report['f3'], epoched=True)
        extra_html_string = extra_html_string1 + extra_html_string2 + extra_html_string3


#     #case: 2 plots: mag or grad, with epoching
#     if (default_section['do_for'] == 'mags' or default_section['do_for'] == 'grads') and default_section['stim_channel'] == '*':
#         extra_html_string = '''
#                 <!-- *** Section *** --->
#                 <br></br>
#                 <br></br>
#                 <br></br>
#                 <h2>'''+heading+''' over epochs</h2>
#                 ''' + figures_report['f1'] + '''
#                 <p>graph description...</p>
            
# '''
#     #case: 2 plots: mag + grad, no epoching
#     elif (default_section['do_for'] == 'mags' and default_section['do_for'] == 'grads') and default_section['stim_channel'] == '':
#         extra_html_string = '''
#                 <!-- *** Section *** --->
#                 <br></br>
#                 <br></br>
#                 <br></br>
#                 <h2>'''+heading+''' over all data</h2>
#                 ''' + figures_report['f1'] + '''
#                 <p>graph description...</p>'''

#     #case: 4 plots: mag + grad, with epoching
#     elif (default_section['do_for'] == 'mags' or default_section['do_for'] == 'grads') and default_section['stim_channel'] == '*':
#         extra_html_string = '''
#         <!doctype html>
#         <html>
#             <head>
#                 <meta charset="UTF-8">
#                 <title>MEG QC: '''+heading+''' Report</title>
#                 <style>body{ margin:0 100;}</style>
#             </head>
            
#             <body style="font-family: Arial">
#                 <center>
#                 <h1>MEG data quality analysis report</h1>
#                 <br></br>
#                 <!-- *** Section 1 *** --->
#                 <h2>'''+heading+''' over the entire data</h2>
#                 ''' + figures_report['f0'] + '''
#                 <p>graph description...</p>

#                 <!-- *** Section 2 *** --->
#                 <br></br>
#                 <br></br>
#                 <br></br>
#                 <h2>'''+heading+''' over epochs</h2>
#                 ''' + figures_report['f1'] + '''
#                 <p>graph description...</p>
#                 </center>
#             </body>
#         </html>'''


#     elif len(list_of_figure_paths) == 2 and what_data == 'peaks':
#         html_string = '''
#         <!doctype html>
#         <html>
#             <head>
#                 <meta charset="UTF-8">
#                 <title>MEG QC: Peak-to-peak amplitudes Report</title>
#                 <style>body{ margin:0 100;}</style>
#             </head>
            
#             <body style="font-family: Arial">
#                 <center>
#                 <h1>MEG data quality analysis report</h1>
#                 <br></br>
#                 <!-- *** Section 1 *** --->
#                 <h2>Peak-to-peak amplitudes over epochs</h2>
#                 ''' + figures_report['f0'] + '''
#                 <p>graph description...</p>

#                 <br></br>
#                 ''' + figures_report['f1'] + '''
#                 <p>graph description...</p>
#                 </center>
#             </body>
#         </html>'''



    else:
        print('Check the number of figure paths! Must be 1, 2 or 4.')
        return

    end_string = '''
                     </center>
            </body>
        </html>'''

    html_string = header_html_string + single_html_string + extra_html_string + end_string

    with open('../derivatives/sub-'+sid+'/megqc/reports/report_'+what_data+'.html', 'w', encoding = 'utf8') as f:
        f.write(html_string)



def make_peak_html_report(sid: str, what_data: str, list_of_figure_paths: list):

    if what_data=='stds':
        heading='Standard deviation'
        fig_tit='STD'
    elif what_data=='peaks':
        heading='Peak-to-peak amplitudes'
        fig_tit='PP_amplitude'

    heading='Peak-to-peak amplitudes'
    fig_tit='PP_amplitude'

    with open(list_of_figure_paths[0], 'r') as f1m:
        fig1m = f1m.read()

    with open(list_of_figure_paths[1], 'r') as f1g:
        fig1g = f1g.read()
        

    html_string = '''
    <!doctype html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>MEG QC: '''+heading+''' Report</title>
            <style>body{ margin:0 100;}</style>
        </head>
        
        <body style="font-family: Arial">
            <center>
            <h1>MEG data quality analysis report</h1>
            <br></br>
            <!-- *** Section 1 *** --->
            <h2>'''+heading+''' over epochs</h2>
            ''' + fig1m + '''
            <p>graph description...</p>

            <br></br>
            ''' + fig1g + '''
            <p>graph description...</p>
        
        </body>
    </html>'''

    with open('../derivatives/sub-'+sid+'/megqc/reports/report_'+fig_tit+'.html', 'w', encoding = 'utf8') as f:
        f.write(html_string)



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


def make_RMSE_html_report(sid: str, what_data: str, list_of_figure_paths: list):

    '''Create an html report with figures
    Example: https://towardsdatascience.com/automated-interactive-reports-with-plotly-and-python-88dbe3aae5
    '''

    if what_data=='stds':
        heading='Standard deviation'
        fig_tit='STD'
    elif what_data=='peaks':
        heading='Peak-to-peak amplitudes'
        fig_tit='PP_amplitude'

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
            <title>MEG QC: '''+heading+''' Report</title>
            <style>body{ margin:0 100;}</style>
        </head>
        
        <body style="font-family: Arial">
            <center>
            <h1>MEG data quality analysis report</h1>
            <br></br>
            <!-- *** Section 1 *** --->
            <h2>'''+heading+''' over the entire data</h2>
            ''' + fig1m + '''
            <p>graph description...</p>

            <br></br>
            ''' + fig1g + '''
            <p>graph description...</p>
            
            <!-- *** Section 2 *** --->
            <br></br>
            <br></br>
            <br></br>
            <h2>'''+heading+''' over epochs</h2>
            ''' + fig2m + '''
            <p>graph description...</p>
            <br></br>
            ''' + fig2g + '''
            <p>graph description...</p>
            </center>
        
        </body>
    </html>'''

    with open('../derivatives/sub-'+sid+'/megqc/reports/report_'+fig_tit+'.html', 'w', encoding = 'utf8') as f:
        f.write(html_string)



def make_peak_html_report(sid: str, what_data: str, list_of_figure_paths: list):

    '''Create an html report with figures
    Example: https://towardsdatascience.com/automated-interactive-reports-with-plotly-and-python-88dbe3aae5
    '''

    if what_data=='stds':
        heading='Standard deviation'
        fig_tit='STD'
    elif what_data=='peaks':
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
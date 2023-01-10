import plotly
import plotly.graph_objects as go
import mpld3
import base64
from io import BytesIO
import pandas as pd
import numpy as np
import warnings

def get_tit_and_unit(m_or_g):
    if m_or_g=='mag':
        m_or_g_tit="Magnetometers"
        unit='T/Hz'
    elif m_or_g=='grad':
        m_or_g_tit='Gradiometers'
        unit='T/m / Hz'
    else:
        m_or_g_tit='?'
        unit='?'

    return m_or_g_tit, unit

class QC_derivative:

    def __init__(self, content, description, filepath, content_type) -> None:
        self.content =  content
        self.description = description
        self.filepath = filepath
        self.content_type = content_type
        self.description_for_user = 'Add measurement description for a user here...'

    def __repr__(self):
        return 'MEG QC derivative: \n content: ' + str(type(self.content)) + '\n description: ' + self.description + '\n filepath: ' + str(self.filepath) + '\n type: ' + self.content_type + '\n description for user: ' + self.description_for_user + '\n '

    def convert_fig_to_html(self):
        if self.content_type == 'plotly':
            return plotly.io.to_html(self.content, full_html=False)
        elif self.content_type == 'matplotlib':
            tmpfile = BytesIO()
            self.content.savefig(tmpfile, format='png', dpi=130) #writing image into a temporary file
            encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
            return html
            # return mpld3.fig_to_html(self.content)
        elif not self.content_type:
            warnings.warn("Empty content_type of this QC_derivative instance")
        else:
            return None

    def convert_fig_to_html_add_description(self):

        figure_report = self.convert_fig_to_html()

        return '''<br></br>'''+ figure_report + '''<p>'''+self.description_for_user+'''</p>'''


    def get_section(self):
        if 'std' in self.description or 'rmse' in self.description or 'STD' in self.description or 'RMSE' in self.description:
            return 'std'
        elif 'ptp_manual' in self.description or 'pp_manual' in self.description or 'PTP_manual' in self.description or 'PP_manual'in self.description:
            return 'ptp_manual'
        elif 'ptp_auto' in self.description or 'pp_auto' in self.description or 'PTP_auto' in self.description or 'PP_auto' in self.description:
            return 'ptp_auto'
        elif 'psd' in self.description or 'PSD' in self.description:
            return 'psd'
        elif 'eog' in self.description or 'EOG' in self.description:
            return 'eog'
        elif 'ecg' in self.description or 'ECG' in self.description:
            return 'ecg'
        elif 'head' in self.description or 'HEAD' in self.description:
            return 'head'
        elif 'muscle' in self.description or 'MUSCLE' in self.description:
            return 'muscle'
        else:  
            warnings.warn("Check description of this QC_derivative instance: " + self.description)
        

def boxplot_channel_epoch_hovering_plotly(df_mg: pd.DataFrame, ch_type: str, what_data: str) -> QC_derivative:

    '''
    Creates representation of calculated data as multiple boxplots: 
    each box represents 1 channel, each dot is std of 1 epoch in this channel
    Implemented with plotly: https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Box.html
    The figure will be saved as an interactive html file.

    Args:
    df_mg(pd.DataFrame): data frame containing data (stds, peak-to-peak amplitudes, etc) for each epoch, each channel, 
        mags OR grads, not together
    ch_type (str): title, like "Magnetometers", or "Gradiometers", 
    what_data (str): 'peaks' for peak-to-peak amplitudes or 'stds'

    Returns:
    fig (go.Figure): plottly figure
    fig_path (str): path where the figure is saved as html file
    '''

    if ch_type=='Magnetometers':
        unit='Tesla'
    elif ch_type=='Gradiometers':
        unit='Tesla/meter'
    else:
        unit='?unknown unit?'
        print('Please check ch_type input. Has to be "Magnetometers" or "Gradiometers"')

    if what_data=='peaks':
        hover_tit='Amplitude'
        y_ax_and_fig_title='Peak-to-peak amplitude'
        fig_name='PP_manual_epoch_per_channel_'+ch_type
    elif what_data=='stds':
        hover_tit='STD'
        y_ax_and_fig_title='Standard deviation'
        fig_name='STD_epoch_per_channel_'+ch_type

    #transpose the data to plot the boxplots on x axes
    df_mg_transposed = df_mg.T 

    #collect all names of original df into a list to use as tick labels:
    ch_names=list(df_mg_transposed) 

    fig = go.Figure()

    for col in df_mg_transposed:
        fig.add_trace(go.Box(y=df_mg_transposed[col].values, 
        name=df_mg_transposed[col].name, 
        opacity=0.7, 
        boxpoints="all", 
        pointpos=0,
        marker_size=3,
        line_width=1,
        text=df_mg_transposed[col].index,
        ))
        fig.update_traces(hovertemplate='Epoch: %{text}<br>'+hover_tit+': %{y: .2e}')

    
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [v for v in range(0, len(ch_names))],
            ticktext = ch_names,
            rangeslider=dict(visible=True)
        ),
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'),
        yaxis_title=y_ax_and_fig_title+' in '+unit,
        title={
            'text': y_ax_and_fig_title+' over epochs for '+ch_type,
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
        

    fig_path='../derivatives/megqc/figures/'+fig_name+'.html'

    #fig.show()
    #fig.write_html(fig_path)

    qc_derivative = QC_derivative(content=fig, description=fig_name, filepath=fig_path, content_type='plotly')

    return qc_derivative


def boxplot_std_hovering_plotly(std_data: list, ch_type: str, channels: list, what_data: str):

    '''Creates representation of calculated std data as a boxplot (box containd magnetometers or gradiomneters, not together): 
    each dot represents 1 channel: name: std value over whole data of this channel. Too high/low stds are outliers.
    Implemebted with plotly: https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Box.html
    The figure will be saved as an interactive html file.

    Args:
    std_data (list): stds for mags or grads calculated in RMSE_meg_all, 
    ch_type (str): "Magnetometers" or "Gradiometers", 
    channels (list of tuples): magnetometer channel name + its index, 
    what_data (str): 'peaks' or 'stds'

    Returns:
    fig (go.Figure): plottly figure
    fig_path (str): path where the figure is saved as html file
    fig_name (str): figure name
    '''

    if ch_type=='Magnetometers':
        unit='Tesla'
    elif ch_type=='Gradiometers':
        unit='Tesla/meter'
    else:
        unit='?unknown unit?'
        print('Please check ch_type input. Has to be "Magnetometers" or "Gradiometers"')


    if what_data=='peaks':
        hover_tit='PP_Amplitude'
        y_ax_and_fig_title='Peak-to-peak amplitude'
        fig_name='PP_manual_all_data_'+ch_type
    elif what_data=='stds':
        hover_tit='STD'
        y_ax_and_fig_title='Standard deviation'
        fig_name='STD_epoch_all_data_'+ch_type

    df = pd.DataFrame (std_data, index=channels, columns=[hover_tit])

    fig = go.Figure()

    fig.add_trace(go.Box(x=df[hover_tit],
    name="",
    text=df[hover_tit].index, 
    opacity=0.7, 
    boxpoints="all", 
    pointpos=0,
    marker_size=5,
    line_width=1))
    fig.update_traces(hovertemplate='%{text}<br>'+hover_tit+': %{x: .0f}')
        

    fig.update_layout(
        yaxis={'visible': False, 'showticklabels': False},
        xaxis = dict(
        showexponent = 'all',
        exponentformat = 'e'),
        xaxis_title=y_ax_and_fig_title+" in "+unit,
        title={
        'text': y_ax_and_fig_title+' of the data for '+ch_type+' over the entire time series',
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
        
    #fig.show()

    fig_path='../derivatives/megqc/figures/'+fig_name+'.html'

    qc_derivative = QC_derivative(content=fig, description=fig_name, filepath=fig_path, content_type='plotly')

    return qc_derivative

#%%
def Plot_periodogram(m_or_g:str, freqs: np.ndarray, psds:np.ndarray, mg_names: list):

    '''Plotting periodogram on the data.

    Args:
    m_or_g (str): 'mag' or 'grad', 
    freqs (np.ndarray): numpy array of frequencies after performing Welch (or other method) psd decomposition
    psds (np.ndarray): numpy array of psds after performing Welch (or other method) psd decomposition
    mg_names (list of tuples): channel name + its index

    Returns:
    fig (go.Figure): plottly figure
    fig_path (str): path where the figure is saved as html file
    '''

    if m_or_g=='mag':
        tit='Magnetometers'
        unit='T/Hz'
    elif m_or_g=='grad':
        tit='Gradiometers'
        unit='T/m / Hz'
    else:
        tit='?'
        unit='?'

    df_psds=pd.DataFrame(np.sqrt(psds.T), columns=mg_names)

    fig = go.Figure()

    for col in df_psds:
        fig.add_trace(go.Scatter(x=freqs, y=df_psds[col].values, name=df_psds[col].name));

    #fig.update_xaxes(type="log")
    #fig.update_yaxes(type="log")
    
    fig.update_layout(
    title={
    'text': "Welch's periodogram for all "+tit,
    'y':0.85,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'},
    yaxis_title="Amplitude, "+unit,
    yaxis = dict(
        showexponent = 'all',
        exponentformat = 'e'),
    xaxis_title="Frequency (Hz)")
    fig.update_traces(hovertemplate='Frequency: %{x} Hz<br>Amplitude: %{y: .2e} T/Hz')

    #fig.show()
    
    fig_name='PSD_all_data_'+tit
    fig_path='../derivatives/megqc/figures/'+fig_name+'.html'
    
    #fig.write_html(fig_path)

    qc_derivative = QC_derivative(content=fig, description=fig_name, filepath=fig_path, content_type='plotly')

    return qc_derivative


def plot_pie_chart_freq(mean_relative_freq: list, tit: str, bands_names:str):
    
    ''''Pie chart representation of relative power of each frequency band in given data - in the entire 
    signal of mags or of grads, not separated by individual channels.

    Args:
    mean_relative_freq (list): list of power of each band like: [rel_power_of_delta, rel_power_of_gamma, etc...] - in relative  
        (percentage) values: what percentage of the total power does this band take,
    tit (str): title, like "Magnetometers", or "Gradiometers", 
    
    Returns:
    fig (go.Figure): plottly piechart figure
    fig_path (str): path where the figure is saved as html file
    '''

    #If mean relative percentages dont sum up into 100%, add the 'unknown' part.
    mean_relative_unknown=[v * 100 for v in mean_relative_freq]  #in percentage
    power_unknown_m=100-(sum(mean_relative_freq))*100
    if power_unknown_m>0:
        mean_relative_unknown.append(power_unknown_m)
        bands_names=bands_names+['unknown']


    fig = go.Figure(data=[go.Pie(labels=bands_names, values=mean_relative_unknown)])
    fig.update_layout(
    title={
    'text': "Relative power of each band: "+tit,
    'y':0.85,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'})

    #fig.show()

    fig_name='PSD_Relative_bandpower_all_channels_'+tit
    fig_path='../derivatives/megqc/figures/'+fig_name+'.html'
    
    #fig.write_html(fig_path)

    qc_derivative = QC_derivative(content=fig, description=fig_name, filepath=fig_path, content_type='plotly')

    return qc_derivative


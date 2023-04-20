import plotly
import plotly.graph_objects as go
import base64
from io import BytesIO
import pandas as pd
import mne
import warnings

def get_tit_and_unit(m_or_g: str, psd: bool = False):

    """
    Return title and unit for a given type of data (magnetometers or gradiometers) and type of plot (psd or not)
    
    Parameters
    ----------
    m_or_g : str
        'mag' or 'grad'
    psd : bool, optional
        True if psd plot, False if not, by default False

    Returns
    -------
    m_or_g_tit : str
        'Magnetometers' or 'Gradiometers'
    unit : str
        'T' or 'T/m' or 'T/Hz' or 'T/m / Hz'

    """
    
    if m_or_g=='mag':
        m_or_g_tit='Magnetometers'
        if psd is False:
            unit='Tesla'
        elif psd is True:
            unit='Tesla/Hz'
    elif m_or_g=='grad':
        m_or_g_tit='Gradiometers'
        if psd is False:
            unit='Tesla/m'
        elif psd is True:
            unit='Tesla/m / Hz'
    else:
        m_or_g_tit = '?'
        unit='?'

    return m_or_g_tit, unit

class QC_derivative:

    """ 
    Derivative of a QC measurement, main content of which is figure, data frame (saved later as csv) or html string.

    Attributes
    ----------
    content : figure, pd.DataFrame or str
        The main content of the derivative.
    name : str
        The name of the derivative (used to save in to file system)
    content_type : str
        The type of the content: 'plotly', 'matplotlib', 'csv', 'report' or 'mne_report'.
        Used to choose the right way to save the derivative in main function.
    description_for_user : str, optional
        The description of the derivative, by default 'Add measurement description for a user...'
        Used in the report to describe the derivative.
    

    """

    def __init__(self, content, name, content_type, description_for_user = 'Add measurement description for a user...'):

        """
        Constructor method
        
        Parameters
        ----------
        content : figure, pd.DataFrame or str
            The main content of the derivative.
        name : str
            The name of the derivative (used to save in to file system)
        content_type : str
            The type of the content: 'plotly', 'matplotlib', 'csv', 'report' or 'mne_report'.
            Used to choose the right way to save the derivative in main function.
        description_for_user : str, optional
            The description of the derivative, by default 'Add measurement description for a user...'
            Used in the report to describe the derivative.

        """

        self.content =  content
        self.name = name
        self.content_type = content_type
        self.description_for_user = description_for_user

    def __repr__(self):

        """
        Returns the string representation of the object.
        
        """

        return 'MEG QC derivative: \n content: ' + str(type(self.content)) + '\n name: ' + self.name + '\n type: ' + self.content_type + '\n description for user: ' + self.description_for_user + '\n '

    def convert_fig_to_html(self):

        """
        Converts figure to html string.
        
        Returns
        -------
        html : str or None
            Html string or None if content_type is not 'plotly' or 'matplotlib'.

        """

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

        """
        Converts figure to html string and adds description.

        Returns
        -------
        html : str or None
            Html string: fig + description or None + description if content_type is not 'plotly' or 'matplotlib'.

        """

        figure_report = self.convert_fig_to_html()

        return """<br></br>"""+ figure_report + """<p>"""+self.description_for_user+"""</p>"""


    def get_section(self):

        """ 
        Return a section of the report based on the info saved in the name. Normally not used. Use if cant figure out the derivative type.
        
        Returns
        -------
        section : str
            'RMSE', 'PTP_MANUAL', 'PTP_AUTO', 'PSD', 'EOG', 'ECG', 'MUSCLE', 'HEAD'.

        """

        if 'std' in self.name or 'rmse' in self.name or 'STD' in self.name or 'RMSE' in self.name:
            return 'RMSE'
        elif 'ptp_manual' in self.name or 'pp_manual' in self.name or 'PTP_manual' in self.name or 'PP_manual'in self.name:
            return 'PTP_MANUAL'
        elif 'ptp_auto' in self.name or 'pp_auto' in self.name or 'PTP_auto' in self.name or 'PP_auto' in self.name:
            return 'PTP_AUTO'
        elif 'psd' in self.name or 'PSD' in self.name:
            return 'PSD'
        elif 'eog' in self.name or 'EOG' in self.name:
            return 'EOG'
        elif 'ecg' in self.name or 'ECG' in self.name:
            return 'ECG'
        elif 'head' in self.name or 'HEAD' in self.name:
            return 'HEAD'
        elif 'muscle' in self.name or 'MUSCLE' in self.name:
            return 'MUSCLE'
        else:  
            warnings.warn("Check description of this QC_derivative instance: " + self.name)
        
def plot_time_series(raw: mne.io.Raw, m_or_g_chosen: str):

    """
    Plots time series of the chosen channels.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw file to be plotted.
    m_or_g_chosen : str
        The type of the channels to be plotted: 'mag' or 'grad'.
    
    Returns
    -------
    qc_derivative : list
        A list of QC_derivative objects containing the plotly figures with the sensor locations.

    """
    qc_derivative = []
    tit, unit = get_tit_and_unit(m_or_g_chosen)

    picked_channels = mne.pick_types(raw.info, meg=m_or_g_chosen)

    # Downsample data
    raw_resampled = raw.resample(100, npad='auto') #downsample the data to 100 Hz. The `npad` parameter is set to `'auto'` to automatically determine the amount of padding to use during the resampling process

    data = raw_resampled.get_data(picks=picked_channels) 

    fig = go.Figure()

    for i in range(data.shape[0]):
        fig.add_trace(go.Scatter(x=raw.times, y=data[i], mode='lines', name=raw.ch_names[picked_channels[i]]))

    # Add title, x axis title, x axis slider and y axis units+title:
    fig.update_layout(
        title={
            'text': tit+' Time Series',
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},

        xaxis_title='Time (s)',

        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="linear"),

        yaxis = dict(
                showexponent = 'all',
                exponentformat = 'e'),
            yaxis_title = unit) 
    
    qc_derivative += [QC_derivative(content=fig, name=tit+'_time_series', content_type='plotly', description_for_user = 'For this visialisation the data is resampled to 100Hz but not filtered. If cropping was chosen in settings the cropped raw is presented here, otherwise - entire duration.')]

    return qc_derivative



def switch_names_on_off(fig: go.Figure):

    """
    Switches between showing channel names when hovering and always showing channel names.
    
    Parameters
    ----------
    fig : go.Figure
        The figure to be modified.
        
    Returns
    -------
    fig : go.Figure
        The modified figure.
        
    """

    # Define the buttons
    buttons = [
    dict(label='Show channel names when hovering',
         method='update',
         args=[{'mode': 'markers'}]),
    dict(label='Always show channel names',
         method='update',
         args=[{'mode': 'markers+text'}])
    ]

    # Add the buttons to the layout
    fig.update_layout(updatemenus=[dict(type='buttons',
                                        showactive=True,
                                        buttons=buttons)])

    return fig


def plot_sensors_3d(raw: mne.io.Raw, m_or_g_chosen: str):

    """
    Plots the 3D locations of the sensors in the raw file.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw file to be plotted.
    m_or_g_chosen : str
        The type of the channels to be plotted: 'mag' or 'grad'.
    
    Returns
    -------
    qc_derivative : list
        A list of QC_derivative objects containing the plotly figures with the sensor locations.

    """
    qc_derivative = []

    # Check if there are magnetometers and gradiometers in the raw file:
    if 'mag' in m_or_g_chosen:

        # Extract the sensor locations and names for magnetometers
        mag_locs = raw.copy().pick_types(meg='mag').info['chs']
        mag_pos = [ch['loc'][:3] for ch in mag_locs]
        mag_names = [ch['ch_name'] for ch in mag_locs]

        # Create the magnetometer plot with markers only

        mag_fig = go.Figure(data=[go.Scatter3d(x=[pos[0] for pos in mag_pos],
                                            y=[pos[1] for pos in mag_pos],
                                            z=[pos[2] for pos in mag_pos],
                                            mode='markers',
                                            marker=dict(size=5),
                                            text=mag_names,
                                            hovertemplate='%{text}')],
                                            layout=go.Layout(width=1200, height=1200))

        mag_fig.update_layout(
            title={
            'text': 'Magnetometers positions',
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

        mag_fig = switch_names_on_off(mag_fig)

        qc_derivative += [QC_derivative(content=mag_fig, name='Magnetometers_positions', content_type='plotly')]

    if 'grad' in m_or_g_chosen:

        # Extract the sensor locations and names for gradiometers
        grad_locs = raw.copy().pick_types(meg='grad').info['chs']
        grad_pos = [ch['loc'][:3] for ch in grad_locs]
        grad_names = [ch['ch_name'] for ch in grad_locs]

        #since grads have 2 sensors located in the same spot - need to put their names together to make pretty plot labels:

        grad_pos_together = []
        grad_names_together = []

        for i in range(len(grad_pos)-1):
            if all(x == y for x, y in zip(grad_pos[i], grad_pos[i+1])):
                grad_pos_together += [grad_pos[i]]
                grad_names_together += [grad_names[i]+', '+grad_names[i+1]]
            else:
                pass


        # Add both sets of gradiometer positions to the plot:
        grad_fig = go.Figure(data=[go.Scatter3d(x=[pos[0] for pos in grad_pos_together],
                                                y=[pos[1] for pos in grad_pos_together],
                                                z=[pos[2] for pos in grad_pos_together],
                                                mode='markers',
                                                marker=dict(size=5),
                                                text=grad_names_together,
                                                hovertemplate='%{text}')],
                                                layout=go.Layout(width=1200, height=1200))

        grad_fig.update_layout(
            title={
            'text': 'Gradiometers positions',
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})


        # Add the button to have names show up on hover or always:
        grad_fig = switch_names_on_off(grad_fig)

        qc_derivative += [QC_derivative(content=grad_fig, name='Gradiometers_positions', content_type='plotly')]

    return qc_derivative



def boxplot_epochs(df_mg: pd.DataFrame, ch_type: str, what_data: str, x_axis_boxes: str):

    """
    Creates representation of calculated data as multiple boxplots. Used in RMSE and PtP_manual measurements. 

    - If x_axis_boxes is 'channels', each box represents 1 epoch, each dot is std of 1 channel for this epoch
    - If x_axis_boxes is 'epochs', each box represents 1 channel, each dot is std of 1 epoch for this channel

    
    Parameters
    ----------
    df_mg : pd.DataFrame
        Data frame with std or peak-to-peak values for each channel and epoch. Columns are epochs, rows are channels.
    ch_type : str
        Type of the channel: 'mag', 'grad'
    what_data : str
        Type of the data: 'peaks' or 'stds'
    x_axis_boxes : str
        What to plot as boxplot on x axis: 'channels' or 'epochs'

    Returns
    -------
    fig_deriv : QC_derivative 
        derivative containing plotly figure
    
    """

    ch_tit, unit = get_tit_and_unit(ch_type)

    if what_data=='peaks':
        hover_tit='Amplitude'
        y_ax_and_fig_title='Peak-to-peak amplitude'
        fig_name='PP_manual_epoch_per_channel_'+ch_tit
    elif what_data=='stds':
        hover_tit='STD'
        y_ax_and_fig_title='Standard deviation'
        fig_name='STD_epoch_per_channel_'+ch_tit
    else:
        print('what_data should be either peaks or stds')

    if x_axis_boxes=='channels':
        #transpose the data to plot channels on x axes
        df_mg = df_mg.T
        legend_title = ''
        hovertemplate='Epoch: %{text}<br>'+hover_tit+': %{y: .2e}'
    elif x_axis_boxes=='epochs':
        legend_title = 'Epochs'
        hovertemplate='%{text}<br>'+hover_tit+': %{y: .2e}'
    else:
        print('x_axis_boxes should be either channels or epochs')

    #collect all names of original df into a list to use as tick labels:
    boxes_names = df_mg.columns.tolist() #list of channel names or epoch names
    #boxes_names=list(df_mg) 

    fig = go.Figure()

    for col in df_mg:
        fig.add_trace(go.Box(y=df_mg[col].values, 
        name=str(df_mg[col].name), 
        opacity=0.7, 
        boxpoints="all", 
        pointpos=0,
        marker_size=3,
        line_width=1,
        text=df_mg[col].index,
        ))
        fig.update_traces(hovertemplate=hovertemplate)

    
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [v for v in range(0, len(boxes_names))],
            ticktext = boxes_names,
            rangeslider=dict(visible=True)
        ),
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'),
        yaxis_title=y_ax_and_fig_title+' in '+unit,
        title={
            'text': y_ax_and_fig_title+' over epochs for '+ch_tit,
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        legend_title=legend_title)
        
    #fig.show()

    fig_deriv = QC_derivative(content=fig, name=fig_name, content_type='plotly')

    return fig_deriv


def boxplot_epochs_old(df_mg: pd.DataFrame, ch_type: str, what_data: str) -> QC_derivative:

    """
    Create representation of calculated data as multiple boxplots: 
    each box represents 1 channel, each dot is std of 1 epoch in this channel
    Implemented with plotly: https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Box.html
    The figure will be saved as an interactive html file.

    Parameters
    ----------
    df_mg : pd.DataFrame
        data frame containing data (stds, peak-to-peak amplitudes, etc) for each epoch, each channel, mags OR grads, not together
    ch_type : str 
        title, like "Magnetometers", or "Gradiometers", 
    what_data : str
        'peaks' for peak-to-peak amplitudes or 'stds'

    Returns
    -------
    fig : go.Figure
        plottly figure

    """

    ch_tit, unit = get_tit_and_unit(ch_type)

    if what_data=='peaks':
        hover_tit='Amplitude'
        y_ax_and_fig_title='Peak-to-peak amplitude'
        fig_name='PP_manual_epoch_per_channel_'+ch_tit
    elif what_data=='stds':
        hover_tit='STD'
        y_ax_and_fig_title='Standard deviation'
        fig_name='STD_epoch_per_channel_'+ch_tit

    #collect all names of original df into a list to use as tick labels:
    epochs = df_mg.columns.tolist()

    fig = go.Figure()

    for col in df_mg:
        fig.add_trace(go.Box(y=df_mg[col].values, 
        name=str(df_mg[col].name), 
        opacity=0.7, 
        boxpoints="all", 
        pointpos=0,
        marker_size=3,
        line_width=1,
        text=df_mg[col].index,
        ))
        fig.update_traces(hovertemplate='%{text}<br>'+hover_tit+': %{y: .2e}')

    
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [v for v in range(0, len(epochs))],
            ticktext = epochs,
            rangeslider=dict(visible=True)
        ),
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'),
        yaxis_title=y_ax_and_fig_title+' in '+unit,
        title={
            'text': y_ax_and_fig_title+' of epochs for '+ch_tit,
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        legend_title="Epochs")
        
    #fig.show()

    qc_derivative = QC_derivative(content=fig, name=fig_name, content_type='plotly')

    return qc_derivative


def boxplot_std_hovering_plotly(std_data: list, ch_type: str, channels: list, what_data: str):

    """
    Create representation of calculated std data as a boxplot (box containd magnetometers or gradiomneters, not together): 
    each dot represents 1 channel: name: std value over whole data of this channel. Too high/low stds are outliers.

    Parameters
    ----------
    std_data : list
        list of std values for each channel
    ch_type : str
        'mag' or 'grad'
    channels : list
        list of channel names
    what_data : str
        'peaks' for peak-to-peak amplitudes or 'stds'

    Returns
    -------
    QC_derivative
        QC_derivative object with plotly figure as content

    """

    ch_tit, unit = get_tit_and_unit(ch_type)

    if what_data=='peaks':
        hover_tit='PP_Amplitude'
        y_ax_and_fig_title='Peak-to-peak amplitude'
        fig_name='PP_manual_all_data_'+ch_tit
    elif what_data=='stds':
        hover_tit='STD'
        y_ax_and_fig_title='Standard deviation'
        fig_name='STD_epoch_all_data_'+ch_tit

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
        'text': y_ax_and_fig_title+' of the data for '+ch_tit+' over the entire time series',
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
        
    #fig.show()

    qc_derivative = QC_derivative(content=fig, name=fig_name, content_type='plotly')

    return qc_derivative




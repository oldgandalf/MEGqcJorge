import plotly
import plotly.graph_objects as go
import base64
from io import BytesIO
import numpy as np
import pandas as pd
import mne
import warnings
from IPython.display import display
import random
import copy

def check_num_channels_correct(chs_by_lobe: dict, note: str):

    """ 
    Print total number of channels in all lobes for 1 ch type (must be 102 mag and 204 grad in Elekta/Neuromag)
    
    Parameters
    ----------
    chs_by_lobe : dict
        A dictionary of channels sorted by ch type and lobe.
    note : str
        A note to print with the total number of channels.
    
    Returns
    -------
    
    """
    for m_or_g in ['mag', 'grad']:
        total_number = sum([len(chs_by_lobe[m_or_g][key]) for key in chs_by_lobe[m_or_g].keys()])
        print("_______"+note+"_______total number in " + m_or_g, total_number)
        print("_______"+note+"_______must be 102 mag and 204 grad in Elekta/Neuromag")

    return 

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
    elif m_or_g == 'ECG':
        m_or_g_tit = 'ECG channel'
        unit = 'V'
    elif m_or_g == 'EOG':
        m_or_g_tit = 'EOG channel'
        unit = 'V'
    else:
        m_or_g_tit = '?'
        unit='?'

    return m_or_g_tit, unit

def get_ch_color_knowing_name(ch_name: str, chs_by_lobe: dict):

    """
    Get channel color from chs_by_lobe knowing its name.
    Currently not used in pipeline. Might be useful later.

    Parameters
    ----------
    ch_name : str
        channel name
    chs_by_lobe : dict
        dictionary with channel objects sorted by lobe
    
    Returns
    -------
    color : str
        color of the channel

    """

    color = 'black'
    for lobe, ch_obj_list in chs_by_lobe.items():
        for ch_obj in ch_obj_list:
            if ch_obj.name == ch_name:
                color = ch_obj.lobe_color
                break

    return color

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

    def __init__(self, content, name, content_type, description_for_user = ''):

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

def plot_df_of_channels_data_as_lines_by_lobe_OLD(chs_by_lobe: dict, df_data: pd.DataFrame, x_values):

    """
    Plots data from a data frame as lines, each lobe has own color as set in chs_by_lobe.
    Old version. Here we plot all channels of one lobe together, then all channels of next lobe - gives less visual separation of traces since they blend together.

    Parameters
    ----------
    chs_by_lobe : dict
        Dictionary with lobes as keys and lists of channels as values.
    df_data : pd.DataFrame
        Data frame with data to plot.
    x_values : list
        List of x values for the plot.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure.

    """

    fig = go.Figure()

    for lobe, ch_list in chs_by_lobe.items():
        
        #Add lobe as a category to the plot
        #No unfortunatelly you can make it so when you click on lobe you activate/hide all related channels. It is not a proper category in plotly, it is in fact just one more trace.
        fig.add_trace(go.Scatter(x=x_values, y=[None]*len(x_values), mode='markers', marker=dict(size=5, color=ch_list[0].lobe_color), showlegend=True, name=lobe.upper()))

        for ch_obj in ch_list:
            if ch_obj.name in df_data.columns:
                ch_data=df_data[ch_obj.name].values
                color = ch_obj.lobe_color 
                # normally color must be same for all channels in lobe, so we could assign it before the loop as the color of the first channel,
                # but here it is done explicitly for every channel so that if there is any color error in chs_by_lobe, it will be visible

                fig.add_trace(go.Scatter(x=x_values, y=ch_data, line=dict(color=color), name=ch_obj.name))

    return fig


def plot_df_of_channels_data_as_lines_by_lobe(chs_by_lobe: dict, df_data: pd.DataFrame, x_values):

    """
    Plots data from a data frame as lines, each lobe has own color as set in chs_by_lobe.

    Parameters
    ----------
    chs_by_lobe : dict
        Dictionary with lobes as keys and lists of channels as values.
    df_data : pd.DataFrame
        Data frame with data to plot.
    x_values : list
        List of x values for the plot.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure.

    """

    fig = go.Figure()
    traces_lobes=[]
    traces_chs=[]
    for lobe, ch_list in chs_by_lobe.items():
        
        #Add lobe as a category to the plot
        
        for ch_obj in ch_list:
            if ch_obj.name in df_data.columns:
                ch_data=df_data[ch_obj.name].values
                color = ch_obj.lobe_color 
                # normally color must be same for all channels in lobe, so we could assign it before the loop as the color of the first channel,
                # but here it is done explicitly for every channel so that if there is any color error in chs_by_lobe, it will be visible

                traces_chs += [go.Scatter(x=x_values, y=ch_data, line=dict(color=color), name=ch_obj.name, legendgroup=ch_obj.lobe, legendgrouptitle=dict(text=lobe.upper(), font=dict(color=color)))]
                #legendgrouptitle is group tile on the plot. legendgroup is not visible on the plot - it s used for sorting the legend items in update_layout() below.

    # sort traces in random order:
    # When you plot traves right away in the order of the lobes, all the traces of one color lay on top of each other and yu can't see them all.
    # This is why they are not plotted in the loop. So we sort them in random order, so that traces of different colors are mixed.
    traces = traces_lobes + sorted(traces_chs, key=lambda x: random.random())

    # Now first add these traces to the figure and only after that update the layout to make sure that the legend is grouped by lobe.
    fig = go.Figure(data=traces)

    fig.update_layout(legend_traceorder='grouped', legend_tracegroupgap=12, legend_groupclick='toggleitem')
    #You can make it so when you click on lobe title or any channel in lobe you activate/hide all related channels if u set legend_groupclick='togglegroup'.
    #But then you cant see individual channels, it turn on/off the whole group. There is no option to tun group off by clicking on group title. Grup title and group items behave the same.

    #to see the legend: there is really nothing to sort here. The legend is sorted by default by the order of the traces in the figure. The onl way is to group the traces by lobe.
    #print(fig['layout'])

    #https://plotly.com/python/reference/?_ga=2.140286640.2070772584.1683497503-1784993506.1683497503#layout-legend-traceorder
    

    return fig
        

def plot_time_series(raw: mne.io.Raw, m_or_g: str, chs_by_lobe: dict):

    """
    Plots time series of the chosen channels.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw file to be plotted.
    m_or_g_chosen : str
        The type of the channels to be plotted: 'mag' or 'grad'.
    chs_by_lobe : dict
        A dictionary with the keys as the names of the lobes and the values as the lists of the channels in the lobe.
    
    Returns
    -------
    qc_derivative : list
        A list of QC_derivative objects containing the plotly figure with interactive time series of each channel.

    """
    qc_derivative = []
    tit, unit = get_tit_and_unit(m_or_g)

    picked_channels = mne.pick_types(raw.info, meg=m_or_g)

    # Downsample data
    raw_resampled = raw.copy().resample(100, npad='auto') 
    #downsample the data to 100 Hz. The `npad` parameter is set to `'auto'` to automatically determine the amount of padding to use during the resampling process

    data = raw_resampled.get_data(picks=picked_channels) 

    ch_names=[]
    for i in range(data.shape[0]):
        ch_names.append(raw.ch_names[picked_channels[i]])


    #put data in data frame with ch_names as columns:
    df_data=pd.DataFrame(data.T, columns=ch_names)

    fig = plot_df_of_channels_data_as_lines_by_lobe(chs_by_lobe, df_data, raw_resampled.times)

    # Add title, x axis title, x axis slider and y axis units+title:
    fig.update_layout(
        title={
            'text': tit+' time series per channel',
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
    
    qc_derivative += [QC_derivative(content=fig, name=tit+'_time_series', content_type='plotly')]

    return qc_derivative


def plot_time_series_avg(raw: mne.io.Raw, m_or_g: str):

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
        A list of QC_derivative objects containing the plotly figure with interactive average time series.

    """
    qc_derivative = []
    tit, unit = get_tit_and_unit(m_or_g)

    picked_channels = mne.pick_types(raw.info, meg=m_or_g)

    # Downsample data
    raw_resampled = raw.copy().resample(100, npad='auto') 
    #downsample the data to 100 Hz. The `npad` parameter is set to `'auto'` to automatically determine the amount of padding to use during the resampling process

    t = raw_resampled.times
    data = raw_resampled.get_data(picks=picked_channels) 

    #average the data over all channels:
    data_avg = np.mean(data, axis = 0)

    #plot:
    trace = go.Scatter(x=t, y=data_avg, mode='lines', name=tit)
    fig = go.Figure(data=trace)

    # Add title, x axis title, x axis slider and y axis units+title:
    fig.update_layout(
        title={
            'text': tit+': time series averaged over all channels',
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
    
    qc_derivative += [QC_derivative(content=fig, name=tit+'_time_series_avg', content_type='plotly')]

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
    dict(label='Show channels names on hover',
         method='update',
         args=[{'mode': 'markers'}]),
    dict(label='Always show channels names',
         method='update',
         args=[{'mode': 'markers+text'}])
    ]

    # Add the buttons to the layout
    fig.update_layout(updatemenus=[dict(type='buttons',
                                        showactive=True,
                                        buttons=buttons)])

    return fig


def plot_sensors_3d_separated(raw: mne.io.Raw, m_or_g_chosen: str):

    """
    Plots the 3D locations of the sensors in the raw file.
    Not used any more. As it plots mag and grad sensors separately and only if both are chosen for analysis. 
    Also it doesnt care for the lobe areas.

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
                                            layout=go.Layout(width=800, height=800))

        mag_fig.update_layout(
            title={
            'text': 'Magnetometers positions',
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            hoverlabel=dict(font=dict(size=10)))
        

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
                                                layout=go.Layout(width=800, height=800))

        grad_fig.update_layout(
            title={
            'text': 'Gradiometers positions',
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            hoverlabel=dict(font=dict(size=10)))


        # Add the button to have names show up on hover or always:
        grad_fig = switch_names_on_off(grad_fig)

        qc_derivative += [QC_derivative(content=grad_fig, name='Gradiometers_positions', content_type='plotly')]

    return qc_derivative

def keep_unique_locs(ch_list):

    channel_names = [ch.name for ch in ch_list]
    channel_locations = [ch.loc for ch in ch_list]
    channel_colors = [ch.lobe_color for ch in ch_list]
    channel_lobes = [ch.lobe for ch in ch_list]

    # Create dictionaries to store unique locations and combined channel names
    unique_locations = {}
    combined_names = {}
    unique_colors = {}
    unique_lobes = {}

    # Loop through each channel and its location
    for i, (name, loc, color, lobe) in enumerate(zip(channel_names, channel_locations, channel_colors, channel_lobes)):
        # Convert location to a tuple for use as a dictionary key
        loc_key = tuple(loc)
        
        # If location is already in the unique_locations dictionary, add channel name to combined_names
        if loc_key in unique_locations:
            combined_names[unique_locations[loc_key]].append(name)
        # Otherwise, add location to unique_locations and channel name to combined_names
        else:
            unique_locations[loc_key] = i
            combined_names[i] = [name]
            unique_colors[i] = color
            unique_lobes[i] = lobe

    # Create new lists of unique locations and combined channel names
    new_locations = list(unique_locations.keys())
    new_names = [' & '.join(combined_names[i]) for i in combined_names]
    new_colors = [unique_colors[i] for i in unique_colors]
    new_lobes = [unique_lobes[i] for i in unique_lobes]

    return new_locations, new_names, new_colors, new_lobes 


def make_3d_sensors_trace(d3_locs: list, names: list, color: str, textsize: int, legend_category: str = 'channels', symbol: str = 'circle', textposition: str = 'top right'):

    """ Since grads have 2 sensors located in the same spot - need to put their names together to make pretty plot labels.

    Parameters
    ----------
    d3_locs : list
        A list of 3D locations of the sensors.
    names : list
        A list of names of the sensors.
    color : str
        A color of the sensors.
    textsize : int
        A size of the text.
    ch_type : str
        A type of the channels.
    symbol : str
        A symbol of the sensors.
    textposition : str
        A position of the text.
    
    Returns
    -------
    trace : plotly.graph_objs._scatter3d.Scatter3d
        A trace of the sensors.
    
    
    """

    trace = go.Scatter3d(
    x=[loc[0] for loc in d3_locs],
    y=[loc[1] for loc in d3_locs],
    z=[loc[2] for loc in d3_locs],
    mode='markers',
    marker=dict(
        color=color,
        size=8,
        symbol=symbol,
    ),
    text=names,
    hoverinfo='text',
    name=legend_category,
    textposition=textposition,
    textfont=dict(size=textsize, color=color))

    return trace


def plot_sensors_3d(chs_by_lobe: dict):

    """
    Plots the 3D locations of the sensors in the raw file. Plot both mags and grads (if both present) in 1 figure. 
    Can turn mags/grads visialisation on and off.
    Separete channels into brain areas by color coding.


    Parameters
    ----------
    chs_by_lobe : dict
        A dictionary of channels by ch type and lobe.
    
    Returns
    -------
    qc_derivative : list
        A list of QC_derivative objects containing the plotly figures with the sensor locations.

    """

    chs_by_lobe_copy = copy.deepcopy(chs_by_lobe)
    #otherwise we will change the original dict here and keep it messed up for the next function

    qc_derivative = []

    # Put all channels into a simplier dictiary: separatin by lobe byt not by ch type any more as we plot all chs in 1 fig here:
    lobes_dict = {}
    for ch_type in chs_by_lobe_copy:
        for lobe in chs_by_lobe_copy[ch_type]:
            if lobe not in lobes_dict:
                lobes_dict[lobe] = chs_by_lobe_copy[ch_type][lobe]
            else:
                lobes_dict[lobe] += chs_by_lobe_copy[ch_type][lobe]

    traces = []

    if len(lobes_dict)>1: #if there are lobes - we use color coding: one clor pear each lobe
        for lobe in lobes_dict:
            ch_locs, ch_names, ch_color, ch_lobe = keep_unique_locs(lobes_dict[lobe])
            traces.append(make_3d_sensors_trace(ch_locs, ch_names, ch_color[0], 10, ch_lobe[0], 'circle', 'top left'))
            #here color and lobe must be identical for all channels in 1 trace, thi is why we take the first element of the list
            # TEXT SIZE set to 10. This works for the "Always show names" option but not for "Show names on hover" option

    else: #if there are no lobes - we use random colors previously assigned to channels, channel names will be used instead of lobe names in make_3d_trace function
        ch_locs, ch_names, ch_color, ch_lobe = keep_unique_locs(lobes_dict[lobe])
        for i, _ in enumerate(ch_locs):
            traces.append(make_3d_sensors_trace([ch_locs[i]], ch_names[i], ch_color[i], 10, ch_names[i], 'circle', 'top left'))


    fig = go.Figure(data=traces)

    fig.update_layout(
        width=900, height=900,
        title={
        'text': 'Sensors positions',
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    
    fig.update_layout(
        scene = dict(
        xaxis = dict(visible=False),
        yaxis = dict(visible=False),
        zaxis =dict(visible=False)
        )
    )

    #check_num_channels_correct(chs_by_lobe, 'END_PLOT') #check we didnt change the original dict


    # Add the button to have names show up on hover or always:
    fig = switch_names_on_off(fig)

    fig.update_traces(hoverlabel=dict(font=dict(size=10))) #TEXT SIZE set to 10 again. This works for the "Show names on hover" option, but not for "Always show names" option

    qc_derivative += [QC_derivative(content=fig, name='Sensors_positions', content_type='plotly', description_for_user="Magnetometers names end with '1' like 'MEG0111'. Gradiometers names end with '2' and '3' like 'MEG0112', 'MEG0113'. ")]

    return qc_derivative


def boxplot_epochs(df_mg: pd.DataFrame, ch_type: str, what_data: str, x_axis_boxes: str, verbose_plots: bool):

    """
    Creates representation of calculated data as multiple boxplots. Used in STD and PtP_manual measurements. 

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
    verbose_plots : bool
        True for showing plot in notebook.

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
        xaxis_title='Experimental epochs',
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
        
    if verbose_plots is True:
        fig.show()

    fig_deriv = QC_derivative(content=fig, name=fig_name, content_type='plotly')

    return fig_deriv


def boxplot_epoched_xaxis_channels(chs_by_lobe: dict, df_std_ptp: pd.DataFrame, ch_type: str, what_data: str, verbose_plots: bool):

    """
    Creates representation of calculated data as multiple boxplots. Used in STD and PtP_manual measurements. 
    Color tagged channels by lobes. 
    One box is one channel, boxes are on x axis. Epoch are inside as dots. Y axis shows the STD/PtP value.
    
    Parameters
    ----------
    chs_by_lobe : dict
        Dictionary with channel objects sorted by lobe.
    df_std_ptp : pd.DataFrame
        Data Frame containing std or ptp value for each chnnel and each epoch
    ch_type : str
        Type of the channel: 'mag', 'grad'
    what_data : str
        Type of the data: 'peaks' or 'stds'
    x_axis_boxes : str
        What to plot as boxplot on x axis: 'channels' or 'epochs'
    verbose_plots : bool
        True for showing plot in notebook.

    Returns
    -------
    fig_deriv : QC_derivative 
        derivative containing plotly figure
    
    """

    epochs_names = df_std_ptp.columns.tolist()
    

    ch_tit, unit = get_tit_and_unit(ch_type)

    if what_data=='peaks':
        hover_tit='PtP Amplitude'
        y_ax_and_fig_title='Peak-to-peak amplitude'
        fig_name='PP_manual_epoch_per_channel_'+ch_tit
    elif what_data=='stds':
        hover_tit='STD'
        y_ax_and_fig_title='Standard deviation'
        fig_name='STD_epoch_per_channel_'+ch_tit
    else:
        print('what_data should be either peaks or stds')

    x_axis_boxes = 'channels'
    if x_axis_boxes=='channels':
        hovertemplate='Epoch: %{text}<br>'+hover_tit+': %{y: .2e}'
    elif x_axis_boxes=='epochs':
        #legend_title = 'Epochs'
        hovertemplate='%{text}<br>'+hover_tit+': %{y: .2e}'
    else:
        print('x_axis_boxes should be either channels or epochs')


    fig = go.Figure()

    #Here each trace is 1 box representing 1 channel. Epochs inside the box are automatically plotted given argument boxpoints="all":
    #Boxes are groupped by lobe. So first each channel fo lobe 1 is plotted, then each of lobe 2, etc..
    boxes_names = []
    for lobe,  ch_list in chs_by_lobe.items():
        for ch in ch_list:
            if what_data == 'stds':
                data = ch.std_epoch
            elif what_data == 'peaks':
                data = ch.ptp_epoch
            
            boxes_names += [ch.name]

            fig.add_trace(go.Box(y=data, 
            name=ch.name, 
            opacity=0.7, 
            boxpoints="all", 
            pointpos=0,
            marker_color=ch.lobe_color,
            marker_size=3,
            legendgroup=ch.lobe, 
            legendgrouptitle=dict(text=lobe.upper()),
            line_width=0.8,
            line_color=ch.lobe_color,
            text=epochs_names))

    fig.update_traces(hovertemplate=hovertemplate)

    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [v for v in range(0, len(boxes_names))],
            ticktext = boxes_names,
            rangeslider=dict(visible=True)),
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'),
        yaxis_title=y_ax_and_fig_title+' in '+unit,
        title={
            'text': y_ax_and_fig_title+' over epochs for '+ch_tit,
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},)
        #legend_title=legend_title)
        
    if verbose_plots is True:
        fig.show()

    fig_deriv = QC_derivative(content=fig, name=fig_name, content_type='plotly')

    return fig_deriv

def assign_epoched_std_ptp_to_channels(what_data, chs_by_lobe, df_std_ptp):

    """
    Assign std or ptp va;ues of each epoch as list to each channel. 
    This is done for easier plotting when need to plot epochs per channel and also color coded by lobes.
    
    Parameters
    ----------
    what_data : str
        'peaks' for peak-to-peak amplitudes or 'stds'
    chs_by_lobe : dict
        dictionary with channel objects sorted by lobe.
    df_std_ptp : pd.DataFrame
        Data Frame containing std or ptp value for each chnnel and each epoch
    
        
    Returns
    -------
    chs_by_lobe : dict
        updated dictionary with channel objects sorted by lobe - with info about std or ptp of epochs.
    """

    if what_data=='peaks':
        #Add the data about std of each epoch (as a list, 1 std for 1 epoch) into each channel object inside the chs_by_lobe dictionary:
        for lobe in chs_by_lobe:
            for ch in chs_by_lobe[lobe]:
                ch.ptp_epoch = df_std_ptp.loc[ch.name].values
    elif what_data=='stds':
        for lobe in chs_by_lobe:
            for ch in chs_by_lobe[lobe]:
                ch.std_epoch = df_std_ptp.loc[ch.name].values
    else:
        print('what_data should be either peaks or stds')

    return chs_by_lobe


def boxplot_epoched_xaxis_epochs(chs_by_lobe: dict, df_std_ptp: pd.DataFrame, ch_type: str, what_data: str, verbose_plots: bool):

    """
    Represent std of epochs for each channel as box plots, where each box on x axis is 1 epoch. Dots inside the box are channels.
    
    Process: 
    Each box need to be plotted as a separate trace first.
    Each channels inside each box has to be plottted as separate trace to allow diffrenet color coding
    
    For each box_representing_epoch:
        box trace
        For each color coded lobe:
            For each dot_representing_channel in lobe:
                dot trace

    Add all traces to plotly figure


    Parameters
    ----------
    chs_by_lobe : dict
        dictionary with channel objects sorted by lobe.
    df_std_ptp : pd.DataFrame
        Data Frame containing std or ptp value for each chnnel and each epoch
    ch_type : str
        'mag' or 'grad'
    what_data : str
        'peaks' for peak-to-peak amplitudes or 'stds'
    verbose_plots : bool
        True for showing plot in notebook.

    Returns
    -------
    QC_derivative
        QC_derivative object with plotly figure as content

    """

    epochs_names = df_std_ptp.columns.tolist()

    ch_tit, unit = get_tit_and_unit(ch_type)

    if what_data=='peaks':
        hover_tit='PtP Amplitude'
        y_ax_and_fig_title='Peak-to-peak amplitude'
        fig_name='PP_manual_epoch_per_channel_2_'+ch_tit
    elif what_data=='stds':
        hover_tit='STD'
        y_ax_and_fig_title='Standard deviation'
        fig_name='STD_epoch_per_channel_2_'+ch_tit
    else:
        print('what_data should be either peaks or stds')


    boxwidth=0.5 #the area around which the data dots are scattered depends on the width of the box.

    # For this plot have to separately create a box (no data points plotted) as 1 trace
    # Then separately create for each cannel (dot) a separate trace. It s the only way to make them all different lobe colors.
    # Additionally, the dots are scattered along the x axis inside each box, this is done for visualisation only, x position does not hold information.
    
    # Put all data dots in a list of traces groupped by lobe:
    
    dot_traces = []
    box_traces = []

    for ep_number, ep_name in enumerate(epochs_names):
        dots_in_1_box=[]
        for lobe,  ch_list in chs_by_lobe.items():
            for ch in ch_list:
                if what_data == 'stds':
                    data = ch.std_epoch[ep_number]
                elif what_data == 'peaks':
                    data = ch.ptp_epoch[ep_number]
                dots_in_1_box += [data]

                x = ep_number + random.uniform(-0.2*boxwidth, 0.2*boxwidth) 
                #here create random y values for data dots, they dont have a meaning, just used so that dots are scattered around the box plot and not in 1 line.
                
                dot_traces += [go.Scatter(x=[x], y=[data], mode='markers', marker=dict(size=4, color=ch.lobe_color), opacity=0.8, name=ch.name, text=str(ep_name), legendgroup=ch.lobe, legendgrouptitle=dict(text=lobe.upper()), hovertemplate='Epoch: '+str(ep_name)+'<br>'+hover_tit+': %{y: .2e}')]

        # create box plot trace
        box_traces += [go.Box(x0=ep_number, y=dots_in_1_box, orientation='v', name=ep_name, line_width=1.8, opacity=0.8, boxpoints=False, width=boxwidth, showlegend=False)]
    
    #Collect all traces and add them to the figure:

    all_traces = box_traces+dot_traces
    fig = go.Figure(data=all_traces)
        
    #more settings:
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [v for v in range(0, len(epochs_names))],
            ticktext = epochs_names,
            rangeslider=dict(visible=True)
        ),
        xaxis_title='Experimental epochs',
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
        legend_groupclick='togglegroup') #this setting allowes to select the whole group when clicking on 1 element of the group. But then you can not select only 1 element.
    
    if verbose_plots is True:
        fig.show()

    qc_derivative = QC_derivative(content=fig, name=fig_name, content_type='plotly')

    return qc_derivative



def boxplot_epochs_old(df_mg: pd.DataFrame, ch_type: str, what_data: str, verbose_plots: bool) -> QC_derivative:

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
    verbose_plots : bool
        True for showing plot in notebook.

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
        
    if verbose_plots is True:
        fig.show()

    qc_derivative = QC_derivative(content=fig, name=fig_name, content_type='plotly')

    return qc_derivative


def boxplot_all_time_OLD(std_data_named: dict, ch_type: str, channels: list, what_data: str, verbose_plots: bool):

    """
    Create representation of calculated std data as a boxplot (box containd magnetometers or gradiomneters, not together): 
    each dot represents 1 channel: name: std value over whole data of this channel. Too high/low stds are outliers.
    OLD but still working version, currently not used. Other principal than the current one so left for reference.

    Parameters
    ----------
    std_data_named : dict
        std values for each channel
    ch_type : str
        'mag' or 'grad'
    channels : list
        list of channel names
    what_data : str
        'peaks' for peak-to-peak amplitudes or 'stds'
    verbose_plots : bool
        True for showing plot in notebook.

    Returns
    -------
    QC_derivative
        QC_derivative object with plotly figure as content

    """
    # Put all values in 1 array from the dictionsry:
    std_data = np.array(list(std_data_named.values()))

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
        
    if verbose_plots is True:
        fig.show()

    qc_derivative = QC_derivative(content=fig, name=fig_name, content_type='plotly')

    return qc_derivative


def boxplot_all_time(chs_by_lobe: dict, ch_type: str, what_data: str, verbose_plots: bool):

    """
    Create representation of calculated std data as a boxplot over the whoe time series, not epoched.
    (box contains magnetometers or gradiomneters, not together): 
    each dot represents 1 channel (std value over whole data of this channel). Too high/low stds are outliers.

    Parameters
    ----------
    chs_by_lobe : dict
        dictionary with channel objects sorted by lobe.
    ch_type : str
        'mag' or 'grad'
    channels : list
        list of channel names
    what_data : str
        'peaks' for peak-to-peak amplitudes or 'stds'
    verbose_plots : bool
        True for showing plot in notebook.

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
    else:
        raise ValueError('what_data must be set to "stds" or "peaks"')

    boxwidth=0.4 #the area around which the data dots are scattered depends on the width of the box.

    # For this plot have to separately create a box (no data points plotted) as 1 trace
    # Then separately create for each cannel (dot) a separate trace. It s the only way to make them all different lobe colors.
    # Additionally, the dots are scattered along the y axis, this is done for visualisation only, y position does not hold information.
    
    # Put all data dots in a list of traces groupped by lobe:
    values_all=[]
    traces = []

    for lobe,  ch_list in chs_by_lobe.items():
        for ch in ch_list:
            if what_data == 'stds':
                data = ch.std_overall
            elif what_data == 'peaks':
                data = ch.ptp_overall
            values_all += [data]

            y = random.uniform(-0.2*boxwidth, 0.2*boxwidth) 
            #here create random y values for data dots, they dont have a meaning, just used so that dots are scattered around the box plot and not in 1 line.
            
            traces += [go.Scatter(x=[data], y=[y], mode='markers', marker=dict(size=5, color=ch.lobe_color), name=ch.name, legendgroup=ch.lobe, legendgrouptitle=dict(text=lobe.upper()))]


    # create box plot trace
    box_trace = go.Box(x=values_all, y0=0, orientation='h', name='box', line_width=1, opacity=0.7, boxpoints=False, width=boxwidth, showlegend=False)
    
    #Colllect all traces and add them to the figure:
    all_traces = [box_trace]+traces
    fig = go.Figure(data=all_traces)

    #Add hover text to the dots, remove too many digits after coma.
    fig.update_traces(hovertemplate=hover_tit+': %{x: .2e}')
        
    #more settings:
    fig.update_layout(
        yaxis_range=[-0.5,0.5],
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
        'yanchor': 'top'},
        legend_groupclick='togglegroup') #this setting allowes to select the whole group when clicking on 1 element of the group. But then you can not select only 1 element.
    
    if verbose_plots is True:
        fig.show()

    description_for_user = 'Positions of points on the Y axis do not hold information, made for visialisation only.'
    qc_derivative = QC_derivative(content=fig, name=fig_name, content_type='plotly', description_for_user = description_for_user)

    return qc_derivative




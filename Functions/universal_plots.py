
import plotly.graph_objects as go
import pandas as pd

def boxplot_channel_epoch_hovering_plotly(df_mg: pd.DataFrame, tit: str, sid: str, what_data: str):

    '''
    Creates representation of calculated data as multiple boxplots: 
    each box represents 1 channel, each dot is std of 1 epoch in this channel
    Implemented with plotly: https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Box.html
    The figure will be saved as an interactive html file.

    Args:
    df_mg(pd.DataFrame): data frame containing data (stds, peak-to-peak amplitudes, etc) for each epoch, each channel, 
        mags OR grads, not together
    tit (str): title, like "Magnetometers", or "Gradiometers", 
    sid (str): subject id number, like '1'
    what_data (str): 'peaks' for peak-to-peak amplitudes or 'stds'

    Returns:
    fig (go.Figure): plottly figure
    fig_path (str): path where the figure is saved as html file
    '''

    unit='?'
    if tit=='Magnetometers':
        unit='T'
    elif tit=='Gradiometers':
        unit='T/m'
    else:
        print('Please check tit input. Has to be "Magnetoneters" or "Gradiometers"')

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
        text=df_mg_transposed[col].index))

        if what_data=='peaks':
            fig.update_traces(hovertemplate='Epoch: %{text}'+'<br>Amplitude: %{y: .0f}')
        elif what_data=='stds':
            fig.update_traces(hovertemplate='Epoch: %{text}'+'<br>STD: %{y: .0f}')
        

    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [v for v in range(0, len(ch_names))],
            ticktext = ch_names,
            rangeslider=dict(visible=True)
        ),
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'))
        
    if what_data=='peaks':
        fig.update_layout(
        yaxis_title="Peak-to-peak amplitude in "+unit,
        title={
            'text': "Peak-to-peak amplitude over epochs for "+tit,
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    elif what_data=='stds':
        fig.update_layout(
        yaxis_title="standard deviation in "+unit,
        title={
            'text': "STDs over epochs for "+tit,
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
        
    fig.show()

    if what_data=='peaks':
        fig_name='PP_amplitude_epochs_per_channel_'+tit+'.html'
    elif what_data=='stds':
        fig_name='Stds_epochs_per_channel_'+tit+'.html'

    fig_path='../derivatives/sub-'+sid+'/megqc/figures/'+fig_name
    fig.write_html(fig_path)

    return(fig, fig_path)
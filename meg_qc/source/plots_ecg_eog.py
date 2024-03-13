import pandas as pd
import numpy as np
import plotly.graph_objects as go

from meg_qc.source.universal_plots import QC_derivative, get_tit_and_unit, plot_df_of_channels_data_as_lines_by_lobe_csv

def figure_x_axis(df, metric):
     
    if metric.lower() == 'psd':
        # Figure out frequencies:
        freq_cols = [column for column in df if column.startswith('PSD_Hz_')]
        freqs = np.array([float(x.replace('PSD_Hz_', '')) for x in freq_cols])
        return freqs
    
    elif metric.lower() == 'eog' or metric.lower() == 'ecg' or metric.lower() == 'muscle' or metric.lower() == 'head':
        if metric.lower() == 'ecg':
            prefix = 'mean_ecg_sec_'
        elif metric.lower() == 'eog': 
            prefix = 'mean_eog_sec_'
        elif metric.lower() == 'smoothed_ecg' or metric.lower() == 'ecg_smoothed':
            prefix = 'smoothed_mean_ecg_sec_'
        elif metric.lower() == 'smoothed_eog' or metric.lower() == 'eog_smoothed':
            prefix = 'smoothed_mean_eog_sec_'
        elif metric.lower() == 'muscle':
            prefix = 'Muscle_sec_'
        elif metric.lower() == 'head':
            prefix = 'Head_sec_'
        
        time_cols = [column for column in df if column.startswith(prefix)]
        time_vec = np.array([float(x.replace(prefix, '')) for x in time_cols])

        return time_vec
    
    else:
        print('Oh well IDK! figure_x_axis()')
        return None
    

def split_correlated_artifacts_into_3_groups_csv(df, metric):

    """
    Collect artif_per_ch into 3 lists - for plotting:
    - a third of all channels that are the most correlated with mean_rwave
    - a third of all channels that are the least correlated with mean_rwave
    - a third of all channels that are in the middle of the correlation with mean_rwave

    Parameters
    ----------
    artif_per_ch : list
        List of objects of class Avg_artif

    Returns
    -------
    artif_per_ch : list
        List of objects of class Avg_artif, ranked by correlation coefficient
    most_correlated : list
        List of objects of class Avg_artif that are the most correlated with mean_rwave
    least_correlated : list
        List of objects of class Avg_artif that are the least correlated with mean_rwave
    middle_correlated : list
        List of objects of class Avg_artif that are in the middle of the correlation with mean_rwave
    corr_val_of_last_least_correlated : float
        Correlation value of the last channel in the list of the least correlated channels
    corr_val_of_last_middle_correlated : float
        Correlation value of the last channel in the list of the middle correlated channels
    corr_val_of_last_most_correlated : float
        Correlation value of the last channel in the list of the most correlated channels


    """

    #sort by correlation coef. Take abs of the corr coeff, because the channels might be just flipped due to their location against magnetic field::
    #artif_per_ch.sort(key=lambda x: abs(x.corr_coef), reverse=True)

    if metric.lower() != 'ecg' and metric.lower() != 'eog':
        print('Wrong metric in split_correlated_artifacts_into_3_groups_csv()')


    df_sorted = df.copy()    
    df_sorted.sort_values(by = metric.lower()+'_corr_coeff') 

    most_correlated = df_sorted.copy()[:int(len(df_sorted)/3)]
    least_correlated = df_sorted.copy()[-int(len(df_sorted)/3):]
    middle_correlated = df_sorted.copy()[int(len(df_sorted)/3):-int(len(df_sorted)/3)]

    #get correlation values of all most correlated channels:
    all_most_correlated = most_correlated[metric.lower()+'_corr_coeff'].abs().tolist()
    all_middle_correlated = middle_correlated[metric.lower()+'_corr_coeff'].abs().tolist()
    all_least_correlated = least_correlated[metric.lower()+'_corr_coeff'].abs().tolist()

    #find the correlation value of the last channel in the list of the most correlated channels:
    # this is needed for plotting correlation values, to know where to put separation rectangles.
    corr_val_of_last_most_correlated = max(all_most_correlated)
    corr_val_of_last_middle_correlated = max(all_middle_correlated)
    corr_val_of_last_least_correlated = max(all_least_correlated)

    return most_correlated, middle_correlated, least_correlated, corr_val_of_last_most_correlated, corr_val_of_last_middle_correlated, corr_val_of_last_least_correlated


def plot_affected_channels_csv(df, artifact_lvl: float, t: np.ndarray, m_or_g: str, ecg_or_eog: str, title: str, flip_data: bool or str = 'flip', smoothed: bool = False, verbose_plots: bool = True):

    """
    Plot the mean artifact amplitude for all affected (not affected) channels in 1 plot together with the artifact_lvl.
    
    Parameters
    ----------
    artif_affected_channels : list
        List of ECG/EOG artifact affected channels.
    artifact_lvl : float
        The threshold for the artifact amplitude: average over all channels*norm_lvl.
    t : np.ndarray
        Time vector.
    m_or_g : str
        Either 'mag' or 'grad'.
    fig_tit: str
        The title of the figure.
    chs_by_lobe : dict
        dictionary with channel objects sorted by lobe
    flip_data : bool
        If True, the absolute value of the data will be used for the calculation of the mean artifact amplitude. Default to 'flip'. 
        'flip' means that the data will be flipped if the peak of the artifact is negative. 
        This is donr to get the same sign of the artifact for all channels, then to get the mean artifact amplitude over all channels and the threshold for the artifact amplitude onbase of this mean
        And also for the reasons of visualization: the artifact amplitude is always positive.
    smoothed: bool
        Plot smoothed data (true) or nonrmal (false)
    verbose_plots : bool
        True for showing plot in notebook.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure with the mean artifact amplitude for all affected (not affected) channels in 1 plot together with the artifact_lvl.

        
    """

    fig_tit=ecg_or_eog+title

    #if df and not df.empty: #if affected channels present:
    if df is not None:
        if smoothed is True:
            metric = ecg_or_eog+'_smoothed'
        elif smoothed is False:
            metric = ecg_or_eog
        fig = plot_df_of_channels_data_as_lines_by_lobe_csv(None, metric, t, m_or_g, df)

        #decorate the plot:
        ch_type_tit, unit = get_tit_and_unit(m_or_g)
        fig.update_layout(
            xaxis_title='Time in seconds',
            yaxis = dict(
                showexponent = 'all',
                exponentformat = 'e'),
            yaxis_title='Mean artifact magnitude in '+unit,
            title={
                'text': fig_tit+str(len(df))+' '+ch_type_tit,
                'y':0.85,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})


    else:
        fig=go.Figure()
        ch_type_tit, _ = get_tit_and_unit(m_or_g)
        title=fig_tit+'0 ' +ch_type_tit
        fig.update_layout(
            title={
            'text': title,
            'x': 0.5,
            'y': 0.9,
            'xanchor': 'center',
            'yanchor': 'top'})
        
    #in any case - add the threshold on the plot
    fig.add_trace(go.Scatter(x=t, y=[(artifact_lvl)]*len(t), line=dict(color='red'), name='Thres=mean_peak/norm_lvl')) #add threshold level

    if flip_data is False and artifact_lvl is not None: 
        fig.add_trace(go.Scatter(x=t, y=[(-artifact_lvl)]*len(t), line=dict(color='black'), name='-Thres=mean_peak/norm_lvl'))

    if verbose_plots is True:
        fig.show()

    return fig

def plot_artif_per_ch_correlated_lobes_csv(f_path: str, m_or_g: str, ecg_or_eog: str, flip_data: bool, verbose_plots: bool):

    """
    THE FINAL func

    TODO:
    maybe remove reading csv and pass directly the df here?
    adjust docstrings


    Plot average artifact for each channel, colored by lobe, 
    channels are split into 3 separate plots, based on their correlation with mean_rwave: equal number of channels in each group.

    Parameters
    ----------
    artif_per_ch : list
        List of objects of class Avg_artif
    tmin : float
        Start time of the epoch (negative value)
    tmax : float
        End time of the epoch
    m_or_g : str
        Type of the channel: mag or grad
    ecg_or_eog : str
        Type of the artifact: ECG or EOG
    chs_by_lobe : dict
        Dictionary with channels split by lobe
    flip_data : bool
        Use True or False, doesnt matter here. It is only passed into the plotting function and influences the threshold presentation. But since treshold is not used in correlation method, this is not used.
    verbose_plots : bool
        If True, plots are shown in the notebook.

    Returns
    -------
    artif_per_ch : list
        List of objects of class Avg_artif
    affected_derivs : list
        List of objects of class QC_derivative (plots)
    

    """


    ecg_or_eog = ecg_or_eog.lower()

    df = pd.read_csv(f_path, sep='\t') #TODO: maybe remove reading csv and pass directly the df here?
    df = df.drop(df[df['Type'] != m_or_g].index) #remove non needed channel kind

    artif_time_vector = figure_x_axis(df, metric=ecg_or_eog)

    most_correlated, middle_correlated, least_correlated, _, _, _ = split_correlated_artifacts_into_3_groups_csv(df, ecg_or_eog)

    smoothed = True
    fig_most_affected = plot_affected_channels_csv(most_correlated, None, artif_time_vector, m_or_g, ecg_or_eog, title = ' most affected channels (smoothed): ', flip_data=flip_data, smoothed = smoothed, verbose_plots=False)
    fig_middle_affected = plot_affected_channels_csv(middle_correlated, None, artif_time_vector, m_or_g, ecg_or_eog, title = ' middle affected channels (smoothed): ', flip_data=flip_data, smoothed = smoothed, verbose_plots=False)
    fig_least_affected = plot_affected_channels_csv(least_correlated, None, artif_time_vector, m_or_g, ecg_or_eog, title = ' least affected channels (smoothed): ', flip_data=flip_data, smoothed = smoothed, verbose_plots=False)


    #set the same Y axis limits for all 3 figures for clear comparison:

    if ecg_or_eog.lower() == 'ecg' and smoothed is False:
        prefix = 'mean_ecg_sec_'
    elif ecg_or_eog.lower() == 'ecg' and smoothed is True:
        prefix = 'smoothed_mean_ecg_sec_'
    elif ecg_or_eog.lower() == 'eog' and smoothed is False:
        prefix = 'mean_eog_sec_'
    elif ecg_or_eog.lower() == 'eog' and smoothed is True:
        prefix = 'smoothed_mean_eog_sec_'

    cols = [column for column in df if column.startswith(prefix)]
    cols = ['Name']+cols

    limits_df = df[cols]

    ymax = limits_df.loc[:, limits_df.columns != 'Name'].max().max()
    ymin = limits_df.loc[:, limits_df.columns != 'Name'].min().min()

    ylim = [ymin*.95, ymax*1.05]

    # update the layout of all three figures with the same y-axis limits
    fig_most_affected.update_layout(yaxis_range=ylim)
    fig_middle_affected.update_layout(yaxis_range=ylim)
    fig_least_affected.update_layout(yaxis_range=ylim)

    if verbose_plots is True:
        fig_most_affected.show()
        fig_middle_affected.show()
        fig_least_affected.show()
    
    affected_derivs = []
    affected_derivs += [QC_derivative(fig_most_affected, ecg_or_eog+'most_affected_channels_'+m_or_g, 'plotly')]
    affected_derivs += [QC_derivative(fig_middle_affected, ecg_or_eog+'middle_affected_channels_'+m_or_g, 'plotly')]
    affected_derivs += [QC_derivative(fig_least_affected, ecg_or_eog+'least_affected_channels_'+m_or_g, 'plotly')]

   
    return affected_derivs


def plot_correlation_csv(f_path, ecg_or_eog, m_or_g, verbose_plots=False):

    """
    Plot correlation coefficient and p-value between mean R wave and each channel in artif_per_ch.

    Parameters
    ----------
    artif_per_ch : list
        List of channels with Avg_artif objects.
    ecg_or_eog : str
        Either 'ECG' or 'EOG'.
    m_or_g : str
        Either 'mag' or 'grad'.
    verbose_plots : bool
        If True, plot will be displayed in a notebook.

    Returns
    -------
    corr_derivs : list
        List with 1 QC_derivative instance: Figure with correlation coefficient and p-value between mean R wave and each channel in artif_per_ch.
    
    """

    ecg_or_eog = ecg_or_eog.lower()

    df = pd.read_csv(f_path, sep='\t') #TODO: maybe remove reading csv and pass directly the df here?
    df = df.drop(df[df['Type'] != m_or_g].index) #remove non needed channel kind

    _, _, _, corr_val_of_last_most_correlated, corr_val_of_last_middle_correlated, corr_val_of_last_least_correlated = split_correlated_artifacts_into_3_groups_csv(df, ecg_or_eog)

    traces = []

    tit, _ = get_tit_and_unit(m_or_g)

    for index, row in df.iterrows():
        traces += [go.Scatter(x=[abs(row[ecg_or_eog.lower()+'_corr_coeff'])], y=[row[ecg_or_eog.lower()+'_pval']], mode='markers', marker=dict(size=5, color=row['Lobe Color']), name=row['Name'], legendgroup=row['Lobe Color'], legendgrouptitle=dict(text=row['Lobe'].upper()), hovertemplate='Corr coeff: '+str(row[ecg_or_eog.lower()+'_corr_coeff'])+'<br>p-value: '+str(abs(row[ecg_or_eog.lower()+'_pval'])))]

    fig = go.Figure(data=traces)

    fig.add_shape(type="rect", xref="x", yref="y", x0=0, y0=-0.1, x1=corr_val_of_last_least_correlated, y1=1.1, line=dict(color="Green", width=2), fillcolor="Green", opacity=0.1)
    fig.add_shape(type="rect", xref="x", yref="y", x0=corr_val_of_last_least_correlated, y0=-0.1, x1=corr_val_of_last_middle_correlated, y1=1.1, line=dict(color="Yellow", width=2), fillcolor="Yellow", opacity=0.1)
    fig.add_shape(type="rect", xref="x", yref="y", x0=corr_val_of_last_middle_correlated, y0=-0.1, x1=1, y1=1.1, line=dict(color="Red", width=2), fillcolor="Red", opacity=0.1)

    #set axis titles:
    fig.update_xaxes(title_text='Correlation coefficient')
    fig.update_yaxes(title_text='P-value')

    #set title:
    fig.update_layout(title_text=tit+': Pearson correlation between reference '+ecg_or_eog+' epoch and '+ecg_or_eog+' epoch in each channel')

    if verbose_plots is True:
        fig.show()

    corr_derivs = [QC_derivative(fig, 'Corr_values_'+ecg_or_eog, 'plotly', description_for_user='Absolute value of the correlation coefficient is shown here. The sign would only represent the position of the channel towards magnetic field. <p>- Green: 33% of all channels that have the weakest correlation with mean ' +ecg_or_eog +'; </p> <p>- Yellow: 33% of all channels that have mild correlation with mean ' +ecg_or_eog +';</p> <p>- Red: 33% of all channels that have the stronges correlation with mean ' +ecg_or_eog +'. </p>')]

    return corr_derivs
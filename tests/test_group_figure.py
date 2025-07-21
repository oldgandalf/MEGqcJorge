import os
import pandas as pd
from meg_qc.calculation.metrics.summary_report_GQI import create_group_metrics_figure


def test_create_group_metrics_figure(tmp_path):
    df = pd.DataFrame({
        'GQI': [80, 90],
        'GQI_penalty_ch': [10, 5],
        'GQI_penalty_corr': [0, 5],
        'GQI_penalty_mus': [5, 0],
        'GQI_penalty_psd': [2, 3],
        'GQI_bad_pct': [1.0, 2.0],
        'GQI_ecg_pct': [0.1, 0.2],
        'GQI_eog_pct': [0.1, 0.0],
        'GQI_muscle_pct': [0.01, 0.02],
        'GQI_psd_noise_pct': [3.0, 4.0],
    })
    tsv = tmp_path / 'data.tsv'
    df.to_csv(tsv, sep='\t', index=False)
    png = tmp_path / 'out.png'
    create_group_metrics_figure(tsv, png)
    assert png.exists()


def test_group_figure_handles_missing_columns(tmp_path):
    df = pd.DataFrame({
        'GQI_bad_pct': [1.0, 2.0],
        'GQI_psd_noise_pct': [3.0, 4.0],
    })
    tsv = tmp_path / 'data2.tsv'
    df.to_csv(tsv, sep='\t', index=False)
    png = tmp_path / 'out2.png'
    create_group_metrics_figure(tsv, png)
    assert png.exists()

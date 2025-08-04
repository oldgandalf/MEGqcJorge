import pandas as pd
import numpy as np
from meg_qc.calculation.between_sample_analysis import _mutual_information, analyze_mutual_information


def test_mutual_information(tmp_path):
    df = pd.DataFrame({
        'GQI_std_pct': [0.1, 0.2, 0.15],
        'GQI_ptp_pct': [0.3, 0.4, 0.35],
        'GQI_ecg_pct': [0.05, 0.1, 0.08],
        'GQI_eog_pct': [0.02, 0.03, 0.025],
        'GQI_muscle_pct': [0.01, 0.02, 0.015],
        'GQI_psd_noise_pct': [0.5, 0.6, 0.55],
    })
    tsv = tmp_path / 'mi.tsv'
    png = tmp_path / 'mi.png'
    _mutual_information(df, list(df.columns), png, tsv)
    assert tsv.exists()
    assert png.exists()


def test_mutual_information_with_permutation():
    df = pd.DataFrame({
        'GQI_std_pct': [0.1, 0.2, 0.15],
        'GQI_ptp_pct': [0.3, 0.4, 0.35],
        'GQI_ecg_pct': [0.05, 0.1, 0.08],
        'GQI_eog_pct': [0.02, 0.03, 0.025],
        'GQI_muscle_pct': [0.01, 0.02, 0.015],
        'GQI_psd_noise_pct': [0.5, 0.6, 0.55],
    })
    res = analyze_mutual_information(
        df,
        list(df.columns),
        n_permutations=10,
        permutation_test=True,
        seed=0,
    )
    assert np.allclose(res["mi"], res["mi"].T, equal_nan=True)
    assert res["p"].notna().values.sum() > 0
    assert res["p_fdr"].notna().values.sum() > 0
    assert res["z"].notna().values.sum() > 0

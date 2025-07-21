import json
from meg_qc.calculation.meg_qc_pipeline import flatten_summary_metrics

def test_flatten_summary_metrics():
    js = {
        "GQI": 80.0,
        "STD_time_series": [
            {"Metric": "Flat Channels", "MAGNETOMETERS": "18 (17.6%)", "GRADIOMETERS": "0 (0.0%)"},
            {"Metric": "Noisy Channels", "MAGNETOMETERS": "16 (15.7%)", "GRADIOMETERS": "11 (5.4%)"}
        ],
        "PTP_time_series": [
            {"Metric": "Flat Channels", "MAGNETOMETERS": "6 (5.9%)", "GRADIOMETERS": "0 (0.0%)"},
            {"Metric": "Noisy Channels", "MAGNETOMETERS": "20 (19.6%)", "GRADIOMETERS": "19 (9.3%)"}
        ],
        "STD_epoch_summary": [
            {"Sensor Type": "MAGNETOMETERS", "Noisy Epochs": "16 (15.7%)", "Flat Epochs": "18 (17.6%)"},
            {"Sensor Type": "GRADIOMETERS", "Noisy Epochs": "11 (5.4%)", "Flat Epochs": "0 (0.0%)"}
        ],
        "PTP_epoch_summary": [
            {"Sensor Type": "MAGNETOMETERS", "Noisy Epochs": "20 (19.6%)", "Flat Epochs": "6 (5.9%)"},
            {"Sensor Type": "GRADIOMETERS", "Noisy Epochs": "19 (9.3%)", "Flat Epochs": "0 (0.0%)"}
        ],
        "ECG_correlation_summary": [
            {"Sensor Type": "MAGNETOMETERS", "# |High Correlations| > 0.8": "0 (0.0%)", "Total Channels": 102},
            {"Sensor Type": "GRADIOMETERS", "# |High Correlations| > 0.8": "0 (0.0%)", "Total Channels": 204}
        ],
        "EOG_correlation_summary": [
            {"Sensor Type": "MAGNETOMETERS", "# |High Correlations| > 0.8": "0 (0.0%)", "Total Channels": 102},
            {"Sensor Type": "GRADIOMETERS", "# |High Correlations| > 0.8": "0 (0.0%)", "Total Channels": 204}
        ],
        "PSD_noise_summary": [
            {"Metric": "Noise Power", "MAGNETOMETERS": "49.48%", "GRADIOMETERS": "17.91%"}
        ],
        "Muscle_events": {"# Muscle Events": 1, "total_number_of_events": 542001},
        "GQI_penalties": {"ch": 0.0, "corr": 0.0, "mus": 0.0, "psd": 20.0},
        "GQI_metrics": {
            "bad_pct": 9.1875,
            "std_pct": 4.0,
            "ptp_pct": 5.0,
            "ecg_pct": 0.0,
            "eog_pct": 0.0,
            "muscle_pct": 0.0001845015046097701,
            "psd_noise_pct": 33.695
        },
        "parameters": {"std_lvl": 1, "ptp_lvl": 1.0, "std_epoch_lvl": 1.2, "ptp_epoch_lvl": 1.2}
    }

    row = flatten_summary_metrics(js)
    assert row["GQI"] == 80.0
    assert row["STD_ts_flat_channels_mag_num"] == 18
    assert row["STD_ts_flat_channels_mag_percentage"] == 17.6
    assert row["PTP_ts_noisy_channels_grad_num"] == 19
    assert row["PSD_noise_mag_percentage"] == 49.48
    assert row["Muscle_events_num"] == 1
    assert row["GQI_penalty_psd"] == 20.0
    assert row["GQI_std_pct"] == 4.0
    assert row["GQI_ptp_pct"] == 5.0


import json
import os
from meg_qc.calculation.meg_qc_pipeline import create_summary_report

def test_create_summary_report_handles_missing_corr(tmp_path):
    data = {
        "STD": {
            "STD_all_time_series": {
                "mag": {
                    "number_of_noisy_ch": 0,
                    "percent_of_noisy_ch": 0,
                    "number_of_flat_ch": 0,
                    "percent_of_flat_ch": 0,
                    "std_lvl": 0
                },
                "grad": {
                    "number_of_noisy_ch": 0,
                    "percent_of_noisy_ch": 0,
                    "number_of_flat_ch": 0,
                    "percent_of_flat_ch": 0
                }
            },
            "STD_epoch": {
                "mag": {"noisy_channel_multiplier": 0},
                "grad": {"noisy_channel_multiplier": 0}
            }
        },
        "PTP_MANUAL": {
            "ptp_manual_all": {
                "mag": {
                    "number_of_noisy_ch": 0,
                    "percent_of_noisy_ch": 0,
                    "number_of_flat_ch": 0,
                    "percent_of_flat_ch": 0,
                    "ptp_lvl": 0
                },
                "grad": {
                    "number_of_noisy_ch": 0,
                    "percent_of_noisy_ch": 0,
                    "number_of_flat_ch": 0,
                    "percent_of_flat_ch": 0
                }
            },
            "ptp_manual_epoch": {
                "mag": {"noisy_channel_multiplier": 0},
                "grad": {"noisy_channel_multiplier": 0}
            }
        },
        "PSD": {
            "PSD_global": {
                "mag": {"details": {}},
                "grad": {"details": {}},
            }
        },
        "MUSCLE": {
            "zscore_thresholds": {"number_muscle_events": 0},
            "total_number_of_events": 100,
        },
        "ECG": {"description": "ECG channel noisy"},
        "EOG": {"description": "EOG channel noisy"}
    }
    json_file = tmp_path / "metrics.json"
    html_file = tmp_path / "report.html"
    summary_json = tmp_path / "summary.json"
    json_file.write_text(json.dumps(data))

    gqi = {
        "ch": {"start": 5.0, "end": 30.0, "weight": 0.32},
        "corr": {"start": 5.0, "end": 25.0, "weight": 0.24},
        "mus": {"start": 1.0, "end": 10.0, "weight": 0.24},
        "psd": {"start": 1.0, "end": 5.0, "weight": 0.2},
    }
    create_summary_report(str(json_file), str(html_file), str(summary_json), gqi)

    assert html_file.exists()
    with open(summary_json) as f:
        summary = json.load(f)
    assert summary["ECG_correlation_summary"][0]["Total Channels"] == 0
    assert summary["EOG_correlation_summary"][0]["Total Channels"] == 0
    assert summary["Muscle_events"]["total_number_of_events"] == 100
    assert summary["PSD_noise_summary"][0]["MAGNETOMETERS"] == "0.00%"

import json

from meg_qc.plotting.universal_html_report import make_summary_qc_report


def test_make_summary_qc_report(tmp_path):
    report_path = tmp_path / "desc-ReportStrings_meg.json"
    simple_path = tmp_path / "desc-SimpleMetrics_meg.json"

    report_content = {"INITIAL_INFO": "Data resampled to 1000 Hz."}
    simple_content = {
        "STD": {
            "measurement_unit_mag": "Tesla",
            "STD_all_time_series": {
                "description": "Standard deviation",
                "mag": {
                    "number_of_noisy_ch": 1,
                    "percent_of_noisy_ch": 20.0,
                    "number_of_flat_ch": 0,
                    "percent_of_flat_ch": 0.0,
                    "std_lvl": 1,
                    "details": {"noisy_ch": {"MEG0121": 0.1}, "flat_ch": {}}
                },
            },
        }
    }

    report_path.write_text(json.dumps(report_content))
    simple_path.write_text(json.dumps(simple_content))

    html = make_summary_qc_report(str(report_path), str(simple_path))

    assert "Data resampled to 1000 Hz." in html
    assert "MAGNETOMETERS" in html
    assert "MEG0121" in html
    assert "<table" in html


def test_make_summary_qc_report_handles_list(tmp_path):
    report_path = tmp_path / "desc-ReportStrings_meg.json"
    simple_path = tmp_path / "desc-SimpleMetrics_meg.json"

    report_content = {"INITIAL_INFO": "Data resampled to 1000 Hz."}
    simple_content = {"STD": [], "PSD": [{"measurement_unit_mag": "Tesla"}]}

    report_path.write_text(json.dumps(report_content))
    simple_path.write_text(json.dumps(simple_content))

    html = make_summary_qc_report(str(report_path), str(simple_path))

    assert "<strong>STD</strong>" in html
    assert "measurement_unit_mag" in html


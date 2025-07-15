# -*- coding: utf-8 -*-
"""Utilities for generating Global Quality Index reports."""

import os
import json
import glob
import shutil
import configparser
from statistics import mean
from typing import Union, Optional, Dict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

from meg_qc.calculation.initial_meg_qc import get_all_config_params


def create_summary_report(
    json_file: Union[str, os.PathLike],
    html_output: str | None = None,
    json_output: str = "first_sight_report.json",
    gqi_settings: Optional[Dict[str, Dict[str, float]]] = None,
):
    """Create a human readable QC summary from the metrics JSON file."""

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_events = data.get("MUSCLE", {}).get("total_number_of_events")

    psd_details_mag = (
        data.get("PSD", {}).get("PSD_global", {}).get("mag", {}).get("details", {})
    )
    psd_details_grad = (
        data.get("PSD", {}).get("PSD_global", {}).get("grad", {}).get("details", {})
    )
    noisy_power_mag = sum(
        d.get("percent_of_this_noise_ampl_relative_to_all_signal_global", 0)
        for d in psd_details_mag.values()
    )
    noisy_power_grad = sum(
        d.get("percent_of_this_noise_ampl_relative_to_all_signal_global", 0)
        for d in psd_details_grad.values()
    )
    M_psd = mean([noisy_power_mag, noisy_power_grad])

    metrics_to_check = ["STD", "PSD", "PTP_MANUAL", "MUSCLE"]
    compute_gqi = True
    include_corr = True
    if gqi_settings is not None:
        compute_gqi = gqi_settings.get("compute_gqi", True)
        include_corr = gqi_settings.get("include_ecg_eog", True)
    if compute_gqi and include_corr:
        metrics_to_check.extend(["ECG", "EOG"])
    missing = [m for m in metrics_to_check if not data.get(m)]
    if missing:
        print(
            f"___MEGqc___: Skipping GlobalSummaryReport for {json_file}. "
            f"Missing metrics: {', '.join(missing)}"
        )
        return
    if html_output is not None:
        html_name = os.path.splitext(os.path.basename(json_output))[0].replace(
            "-GlobalSummaryReport_meg",
            "",
        )

    def build_summary_table(source):
        rows = []
        for sensor_type in ["mag", "grad"]:
            n_noisy = source[sensor_type]["number_of_noisy_ch"]
            p_noisy = source[sensor_type]["percent_of_noisy_ch"]
            n_flat = source[sensor_type]["number_of_flat_ch"]
            p_flat = source[sensor_type]["percent_of_flat_ch"]
            rows.append({"Metric": "Noisy Channels", sensor_type: f"{n_noisy} ({p_noisy:.1f}%)"})
            rows.append({"Metric": "Flat Channels", sensor_type: f"{n_flat} ({p_flat:.1f}%)"})
        df = pd.DataFrame(rows)
        df = df.groupby("Metric").first().reset_index()
        df.rename(columns={"mag": "MAGNETOMETERS", "grad": "GRADIOMETERS"}, inplace=True)
        return df

    general_df = build_summary_table(data["STD"]["STD_all_time_series"])
    ptp_df = build_summary_table(data["PTP_MANUAL"]["ptp_manual_all"])

    def build_psd_summary(noise_mag, noise_grad):
        df = pd.DataFrame([
            {"Metric": "Noise Power", "mag": noise_mag, "grad": noise_grad}
        ])
        df["mag"] = df["mag"].map(lambda v: f"{v:.2f}%")
        df["grad"] = df["grad"].map(lambda v: f"{v:.2f}%")
        df.rename(columns={"mag": "MAGNETOMETERS", "grad": "GRADIOMETERS"}, inplace=True)
        return df

    psd_df = build_psd_summary(noisy_power_mag, noisy_power_grad)

    if gqi_settings is not None:
        thresholds = {k: v for k, v in gqi_settings.items() if isinstance(v, dict)}
    else:
        thresholds = {
            "ch": {"start": 5.0, "end": 30.0, "weight": 0.32},
            "corr": {"start": 5.0, "end": 25.0, "weight": 0.24},
            "mus": {"start": 1.0, "end": 10.0, "weight": 0.24},
            "psd": {"start": 1.0, "end": 5.0, "weight": 0.2},
        }

    def quality_q(M, start, end):
        if M <= start:
            return 1.0
        if M >= end:
            return 0.0
        f = (M - start) / (end - start)
        return 1.0 - f

    def count_high_correlations_from_details(section, contamination_key):
        results = []
        percentages = []
        for sensor_type in ["mag", "grad"]:
            entries = (
                data.get(section, {})
                .get(contamination_key, {})
                .get(sensor_type, {})
                .get("details", {})
            )
            if not isinstance(entries, dict):
                entries = {}
            total = len(entries)
            high_corr = sum(
                1
                for _, pair in entries.items()
                if isinstance(pair, (list, tuple)) and pair and abs(pair[0]) > 0.8
            )
            percent = 100 * high_corr / total if total > 0 else 0
            percentages.append(percent)
            results.append(
                {
                    "Sensor Type": "MAGNETOMETERS" if sensor_type == "mag" else "GRADIOMETERS",
                    "# |High Correlations| > 0.8": f"{high_corr} ({percent:.1f}%)",
                    "Total Channels": total,
                }
            )
        return pd.DataFrame(results), percentages

    ecg_df, ecg_percents = count_high_correlations_from_details(
        "ECG", "all_channels_ranked_by_ECG_contamination_level"
    )
    eog_df, eog_percents = count_high_correlations_from_details(
        "EOG", "all_channels_ranked_by_EOG_contamination_level"
    )

    def is_noisy(desc: str) -> bool:
        noisy_markers = ["too noisy", "does not have expected", "can not be detected"]
        return any(m in desc for m in noisy_markers)

    ecg_desc = str(data.get("ECG", {}).get("description", "")).lower()
    eog_desc = str(data.get("EOG", {}).get("description", "")).lower()
    ecg_noisy = is_noisy(ecg_desc)
    eog_noisy = is_noisy(eog_desc)

    bad_pct = mean([
        data["STD"]["STD_all_time_series"]["mag"]["percent_of_noisy_ch"],
        data["STD"]["STD_all_time_series"]["mag"]["percent_of_flat_ch"],
        data["STD"]["STD_all_time_series"]["grad"]["percent_of_noisy_ch"],
        data["STD"]["STD_all_time_series"]["grad"]["percent_of_flat_ch"],
        data["PTP_MANUAL"]["ptp_manual_all"]["mag"]["percent_of_noisy_ch"],
        data["PTP_MANUAL"]["ptp_manual_all"]["mag"]["percent_of_flat_ch"],
        data["PTP_MANUAL"]["ptp_manual_all"]["grad"]["percent_of_noisy_ch"],
        data["PTP_MANUAL"]["ptp_manual_all"]["grad"]["percent_of_flat_ch"],
    ])

    def mean_or_end(percs):
        vals = [p for p in percs if p is not None]
        return mean(vals) if vals else thresholds["corr"]["end"]

    ecg_pct = mean_or_end(ecg_percents)
    eog_pct = mean_or_end(eog_percents)

    weight_corr_each = thresholds["corr"]["weight"] / 2
    ecg_present = bool(data.get("ECG"))
    eog_present = bool(data.get("EOG"))
    weight_ecg = weight_corr_each if include_corr and ecg_present else 0.0
    weight_eog = weight_corr_each if include_corr and eog_present else 0.0

    q_corr_ecg = (
        0.5 if ecg_noisy else quality_q(ecg_pct, thresholds["corr"]["start"], thresholds["corr"]["end"])
        if weight_ecg
        else 1.0
    )
    q_corr_eog = (
        0.5 if eog_noisy else quality_q(eog_pct, thresholds["corr"]["start"], thresholds["corr"]["end"])
        if weight_eog
        else 1.0
    )

    muscle_events = data["MUSCLE"]["zscore_thresholds"]["number_muscle_events"]
    muscle_pct = 100.0 * muscle_events / total_events if total_events else float(muscle_events)

    q_ch = quality_q(bad_pct, thresholds["ch"]["start"], thresholds["ch"]["end"])
    q_mus = quality_q(muscle_pct, thresholds["mus"]["start"], thresholds["mus"]["end"])
    q_psd = quality_q(M_psd, thresholds["psd"]["start"], thresholds["psd"]["end"])

    weights_sum = (
        thresholds["ch"]["weight"] + weight_ecg + weight_eog + thresholds["mus"]["weight"] + thresholds["psd"]["weight"]
    )
    if weights_sum == 0:
        weights_sum = 1.0

    if compute_gqi:
        GQI = round(
            100.0
            * (
                thresholds["ch"]["weight"] * q_ch
                + weight_ecg * q_corr_ecg
                + weight_eog * q_corr_eog
                + thresholds["mus"]["weight"] * q_mus
                + thresholds["psd"]["weight"] * q_psd
            )
            / weights_sum,
            2,
        )
        penalties = {
            "ch": 100 * thresholds["ch"]["weight"] * (1 - q_ch) / weights_sum,
            "corr": 100 * (weight_ecg * (1 - q_corr_ecg) + weight_eog * (1 - q_corr_eog)) / weights_sum,
            "mus": 100 * thresholds["mus"]["weight"] * (1 - q_mus) / weights_sum,
            "psd": 100 * thresholds["psd"]["weight"] * (1 - q_psd) / weights_sum,
        }
    else:
        GQI = None
        penalties = {}

    std_epoch_df = pd.DataFrame(data["STD"]["STD_epoch"])
    ptp_epoch_df = pd.DataFrame(data["PTP_MANUAL"]["ptp_manual_epoch"])
    muscle_df = pd.DataFrame([
        {"# Muscle Events": muscle_events, "Total Events": total_events if total_events is not None else 0}
    ])

    std_lvl = data["STD"]["STD_all_time_series"]["mag"].get("std_lvl", "NA")
    ptp_lvl = data["PTP_MANUAL"]["ptp_manual_all"]["mag"].get("ptp_lvl", "NA")
    std_epoch_lvl = data["STD"]["STD_epoch"]["mag"].get("noisy_channel_multiplier", "NA")
    ptp_epoch_lvl = data["PTP_MANUAL"]["ptp_manual_epoch"]["mag"].get("noisy_channel_multiplier", "NA")

    if html_output is not None:
        style = """
            <style>
                body { font-family: Arial, sans-serif; margin: 10px; font-size: 16px; }
                h1 { color: #003366; font-size: 25px; margin-bottom: 6px; font-weight: bold; }
                h2 { color: #004d99; font-size: 19px; margin: 12px 0 6px 0; }
                table { border-collapse: collapse; margin: 0 0 8px 0; font-size: 16px; }
                th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: center; }
                th { background-color: #f2f2f2; }
                .table-flex { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; margin-bottom: 12px; }
                .table-box { flex: 1; min-width: 300px; }
                .file-label { font-size: 18px; font-weight: bold; margin: 0 0 2px 12px; }
                .subtitle { font-size: 19px; font-weight: bold; color: #222; margin: 0 0 12px 12px; }
                .header-grid { display: grid; grid-template-columns: 1fr 1fr; align-items: start; margin-bottom: 0; }
            </style>
        """
        with open(html_output, "w", encoding="utf-8") as f:
            f.write("</div></div>")
            f.write("<html><head><meta charset='UTF-8'>" + style + "</head><body>")
            f.write("<div class='header-grid'>")
            f.write("<div><h1>MEGQC Global Quality Report</h1></div>")
            f.write(f"<div><div class='file-label'>File: {html_name}</div>")
            f.write(f"<div class='subtitle'>Global Quality Index (GQI): {GQI}</div></div></div>")
            f.write("<h2>GQI Penalties</h2>")
            f.write(
                pd.DataFrame([
                    {"Metric": "Bad Channels", "Penalty (%)": f"{penalties['ch']:.2f}"},
                    {"Metric": "Correlation", "Penalty (%)": f"{penalties['corr']:.2f}"},
                    {"Metric": "Muscle", "Penalty (%)": f"{penalties['mus']:.2f}"},
                    {"Metric": "PSD Noise", "Penalty (%)": f"{penalties['psd']:.2f}"},
                ]).to_html(index=False)
            )
            f.write("<div class='table-flex'>")
            f.write(f"<div class='table-box'><h2>STD Time-Series (STD level: {std_lvl})</h2>")
            f.write(general_df.to_html(index=False))
            f.write(f"</div><div class='table-box'><h2>PTP Time-Series (STD level: {ptp_lvl})</h2>")
            f.write(ptp_df.to_html(index=False))
            f.write("</div></div>")
            f.write("<h2>PSD Noise Summary</h2>")
            f.write(psd_df.to_html(index=False))
            f.write("<div class='table-flex'>")
            f.write(f"<div class='table-box'><h2>STD Epoch Summary (STD level: {std_epoch_lvl})</h2>")
            f.write(std_epoch_df.to_html(index=False))
            f.write(f"</div><div class='table-box'><h2>PTP Epoch Summary (STD level: {ptp_epoch_lvl})</h2>")
            f.write(ptp_epoch_df.to_html(index=False))
            f.write("</div></div>")
            f.write("<div class='table-flex'>")
            f.write("<div class='table-box'><h2>ECG Correlation Summary</h2>")
            f.write(ecg_df.to_html(index=False))
            f.write("</div><div class='table-box'><h2>EOG Correlation Summary</h2>")
            f.write(eog_df.to_html(index=False))
            f.write("</div></div>")
            f.write("<h2>Muscle Events Summary</h2>")
            f.write(muscle_df.to_html(index=False))
            f.write("</body></html>")

    summary_data = {
        "file_name": os.path.basename(json_output),
        "GQI": GQI,
        "STD_time_series": general_df.to_dict(orient="records"),
        "PTP_time_series": ptp_df.to_dict(orient="records"),
        "STD_epoch_summary": std_epoch_df.to_dict(orient="records"),
        "PTP_epoch_summary": ptp_epoch_df.to_dict(orient="records"),
        "ECG_correlation_summary": ecg_df.to_dict(orient="records"),
        "EOG_correlation_summary": eog_df.to_dict(orient="records"),
        "PSD_noise_summary": psd_df.to_dict(orient="records"),
        "Muscle_events": {"# Muscle Events": muscle_events, "total_number_of_events": total_events},
        "GQI_penalties": penalties,
        "GQI_metrics": {
            "bad_pct": bad_pct,
            "ecg_pct": ecg_pct,
            "eog_pct": eog_pct,
            "muscle_pct": muscle_pct,
            "psd_noise_pct": M_psd,
        },
        "parameters": {
            "std_lvl": std_lvl,
            "ptp_lvl": ptp_lvl,
            "std_epoch_lvl": std_epoch_lvl,
            "ptp_epoch_lvl": ptp_epoch_lvl,
        },
    }

    with open(json_output, "w", encoding="utf-8") as f_json:
        json.dump(summary_data, f_json, indent=4)

    print(f"HTML successfully generated: {html_output}")
    print(f"JSON summary successfully generated: {json_output}")


def create_group_metrics_figure(tsv_path: Union[str, os.PathLike], output_png: Union[str, os.PathLike]) -> None:
    """Generate violin plot of group GQI metrics."""
    df = pd.read_csv(tsv_path, sep="\t")

    cols = [
        "GQI",
        "GQI_penalty_ch",
        "GQI_penalty_corr",
        "GQI_penalty_mus",
        "GQI_penalty_psd",
        "GQI_bad_pct",
        "GQI_ecg_pct",
        "GQI_eog_pct",
        "GQI_muscle_pct",
        "GQI_psd_noise_pct",
    ]
    available_cols = [c for c in cols if c in df.columns]
    data = df[available_cols].apply(pd.to_numeric, errors="coerce")
    violin_data = [data[c].dropna().values for c in available_cols]

    palette = cm.get_cmap("tab10", len(available_cols))

    plt.figure(figsize=(22, 10))
    parts = plt.violinplot(
        violin_data,
        showmeans=True,
        showextrema=True,
        showmedians=False,
        widths=0.8,
    )

    for i, pc in enumerate(parts["bodies"]):
        color = palette(i)
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.3)

    parts["cmeans"].set_linewidth(3)
    parts["cmeans"].set_color("black")
    parts["cbars"].set_color("black")

    for i, y in enumerate(violin_data, start=1):
        x = np.random.normal(i, 0.08, size=len(y))
        plt.scatter(
            x,
            y,
            s=40,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.6,
            facecolor=palette(i - 1),
        )

    plt.xticks(range(1, len(available_cols) + 1), available_cols, rotation=35, ha="right", fontsize=20, fontweight="bold")
    plt.yticks(fontsize=18)
    plt.ylabel("Value", fontsize=22, fontweight="bold")
    plt.title("Violin Plot of GQI Metrics with Individual Data Points", fontsize=26, pad=25)

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()



def generate_gqi_summary(dataset_path: str, config_file: str) -> None:
    """Generate Global Quality Index summaries from existing metrics."""
    qc_params = get_all_config_params(config_file)
    if qc_params is None:
        return
    gqi_params = qc_params.get("GlobalQualityIndex")

    calc_dir = os.path.join(dataset_path, "derivatives", "Meg_QC", "calculation")
    reports_root = os.path.join(dataset_path, "derivatives", "Meg_QC", "summary_reports")
    os.makedirs(reports_root, exist_ok=True)

    existing = glob.glob(os.path.join(reports_root, "global_quality_index_*"))
    numbers = [int(os.path.basename(p).split("_")[-1]) for p in existing if os.path.basename(p).split("_")[-1].isdigit()]
    attempt = max(numbers) + 1 if numbers else 1

    attempt_dir = os.path.join(reports_root, f"global_quality_index_{attempt}")
    os.makedirs(attempt_dir, exist_ok=True)

    pattern = os.path.join(calc_dir, "sub-*", "*SimpleMetrics_meg.json")
    summary_paths = []
    for json_path in glob.glob(pattern):
        sub_dir = os.path.basename(os.path.dirname(json_path))
        out_sub = os.path.join(attempt_dir, sub_dir)
        os.makedirs(out_sub, exist_ok=True)
        base = os.path.basename(json_path).replace("SimpleMetrics", f"GlobalSummaryReport_attempt{attempt}")
        out_json = os.path.join(out_sub, base)
        create_summary_report(json_path, None, out_json, gqi_params)
        summary_paths.append(out_json)

    group_dir = os.path.join(reports_root, "group_metrics")
    os.makedirs(group_dir, exist_ok=True)
    from meg_qc.calculation.meg_qc_pipeline import flatten_summary_metrics
    rows = []
    for path in summary_paths:
        with open(path, "r", encoding="utf-8") as f:
            js = json.load(f)
        subject = os.path.basename(os.path.dirname(path))
        row = {"subject": subject}
        row.update(flatten_summary_metrics(js))
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        tsv_file = os.path.join(group_dir, f"Global_Quality_Index_attempt_{attempt}.tsv")
        df.to_csv(tsv_file, sep="\t", index=False)
        png_file = os.path.join(group_dir, f"Global_Quality_Index_attempt_{attempt}.png")
        create_group_metrics_figure(tsv_file, png_file)

    config_dir = os.path.join(reports_root, "config")
    os.makedirs(config_dir, exist_ok=True)
    cfg = configparser.ConfigParser()
    src = configparser.ConfigParser()
    src.read(config_file)
    if src.has_section("GlobalQualityIndex"):
        cfg["GlobalQualityIndex"] = src["GlobalQualityIndex"]
    with open(os.path.join(config_dir, f"global_quality_index_{attempt}.ini"), "w") as f:
        cfg.write(f)

    print(f"Attempt {attempt} completed. Reports saved to {attempt_dir}")

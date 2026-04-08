import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

# =====================================
# 1. CONFIGURATION
# =====================================

# NOTE:
# This pipeline expects user-provided .mat files in DATA_DIR.
# Input data are not included in the repository for privacy/research reasons.
DATA_DIR = "data"

OUTPUT_DIR = os.path.join(DATA_DIR, "output_pipeline_v8")
FIGURES_INDIVIDUAL_PATH = os.path.join(OUTPUT_DIR, "figures_individual")
FIGURES_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "figures_summary")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_INDIVIDUAL_PATH, exist_ok=True)
os.makedirs(FIGURES_SUMMARY_PATH, exist_ok=True)

# Analysis convention
# baseline: -20 to 0 s relative to task onset
# task window: 0 to 60 s relative to task onset
BASELINE_WINDOW = (-20, 0)
ANALYSIS_WINDOW = (0, 60)
MIN_RESPONSE_WINDOW_S = 30  # minimum acceptable segment duration


# =====================================
# 2. HELPER FUNCTIONS
# =====================================

def normalize_channel_name(name):
    """Standardize channel names for robust matching."""
    name = str(name).strip().lower()
    name = re.sub(r"\s+", " ", name)
    return name


def normalize_text(text):
    """Normalize generic text fields."""
    return str(text).strip().lower()


def clean_filename(name):
    """Remove file extension and replace invalid characters."""
    name = str(name)
    name = name.replace(".mat", "")
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return name


def extract_metadata(filename):
    """
    Extract group, posture, and task information from the filename.

    Outputs:
        group: CTRL / DP / NA
        posture: supine / orthostatic / NA
        task: active / passive / NA
    """
    name = filename.lower()

    if "ctrl" in name or "controle" in name:
        group = "CTRL"
    elif "pd" in name or "dp" in name or "parkinson" in name:
        group = "DP"
    else:
        group = "NA"

    if "supino" in name or "supine" in name:
        posture = "supine"
    elif (
        "ort" in name
        or "orto" in name
        or "ortostat" in name
        or "empe" in name
        or "em_pe" in name
        or "standing" in name
        or "upright" in name
    ):
        posture = "orthostatic"
    else:
        posture = "NA"

    if "ativo" in name or "active" in name:
        task = "active"
    elif "passivo" in name or "passive" in name:
        task = "passive"
    else:
        task = "NA"

    return group, posture, task


def mean_per_second(signal, time):
    """Compute per-second mean values from a continuous signal."""
    duration = int(np.floor(time[-1]))
    means = []
    times = []

    for second in range(duration + 1):
        idx = (time >= second) & (time < second + 1)
        if np.any(idx):
            means.append(np.mean(signal[idx]))
            times.append(second)

    return np.array(times), np.array(means)


def mean_in_window(df, col, t0, t1):
    """Compute the mean value of a dataframe column within a time window."""
    x = df.loc[(df["t"] >= t0) & (df["t"] < t1), col].dropna()
    return float(x.mean()) if len(x) else np.nan


def percent_of_baseline(df, var):
    """Express a variable as percentage of its baseline mean."""
    baseline = mean_in_window(df, var, *BASELINE_WINDOW)
    if np.isnan(baseline) or baseline == 0:
        return pd.Series(np.nan, index=df.index)
    return (df[var] / baseline) * 100


def find_channel_by_name(titles, candidates):
    """
    Find channel index by matching candidate names against available titles.

    Matching strategy:
    1. exact match
    2. partial match
    """
    normalized_titles = [normalize_channel_name(t) for t in titles]
    normalized_candidates = [normalize_channel_name(c) for c in candidates]

    for i, title in enumerate(normalized_titles):
        if title in normalized_candidates:
            return i

    for i, title in enumerate(normalized_titles):
        for candidate in normalized_candidates:
            if candidate in title:
                return i

    return None


# =====================================
# 3. ROBUST .MAT FILE LOADING (LABCHART)
# =====================================

def load_labchart_mat(file_path):
    """
    Load LabChart .mat file and detect its structure format.

    Supported formats:
    - simple
    - block (data_block1)

    Returns:
        dict with standardized structure
    """
    mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)

    # ===== FORMAT 1: SIMPLE =====
    if all(k in mat for k in ["data", "titles", "datastart", "dataend", "samplerate"]):
        return {
            "type": "simple",
            "mat": mat,
            "data": mat["data"],
            "titles": np.atleast_1d(mat["titles"]),
            "datastart": np.atleast_1d(mat["datastart"]),
            "dataend": np.atleast_1d(mat["dataend"]),
            "samplerate": np.atleast_1d(mat["samplerate"]),
        }

    # ===== FORMAT 2: BLOCK =====
    if all(k in mat for k in ["data_block1", "titles_block1", "ticktimes_block1"]):
        return {
            "type": "block",
            "mat": mat,
            "data_block": np.asarray(mat["data_block1"], dtype=float),
            "titles_block": np.atleast_1d(mat["titles_block1"]),
            "ticktimes_block": np.atleast_1d(mat["ticktimes_block1"]),
            "comtext_block1": mat.get("comtext_block1", None),
            "comtick_block1": mat.get("comtick_block1", None),
            "comchan_block1": mat.get("comchan_block1", None),
        }

    raise ValueError(f"Unrecognized .mat format. Available keys: {sorted(mat.keys())}")


def extract_labchart_channels(file_path):
    """
    Extract MCAv, MAP, and HR channels from LabChart .mat file.
    Works for both simple and block formats.
    """
    info = load_labchart_mat(file_path)
    mat = info["mat"]

    # ========= SIMPLE FORMAT =========
    if info["type"] == "simple":
        data = info["data"]
        titles = info["titles"]
        datastart = info["datastart"]
        dataend = info["dataend"]
        samplerate = info["samplerate"]

        idx_mcav = find_channel_by_name(
            titles, ["MCAv", "MCAv E", "MCAV", "MCAV E", "MCAv1", "MCAv 1"]
        )
        idx_hr = find_channel_by_name(
            titles, ["FC", "Heart Rate", "HR"]
        )
        idx_map = find_channel_by_name(
            titles, ["MAP", "Mean Arterial", "Mean Arterial Pressure", "PAM"]
        )

        if idx_mcav is None:
            raise ValueError("MCAv channel not found.")
        if idx_hr is None:
            raise ValueError("HR channel not found.")
        if idx_map is None:
            raise ValueError("MAP channel not found.")

        def get_channel_simple(i):
            start = int(datastart[i]) - 1
            end = int(dataend[i])
            y = np.asarray(data[start:end], dtype=float)
            fs = float(samplerate[i])
            name = str(titles[i]).strip()
            return {"name": name, "y": y, "fs": fs}

        channels = {
            "mcav": get_channel_simple(idx_mcav),
            "hr": get_channel_simple(idx_hr),
            "map": get_channel_simple(idx_map),
        }

        return mat, channels, info

    # ========= BLOCK FORMAT =========
    elif info["type"] == "block":
        data_block = np.asarray(info["data_block"], dtype=float)
        titles_block = np.atleast_1d(info["titles_block"])
        ticktimes_block = np.asarray(info["ticktimes_block"], dtype=float).squeeze()
        t = np.ravel(ticktimes_block)

        if data_block.ndim == 1:
            data_block = data_block[:, None]

        if data_block.shape[0] == len(titles_block) and data_block.shape[1] == len(t):
            data_block = data_block.T
        elif data_block.shape[1] == len(titles_block) and data_block.shape[0] == len(t):
            pass
        elif data_block.shape[0] == len(titles_block):
            data_block = data_block.T
        elif data_block.shape[1] == len(titles_block):
            pass
        else:
            raise ValueError(
                f"Unexpected block format: data_block shape={data_block.shape}, "
                f"n_titles={len(titles_block)}, n_ticks={len(t)}"
            )

        idx_mcav = find_channel_by_name(
            titles_block, ["MCAv", "MCAv E", "MCAV", "MCAV E", "MCAv1", "MCAv 1"]
        )
        idx_hr = find_channel_by_name(
            titles_block, ["FC", "Heart Rate", "HR"]
        )
        idx_map = find_channel_by_name(
            titles_block, ["MAP", "Mean Arterial", "Mean Arterial Pressure", "PAM"]
        )

        if idx_mcav is None:
            raise ValueError("MCAv channel not found (block).")
        if idx_hr is None:
            raise ValueError("HR channel not found (block).")
        if idx_map is None:
            raise ValueError("MAP channel not found (block).")

        def get_channel_block(i):
            y = np.asarray(data_block[:, i], dtype=float)
            name = str(titles_block[i]).strip()

            dt = np.diff(t)
            dt = dt[np.isfinite(dt)]
            dt = dt[dt > 0]

            if len(dt) == 0:
                raise ValueError(f"Could not estimate sampling rate for channel {name}.")

            fs = 1.0 / np.median(dt)
            return {"name": name, "y": y, "fs": fs, "t": t}

        channels = {
            "mcav": get_channel_block(idx_mcav),
            "hr": get_channel_block(idx_hr),
            "map": get_channel_block(idx_map),
        }

        return mat, channels, info


def inspect_mat_file(file_path):
    """
    Print diagnostic information about .mat file structure.
    Useful for debugging unknown formats.
    """
    mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)

    print("Available keys:")
    print(sorted(mat.keys()))

    if "titles" in mat:
        print("\nChannel titles:")
        for i, t in enumerate(np.atleast_1d(mat["titles"])):
            print(i, str(t))

    if "titles_block1" in mat:
        print("\nChannel titles (block1):")
        for i, t in enumerate(np.atleast_1d(mat["titles_block1"])):
            print(i, str(t))


# =====================================
# 4. EVENTS / TASK MARKERS
# =====================================

def extract_comments(mat):
    """
    Extract event/comment markers from LabChart .mat structures.

    Returns a sorted list of tuples: (time, text)
    """
    comments = []

    possible_pairs = [
        ("comtime", "comtext"),
        ("commenttime", "commenttext"),
        ("eventtime", "eventtext"),
    ]

    for time_key, text_key in possible_pairs:
        if time_key in mat and text_key in mat:
            times = np.atleast_1d(mat[time_key])
            texts = np.atleast_1d(mat[text_key])

            n = min(len(times), len(texts))
            for i in range(n):
                try:
                    t = float(times[i])
                    txt = normalize_text(texts[i])
                    comments.append((t, txt))
                except Exception:
                    pass

    if "comtick_block1" in mat and "comtext_block1" in mat and "ticktimes_block1" in mat:
        comtick = np.atleast_1d(mat["comtick_block1"])
        comtext = np.atleast_1d(mat["comtext_block1"])
        ticktimes = np.atleast_1d(mat["ticktimes_block1"]).squeeze()
        ticktimes = np.ravel(ticktimes)

        n = min(len(comtick), len(comtext))
        for i in range(n):
            try:
                idx = int(comtick[i]) - 1
                if 0 <= idx < len(ticktimes):
                    t = float(ticktimes[idx])
                    txt = normalize_text(comtext[i])
                    comments.append((t, txt))
            except Exception:
                pass

    comments = sorted(comments, key=lambda x: x[0])
    return comments


def find_task_window_from_comments(mat, expected_task):
    """
    Determine task start and end times based on event comments.
    """
    comments = extract_comments(mat)
    expected_task = normalize_text(expected_task)

    task_comments = [(t, txt) for t, txt in comments if txt == expected_task]

    if len(task_comments) >= 2:
        start_time_abs = task_comments[0][0]
        end_time_abs = task_comments[1][0]
        source = "comments"
        return start_time_abs, end_time_abs, comments, source

    elif len(task_comments) == 1:
        start_time_abs = task_comments[0][0]
        end_time_abs = start_time_abs + 60.0
        source = "single_comment"
        return start_time_abs, end_time_abs, comments, source

    else:
        start_time_abs = 20.0
        end_time_abs = 80.0
        source = "fallback_20_80"
        return start_time_abs, end_time_abs, comments, source


# =====================================
# 5. PROCESS A SINGLE .MAT FILE
# =====================================

def process_mat_file(file_path, expected_task):
    """
    Process a single LabChart .mat file and compute key physiological metrics.
    """
    mat, channels, info = extract_labchart_channels(file_path)

    mcav = channels["mcav"]["y"]
    hr = channels["hr"]["y"]
    map_signal = channels["map"]["y"]

    if info["type"] == "block":
        t_mcav_abs = np.asarray(channels["mcav"]["t"], dtype=float)
        t_hr_abs = np.asarray(channels["hr"]["t"], dtype=float)
        t_map_abs = np.asarray(channels["map"]["t"], dtype=float)
    else:
        t_mcav_abs = np.arange(len(mcav)) / channels["mcav"]["fs"]
        t_hr_abs = np.arange(len(hr)) / channels["hr"]["fs"]
        t_map_abs = np.arange(len(map_signal)) / channels["map"]["fs"]

    t_mcav_1hz_abs, mcav_1hz = mean_per_second(mcav, t_mcav_abs)
    t_hr_1hz_abs, hr_1hz = mean_per_second(hr, t_hr_abs)
    t_map_1hz_abs, map_1hz = mean_per_second(map_signal, t_map_abs)

    start_time_abs, end_time_abs, comments, window_source = find_task_window_from_comments(
        mat, expected_task
    )

    t_mcav_rel = t_mcav_1hz_abs - start_time_abs
    t_hr_rel = t_hr_1hz_abs - start_time_abs
    t_map_rel = t_map_1hz_abs - start_time_abs

    hr_1hz_aligned = np.interp(t_mcav_rel, t_hr_rel, hr_1hz)
    map_1hz_aligned = np.interp(t_mcav_rel, t_map_rel, map_1hz)

    n = min(len(t_mcav_rel), len(mcav_1hz), len(hr_1hz_aligned), len(map_1hz_aligned))

    df_1hz = pd.DataFrame({
        "t": t_mcav_rel[:n],
        "mcav": mcav_1hz[:n],
        "hr": hr_1hz_aligned[:n],
        "map": map_1hz_aligned[:n],
    })

    df_1hz["cvci"] = df_1hz["mcav"] / df_1hz["map"]

    mcav_baseline = mean_in_window(df_1hz, "mcav", *BASELINE_WINDOW)
    map_baseline = mean_in_window(df_1hz, "map", *BASELINE_WINDOW)
    cvci_baseline = mcav_baseline / map_baseline if not np.isnan(map_baseline) and map_baseline != 0 else np.nan

    max_available_time = float(df_1hz["t"].max())
    analysis_end = min(ANALYSIS_WINDOW[1], max_available_time)

    response_df = df_1hz[
        (df_1hz["t"] >= ANALYSIS_WINDOW[0]) & (df_1hz["t"] <= analysis_end)
    ].copy()

    if response_df.empty or analysis_end < MIN_RESPONSE_WINDOW_S:
        raise ValueError("Insufficient response window after alignment.")

    peak_mcav = float(response_df["mcav"].max())
    peak_idx = response_df["mcav"].idxmax()
    ttp = float(response_df.loc[peak_idx, "t"])

    map_at_peak = float(response_df.loc[peak_idx, "map"])
    cvci_peak = peak_mcav / map_at_peak if map_at_peak != 0 else np.nan

    delta_mcav_pct = (
        ((peak_mcav - mcav_baseline) / mcav_baseline) * 100
        if not np.isnan(mcav_baseline) and mcav_baseline != 0
        else np.nan
    )
    delta_map_pct = (
        ((map_at_peak - map_baseline) / map_baseline) * 100
        if not np.isnan(map_baseline) and map_baseline != 0
        else np.nan
    )

    if np.isnan(cvci_baseline) or cvci_baseline == 0:
        delta_cvci_pct = np.nan
    else:
        delta_cvci_pct = ((cvci_peak - cvci_baseline) / cvci_baseline) * 100

    metrics = {
        "baseline_mcav": mcav_baseline,
        "baseline_map": map_baseline,
        "baseline_cvci": cvci_baseline,
        "peak_mcav": peak_mcav,
        "ttp": ttp,
        "delta_mcav_pct": delta_mcav_pct,
        "delta_map_pct": delta_map_pct,
        "delta_cvci_pct": delta_cvci_pct,
        "task_start_relative_s": 0.0,
        "task_end_relative_s": analysis_end,
        "window_source": window_source,
        "mcav_channel": channels["mcav"]["name"],
        "hr_channel": channels["hr"]["name"],
        "map_channel": channels["map"]["name"],
        "mat_format": info["type"],
        "task_start_absolute_s": start_time_abs,
        "task_end_absolute_s": end_time_abs,
        "analysis_window_duration_s": analysis_end,
    }

    return df_1hz, metrics, comments


# =====================================
# 6. INDIVIDUAL FIGURES
# =====================================

def save_panel_figure(df, file_name, analysis_end):
    """Save 3-panel figure (MCAv, MAP, CVCi) normalized to baseline (%)."""
    clean_name = clean_filename(file_name)
    output_path = os.path.join(FIGURES_INDIVIDUAL_PATH, f"{clean_name}_panels.png")

    mcav_pct = percent_of_baseline(df, "mcav")
    map_pct = percent_of_baseline(df, "map")
    cvci_pct = percent_of_baseline(df, "cvci")

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axs[0].plot(df["t"], mcav_pct, linewidth=2)
    axs[0].axhline(100, linestyle="--")
    axs[0].axvline(0, linestyle="--")
    axs[0].axvline(analysis_end, linestyle="--")
    axs[0].set_ylabel("MCAv (% baseline)")
    axs[0].set_title(file_name)

    axs[1].plot(df["t"], map_pct, linewidth=2)
    axs[1].axhline(100, linestyle="--")
    axs[1].axvline(0, linestyle="--")
    axs[1].axvline(analysis_end, linestyle="--")
    axs[1].set_ylabel("MAP (% baseline)")

    axs[2].plot(df["t"], cvci_pct, linewidth=2)
    axs[2].axhline(100, linestyle="--")
    axs[2].axvline(0, linestyle="--")
    axs[2].axvline(analysis_end, linestyle="--")
    axs[2].set_ylabel("CVCi (% baseline)")
    axs[2].set_xlabel("Time (s)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_mcav_ttp_figure(df, file_name, analysis_end):
    """Save MCAv time-series figure highlighting baseline, task window and TTP."""
    clean_name = clean_filename(file_name)
    output_path = os.path.join(FIGURES_INDIVIDUAL_PATH, f"{clean_name}_mcav_ttp.png")

    mcav_baseline = mean_in_window(df, "mcav", *BASELINE_WINDOW)
    response = df[(df["t"] >= 0) & (df["t"] <= analysis_end)].copy()

    peak_idx = response["mcav"].idxmax()
    peak_mcav = float(response.loc[peak_idx, "mcav"])
    ttp = float(response.loc[peak_idx, "t"])

    fig = plt.figure(figsize=(9, 6))

    plt.axvspan(-20, 0, alpha=0.15, label="Baseline")
    plt.axvspan(0, analysis_end, alpha=0.10, label="Task")
    plt.plot(df["t"], df["mcav"], linewidth=2.5, label="MCAv")
    plt.axhline(mcav_baseline, linestyle="--", linewidth=1.2, label="Baseline MCAv")
    plt.axvline(0, linestyle="--", linewidth=1.2, label="Task start")
    plt.axvline(analysis_end, linestyle="--", linewidth=1.2, label="Window end")
    plt.axvline(ttp, linestyle=":", linewidth=1.8, label="TTP")
    plt.scatter([ttp], [peak_mcav], s=80, zorder=5)
    plt.text(ttp + 1, peak_mcav, f"TTP = {ttp:.0f}s", va="bottom")

    plt.xlabel("Time (s)")
    plt.ylabel("MCAv")
    plt.title(file_name)
    plt.legend()
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================
# 7. BATCH PROCESSING
# =====================================

mat_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".mat")]
if len(mat_files) == 0:
    raise ValueError("No .mat files found in DATA_DIR.")

results = []
errors = []

for file_name in mat_files:
    file_path = os.path.join(DATA_DIR, file_name)
    group, posture, task = extract_metadata(file_name)

    row = {
        "file": file_name,
        "group": group,
        "posture": posture,
        "task": task,
        "baseline_mcav": np.nan,
        "baseline_map": np.nan,
        "baseline_cvci": np.nan,
        "peak_mcav": np.nan,
        "ttp": np.nan,
        "delta_mcav_pct": np.nan,
        "delta_map_pct": np.nan,
        "delta_cvci_pct": np.nan,
        "task_start_relative_s": np.nan,
        "task_end_relative_s": np.nan,
        "window_source": "",
        "mcav_channel": "",
        "hr_channel": "",
        "map_channel": "",
        "mat_format": "",
        "task_start_absolute_s": np.nan,
        "task_end_absolute_s": np.nan,
        "analysis_window_duration_s": np.nan,
        "status": "ok",
        "error_message": "",
    }

    try:
        df_1hz, metrics, comments = process_mat_file(file_path, task)
        row.update(metrics)

        try:
            analysis_end = metrics["analysis_window_duration_s"]
            save_panel_figure(df_1hz, file_name, analysis_end)
            save_mcav_ttp_figure(df_1hz, file_name, analysis_end)
        except Exception as fig_error:
            print(f"[FIGURE WARNING] {file_name}: {fig_error}")

    except Exception as e:
        row["status"] = "error"
        row["error_message"] = str(e)

        errors.append({
            "file": file_name,
            "group": group,
            "posture": posture,
            "task": task,
            "error_message": str(e),
        })

        print(f"[ERROR] {file_name}: {e}")

    results.append(row)


# =====================================
# 8. FINAL TABLES
# =====================================

df_results = pd.DataFrame(results)
df_errors = pd.DataFrame(errors)

expected_columns = [
    "file", "group", "posture", "task",
    "baseline_mcav", "baseline_map", "baseline_cvci",
    "peak_mcav", "ttp",
    "delta_mcav_pct", "delta_map_pct", "delta_cvci_pct",
    "task_start_relative_s", "task_end_relative_s", "window_source",
    "mcav_channel", "hr_channel", "map_channel", "mat_format",
    "task_start_absolute_s", "task_end_absolute_s", "analysis_window_duration_s",
    "status", "error_message"
]

for col in expected_columns:
    if col not in df_results.columns:
        df_results[col] = np.nan

df_results = df_results[expected_columns]

print("\nFinal results table:")
print(df_results.head())

if not df_errors.empty:
    print("\nFiles with errors:")
    print(df_errors)


# =====================================
# 9. SAVE OUTPUT TABLES
# =====================================

results_output_path = os.path.join(OUTPUT_DIR, "pipeline_results_v8.xlsx")
df_results.to_excel(results_output_path, index=False)

if not df_errors.empty:
    errors_output_path = os.path.join(OUTPUT_DIR, "pipeline_errors_v8.xlsx")
    df_errors.to_excel(errors_output_path, index=False)


# =====================================
# 10. AUTOMATIC SUMMARIES
# =====================================

df_valid = df_results[df_results["status"] == "ok"].copy()

if not df_valid.empty:
    summary_by_group = (
        df_valid.groupby("group")[["ttp", "delta_mcav_pct", "delta_map_pct", "delta_cvci_pct"]]
        .agg(["mean", "std", "count"])
        .round(2)
    )

    summary_by_group_task = (
        df_valid.groupby(["group", "task"])[["ttp", "delta_mcav_pct", "delta_map_pct", "delta_cvci_pct"]]
        .agg(["mean", "std", "count"])
        .round(2)
    )

    summary_by_group_posture_task = (
        df_valid.groupby(["group", "posture", "task"])[["ttp", "delta_mcav_pct", "delta_map_pct", "delta_cvci_pct"]]
        .agg(["mean", "std", "count"])
        .round(2)
    )

    summary_by_group.to_excel(os.path.join(OUTPUT_DIR, "summary_by_group.xlsx"))
    summary_by_group_task.to_excel(os.path.join(OUTPUT_DIR, "summary_by_group_task.xlsx"))
    summary_by_group_posture_task.to_excel(
        os.path.join(OUTPUT_DIR, "summary_by_group_posture_task.xlsx")
    )

    print("\nSummary by group:")
    print(summary_by_group)

    print("\nSummary by group and task:")
    print(summary_by_group_task)

    print("\nSummary by group, posture, and task:")
    print(summary_by_group_posture_task)


# =====================================
# 11. SUMMARY FIGURES
# =====================================

if not df_valid.empty:
    df_plot = df_valid.copy()

    group_order = ["CTRL", "DP"]
    df_plot["group"] = pd.Categorical(df_plot["group"], categories=group_order, ordered=True)

    # Figure 1: TTP by group
    fig = plt.figure(figsize=(6, 5))
    for i, grp in enumerate(group_order):
        values = df_plot[df_plot["group"] == grp]["ttp"].dropna().values
        if len(values) == 0:
            continue
        mean = values.mean()
        sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
        plt.bar(i, mean, yerr=sem, capsize=5)
        jitter = np.random.normal(i, 0.04, size=len(values))
        plt.scatter(jitter, values, s=40, zorder=3)

    plt.xticks([0, 1], group_order)
    plt.ylabel("TTP (s)")
    plt.title("Time-to-Peak by group")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_SUMMARY_PATH, "ttp_by_group.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Figure 2: ΔMCAv by group
    fig = plt.figure(figsize=(6, 5))
    for i, grp in enumerate(group_order):
        values = df_plot[df_plot["group"] == grp]["delta_mcav_pct"].dropna().values
        if len(values) == 0:
            continue
        mean = values.mean()
        sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
        plt.bar(i, mean, yerr=sem, capsize=5)
        jitter = np.random.normal(i, 0.04, size=len(values))
        plt.scatter(jitter, values, s=40, zorder=3)

    plt.xticks([0, 1], group_order)
    plt.ylabel("ΔMCAv (%)")
    plt.title("ΔMCAv by group")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_SUMMARY_PATH, "delta_mcav_by_group.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Figure 3: TTP by group and task
    plot_df = df_plot[df_plot["task"].isin(["active", "passive"])].copy()
    groups = ["CTRL", "DP"]
    tasks = ["active", "passive"]
    x_positions = np.arange(len(tasks))
    bar_width = 0.35

    fig = plt.figure(figsize=(7, 5))
    for j, grp in enumerate(groups):
        means = []
        sems = []

        for task in tasks:
            values = plot_df[
                (plot_df["group"] == grp) & (plot_df["task"] == task)
            ]["ttp"].dropna().values

            means.append(values.mean() if len(values) else np.nan)
            sems.append(values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0)

        offset = (j - 0.5) * bar_width
        plt.bar(x_positions + offset, means, width=bar_width, yerr=sems, capsize=4, label=grp)

        for i, task in enumerate(tasks):
            values = plot_df[
                (plot_df["group"] == grp) & (plot_df["task"] == task)
            ]["ttp"].dropna().values

            jitter = np.random.normal(x_positions[i] + offset, 0.03, size=len(values))
            plt.scatter(jitter, values, s=35, zorder=3)

    plt.xticks(x_positions, ["Active", "Passive"])
    plt.ylabel("TTP (s)")
    plt.title("TTP by task and group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_SUMMARY_PATH, "ttp_by_task_and_group.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Figure 4: Multi-panel paper-style summary
    variables = [
        ("delta_mcav_pct", "ΔMCAv (%)", "A"),
        ("delta_map_pct", "ΔMAP (%)", "B"),
        ("delta_cvci_pct", "ΔCVCi (%)", "C"),
        ("ttp", "TTP (s)", "D"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for ax, (var, ylabel, label) in zip(axs, variables):
        grouped = df_plot.groupby("group", observed=True)[var]
        mean = grouped.mean().reindex(group_order)
        sem = grouped.sem().reindex(group_order)

        x = np.arange(len(group_order))
        ax.bar(x, mean, yerr=sem, capsize=5, edgecolor="black", linewidth=1.2)

        for i, grp in enumerate(group_order):
            values = df_plot[df_plot["group"] == grp][var].dropna().values
            jitter = np.random.normal(i, 0.05, size=len(values))
            ax.scatter(jitter, values, s=45, edgecolors="black", linewidths=0.8, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(group_order)
        ax.set_ylabel(ylabel)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Cerebrovascular response", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_SUMMARY_PATH, "paper_style_panel.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# =====================================
# 12. FINAL MESSAGES
# =====================================

print("\nResults saved to:")
print(results_output_path)

if not df_errors.empty:
    print("\nErrors saved to:")
    print(errors_output_path)

print("\nIndividual figures saved to:")
print(FIGURES_INDIVIDUAL_PATH)

print("\nSummary figures saved to:")
print(FIGURES_SUMMARY_PATH)

print("\nPipeline v8 completed successfully.")

import os
import re
import shutil
import zipfile
import tempfile
from io import StringIO

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from preprocessing_utils import (
    baseline_correction,
    snv_normalization,
    interpolate_to_reference,
    normalized_correlation_matrix,
    vector_normalization,
    hqi_matrix,
)

# ---------------------------------------------------------
#   Load reference library ONCE at startup (not per request)
# ---------------------------------------------------------
REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_data.pkl")
_ref = joblib.load(REF_PATH)
REFERENCE_AXIS = _ref["reference_axis"]
COMBINED_NC_ARRAY = _ref["combined_data"]
COMBINED_HQI_ARRAY = _ref["combined_data_hqi"]
DATASET_LABELS = _ref["dataset_labels"]

NC_THRESHOLD = 0.8
HQI_THRESHOLD = 0.8


def clean_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.replace(" ", "_").replace("%", "")
    name = name.replace("(", "").replace(")", "")
    return name


def _build_export(source_df, mask, x_coords, y_coords, extra_cols):
    idx = mask.nonzero()[0]
    out = source_df.iloc[idx].copy()
    out.insert(0, "Y", y_coords[mask])
    out.insert(0, "X", x_coords[mask])
    col_pos = 2
    for col_name, values in extra_cols.items():
        out.insert(col_pos, col_name, values[mask])
        col_pos += 1
    return out


def run_pipeline(txt_file, progress=gr.Progress()):
    if txt_file is None:
        raise gr.Error("Please upload a .txt file first.")

    txt_file_path = txt_file.name if hasattr(txt_file, "name") else txt_file
    raw_name = os.path.splitext(os.path.basename(txt_file_path))[0]
    file_base_name = clean_filename(raw_name)

    work_dir = tempfile.mkdtemp(prefix=f"{file_base_name}_")

    # -------------------------------------------------
    # Parse the raw .txt export
    # -------------------------------------------------
    progress(0.05, desc="Reading file...")
    with open(txt_file_path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    filtered_lines = [ln for ln in lines if not ln.strip().startswith("#")]
    data_str = "".join(filtered_lines)

    df_raw = pd.read_csv(StringIO(data_str), sep="\t", header=None)

    header_row = None
    for i, row in df_raw.iterrows():
        if all(str(x).replace(".", "", 1).isdigit() for x in row[2:]):
            header_row = i
            break
    if header_row is None:
        raise gr.Error("Couldn't find a valid header row in this file — is it a HORIBA/LabSpec .txt export?")

    df = pd.read_csv(StringIO(data_str), sep="\t", skiprows=header_row, header=0)

    wavenumber_start_idx = None
    for i, col in enumerate(df.columns):
        try:
            if float(col) > 0:
                wavenumber_start_idx = i
                break
        except ValueError:
            continue
    if wavenumber_start_idx is None:
        raise gr.Error("Couldn't detect wavenumber columns in this file.")

    x_col_idx = wavenumber_start_idx - 2
    df = df.iloc[:, x_col_idx:]
    df.columns = ["X", "Y"] + list(df.columns[2:])

    unknown_df = df.iloc[:, 2:]
    unknown_df = unknown_df[(unknown_df != 0).any(axis=1)]

    if len(unknown_df) == 0:
        raise gr.Error("No non-zero spectra found in this file.")

    # -------------------------------------------------
    # Preprocess (NC path + HQI path)
    # -------------------------------------------------
    progress(0.2, desc="Baseline correcting...")
    unknown_corr = baseline_correction(unknown_df)

    progress(0.4, desc="Normalizing + interpolating...")
    unknown_snv = snv_normalization(unknown_corr.values)
    unknown_interp_nc = interpolate_to_reference(
        pd.DataFrame(unknown_snv, columns=unknown_corr.columns), REFERENCE_AXIS
    )
    unknown_vnorm = vector_normalization(unknown_corr.values)
    unknown_interp_hqi = interpolate_to_reference(
        pd.DataFrame(unknown_vnorm, columns=unknown_corr.columns), REFERENCE_AXIS
    )

    x_coords = df.loc[unknown_df.index, "X"].values
    y_coords = df.loc[unknown_df.index, "Y"].values

    # -------------------------------------------------
    # Scoring (vectorized — fast even on modest hardware)
    # -------------------------------------------------
    progress(0.55, desc="Scoring against reference library...")
    nc_scores_matrix = normalized_correlation_matrix(unknown_interp_nc.values, COMBINED_NC_ARRAY)
    max_nc = nc_scores_matrix.max(axis=1)
    mean_nc = nc_scores_matrix.mean(axis=1)
    best_nc_idx = nc_scores_matrix.argmax(axis=1)
    best_nc_label = [DATASET_LABELS[i] for i in best_nc_idx]

    hqi_scores_matrix = hqi_matrix(unknown_interp_hqi.values, COMBINED_HQI_ARRAY)
    max_hqi = hqi_scores_matrix.max(axis=1)
    best_hqi_idx = hqi_scores_matrix.argmax(axis=1)
    best_hqi_label = [DATASET_LABELS[i] for i in best_hqi_idx]

    # -------------------------------------------------
    # Plots
    # -------------------------------------------------
    progress(0.7, desc="Generating plots...")
    mask06 = max_nc > 0.6
    heatmap_path = os.path.join(work_dir, f"Heatmap_{file_base_name}.png")
    plt.figure(figsize=(6, 4))
    plt.scatter(x_coords, y_coords, color="lightgray", s=80, marker="s")
    plt.scatter(x_coords[mask06], y_coords[mask06], color="red", s=120, marker="s")
    plt.title("Spectra with Max NC > 0.6")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=110)
    plt.close()

    nc_hist_path = os.path.join(work_dir, f"NC_Hist_{file_base_name}.png")
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.hist(max_nc, bins=50, color="dodgerblue", edgecolor="black")
    plt.axvline(0.8, color="red", linestyle="--")
    plt.title("Max Normalized Correlation")
    plt.subplot(1, 2, 2)
    plt.hist(mean_nc, bins=50, color="seagreen", edgecolor="black")
    plt.axvline(0.6, color="red", linestyle="--")
    plt.title("Mean Normalized Correlation")
    plt.tight_layout()
    plt.savefig(nc_hist_path, dpi=110)
    plt.close()

    hqi_hist_path = os.path.join(work_dir, f"HQI_Hist_{file_base_name}.png")
    plt.figure(figsize=(6, 4))
    plt.hist(max_hqi, bins=50, color="steelblue", edgecolor="black")
    plt.axvline(0.8, color="red", linestyle="--")
    plt.title("HQI Distribution")
    plt.tight_layout()
    plt.savefig(hqi_hist_path, dpi=110)
    plt.close()

    flat_sorted = np.argsort(nc_scores_matrix.ravel())[::-1]
    top5 = np.unravel_index(flat_sorted[:5], nc_scores_matrix.shape)
    top5_pairs = list(zip(top5[0], top5[1]))

    overlay_paths = []
    top5_lines = []
    for rank, (u_idx, r_idx) in enumerate(top5_pairs, 1):
        unknown_spec = unknown_interp_nc.iloc[u_idx].values
        ref_spec = COMBINED_NC_ARRAY[r_idx]
        label = DATASET_LABELS[r_idx]
        score = nc_scores_matrix[u_idx][r_idx]
        x, y = x_coords[u_idx], y_coords[u_idx]

        top5_lines.append(f"Top {rank}: Unknown#{u_idx} -> {label}, NC={score:.4f}")

        plt.figure(figsize=(8, 3.2))
        plt.plot(REFERENCE_AXIS, unknown_spec, color="black", label=f"Unknown#{u_idx} (X={x:.1f},Y={y:.1f})")
        plt.plot(REFERENCE_AXIS, ref_spec, color="blue", label=f"Reference: {label}")
        plt.title(f"Top {rank} Match (NC={score:.4f})")
        plt.xlabel("Wavenumber"); plt.ylabel("Intensity")
        plt.legend(fontsize=7)
        out = os.path.join(work_dir, clean_filename(f"Top{rank}_Match_{file_base_name}.png"))
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        plt.close()
        overlay_paths.append(out)

    # -------------------------------------------------
    # Export CSVs: qualifying + rejected, raw / baseline-corrected / interpolated
    # -------------------------------------------------
    progress(0.85, desc="Exporting spectra...")
    qualifying_mask = (max_nc > NC_THRESHOLD) & (max_hqi > HQI_THRESHOLD)
    rejected_mask = ~qualifying_mask
    n_qualifying = int(qualifying_mask.sum())
    n_rejected = int(rejected_mask.sum())

    csv_paths = []

    def _export(source_df, mask, extra_cols, suffix):
        if mask.sum() == 0:
            return
        out = _build_export(source_df, mask, x_coords, y_coords, extra_cols)
        path = os.path.join(work_dir, clean_filename(f"{file_base_name}_{suffix}.csv"))
        out.to_csv(path, index=False)
        csv_paths.append(path)

    _export(unknown_df, qualifying_mask, {"Max_NC": max_nc, "Max_HQI": max_hqi}, "Raw_Qualifying_Spectra")
    _export(unknown_corr, qualifying_mask, {"Max_NC": max_nc, "Max_HQI": max_hqi}, "BaselineCorrected_Qualifying_Spectra")
    _export(unknown_interp_nc, qualifying_mask, {"Max_NC": max_nc, "Best_NC_Match": np.array(best_nc_label)}, "Interpolated_NC_Qualifying_Spectra")
    _export(unknown_interp_hqi, qualifying_mask, {"Max_HQI": max_hqi, "Best_HQI_Match": np.array(best_hqi_label)}, "Interpolated_HQI_Qualifying_Spectra")

    _export(unknown_df, rejected_mask, {"Max_NC": max_nc, "Max_HQI": max_hqi}, "Raw_Rejected_Spectra")
    _export(unknown_corr, rejected_mask, {"Max_NC": max_nc, "Max_HQI": max_hqi}, "BaselineCorrected_Rejected_Spectra")
    _export(unknown_interp_nc, rejected_mask, {"Max_NC": max_nc, "Best_NC_Match": np.array(best_nc_label)}, "Interpolated_NC_Rejected_Spectra")
    _export(unknown_interp_hqi, rejected_mask, {"Max_HQI": max_hqi, "Best_HQI_Match": np.array(best_hqi_label)}, "Interpolated_HQI_Rejected_Spectra")

    # -------------------------------------------------
    # Summary text
    # -------------------------------------------------
    summary_path = os.path.join(work_dir, f"{file_base_name}_NC_HQI_Results.txt")
    best_u, best_r = top5_pairs[0]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"File: {raw_name}\n\n")
        f.write(f"Total spectra: {len(unknown_df)}\n")
        f.write(f"Qualifying (Max_NC > {NC_THRESHOLD} AND Max_HQI > {HQI_THRESHOLD}): {n_qualifying}\n")
        f.write(f"Rejected: {n_rejected}\n\n")
        f.write("===== Top 5 Matches =====\n")
        f.write("\n".join(top5_lines) + "\n\n")
        f.write(f"Best overall match: Unknown #{best_u} -> {DATASET_LABELS[best_r]}, "
                f"NC={nc_scores_matrix[best_u][best_r]:.4f}\n")
    csv_paths.append(summary_path)

    # -------------------------------------------------
    # Zip everything for one-click download
    # -------------------------------------------------
    zip_path = os.path.join(tempfile.gettempdir(), f"{file_base_name}_results.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in [heatmap_path, nc_hist_path, hqi_hist_path] + overlay_paths + csv_paths:
            zf.write(p, arcname=os.path.basename(p))

    progress(1.0, desc="Done!")

    summary_md = (
        f"### Results for `{raw_name}`\n\n"
        f"- **Total spectra analyzed:** {len(unknown_df)}\n"
        f"- **Qualifying (bacteria-like) spectra:** {n_qualifying}\n"
        f"- **Rejected spectra:** {n_rejected}\n\n"
        f"**Best overall match:** {DATASET_LABELS[best_r]}  \n"
        f"**NC score:** {nc_scores_matrix[best_u][best_r]:.4f}\n\n"
        f"Download the button below for all CSVs, plots, and the full text summary."
    )

    gallery_images = [heatmap_path, nc_hist_path, hqi_hist_path] + overlay_paths

    return summary_md, gallery_images, zip_path


# ---------------------------------------------------------
#                       UI
# ---------------------------------------------------------
with gr.Blocks(title="Raman/SERS Bacterial Spectra Matcher") as demo:
    gr.Markdown(
        "# Raman/SERS Bacterial Spectra Matcher\n"
        "Upload a raw `.txt` scan file from the instrument. This tool compares every spectrum "
        "in the file against the reference library and reports how closely each one matches "
        "known bacterial spectra (E. coli / Salmonella)."
    )

    with gr.Row():
        file_input = gr.File(label="Upload unknown .txt file", file_types=[".txt"])

    run_btn = gr.Button("Run Analysis", variant="primary")

    summary_output = gr.Markdown()
    gallery_output = gr.Gallery(label="Plots", columns=3, height=400)
    zip_output = gr.File(label="Download all results (.zip)")

    run_btn.click(
        fn=run_pipeline,
        inputs=[file_input],
        outputs=[summary_output, gallery_output, zip_output],
    )

if __name__ == "__main__":
    demo.launch()

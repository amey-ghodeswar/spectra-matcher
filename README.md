---
title: Raman SERS Bacterial Spectra Matcher
emoji: 🧫
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Raman/SERS Bacterial Spectra Matcher

Upload a raw HORIBA/LabSpec `.txt` scan file. This app compares every spectrum in the
file against a reference library (E. coli / Salmonella) using two similarity metrics —
Normalized Correlation (NC) and Hit Quality Index (HQI) — and reports:

- % similarity (NC/HQI) for every spectrum against the closest reference match
- Which spectra qualify as high-confidence bacterial matches (Max_NC > 0.8 AND Max_HQI > 0.8)
- Downloadable CSVs of qualifying and rejected spectra, in raw, baseline-corrected, and
  interpolated (500-point) forms
- Diagnostic plots: spatial heatmap, NC/HQI histograms, top-5 match overlays

## Setup notes (for whoever deploys this)

`reference_data.pkl` must be committed to this Space's repo (same folder as `app.py`) —
it is not included by default since it's specific to your lab's reference library and
may be updated over time. Use `git lfs` for this file since it's large (tens of MB).

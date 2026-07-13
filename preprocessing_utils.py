#!/usr/bin/env python
# coding: utf-8

##----------------PROCESSING Ag DATASETS---------------------##
import os
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from scipy.interpolate import interp1d
from pybaselines.polynomial import modpoly

# ---------- Preprocessing Functions ----------
# Baseline correction
def baseline_correction(df):
    corrected_spectra = []
    for index, spectrum in df.iterrows():
        baseline, _ = modpoly(spectrum.values, poly_order=5)
        corrected = spectrum.values - baseline
        corrected_spectra.append(corrected)
    corrected_df = pd.DataFrame(corrected_spectra, columns=df.columns)
    return corrected_df

# SNV normalization
def snv_normalization(spectra):
    mean = np.mean(spectra, axis=1, keepdims=True)
    std = np.std(spectra, axis=1, keepdims=True)
    return (spectra - mean) / (std + 1e-8)

# Interpolation to reference axis
def interpolate_to_reference(df, reference_axis):
    interpolated_spectra = []
    original_axis = df.columns.astype(float)
    for _, row in df.iterrows():
        interpolator = interp1d(original_axis, row.values, kind='linear', fill_value="extrapolate")
        interpolated = interpolator(reference_axis)
        interpolated_spectra.append(interpolated)
    return pd.DataFrame(interpolated_spectra, columns=reference_axis)


def normalized_correlation(a, b):
    """
    Compute normalized correlation between two 1D numpy arrays.
    Returns a value between 0 and 1. Kept for backward compatibility /
    single-pair use; for scoring many-vs-many, use
    normalized_correlation_matrix instead (same math, vectorized).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("Vectors must be the same length")
    num = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return abs(num / denom) if denom != 0 else 0


def normalized_correlation_matrix(U, R):
    """
    Vectorized normalized correlation between every row of U (n_unknown, n_points)
    and every row of R (n_reference, n_points). Returns an (n_unknown, n_reference)
    matrix, numerically identical (to floating-point precision) to computing
    normalized_correlation(u, r) for every pair with nested loops, but far faster.
    """
    U = np.asarray(U)
    R = np.asarray(R)
    dot = U @ R.T
    norm_u = np.linalg.norm(U, axis=1, keepdims=True)
    norm_r = np.linalg.norm(R, axis=1, keepdims=True)
    denom = norm_u @ norm_r.T
    denom = np.where(denom == 0, 1e-12, denom)
    return np.abs(dot) / denom


from sklearn.preprocessing import Normalizer

def vector_normalization(spectra):
    """ Normalize each spectrum (row-wise) to unit vector using L2 normalization """
    normalizer = Normalizer(norm='l2')
    return normalizer.fit_transform(spectra)

def hqi(a, b):
    """ Calculate the Hit Quality Index between two normalized spectra.
    Kept for backward compatibility / single-pair use; for scoring
    many-vs-many, use hqi_matrix instead (same math, vectorized). """
    a = np.asarray(a)
    b = np.asarray(b)
    dot_product = np.dot(a, b)
    return (dot_product ** 2) / (np.dot(a, a) * np.dot(b, b))


def hqi_matrix(U, R):
    """
    Vectorized Hit Quality Index between every row of U (n_unknown, n_points)
    and every row of R (n_reference, n_points). Returns an (n_unknown, n_reference)
    matrix, numerically identical (to floating-point precision) to computing
    hqi(u, r) for every pair with nested loops, but far faster.
    """
    U = np.asarray(U)
    R = np.asarray(R)
    dot = U @ R.T
    uu = np.sum(U * U, axis=1, keepdims=True)
    rr = np.sum(R * R, axis=1, keepdims=True)
    denom = uu @ rr.T
    denom = np.where(denom == 0, 1e-12, denom)
    return (dot ** 2) / denom

# In[ ]:

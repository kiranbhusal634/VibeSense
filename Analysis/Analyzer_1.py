#!/usr/bin/env python3


import os
import sys
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, peak_prominences, welch, spectrogram as sp_spectrogram
from scipy.fft import rfft, rfftfreq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.platypus.flowables import Flowable

warnings.filterwarnings("ignore", category=RuntimeWarning)



BASE_DIR = Path(__file__).resolve().parent

# ---- Input -------------------------------------------------------------------
INPUT_FILE = BASE_DIR / "dataloadimbalanceAC5" / "processed" / "adxl345_5sec_calibrated.csv"

# ---- Acquisition ---------------------------------------------------------
FS = 1559.0                     # Sampling frequency in Hz (measured, NOT assumed)
SENSOR_NAME = "ADXL345"
CONTROLLER_NAME = "ESP32"
UNITS = "g"

# ---- Time-domain waveform zoom window -----------------------------------
ZOOM_START = 0.0                # seconds
ZOOM_DURATION = 3.0             # seconds

# ---- Frequency-domain visualization zoom (NOT an analysis boundary) -----
LOW_FREQ_LIMIT = 100.0          # Hz - used only for zoomed FFT/PSD plots

# ---- Local spectral analysis ---------------------------------------------
LOCAL_BANDWIDTH_HZ = 100.0      # +/- window around each significant peak
MAX_SIGNIFICANT_PEAKS = 5       # top peaks selected for local analysis
MIN_PEAK_DISTANCE_HZ = 5.0      # minimum spacing enforced between peaks

# ---- Adaptive peak-detection parameters -----------------------------------
# The prominence threshold is derived from each dataset's OWN spectrum
# (robust noise-floor + spread), so it is never a fixed absolute number
# and adapts automatically to the amplitude scale of the current dataset.
PROMINENCE_SIGMA_MULT = 6.0     # threshold = noise_floor + MULT * robust_sigma
MIN_RELATIVE_PROMINENCE = 0.05  # additional floor: prominence >= 5% of spectrum max

# ---- Spectral roll-off -----------------------------------------------------
SPECTRAL_ROLLOFF_FRACTION = 0.85   # fraction of cumulative spectral energy

# ---- PSD (Welch) parameters ------------------------------------------------
PSD_WINDOW = "hann"
PSD_NPERSEG_TARGET = 4096        # will be clipped to available sample count
PSD_OVERLAP_FRACTION = 0.5

# ---- Spectrogram parameters ------------------------------------------------
SPEC_NPERSEG_TARGET = 512
SPEC_OVERLAP_FRACTION = 0.75

# ---- Autocorrelation --------------------------------------------------------
AUTOCORR_MAX_LAG_S = 1.0         # maximum lag shown, seconds (readability)

# ---- Spectral energy bands (Hz) --------------------------------------------
# Final band is automatically extended to the actual Nyquist frequency.
ENERGY_BAND_EDGES = [0, 10, 20, 50, 100, 200, 500]  # upper edge of last fixed band

# ---- Output directories -----------------------------------------------------
DATA_DIR = BASE_DIR / "dataloadimbalanceAC5"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = BASE_DIR / "dataloadimbalanceAC5" /  "reports"

# ---- Plot appearance ---------------------------------------------------------
PLOT_DPI = 170
FIGSIZE_SINGLE = (9.6, 3.85)     # full-width single plot (tuned so 3 fit cleanly per page)
FIGSIZE_WIDE = (9.6, 3.85)

AXIS_SIGNALS_ORIGINAL = ["X_g", "Y_g", "Z_g"]
AXIS_SIGNALS_AC = ["X_ac", "Y_ac", "Z_ac"]
ANALYSIS_SIGNALS = ["X_ac", "Y_ac", "Z_ac", "R_g"]   # used for features/FFT/PSD/etc.

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.autolayout": False,
})


def ensure_directories():
    for d in [DATA_DIR, PROCESSED_DIR, RESULTS_DIR, PLOTS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. DATA LOADING / VALIDATION / PREPROCESSING
# ==============================================================================

def load_data(input_file):
    """Load the processed accelerometer CSV. Raises clear errors on failure."""
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        df = pd.read_csv(input_file)
    except Exception as exc:
        raise ValueError(f"Could not read CSV '{input_file}': {exc}")

    if df.empty:
        raise ValueError(f"Input file '{input_file}' is empty.")

    return df


def validate_data(df, fs):
    """Validate required columns, numeric content, and sample sufficiency."""
    required_cols = ["X_g", "Y_g", "Z_g"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. "
                          f"Found columns: {list(df.columns)}")

    for col in required_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n_nan = df[required_cols].isna().sum().sum()
    if n_nan > 0:
        print(f"WARNING: {n_nan} NaN value(s) found in {required_cols}. "
              f"Rows containing NaN will be dropped.")
        df = df.dropna(subset=required_cols).reset_index(drop=True)

    n_inf = np.isinf(df[required_cols].to_numpy()).sum()
    if n_inf > 0:
        print(f"WARNING: {n_inf} infinite value(s) found. Rows containing "
              f"infinite values will be dropped.")
        mask = np.isfinite(df[required_cols].to_numpy()).all(axis=1)
        df = df.loc[mask].reset_index(drop=True)

    n = len(df)
    if n < 16:
        raise ValueError(f"Insufficient samples for analysis (N={n}). "
                          f"Need at least 16 samples.")

    if fs is None or fs <= 0:
        raise ValueError(f"Invalid sampling frequency FS={fs}. Must be > 0.")

    # Minimum samples for a meaningful FFT (power-of-two not required, but
    # a very short record produces unusably coarse frequency resolution).
    if n < 64:
        print(f"WARNING: Only {n} samples available. Frequency resolution "
              f"will be coarse ({fs/n:.3f} Hz/bin).")

    return df.reset_index(drop=True)


def preprocess_data(df, fs):
    """
    Build the time vector and DC-removed AC vibration signals.
    Original processed signals (X_g, Y_g, Z_g) are preserved unmodified.
    """
    n = len(df)
    time_s = np.arange(n) / fs

    x_g = df["X_g"].to_numpy(dtype=float)
    y_g = df["Y_g"].to_numpy(dtype=float)
    z_g = df["Z_g"].to_numpy(dtype=float)

    x_ac = x_g - np.mean(x_g)
    y_ac = y_g - np.mean(y_g)
    z_ac = z_g - np.mean(z_g)

    out = pd.DataFrame({
        "Time_s": time_s,
        "X_g": x_g, "Y_g": y_g, "Z_g": z_g,
        "X_ac": x_ac, "Y_ac": y_ac, "Z_ac": z_ac,
    })
    return out


def calculate_resultant(df):
    """R_g = sqrt(X_ac^2 + Y_ac^2 + Z_ac^2)"""
    r_g = np.sqrt(df["X_ac"] ** 2 + df["Y_ac"] ** 2 + df["Z_ac"] ** 2)
    df = df.copy()
    df["R_g"] = r_g
    return df


def dataset_metadata(df, fs, input_file):
    n = len(df)
    duration = n / fs
    nyquist = fs / 2.0
    freq_res = fs / n
    return {
        "input_file": str(input_file),
        "dataset_name": Path(input_file).stem,
        "n_samples": n,
        "fs": fs,
        "duration_s": duration,
        "nyquist_hz": nyquist,
        "freq_resolution_hz": freq_res,
    }


# ==============================================================================
# 2. TIME-DOMAIN FEATURES
# ==============================================================================

def _safe_div(a, b):
    """Division that returns NaN instead of raising/inf on zero denominators."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(a, b)
    if not np.isfinite(result):
        return np.nan
    return result


def calculate_time_features(x):
    """
    Calculate the full time-domain feature set for a single 1-D signal.
    Returns a dict of feature_name -> value. Undefined values (e.g. zero
    denominators) are returned as NaN rather than raising.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        keys = ["Mean", "Median", "Std_Dev", "Variance", "RMS", "Maximum",
                "Minimum", "Absolute_Peak", "Peak_to_Peak", "Crest_Factor",
                "Shape_Factor", "Impulse_Factor", "Clearance_Factor",
                "Skewness", "Kurtosis", "Mean_Absolute_Value", "Energy",
                "Signal_Power", "Range", "MAD", "Q1", "Q3", "IQR",
                "Zero_Crossing_Rate"]
        return {k: np.nan for k in keys}

    mean_ = np.mean(x)
    median_ = np.median(x)
    std_ = np.std(x, ddof=0)
    var_ = np.var(x, ddof=0)
    rms_ = np.sqrt(np.mean(x ** 2))
    max_ = np.max(x)
    min_ = np.min(x)
    abs_peak = np.max(np.abs(x))
    p2p = max_ - min_
    mean_abs = np.mean(np.abs(x))
    sqrt_abs_mean = np.mean(np.sqrt(np.abs(x)))

    crest = _safe_div(abs_peak, rms_) if rms_ != 0 else np.nan
    shape = _safe_div(rms_, mean_abs) if mean_abs != 0 else np.nan
    impulse = _safe_div(abs_peak, mean_abs) if mean_abs != 0 else np.nan
    clearance_denom = sqrt_abs_mean ** 2
    clearance = _safe_div(abs_peak, clearance_denom) if clearance_denom != 0 else np.nan

    skewness = stats.skew(x, bias=False) if x.size >= 3 else np.nan
    kurtosis = stats.kurtosis(x, fisher=True, bias=False) if x.size >= 4 else np.nan

    energy = np.sum(x ** 2)
    power = np.mean(x ** 2)
    range_ = p2p

    mad = np.median(np.abs(x - median_))
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1

    signs = np.sign(x)
    signs[signs == 0] = 1
    zero_crossings = np.sum(np.abs(np.diff(signs)) > 0)
    zcr = zero_crossings / (x.size - 1) if x.size > 1 else np.nan

    return {
        "Mean": mean_, "Median": median_, "Std_Dev": std_, "Variance": var_,
        "RMS": rms_, "Maximum": max_, "Minimum": min_,
        "Absolute_Peak": abs_peak, "Peak_to_Peak": p2p,
        "Crest_Factor": crest, "Shape_Factor": shape,
        "Impulse_Factor": impulse, "Clearance_Factor": clearance,
        "Skewness": skewness, "Kurtosis": kurtosis,
        "Mean_Absolute_Value": mean_abs, "Energy": energy,
        "Signal_Power": power, "Range": range_, "MAD": mad,
        "Q1": q1, "Q3": q3, "IQR": iqr, "Zero_Crossing_Rate": zcr,
    }


def build_time_feature_table(signals):
    """signals: dict of {name: array}. Returns DataFrame (features x signals)."""
    results = {name: calculate_time_features(sig) for name, sig in signals.items()}
    df = pd.DataFrame(results)
    df.index.name = "Feature"
    return df


# ==============================================================================
# 3. FFT / FREQUENCY-DOMAIN ANALYSIS
# ==============================================================================

def calculate_fft(x, fs):
    """
    Single-sided amplitude spectrum using a Hann window, covering the full
    0 Hz -> Nyquist range (never restricted to a sub-band here).
    Returns freqs, amplitude (window-corrected, physically meaningful units).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    window = np.hanning(n)
    windowed = x * window

    spectrum = rfft(windowed)
    freqs = rfftfreq(n, d=1.0 / fs)

    # Window amplitude correction (coherent gain), then single-sided doubling
    window_sum = np.sum(window)
    amp = np.abs(spectrum) * 2.0 / window_sum
    amp[0] = amp[0] / 2.0                      # DC bin: no doubling
    if n % 2 == 0 and len(amp) > 1:
        amp[-1] = amp[-1] / 2.0                # Nyquist bin: no doubling

    return freqs, amp


def calculate_power_spectrum(amp):
    """Power proportional to amplitude^2, used for spectral-moment features."""
    return amp ** 2


def estimate_noise_floor(power_no_dc):
    """
    Robust background/noise-floor estimate from the power spectrum
    (excluding DC), using the median (robust to sparse strong peaks) and
    a median-absolute-deviation-based robust spread.
    """
    if power_no_dc.size == 0:
        return 0.0, 0.0
    noise_floor = np.median(power_no_dc)
    mad = np.median(np.abs(power_no_dc - noise_floor))
    robust_sigma = 1.4826 * mad
    return noise_floor, robust_sigma


def calculate_frequency_features(freqs, amp, significant_peaks_count):
    """
    Frequency-domain feature set for one signal's full spectrum.
    Spectral moment features (centroid, bandwidth, variance, skew, kurtosis)
    are computed on the power spectrum (amp^2), excluding the DC bin.
    """
    power = calculate_power_spectrum(amp)

    # Exclude DC bin for moment-based / dominant-frequency calculations
    f_nz = freqs[1:]
    a_nz = amp[1:]
    p_nz = power[1:]

    total_power = np.sum(p_nz)

    if a_nz.size == 0 or total_power <= 0:
        return {
            "Dominant_Frequency_Hz": np.nan, "Dominant_Amplitude_g": np.nan,
            "Spectral_Centroid_Hz": np.nan, "Spectral_RMS_Frequency_Hz": np.nan,
            "Spectral_Bandwidth_Hz": np.nan, "Spectral_Energy": np.nan,
            "Spectral_Variance": np.nan, "Spectral_Skewness": np.nan,
            "Spectral_Kurtosis": np.nan, "Spectral_Flatness": np.nan,
            "Spectral_Entropy": np.nan, "Spectral_Rolloff_Hz": np.nan,
            "Spectral_Peak_Power": np.nan,
            "Number_of_Significant_Peaks": significant_peaks_count,
        }

    dom_idx = np.argmax(a_nz)
    dominant_freq = f_nz[dom_idx]
    dominant_amp = a_nz[dom_idx]

    centroid = np.sum(f_nz * p_nz) / total_power
    rms_freq = np.sqrt(np.sum((f_nz ** 2) * p_nz) / total_power)
    variance = np.sum(((f_nz - centroid) ** 2) * p_nz) / total_power
    bandwidth = np.sqrt(variance)

    if bandwidth > 0:
        skewness = (np.sum(((f_nz - centroid) ** 3) * p_nz) / total_power) / (bandwidth ** 3)
        kurtosis = (np.sum(((f_nz - centroid) ** 4) * p_nz) / total_power) / (bandwidth ** 4)
    else:
        skewness = np.nan
        kurtosis = np.nan

    spectral_energy = total_power

    p_positive = p_nz[p_nz > 0]
    if p_positive.size > 0:
        geo_mean = np.exp(np.mean(np.log(p_positive)))
        arith_mean = np.mean(p_positive)
        flatness = _safe_div(geo_mean, arith_mean) if arith_mean != 0 else np.nan
    else:
        flatness = np.nan

    p_dist = p_nz / total_power
    p_dist_nz = p_dist[p_dist > 0]
    entropy_raw = -np.sum(p_dist_nz * np.log2(p_dist_nz)) if p_dist_nz.size > 0 else np.nan
    max_entropy = np.log2(p_dist.size) if p_dist.size > 1 else np.nan
    entropy_norm = _safe_div(entropy_raw, max_entropy) if max_entropy else np.nan

    cumulative_power = np.cumsum(p_nz)
    rolloff_idx = np.searchsorted(cumulative_power, SPECTRAL_ROLLOFF_FRACTION * total_power)
    rolloff_idx = min(rolloff_idx, f_nz.size - 1)
    rolloff_freq = f_nz[rolloff_idx]

    spectral_peak_power = np.max(p_nz)

    return {
        "Dominant_Frequency_Hz": dominant_freq,
        "Dominant_Amplitude_g": dominant_amp,
        "Spectral_Centroid_Hz": centroid,
        "Spectral_RMS_Frequency_Hz": rms_freq,
        "Spectral_Bandwidth_Hz": bandwidth,
        "Spectral_Energy": spectral_energy,
        "Spectral_Variance": variance,
        "Spectral_Skewness": skewness,
        "Spectral_Kurtosis": kurtosis,
        "Spectral_Flatness": flatness,
        "Spectral_Entropy": entropy_norm,
        "Spectral_Rolloff_Hz": rolloff_freq,
        "Spectral_Peak_Power": spectral_peak_power,
        "Number_of_Significant_Peaks": significant_peaks_count,
    }


# ==============================================================================
# 4. ADAPTIVE FULL-SPECTRUM SIGNIFICANT PEAK DETECTION
# ==============================================================================

def detect_significant_peaks(freqs, amp, fs, n_samples,
                              min_peak_distance_hz=MIN_PEAK_DISTANCE_HZ,
                              max_peaks=MAX_SIGNIFICANT_PEAKS):
    """
    Adaptive significant-peak detection over the COMPLETE 0 Hz -> Nyquist
    spectrum (DC excluded). The prominence threshold is derived from this
    dataset's own robust noise floor, so it automatically adapts to the
    amplitude scale of whatever motor condition produced this spectrum
    (never a fixed absolute number such as 0.001).

    Procedure:
      1. Work on the full single-sided amplitude spectrum, DC excluded.
      2. Estimate a robust background/noise level from the power spectrum
         (median + MAD-based robust sigma).
      3. Adaptive minimum prominence = noise_floor + K * robust_sigma,
         floored by a small fraction of the spectrum's own peak power so
         extremely quiet/flat spectra still get a sane threshold.
      4. Run scipy.signal.find_peaks with that prominence and a minimum
         bin separation derived from MIN_PEAK_DISTANCE_HZ / freq_resolution.
      5. Rank surviving peaks by prominence (descending) and keep the
         strongest max_peaks.

    Returns a list of dicts, one per accepted peak, each containing index,
    frequency, amplitude, and prominence (in power units and amplitude
    units), sorted by rank (strongest first).
    """
    freq_res = fs / n_samples

    freqs_nz = freqs[1:]
    amp_nz = amp[1:]
    power_nz = amp_nz ** 2

    if amp_nz.size < 3:
        return []

    noise_floor_p, robust_sigma_p = estimate_noise_floor(power_nz)
    adaptive_min_prom_power = noise_floor_p + PROMINENCE_SIGMA_MULT * robust_sigma_p

    min_distance_bins = max(1, int(round(min_peak_distance_hz / freq_res)))

    # The relative-amplitude floor must be scaled against genuine LOCAL-MAXIMUM
    # candidates, not the raw spectrum maximum. A signal such as R_g (a
    # non-negative resultant magnitude) is not zero-mean, so spectral leakage
    # from windowing can dump a very large amount of energy into the bin(s)
    # immediately next to DC. That leakage bin sits at the very edge of the
    # DC-excluded array and can never itself be selected as a peak (find_peaks
    # requires two-sided neighbors), but if it were allowed to set the
    # "relative to spectrum max" floor it would silently suppress every real
    # harmonic elsewhere. So the relative floor is scaled against the
    # strongest actual local-maximum candidate instead.
    candidate_idx, _ = find_peaks(power_nz, distance=min_distance_bins)
    if candidate_idx.size == 0:
        return []
    candidate_max_power = np.max(power_nz[candidate_idx])
    relative_floor = MIN_RELATIVE_PROMINENCE * candidate_max_power

    min_prominence_power = max(adaptive_min_prom_power, relative_floor)

    peak_idx, props = find_peaks(
        power_nz,
        prominence=min_prominence_power,
        distance=min_distance_bins,
    )

    if peak_idx.size == 0:
        return []

    prominences = props["prominences"]
    order = np.argsort(prominences)[::-1]
    peak_idx = peak_idx[order]
    prominences = prominences[order]

    peak_idx = peak_idx[:max_peaks]
    prominences = prominences[:max_peaks]

    results = []
    for rank, (idx, prom_power) in enumerate(zip(peak_idx, prominences), start=1):
        freq_hz = freqs_nz[idx]
        amp_g = amp_nz[idx]
        # Convert power-domain prominence back to an approximate amplitude-
        # domain prominence for reporting (sqrt of power prominence, signed
        # consistently with the amplitude at this bin).
        prom_amp = math.sqrt(prom_power) if prom_power > 0 else 0.0
        results.append({
            "rank": rank,
            "index_in_nz": int(idx),
            "frequency_hz": freq_hz,
            "amplitude_g": amp_g,
            "prominence_power": prom_power,
            "prominence_amplitude": prom_amp,
            "noise_floor_power": noise_floor_p,
            "min_prominence_power_used": min_prominence_power,
        })

    return results


# ==============================================================================
# 5. LOCAL (+/-100 Hz) SPECTRAL ANALYSIS AROUND SIGNIFICANT PEAKS
# ==============================================================================

def calculate_local_spectral_features(freqs, amp, peak, nyquist,
                                       all_peaks, bandwidth_hz=LOCAL_BANDWIDTH_HZ):
    """
    For a single detected significant peak, compute local spectral features
    within an automatically clipped +/- bandwidth_hz window around it.
    This is an additional zoomed analysis; it never replaces the full
    spectrum.
    """
    fp = peak["frequency_hz"]
    lower = max(0.0, fp - bandwidth_hz)
    upper = min(nyquist, fp + bandwidth_hz)

    mask = (freqs >= lower) & (freqs <= upper)
    f_local = freqs[mask]
    a_local = amp[mask]
    p_local = a_local ** 2

    total_p = np.sum(p_local)
    if total_p > 0 and f_local.size > 0:
        local_centroid = np.sum(f_local * p_local) / total_p
        local_rms_freq = np.sqrt(np.sum((f_local ** 2) * p_local) / total_p)
        local_bandwidth = np.sqrt(np.sum(((f_local - local_centroid) ** 2) * p_local) / total_p)
        local_energy = total_p
    else:
        local_centroid = np.nan
        local_rms_freq = np.nan
        local_bandwidth = np.nan
        local_energy = np.nan

    # Nearby significant peaks (other detected peaks that fall in this window)
    nearby = [p for p in all_peaks
              if p is not peak and lower <= p["frequency_hz"] <= upper]
    nearby_freqs = sorted(p["frequency_hz"] for p in nearby)
    if nearby_freqs:
        spacing = min(abs(f - fp) for f in nearby_freqs)
    else:
        spacing = np.nan

    return {
        "Local_Lower_Hz": lower,
        "Local_Upper_Hz": upper,
        "Local_Energy": local_energy,
        "Local_Centroid_Hz": local_centroid,
        "Local_Bandwidth_Hz": local_bandwidth,
        "Local_RMS_Frequency_Hz": local_rms_freq,
        "Nearby_Significant_Peaks": len(nearby),
        "Nearest_Peak_Spacing_Hz": spacing,
        "f_local": f_local,
        "a_local": a_local,
    }


# ==============================================================================
# 6. POWER SPECTRAL DENSITY (Welch)
# ==============================================================================

def calculate_psd(x, fs):
    """Welch PSD over the full valid range 0 -> Nyquist."""
    n = len(x)
    nperseg = int(min(PSD_NPERSEG_TARGET, n))
    nperseg = max(nperseg, 8)
    noverlap = int(nperseg * PSD_OVERLAP_FRACTION)

    f, pxx = welch(x, fs=fs, window=PSD_WINDOW, nperseg=nperseg,
                   noverlap=noverlap, scaling="density")
    params = {"window": PSD_WINDOW, "nperseg": nperseg, "noverlap": noverlap}
    return f, pxx, params


# ==============================================================================
# 7. SPECTROGRAM
# ==============================================================================

def calculate_spectrogram(x, fs):
    n = len(x)
    nperseg = int(min(SPEC_NPERSEG_TARGET, n))
    nperseg = max(nperseg, 8)
    noverlap = int(nperseg * SPEC_OVERLAP_FRACTION)

    f, t, sxx = sp_spectrogram(x, fs=fs, window="hann", nperseg=nperseg,
                                noverlap=noverlap, scaling="density")
    params = {"nperseg": nperseg, "noverlap": noverlap}
    return f, t, sxx, params


# ==============================================================================
# 8. AUTOCORRELATION
# ==============================================================================

def calculate_autocorrelation(x, fs, max_lag_s=AUTOCORR_MAX_LAG_S):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = x.size

    full_corr = np.correlate(x, x, mode="full")
    mid = full_corr.size // 2
    corr = full_corr[mid:]
    if corr[0] != 0:
        corr = corr / corr[0]

    max_lag_samples = min(n - 1, int(round(max_lag_s * fs)))
    lags_s = np.arange(max_lag_samples + 1) / fs
    return lags_s, corr[:max_lag_samples + 1]


# ==============================================================================
# 9. EMPIRICAL CDF
# ==============================================================================

def calculate_ecdf(x):
    x = np.asarray(x, dtype=float)
    x_sorted = np.sort(x)
    n = x_sorted.size
    prob = np.arange(1, n + 1) / n
    return x_sorted, prob


# ==============================================================================
# 10. SPECTRAL ENERGY (overall + frequency bands)
# ==============================================================================

def calculate_spectral_energy(freqs, amp, nyquist, band_edges=ENERGY_BAND_EDGES):
    """
    Total spectral energy (sum of power, DC excluded) plus energy within
    fixed visualization bands. The final band is extended to the actual
    Nyquist frequency. These bands are for visualization only and are not
    used as an analysis boundary elsewhere.
    """
    power = amp ** 2
    freqs_nz = freqs[1:]
    power_nz = power[1:]

    total_energy = np.sum(power_nz)

    edges = list(band_edges)
    if edges[-1] < nyquist:
        edges = edges + [nyquist]
    else:
        edges = [e for e in edges if e < nyquist] + [nyquist]

    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freqs_nz >= lo) & (freqs_nz < hi)
        band_energy = np.sum(power_nz[mask])
        bands.append({
            "Band_Low_Hz": lo,
            "Band_High_Hz": hi,
            "Energy": band_energy,
            "Fraction_of_Total": _safe_div(band_energy, total_energy) if total_energy else np.nan,
        })

    return total_energy, pd.DataFrame(bands)


# ==============================================================================
# 11. PLOTTING HELPERS
# ==============================================================================

def _save_fig(fig, path):
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax):
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_time_plots(df, fs, zoom_start, zoom_duration, plots_dir):
    """Full + zoomed waveform plots for original processed signals X_g,Y_g,Z_g."""
    paths = {}
    t = df["Time_s"].to_numpy()
    zoom_end = zoom_start + zoom_duration
    zoom_mask = (t >= zoom_start) & (t <= zoom_end)

    for col in AXIS_SIGNALS_ORIGINAL:
        y = df[col].to_numpy()

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.plot(t, y, linewidth=0.5, color="#1f4e8c")
        ax.set_title(f"{col} \u2014 Full Time-Domain Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Acceleration, {col} (g)")
        ax.set_xlim(t[0], t[-1])
        _style_axes(ax)
        p_full = plots_dir / f"time_{col}_full.png"
        _save_fig(fig, p_full)

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.plot(t[zoom_mask], y[zoom_mask], linewidth=0.9, color="#1f4e8c")
        ax.set_title(f"{col} \u2014 Time-Domain Waveform ({zoom_start:.0f}\u2013{zoom_end:.0f} s)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Acceleration, {col} (g)")
        if np.any(zoom_mask):
            ax.set_xlim(zoom_start, min(zoom_end, t[-1]))
        _style_axes(ax)
        p_zoom = plots_dir / f"time_{col}_zoom.png"
        _save_fig(fig, p_zoom)

        paths[col] = {"full": p_full, "zoom": p_zoom}

    return paths


def generate_ac_plots(df, fs, zoom_start, zoom_duration, plots_dir):
    """Full + zoomed waveform plots for AC signals X_ac,Y_ac,Z_ac and R_g."""
    paths = {}
    t = df["Time_s"].to_numpy()
    zoom_end = zoom_start + zoom_duration
    zoom_mask = (t >= zoom_start) & (t <= zoom_end)

    for col in AXIS_SIGNALS_AC + ["R_g"]:
        y = df[col].to_numpy()
        color = "#b5451b" if col != "R_g" else "#2f7d32"

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.plot(t, y, linewidth=0.5, color=color)
        label = f"{col} \u2014 Full AC Vibration Waveform" if col != "R_g" \
            else f"{col} \u2014 Full Resultant Vibration Waveform"
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Acceleration, {col} (g)")
        ax.set_xlim(t[0], t[-1])
        _style_axes(ax)
        p_full = plots_dir / f"ac_{col}_full.png"
        _save_fig(fig, p_full)

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.plot(t[zoom_mask], y[zoom_mask], linewidth=0.9, color=color)
        label_z = f"{col} \u2014 AC Vibration Waveform ({zoom_start:.0f}\u2013{zoom_end:.0f} s)" if col != "R_g" \
            else f"{col} \u2014 Resultant Vibration Waveform ({zoom_start:.0f}\u2013{zoom_end:.0f} s)"
        ax.set_title(label_z)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Acceleration, {col} (g)")
        if np.any(zoom_mask):
            ax.set_xlim(zoom_start, min(zoom_end, t[-1]))
        _style_axes(ax)
        p_zoom = plots_dir / f"ac_{col}_zoom.png"
        _save_fig(fig, p_zoom)

        paths[col] = {"full": p_full, "zoom": p_zoom}

    return paths


def generate_fft_plots(fft_data, nyquist, low_freq_limit, plots_dir):
    """
    fft_data: {signal_name: (freqs, amp)}
    Generates full-spectrum (0->Nyquist) and low-frequency-zoom plots.
    The zoom is a VISUALIZATION only; full spectrum remains primary.
    """
    paths = {}
    for name, (freqs, amp) in fft_data.items():
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.plot(freqs, amp, linewidth=0.6, color="#1f4e8c")
        ax.set_title(f"{name} \u2014 Full Spectrum FFT (0\u2013{nyquist:.1f} Hz)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"Amplitude, {name} (g)")
        ax.set_xlim(0, nyquist)
        _style_axes(ax)
        p_full = plots_dir / f"fft_{name}_full.png"
        _save_fig(fig, p_full)

        zoom_mask = freqs <= low_freq_limit
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.plot(freqs[zoom_mask], amp[zoom_mask], linewidth=0.8, color="#1f4e8c")
        ax.set_title(f"{name} \u2014 FFT Zoom (0\u2013{low_freq_limit:.0f} Hz)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"Amplitude, {name} (g)")
        ax.set_xlim(0, low_freq_limit)
        _style_axes(ax)
        p_zoom = plots_dir / f"fft_{name}_zoom.png"
        _save_fig(fig, p_zoom)

        paths[name] = {"full": p_full, "zoom": p_zoom}

    return paths


def generate_local_spectral_plots(name, freqs, amp, peaks, local_features_list, plots_dir):
    """One local +/-100 Hz plot per significant peak for a given signal."""
    out = []
    for peak, local_feat in zip(peaks, local_features_list):
        fp = peak["frequency_hz"]
        lower = local_feat["Local_Lower_Hz"]
        upper = local_feat["Local_Upper_Hz"]
        f_local = local_feat["f_local"]
        a_local = local_feat["a_local"]

        fig, ax = plt.subplots(figsize=(9.4, 3.7))
        ax.plot(f_local, a_local, linewidth=0.8, color="#1f4e8c")
        ax.axvline(fp, color="#c0392b", linestyle="--", linewidth=1.0)
        ax.plot([fp], [peak["amplitude_g"]], marker="o", color="#c0392b", markersize=5, zorder=5)

        annotation = (f"f = {fp:.2f} Hz\n"
                      f"amp = {peak['amplitude_g']:.5g} g\n"
                      f"prom = {peak['prominence_amplitude']:.5g} g")
        ax.annotate(annotation, xy=(fp, peak["amplitude_g"]),
                    xytext=(0.98, 0.95), textcoords="axes fraction",
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.9))

        ax.set_title(f"{name} \u2014 Local Spectrum Around {fp:.1f} Hz "
                     f"(Rank {peak['rank']})")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"Amplitude, {name} (g)")
        ax.set_xlim(lower, upper)
        _style_axes(ax)

        p = plots_dir / f"local_{name}_rank{peak['rank']}_{fp:.1f}Hz.png"
        _save_fig(fig, p)
        out.append({"path": p, "peak": peak, "local_feat": local_feat})

    return out


def generate_psd_plots(psd_data, nyquist, low_freq_limit, plots_dir):
    """psd_data: {name: (f, pxx)}"""
    paths = {}
    for name, (f, pxx) in psd_data.items():
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.semilogy(f, pxx + 1e-30, linewidth=0.7, color="#1f4e8c")
        ax.set_title(f"{name} \u2014 Power Spectral Density (0\u2013{nyquist:.1f} Hz)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (g\u00b2/Hz)")
        ax.set_xlim(0, nyquist)
        _style_axes(ax)
        p_full = plots_dir / f"psd_{name}_full.png"
        _save_fig(fig, p_full)

        mask = f <= low_freq_limit
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.semilogy(f[mask], pxx[mask] + 1e-30, linewidth=0.9, color="#1f4e8c")
        ax.set_title(f"{name} \u2014 PSD Zoom (0\u2013{low_freq_limit:.0f} Hz)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (g\u00b2/Hz)")
        ax.set_xlim(0, low_freq_limit)
        _style_axes(ax)
        p_zoom = plots_dir / f"psd_{name}_zoom.png"
        _save_fig(fig, p_zoom)

        paths[name] = {"full": p_full, "zoom": p_zoom}

    return paths


def generate_spectrogram_plots(spec_data, plots_dir):
    """spec_data: {name: (f, t, sxx)}"""
    paths = {}
    for name, (f, t, sxx) in spec_data.items():
        sxx_db = 10 * np.log10(sxx + 1e-30)
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        mesh = ax.pcolormesh(t, f, sxx_db, shading="gouraud", cmap="viridis")
        ax.set_title(f"{name} \u2014 Spectrogram")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Power/Frequency (dB/Hz)")
        p = plots_dir / f"spectrogram_{name}.png"
        _save_fig(fig, p)
        paths[name] = p

    return paths


def generate_autocorrelation_plots(autocorr_data, plots_dir):
    """autocorr_data: {name: (lags_s, corr)}"""
    paths = {}
    for name, (lags, corr) in autocorr_data.items():
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.plot(lags, corr, linewidth=0.8, color="#1f4e8c")
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.set_title(f"{name} \u2014 Autocorrelation")
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Normalized Autocorrelation")
        ax.set_xlim(lags[0], lags[-1])
        _style_axes(ax)
        p = plots_dir / f"autocorr_{name}.png"
        _save_fig(fig, p)
        paths[name] = p

    return paths


def generate_ecdf_plots(ecdf_data, plots_dir):
    """ecdf_data: {name: (x_sorted, prob)}"""
    paths = {}
    for name, (x_sorted, prob) in ecdf_data.items():
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.plot(x_sorted, prob, linewidth=1.0, color="#1f4e8c")
        ax.set_title(f"{name} \u2014 Empirical CDF")
        ax.set_xlabel(f"Acceleration, {name} (g)")
        ax.set_ylabel("Cumulative Probability")
        ax.set_ylim(0, 1)
        _style_axes(ax)
        p = plots_dir / f"ecdf_{name}.png"
        _save_fig(fig, p)
        paths[name] = p

    return paths


def generate_spectral_energy_plot(energy_band_tables, plots_dir):
    """
    energy_band_tables: {name: DataFrame with Band_Low_Hz/Band_High_Hz/Energy}
    One combined grouped-bar chart (bands on x-axis, one bar group per
    signal) to avoid generating four near-identical redundant figures.
    """
    names = list(energy_band_tables.keys())
    ref_df = energy_band_tables[names[0]]
    band_labels = [f"{int(r.Band_Low_Hz)}\u2013{r.Band_High_Hz:.0f}"
                   for r in ref_df.itertuples()]
    n_bands = len(band_labels)
    n_signals = len(names)

    x = np.arange(n_bands)
    width = 0.8 / n_signals
    colors_list = ["#1f4e8c", "#b5451b", "#2f7d32", "#8e44ad"]

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for i, name in enumerate(names):
        energies = energy_band_tables[name]["Energy"].to_numpy()
        ax.bar(x + i * width - 0.4 + width / 2, energies + 1e-30, width,
               label=name, color=colors_list[i % len(colors_list)])

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, rotation=0)
    ax.set_xlabel("Frequency Band (Hz)")
    ax.set_ylabel("Band Spectral Energy (g\u00b2, log scale)")
    ax.set_title("Frequency-Band Spectral Energy Distribution")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    _style_axes(ax)

    p = plots_dir / "spectral_energy_bands.png"
    _save_fig(fig, p)
    return p


# ==============================================================================
# 12. CSV OUTPUTS
# ==============================================================================

def generate_csv_outputs(df_processed, time_feat_df, freq_feat_df,
                          peaks_by_signal, local_rows, results_dir):
    processed_path = results_dir / "processed_analysis.csv"
    df_processed.to_csv(processed_path, index=False)

    time_feat_path = results_dir / "time_domain_features.csv"
    time_feat_df.to_csv(time_feat_path)

    freq_feat_path = results_dir / "frequency_domain_features.csv"
    freq_feat_df.to_csv(freq_feat_path)

    peak_rows = []
    for signal, peaks in peaks_by_signal.items():
        for p in peaks:
            peak_rows.append({
                "Signal": signal,
                "Rank": p["rank"],
                "Frequency_Hz": p["frequency_hz"],
                "Amplitude_g": p["amplitude_g"],
                "Prominence_Power": p["prominence_power"],
                "Prominence_Amplitude_g": p["prominence_amplitude"],
                "Equivalent_RPM_if_1x": p["frequency_hz"] * 60.0,
                "Noise_Floor_Power": p["noise_floor_power"],
                "Min_Prominence_Power_Used": p["min_prominence_power_used"],
            })
    spectral_peaks_df = pd.DataFrame(peak_rows)
    spectral_peaks_path = results_dir / "spectral_peaks.csv"
    spectral_peaks_df.to_csv(spectral_peaks_path, index=False)

    local_df = pd.DataFrame(local_rows)
    local_path = results_dir / "local_spectral_analysis.csv"
    if not local_df.empty:
        local_df = local_df[[
            "Signal", "Peak_Rank", "Peak_Frequency_Hz", "Peak_Amplitude_g",
            "Peak_Prominence", "Local_Lower_Hz", "Local_Upper_Hz",
            "Local_Energy", "Local_Centroid_Hz", "Local_Bandwidth_Hz",
            "Local_RMS_Frequency_Hz",
        ]]
    local_df.to_csv(local_path, index=False)

    return {
        "processed_analysis": processed_path,
        "time_domain_features": time_feat_path,
        "frequency_domain_features": freq_feat_path,
        "spectral_peaks": spectral_peaks_path,
        "local_spectral_analysis": local_path,
    }


# ==============================================================================
# 13. PDF REPORT GENERATION
# ==============================================================================

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


def _fmt(v, sig=5):
    """Compact, readable numeric formatting for report tables."""
    if v is None:
        return "N/A"
    try:
        if isinstance(v, str):
            return v
        if pd.isna(v):
            return "N/A"
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    av = abs(v)
    if av != 0 and (av < 1e-3 or av >= 1e5):
        return f"{v:.{sig-1}e}"
    return f"{v:.{sig}g}"


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontSize=20, leading=24,
        spaceAfter=6, alignment=TA_CENTER, textColor=colors.HexColor("#12233f")))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", parent=styles["Normal"], fontSize=11.5,
        leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#3a3a3a")))
    styles.add(ParagraphStyle(
        name="SectionHeading", parent=styles["Heading1"], fontSize=14.5,
        leading=18, spaceBefore=6, spaceAfter=8,
        textColor=colors.HexColor("#12233f"),
        borderColor=colors.HexColor("#12233f"), borderWidth=0,
    ))
    styles.add(ParagraphStyle(
        name="SubHeading", parent=styles["Heading2"], fontSize=11.5,
        leading=14, spaceBefore=10, spaceAfter=5,
        textColor=colors.HexColor("#1f4e8c")))
    styles.add(ParagraphStyle(
        name="BodyTextJ", parent=styles["Normal"], fontSize=9.6, leading=13.5,
        alignment=TA_LEFT, spaceAfter=6))
    styles.add(ParagraphStyle(
        name="Caption", parent=styles["Normal"], fontSize=8.6, leading=11,
        alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
        spaceBefore=2, spaceAfter=10))
    styles.add(ParagraphStyle(
        name="CoverMeta", parent=styles["Normal"], fontSize=11, leading=17,
        alignment=TA_CENTER))
    return styles


def _header_footer(canvas_obj, doc, dataset_name):
    canvas_obj.saveState()
    canvas_obj.setStrokeColor(colors.HexColor("#c9c9c9"))
    canvas_obj.setLineWidth(0.5)

    # Header
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(colors.HexColor("#666666"))
    canvas_obj.drawString(MARGIN, PAGE_H - 12 * mm,
                           "Motor Vibration Experimental Data Analysis")
    canvas_obj.drawRightString(PAGE_W - MARGIN, PAGE_H - 12 * mm, dataset_name)
    canvas_obj.line(MARGIN, PAGE_H - 13 * mm, PAGE_W - MARGIN, PAGE_H - 13 * mm)

    # Footer
    canvas_obj.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.drawString(MARGIN, 10 * mm, "Data-centric experimental record \u2014 no automatic interpretation")
    canvas_obj.drawRightString(PAGE_W - MARGIN, 10 * mm, f"Page {doc.page}")
    canvas_obj.restoreState()


def _img_flowable(path, max_width_mm=170, max_height_mm=110):
    """Scale an image to fit within given bounds, preserving aspect ratio."""
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w_px, h_px = im.size
    aspect = h_px / w_px
    width = max_width_mm * mm
    height = width * aspect
    max_h = max_height_mm * mm
    if height > max_h:
        height = max_h
        width = height / aspect
    return Image(str(path), width=width, height=height)


def _add_image_block(story, styles, path, caption, max_width_mm=172, max_height_mm=118):
    img = _img_flowable(path, max_width_mm, max_height_mm)
    cap = Paragraph(caption, styles["Caption"])
    story.append(KeepTogether([img, cap]))


def _df_to_table(df, styles, col_widths=None, index_label="Feature", font_size=8.0):
    """Convert a DataFrame (features as rows) into a styled ReportLab Table."""
    header = [index_label] + list(df.columns)
    data = [header]
    for idx, row in df.iterrows():
        label = str(idx).replace("_", " ")
        data.append([label] + [_fmt(v) for v in row.values])

    n_cols = len(header)
    if col_widths is None:
        first_w = CONTENT_W * 0.30
        other_w = (CONTENT_W - first_w) / (n_cols - 1)
        col_widths = [first_w] + [other_w] * (n_cols - 1)

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12233f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9c9c9")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def generate_pdf_report(meta, df_processed, time_feat_df, freq_feat_df,
                         time_plots, ac_plots, fft_plots, psd_plots,
                         spectrogram_plots, autocorr_plots, ecdf_plots,
                         energy_plot_path, energy_band_tables,
                         peaks_by_signal, local_plot_results,
                         psd_params, spec_params, autocorr_max_lag_s,
                         reports_dir):
    styles = _build_styles()
    dataset_name = meta["dataset_name"]
    out_path = reports_dir / f"{dataset_name}_report.pdf"

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN + 4 * mm, bottomMargin=MARGIN,
        title=f"Motor Vibration Analysis - {dataset_name}",
        author="Experimental Vibration Analysis System",
    )

    def _on_page(canvas_obj, doc_obj):
        _header_footer(canvas_obj, doc_obj, dataset_name)

    story = []

    # -------------------- COVER PAGE --------------------
    story.append(Spacer(1, 42 * mm))
    story.append(Paragraph("MOTOR VIBRATION EXPERIMENTAL<br/>DATA ANALYSIS", styles["ReportTitle"]))
    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="60%", thickness=1, color=colors.HexColor("#c9c9c9"),
                             hAlign="CENTER", spaceAfter=10 * mm))

    cover_rows = [
        ["Dataset", dataset_name],
        ["Sensor", SENSOR_NAME],
        ["Controller", CONTROLLER_NAME],
        ["Sampling Frequency", f"{meta['fs']:.2f} Hz"],
        ["Number of Samples", f"{meta['n_samples']}"],
        ["Duration", f"{meta['duration_s']:.3f} s"],
    ]
    cover_table = Table(cover_rows, colWidths=[55 * mm, 75 * mm], hAlign="CENTER")
    cover_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#12233f")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#dddddd")),
    ]))
    story.append(cover_table)
    story.append(PageBreak())

    # -------------------- 1. DATASET INFORMATION --------------------
    story.append(Paragraph("1. Dataset Information", styles["SectionHeading"]))
    info_rows = [
        ["Parameter", "Value"],
        ["Input file", Path(meta["input_file"]).name],
        ["Samples", f"{meta['n_samples']}"],
        ["Sampling frequency", f"{meta['fs']:.2f} Hz"],
        ["Duration", f"{meta['duration_s']:.3f} s"],
        ["Nyquist frequency", f"{meta['nyquist_hz']:.2f} Hz"],
        ["Frequency resolution", f"{meta['freq_resolution_hz']:.4f} Hz"],
        ["Sensor", SENSOR_NAME],
        ["Controller", CONTROLLER_NAME],
        ["Axes", "X, Y, Z"],
        ["Units", "g (gravitational acceleration)"],
    ]
    info_table = Table(info_rows, colWidths=[70 * mm, 90 * mm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12233f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9c9c9")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(info_table)
    story.append(PageBreak())

    # -------------------- 2. SIGNAL PROCESSING --------------------
    story.append(Paragraph("2. Signal Processing Methodology", styles["SectionHeading"]))
    methodology_text = f"""
    Vibration data were acquired with an {SENSOR_NAME} accelerometer connected to an
    {CONTROLLER_NAME}, sampled at {meta['fs']:.1f} Hz. The input CSV is already calibrated/processed
    and contains acceleration channels X_g, Y_g, Z_g. The mean (DC) component of each channel was
    removed to obtain AC vibration signals X_ac, Y_ac, Z_ac. A resultant vibration signal was formed as
    R_g = sqrt(X_ac<super>2</super> + Y_ac<super>2</super> + Z_ac<super>2</super>). Frequency-domain
    analysis applies a Hann window before computing a single-sided amplitude-spectrum FFT (window-gain
    corrected) over the complete 0&ndash;Nyquist range. Power Spectral Density is estimated with
    Welch's method, and time&ndash;frequency content is examined with a short-time Fourier
    spectrogram. Significant spectral peaks are detected with an adaptive prominence threshold derived
    from each dataset's own robust noise floor (median and median-absolute-deviation of the power
    spectrum), so the detection automatically scales with the amplitude of the current dataset rather
    than relying on a fixed value. All numerical results below are directly measured quantities; the
    report performs no automatic engineering interpretation or fault classification.
    """
    story.append(Paragraph(methodology_text, styles["BodyTextJ"]))
    story.append(PageBreak())

    # -------------------- 3. TIME-DOMAIN WAVEFORMS --------------------
    story.append(Paragraph("3. Time-Domain Waveforms (Processed Signals)", styles["SectionHeading"]))
    story.append(Paragraph(
        "Full-duration waveforms of the processed acceleration channels, followed by a "
        f"{ZOOM_DURATION:.0f}-second zoom window for waveform-shape inspection.",
        styles["BodyTextJ"]))
    for col in AXIS_SIGNALS_ORIGINAL:
        _add_image_block(story, styles, time_plots[col]["full"],
                          f"{col} \u2014 full time-domain waveform.")
    for col in AXIS_SIGNALS_ORIGINAL:
        _add_image_block(story, styles, time_plots[col]["zoom"],
                          f"{col} \u2014 zoomed waveform ({ZOOM_START:.0f}\u2013{ZOOM_START+ZOOM_DURATION:.0f} s).")
    story.append(PageBreak())

    # -------------------- 4. AC VIBRATION WAVEFORMS --------------------
    story.append(Paragraph("4. AC Vibration Waveforms", styles["SectionHeading"]))
    story.append(Paragraph(
        "DC-removed AC vibration signals (X_ac, Y_ac, Z_ac) and the resultant vibration "
        "signal R_g = sqrt(X_ac\u00b2+Y_ac\u00b2+Z_ac\u00b2). Full duration and zoomed views.",
        styles["BodyTextJ"]))
    for col in AXIS_SIGNALS_AC + ["R_g"]:
        _add_image_block(story, styles, ac_plots[col]["full"],
                          f"{col} \u2014 full waveform.")
    for col in AXIS_SIGNALS_AC + ["R_g"]:
        _add_image_block(story, styles, ac_plots[col]["zoom"],
                          f"{col} \u2014 zoomed waveform ({ZOOM_START:.0f}\u2013{ZOOM_START+ZOOM_DURATION:.0f} s).")
    story.append(PageBreak())

    # -------------------- 5. TIME-DOMAIN FEATURE RESULTS --------------------
    story.append(Paragraph("5. Time-Domain Feature Results", styles["SectionHeading"]))
    story.append(Paragraph(
        "Time-domain statistical and shape features computed on X_ac, Y_ac, Z_ac and R_g. "
        "Each value is reported once and is not repeated elsewhere in this report.",
        styles["BodyTextJ"]))
    story.append(_df_to_table(time_feat_df, styles))
    story.append(PageBreak())

    # -------------------- 6. FREQUENCY-DOMAIN ANALYSIS --------------------
    story.append(Paragraph("6. Frequency-Domain Analysis (FFT)", styles["SectionHeading"]))
    story.append(Paragraph(
        f"Single-sided amplitude spectra (Hann-windowed) over the complete measured range, "
        f"0&ndash;{meta['nyquist_hz']:.1f} Hz. The {LOW_FREQ_LIMIT:.0f} Hz zoom below is a "
        "visualization aid only and is not an analysis boundary; full-spectrum peak detection "
        "was performed over the entire range (see Section 13).",
        styles["BodyTextJ"]))
    for name in ANALYSIS_SIGNALS:
        _add_image_block(story, styles, fft_plots[name]["full"],
                          f"{name} \u2014 full spectrum FFT, 0\u2013{meta['nyquist_hz']:.1f} Hz.")
    for name in ANALYSIS_SIGNALS:
        _add_image_block(story, styles, fft_plots[name]["zoom"],
                          f"{name} \u2014 FFT, 0\u2013{LOW_FREQ_LIMIT:.0f} Hz zoom (visualization only).")
    story.append(PageBreak())

    # -------------------- 7. FREQUENCY-DOMAIN FEATURE RESULTS --------------------
    story.append(Paragraph("7. Frequency-Domain Feature Results", styles["SectionHeading"]))
    story.append(Paragraph(
        "Spectral features computed from the full 0\u2013Nyquist amplitude/power spectrum "
        "(DC excluded). Reported once; not repeated elsewhere in this report.",
        styles["BodyTextJ"]))
    story.append(_df_to_table(freq_feat_df, styles, font_size=7.4))
    story.append(PageBreak())

    # -------------------- 8. POWER SPECTRAL DENSITY --------------------
    story.append(Paragraph("8. Power Spectral Density", styles["SectionHeading"]))
    story.append(Paragraph(
        f"Welch PSD (window={psd_params['window']}, nperseg={psd_params['nperseg']}, "
        f"noverlap={psd_params['noverlap']}) over the full valid range, followed by a "
        f"{LOW_FREQ_LIMIT:.0f} Hz visualization zoom. The full PSD is the primary complete-"
        "spectrum analysis.", styles["BodyTextJ"]))
    for name in ANALYSIS_SIGNALS:
        _add_image_block(story, styles, psd_plots[name]["full"],
                          f"{name} \u2014 full PSD, 0\u2013{meta['nyquist_hz']:.1f} Hz.")
    for name in ANALYSIS_SIGNALS:
        _add_image_block(story, styles, psd_plots[name]["zoom"],
                          f"{name} \u2014 PSD, 0\u2013{LOW_FREQ_LIMIT:.0f} Hz zoom.")
    story.append(PageBreak())

    # -------------------- 9. SPECTROGRAM --------------------
    story.append(Paragraph("9. Spectrogram Analysis", styles["SectionHeading"]))
    story.append(Paragraph(
        f"Short-time Fourier spectrograms (nperseg={spec_params['nperseg']}, "
        f"noverlap={spec_params['noverlap']}) over the complete available frequency range.",
        styles["BodyTextJ"]))
    for name, p in spectrogram_plots.items():
        _add_image_block(story, styles, p, f"{name} \u2014 spectrogram.")
    story.append(PageBreak())

    # -------------------- 10. AUTOCORRELATION --------------------
    story.append(Paragraph("10. Autocorrelation Analysis", styles["SectionHeading"]))
    story.append(Paragraph(
        f"Normalized autocorrelation up to a {autocorr_max_lag_s:.1f} s lag, for visual "
        "inspection of periodicity.", styles["BodyTextJ"]))
    for name, p in autocorr_plots.items():
        _add_image_block(story, styles, p, f"{name} \u2014 autocorrelation.", max_height_mm=90)
    story.append(PageBreak())

    # -------------------- 11. EMPIRICAL CDF --------------------
    story.append(Paragraph("11. Empirical CDF", styles["SectionHeading"]))
    story.append(Paragraph(
        "Empirical cumulative distribution functions computed directly from the measured "
        "data (no assumed distribution fitted).", styles["BodyTextJ"]))
    for name, p in ecdf_plots.items():
        _add_image_block(story, styles, p, f"{name} \u2014 empirical CDF.", max_height_mm=90)
    story.append(PageBreak())

    # -------------------- 12. SPECTRAL ENERGY --------------------
    story.append(Paragraph("12. Spectral Energy", styles["SectionHeading"]))
    story.append(Paragraph(
        "Spectral energy distributed across fixed visualization frequency bands (final band "
        f"extended to the actual Nyquist frequency, {meta['nyquist_hz']:.1f} Hz). Bands are "
        "for visualization only and were not used as an analysis boundary elsewhere.",
        styles["BodyTextJ"]))
    _add_image_block(story, styles, energy_plot_path,
                      "Frequency-band spectral energy, all signals.", max_height_mm=100)

    band_header = ["Band (Hz)"] + list(energy_band_tables.keys())
    ref_df = next(iter(energy_band_tables.values()))
    band_rows = [band_header]
    for i in range(len(ref_df)):
        row_label = f"{ref_df['Band_Low_Hz'].iloc[i]:.0f}\u2013{ref_df['Band_High_Hz'].iloc[i]:.1f}"
        row = [row_label]
        for name in energy_band_tables:
            row.append(_fmt(energy_band_tables[name]["Energy"].iloc[i]))
        band_rows.append(row)
    band_table = Table(band_rows, colWidths=[38 * mm] + [33 * mm] * len(energy_band_tables))
    band_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12233f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9c9c9")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(Spacer(1, 4 * mm))
    story.append(band_table)
    story.append(PageBreak())

    # -------------------- 13. LOCAL SPECTRAL ANALYSIS --------------------
    story.append(Paragraph("13. Local Spectral Analysis (Automatic \u00b1100 Hz Zoom)", styles["SectionHeading"]))
    story.append(Paragraph(
        "For the strongest automatically detected significant peaks in each signal's full "
        "spectrum (adaptive prominence, see Section 6/methodology), a local "
        f"\u00b1{LOCAL_BANDWIDTH_HZ:.0f} Hz window is shown with a compact annotation "
        "(frequency, amplitude, prominence). Peaks are ranked by prominence; up to "
        f"{MAX_SIGNIFICANT_PEAKS} peaks are shown per signal. \"Equivalent RPM\" values, where "
        "given, denote frequency \u00d7 60 and describe only what would follow IF that "
        "component corresponded to 1\u00d7 shaft rotation \u2014 no such correspondence is "
        "asserted by this report. Detailed peak data for all signals is saved to "
        "spectral_peaks.csv and local_spectral_analysis.csv for further analysis.",
        styles["BodyTextJ"]))

    any_peaks = False
    for name in ANALYSIS_SIGNALS:
        results = local_plot_results.get(name, [])
        if not results:
            continue
        any_peaks = True
        story.append(Paragraph(f"{name} \u2014 Significant Peaks", styles["SubHeading"]))
        for r in results:
            peak = r["peak"]
            eq_rpm = peak["frequency_hz"] * 60.0
            caption = (f"{name} rank {peak['rank']}: f = {peak['frequency_hz']:.2f} Hz, "
                       f"amplitude = {peak['amplitude_g']:.4g} g, "
                       f"prominence = {peak['prominence_amplitude']:.4g} g "
                       f"(equivalent RPM if 1\u00d7: {eq_rpm:.0f}).")
            _add_image_block(story, styles, r["path"], caption, max_height_mm=80)
    if not any_peaks:
        story.append(Paragraph(
            "No spectral peaks exceeded the adaptive significance threshold for this dataset.",
            styles["BodyTextJ"]))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return out_path


# ==============================================================================
# 14. MAIN
# ==============================================================================

def _print_header():
    print("=" * 60)
    print("MOTOR VIBRATION DATA ANALYSIS")
    print("=" * 60)


def main(input_file=INPUT_FILE, fs=FS):
    ensure_directories()
    _print_header()

    # ---- Load / validate / preprocess ----
    df_raw = load_data(input_file)
    df_raw = validate_data(df_raw, fs)
    df = preprocess_data(df_raw, fs)
    df = calculate_resultant(df)

    meta = dataset_metadata(df, fs, input_file)
    print(f"Input:               {meta['input_file']}")
    print(f"Samples:             {meta['n_samples']}")
    print(f"Sampling Frequency:  {meta['fs']:.2f} Hz")
    print(f"Duration:            {meta['duration_s']:.3f} s")
    print(f"Nyquist Frequency:   {meta['nyquist_hz']:.2f} Hz")
    print(f"Frequency Resolution:{meta['freq_resolution_hz']:.4f} Hz")
    print()

    signals = {name: df[name].to_numpy() for name in ANALYSIS_SIGNALS}

    # ---- Time-domain features ----
    time_feat_df = build_time_feature_table(signals)

    # ---- FFT for each analysis signal ----
    fft_data = {}
    for name, sig in signals.items():
        freqs, amp = calculate_fft(sig, fs)
        fft_data[name] = (freqs, amp)

    # ---- Adaptive peak detection (full spectrum) per signal ----
    peaks_by_signal = {}
    for name, (freqs, amp) in fft_data.items():
        peaks_by_signal[name] = detect_significant_peaks(freqs, amp, fs, meta["n_samples"])

    # ---- Frequency-domain features ----
    freq_feat_results = {}
    for name, (freqs, amp) in fft_data.items():
        freq_feat_results[name] = calculate_frequency_features(
            freqs, amp, len(peaks_by_signal[name]))
    freq_feat_df = pd.DataFrame(freq_feat_results)
    freq_feat_df.index.name = "Feature"

    # ---- Local spectral analysis per peak ----
    local_plot_results = {}
    local_rows = []
    for name, (freqs, amp) in fft_data.items():
        peaks = peaks_by_signal[name]
        local_feats = []
        for peak in peaks:
            lf = calculate_local_spectral_features(
                freqs, amp, peak, meta["nyquist_hz"], peaks)
            local_feats.append(lf)
            local_rows.append({
                "Signal": name,
                "Peak_Rank": peak["rank"],
                "Peak_Frequency_Hz": peak["frequency_hz"],
                "Peak_Amplitude_g": peak["amplitude_g"],
                "Peak_Prominence": peak["prominence_amplitude"],
                "Local_Lower_Hz": lf["Local_Lower_Hz"],
                "Local_Upper_Hz": lf["Local_Upper_Hz"],
                "Local_Energy": lf["Local_Energy"],
                "Local_Centroid_Hz": lf["Local_Centroid_Hz"],
                "Local_Bandwidth_Hz": lf["Local_Bandwidth_Hz"],
                "Local_RMS_Frequency_Hz": lf["Local_RMS_Frequency_Hz"],
            })
        results = generate_local_spectral_plots(name, freqs, amp, peaks, local_feats, PLOTS_DIR)
        local_plot_results[name] = results

    # ---- PSD ----
    psd_data = {}
    psd_params = None
    for name, sig in signals.items():
        f, pxx, params = calculate_psd(sig, fs)
        psd_data[name] = (f, pxx)
        psd_params = params

    # ---- Spectrogram ----
    spec_data = {}
    spec_params = None
    for name in AXIS_SIGNALS_AC + ["R_g"]:
        f, t, sxx, params = calculate_spectrogram(df[name].to_numpy(), fs)
        spec_data[name] = (f, t, sxx)
        spec_params = params

    # ---- Autocorrelation ----
    autocorr_data = {}
    for name, sig in signals.items():
        lags, corr = calculate_autocorrelation(sig, fs)
        autocorr_data[name] = (lags, corr)

    # ---- ECDF ----
    ecdf_data = {}
    for name, sig in signals.items():
        x_sorted, prob = calculate_ecdf(sig)
        ecdf_data[name] = (x_sorted, prob)

    # ---- Spectral energy ----
    energy_band_tables = {}
    for name, (freqs, amp) in fft_data.items():
        total_e, band_df = calculate_spectral_energy(freqs, amp, meta["nyquist_hz"])
        energy_band_tables[name] = band_df

    # ---- Terminal feature summary ----
    for name in ANALYSIS_SIGNALS:
        tf = time_feat_df[name]
        ff = freq_feat_results[name]
        print(f"{name}:")
        print(f"  RMS:               {tf['RMS']:.5g} g")
        print(f"  Absolute Peak:     {tf['Absolute_Peak']:.5g} g")
        print(f"  Peak-to-Peak:      {tf['Peak_to_Peak']:.5g} g")
        print(f"  Crest Factor:      {tf['Crest_Factor']:.5g}")
        print(f"  Kurtosis:          {tf['Kurtosis']:.5g}")
        print(f"  Dominant Frequency:{ff['Dominant_Frequency_Hz']:.3f} Hz")
        print(f"  Dominant Amplitude:{ff['Dominant_Amplitude_g']:.5g} g")
        print()

    print("SIGNIFICANT SPECTRAL PEAKS")
    print(f"{'Signal':<8}{'Rank':<6}{'Frequency (Hz)':<18}{'Amplitude (g)':<16}{'Prominence':<14}")
    for name, peaks in peaks_by_signal.items():
        for p in peaks:
            print(f"{name:<8}{p['rank']:<6}{p['frequency_hz']:<18.3f}"
                  f"{p['amplitude_g']:<16.5g}{p['prominence_amplitude']:<14.5g}")
    print()

    print("LOCAL +/-100 Hz WINDOWS")
    for name, results in local_plot_results.items():
        for r in results:
            lf = r["local_feat"]
            print(f"  {name} rank {r['peak']['rank']}: "
                  f"[{lf['Local_Lower_Hz']:.1f}, {lf['Local_Upper_Hz']:.1f}] Hz")
    print()

    # ---- Generate all plots ----
    time_plots = generate_time_plots(df, fs, ZOOM_START, ZOOM_DURATION, PLOTS_DIR)
    ac_plots = generate_ac_plots(df, fs, ZOOM_START, ZOOM_DURATION, PLOTS_DIR)
    fft_plots = generate_fft_plots(fft_data, meta["nyquist_hz"], LOW_FREQ_LIMIT, PLOTS_DIR)
    psd_plots = generate_psd_plots(psd_data, meta["nyquist_hz"], LOW_FREQ_LIMIT, PLOTS_DIR)
    spectrogram_plots = generate_spectrogram_plots(spec_data, PLOTS_DIR)
    autocorr_plots = generate_autocorrelation_plots(autocorr_data, PLOTS_DIR)
    ecdf_plots = generate_ecdf_plots(ecdf_data, PLOTS_DIR)
    energy_plot_path = generate_spectral_energy_plot(energy_band_tables, PLOTS_DIR)

    # ---- CSV outputs ----
    csv_paths = generate_csv_outputs(df, time_feat_df, freq_feat_df,
                                      peaks_by_signal, local_rows, RESULTS_DIR)

    # ---- PDF report ----
    pdf_path = generate_pdf_report(
        meta, df, time_feat_df, freq_feat_df,
        time_plots, ac_plots, fft_plots, psd_plots,
        spectrogram_plots, autocorr_plots, ecdf_plots,
        energy_plot_path, energy_band_tables,
        peaks_by_signal, local_plot_results,
        psd_params, spec_params, AUTOCORR_MAX_LAG_S,
        REPORTS_DIR,
    )

    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"PDF:")
    print(f"  {pdf_path}")
    print(f"Results:")
    for k, p in csv_paths.items():
        print(f"  {p}")
    print(f"Plots:")
    print(f"  {PLOTS_DIR}")

    return {
        "meta": meta,
        "pdf_path": pdf_path,
        "csv_paths": csv_paths,
        "plots_dir": PLOTS_DIR,
    }


if __name__ == "__main__":
    main()
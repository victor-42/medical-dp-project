import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import skew, kurtosis, entropy


def extract_abp_features(abp, fs, preprocess=True):
    abp = np.asarray(abp).astype(float)

    # --- Statistical descriptors ---
    mean_val = np.mean(abp)
    std_val = np.std(abp)
    skew_val = skew(abp)
    kurt_val = kurtosis(abp)
    p10, p25, p75, p90 = np.percentile(abp, [10, 25, 75, 90])

    # Amplitude range
    rng = np.max(abp) - np.min(abp)

    # --- Entropy-like measures ---
    # Normalize histogram for Shannon entropy
    hist, _ = np.histogram(abp, bins=50, density=True)
    hist = hist[hist > 0]
    shannon_entropy = -np.sum(hist * np.log2(hist))

    # Hjorth parameters
    diff1 = np.diff(abp)
    diff2 = np.diff(diff1)
    var0 = np.var(abp)
    var1 = np.var(diff1)
    var2 = np.var(diff2)
    hjorth_mobility = np.sqrt(var1 / var0) if var0 > 0 else 0
    hjorth_complexity = np.sqrt(var2 / var1) / hjorth_mobility if var1 > 0 else 0

    # --- Frequency-domain features ---
    freqs, psd = welch(abp, fs=fs, nperseg=min(1024, len(abp)))

    total_power = np.trapezoid(psd, freqs)
    band_low = np.trapezoid(psd[(freqs >= 0.04) & (freqs < 0.15)], freqs[(freqs >= 0.04) & (freqs < 0.15)])
    band_high = np.trapezoid(psd[(freqs >= 0.15) & (freqs < 0.4)], freqs[(freqs >= 0.15) & (freqs < 0.4)])
    spectral_entropy = entropy(psd / np.sum(psd))
    
    return [
        mean_val,
    std_val,
    skew_val,
    kurt_val,
    p10,
    p25,
    p75,
    p90,
    rng,

    # Entropy / Hjorth
    shannon_entropy,
    hjorth_mobility,
    hjorth_complexity,

    # Spectral
    total_power,
    band_low,
    band_high,
    spectral_entropy,
    ]
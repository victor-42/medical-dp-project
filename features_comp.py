'''
Edit this script to create functions that compute features based on the input signals.
For each function you create to compute a feature, put a docstring describing the feature
you compute. The docstring can be written in the same style as in the present module. 
'''
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import kurtosis, skew
import pywt
import antropy as ant  # for sample entropy
from tqdm import tqdm
import pickle
import neurokit2 as nk
from scipy.fft import fft, fftfreq
import scipy.signal as ssignal

fs = 250  # Sampling frequency


def compute_stats(x, column=0):
    '''Example of feature computation
    
    Parameters
    ----------
    x: array_like
        2d Input signal where the axis 0 corresponds to the time whereas each column corresponds to 
        an input channel
    column: int
        The column of `x` that should be processed by the function

    Returns
    -------
    output: array_like
        A 1-d array containing features computed by the function over the specified `column` of `x`.
    '''
    # compute some descriptive stats
    stats = []
    mean, std, skw, kurt = np.mean(x[:, column]), np.std(x[:, column]), skew(x[:, column]), kurtosis(x[:, column])
    stats += [mean, std, skw, kurt]

    output = np.array(stats)
    return output


def zero_outliers_iqr(df, columns=None, factor=20.0):
    """
    Replaces extreme outliers with 0.0 using IQR method.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to check for outliers. If None, all numeric columns are used.
    factor : float, optional
        IQR factor. Default=3.0 (conservative).

    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers replaced by 0.0
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        # replace values outside bounds with 0.0
        df.loc[(df[col] < lower) | (df[col] > upper), col] = 0.0

    return df


def compute_features(x):
    '''
    Compute the features to describe a recording.

    Parameters
    ----------
    x: array_like
        2d Input signal where the axis 0 corresponds to the time whereas each column corresponds to 
        an input channel

    Returns
    -------
    output: array_like
        A 1-d array concatenating all the features you computed from the signal.

    '''

    # EDIT YOUR CODE HERE TO COMPUTE FEATURES
    features = []
    # call a function you programmed above to compute some features on a signal
    for f in []:  # handle_fresh_abp_features]:
        r = f(x)
        if r is not None:
            features.extend(r)
    # features=compute_stats(x,column=0) #here just an example
    # print(" Feature len : ", len(features))
    # print(features)
    return features


def prepare_data(x, name=None, filter_outliers=True):
    '''
    Apply compute_features to the data input tensor.

    Parameters
    ----------
    x: array_like
        A 3-d array of waveform signals where the first axis corresponds to the input record, the
        second axis to the time, and the third axis to the input channel of the record.

    Returns
    -------
    X: array_like
        A 2-d array dataset in which the first axis (lines) corresponds to the input recording
        and the second axis (columns) corresponds to the recording features computed with `compute_features`
    '''
    accumulation_types = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max,
        'kurtosis': kurtosis,
        'skewness': skew,
        'median': np.median,
        'sum': np.sum,
        'var': np.var,
        'var_coeff': lambda x: ((np.std(x) + 1e-4) / (np.mean(x) + 1e-6)) * 100,
    }

    def detect_abp_syst_peaks(signal):
        peaks, _ = find_peaks(signal, distance=int(0.3 * fs), height=signal.mean() + 5)
        peaks_inverse, _ = find_peaks(-signal, distance=int(0.3 * fs), height=-signal.mean() + 5, )
        peaks_all = peaks_inverse

        # Filter Diastolic peaks by removing all but the first one after the peak foot
        for i in range(len(peaks) - 1):
            mask = (peaks_all > peaks[i]) & (peaks_all < peaks[i + 1])
            if np.sum(mask) > 1:
                to_remove = peaks_all[mask][1:]
                peaks_all = peaks_all[~np.isin(peaks_all, to_remove)]

        abp_diastolic_new = []
        for i in range(len(peaks) - 1):
            mask = (peaks_inverse > peaks[i]) & (peaks_inverse < peaks[i + 1])
            if np.sum(mask) > 0:
                to_keep = peaks_inverse[mask]
                closest = to_keep[np.argmin(peaks[i + 1] - to_keep)]
                abp_diastolic_new.append(closest)

        abp_diastolic_new = np.array(abp_diastolic_new)

        return peaks, peaks_inverse, abp_diastolic_new

    peaks = []
    for sample in x:
        signal = sample[:, 0]
        syst_peaks, dia_peaks, distolic_peaks = detect_abp_syst_peaks(signal)
        ppg = sample[:, 3]
        ppg_peaks, _ = find_peaks(ppg, distance=int(0.3 * fs), height=ppg.mean() + 3)

        if len(distolic_peaks) == 0:
            placeholder = np.zeros(3, dtype=np.float32)
            placeholder_int = np.zeros(3, dtype=np.int32)
            peaks.append(
                (placeholder_int, placeholder_int, placeholder_int, placeholder, placeholder, placeholder, ppg_peaks))
            continue

        syst_peaks_values = signal[syst_peaks]
        diatronic_peaks_values = signal[dia_peaks]
        distolic_peaks_values = signal[distolic_peaks]
        peaks.append(
            (syst_peaks, dia_peaks, distolic_peaks, syst_peaks_values, diatronic_peaks_values, distolic_peaks_values,
             ppg_peaks))

    def bandpass_filter(signal, lowcut=0.5, highcut=40.0, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    def ecg_bandpass(ecg_signal):
        return bandpass_filter(ecg_signal, lowcut=0.5, highcut=40.0, order=4)

    def get_pp(signal, signal_id):
        systolic = np.array(peaks[signal_id][0])
        diastolic = np.array(peaks[signal_id][2])
        time_diffs = []
        pp_values = []
        for i in range(len(systolic) - 1):
            mask = (diastolic > systolic[i]) & (diastolic < systolic[i + 1])

            if np.sum(mask) > 0:
                diastolic_peak = diastolic[mask][0]
                pp = signal[systolic[i]] - signal[diastolic_peak]
                time_diff = (diastolic_peak - systolic[i]) / fs
                time_diffs.append(time_diff)
                pp_values.append(pp)

        return pp_values, time_diffs

    def get_wavelet_entropies(signal):
        wavelet_features = []
        coeffs = pywt.wavedec(signal, 'bior3.3', level=5)

        for subband in coeffs:
            # Sample entropy
            sampen = ant.sample_entropy(subband)
            # Wavelet entropy (Shannon entropy of normalized coefficient energy)
            e = subband ** 2
            p = e / np.sum(e)
            wavelet_entropy = -np.sum(p * np.log2(p + 1e-12))
            wavelet_features.extend([sampen, wavelet_entropy])

        return wavelet_features

    def get_abp_hr(signal_id):
        hr_values = []
        syst_peaks = peaks[signal_id][0]
        for i in range(1, len(syst_peaks)):
            left_time = (syst_peaks[i] - syst_peaks[i - 1]) / fs
            hr = 60 / (left_time + 1e-6)
            hr_values.append(hr)

        return hr_values

    def get_ptt(signal_id):
        ptt_values = []
        syst_peaks = peaks[signal_id][0]
        ppg_peaks = peaks[signal_id][6]
        for sp in syst_peaks:
            ppg_candidates = ppg_peaks[ppg_peaks > sp]
            if len(ppg_candidates) > 0:
                ptt = (ppg_candidates[0] - sp) / fs
                ptt_values.append(ptt)

        return ptt_values

    def compute_hrv_freq(ecg_signal):
        # R-peak detection (simple, replace with robust detector in practice)
        peaks, _ = ssignal.find_peaks(ecg_signal, distance=int(0.25 * fs))  # ensure integer distance
        if len(peaks) < 2:
            return np.nan, np.nan, np.nan

        rr_intervals = np.diff(peaks) / fs  # in seconds

        # Interpolate RR intervals to evenly sampled series
        t_rr = np.cumsum(rr_intervals)
        t_interp = np.linspace(0, t_rr[-1], len(rr_intervals) * 4)
        rr_interp = np.interp(t_interp, t_rr, rr_intervals)

        # Welch PSD (set nperseg <= length to avoid warnings)
        nperseg = min(256, len(rr_interp))
        f, pxx = ssignal.welch(rr_interp - np.mean(rr_interp), fs=4.0, nperseg=nperseg)

        # Bands
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)

        lf_mask = (f >= lf_band[0]) & (f < lf_band[1])
        hf_mask = (f >= hf_band[0]) & (f < hf_band[1])

        lf_power = np.trapezoid(pxx[lf_mask], f[lf_mask]) if np.any(lf_mask) else np.nan
        hf_power = np.trapezoid(pxx[hf_mask], f[hf_mask]) if np.any(hf_mask) else np.nan
        lf_hf_ratio = lf_power / hf_power if hf_power and hf_power > 0 else np.nan

        return lf_power, hf_power, lf_hf_ratio

    # 2. PPG Harmonic Ratio (H2/H1)
    def compute_ppg_harmonic_ratio(ppg_signal):
        # Segment a single beat (naive: first ~1 sec)
        segment = ppg_signal[:fs]
        N = len(segment)
        if N < 2:
            return np.nan

        yf = np.abs(fft(segment))[:N // 2]
        xf = fftfreq(N, 1 / fs)[:N // 2]

        # Find fundamental frequency (peak in 0.5–3 Hz, typical HR)
        mask = (xf >= 0.5) & (xf <= 3)
        if not np.any(mask):
            return np.nan

        fundamental_idx = np.argmax(yf[mask]) + np.where(mask)[0][0]
        f0 = xf[fundamental_idx]

        # Amplitude at fundamental and 2nd harmonic
        H1 = yf[fundamental_idx]
        H2_idx = np.argmin(np.abs(xf - 2 * f0))
        H2 = yf[H2_idx]

        return H2 / H1 if H1 > 0 else np.nan

    # ---------------------------
    # 3. ABP Spectral Centroid
    # ---------------------------
    def compute_abp_spectral_centroid(abp_signal):
        nperseg = min(1024, len(abp_signal))
        f, pxx = ssignal.welch(abp_signal - np.mean(abp_signal), fs=fs, nperseg=nperseg)
        if np.sum(pxx) == 0:
            return np.nan
        centroid = np.sum(f * pxx) / np.sum(pxx)
        return centroid

    def build_feature_table(data_in, name):
        raw_features = {
            'abp_syst_peaks': lambda signal_id: peaks[signal_id][3],
            'abp_ditr_peaks': lambda signal_id: peaks[signal_id][4],
            'abp_dist_peaks': lambda signal_id: peaks[signal_id][5],
            'pulse_pressure': lambda signal_id: get_pp(data_in[signal_id, :, 0], signal_id)[0],
            'pulse_transit_time': lambda signal_id: get_ptt(signal_id),
            'syst_dist_time_diff': lambda signal_id: get_pp(data_in[signal_id, :, 0], signal_id)[1],
            'abp_hr': lambda signal_id: get_abp_hr(signal_id),
            'ecg_hr': lambda signal_id: nk.ecg_rate(
                nk.ecg_findpeaks(ecg_bandpass(data_in[signal_id, :, 1]), sampling_rate=fs)['ECG_R_Peaks'],
                sampling_rate=fs),
            # 'map': lambda signal_id: peaks[signal_id] + [v/3 for v in get_pp(data_in[signal_id,:,0], signal_id)[0]],

            'raw_ecgii': lambda signal_id: ecg_bandpass(data_in[signal_id, :, 1]),
            'raw_abp': lambda signal_id: data_in[signal_id, :, 0],
            'raw_ppg': lambda signal_id: data_in[signal_id, :, 3],
            'raw_ecgv': lambda signal_id: ecg_bandpass(data_in[signal_id, :, 2]),
        }

        raw_features_no_accum = {
            'wavelet_entropies_ecgii': lambda signal_id: get_wavelet_entropies(data_in[signal_id, :, 1]),
            'wavelet_entropies_ecgii_filtered': lambda signal_id: get_wavelet_entropies(
                ecg_bandpass(data_in[signal_id, :, 1])),
            'wavelet_entropies_ecgv': lambda signal_id: get_wavelet_entropies(ecg_bandpass(data_in[signal_id, :, 2])),
            'wavelet_entropies_abp': lambda signal_id: get_wavelet_entropies(data_in[signal_id, :, 0]),
            'wavelet_entropies_ppg': lambda signal_id: get_wavelet_entropies(data_in[signal_id, :, 3]),

            'ecg_lf_power': lambda signal_id: compute_hrv_freq(ecg_bandpass(data_in[signal_id, :, 1]))[0],
            'ecg_hf_power': lambda signal_id: compute_hrv_freq(ecg_bandpass(data_in[signal_id, :, 1]))[1],
            'ecg_lf_hf_ratio': lambda signal_id: compute_hrv_freq(ecg_bandpass(data_in[signal_id, :, 1]))[2],
            'ecg_lf_hf_ratio_v5': lambda signal_id: compute_hrv_freq(ecg_bandpass(data_in[signal_id, :, 2]))[2],
            'ppg_h2_h1_ratio': lambda signal_id: compute_ppg_harmonic_ratio(data_in[signal_id, :, 3]),
            'abp_spectral_centroid': lambda signal_id: compute_abp_spectral_centroid(data_in[signal_id, :, 0]),
            'ppg_spectral_centroid': lambda signal_id: compute_abp_spectral_centroid(data_in[signal_id, :, 3]),
        }

        # Get combined hash value for the feature maps, for pickle file
        combined_hash = hash(frozenset(raw_features.keys())) ^ hash(frozenset(raw_features_no_accum.keys())) ^ hash(
            frozenset(accumulation_types.keys()))
        pickle_file = f'feature_table_cache_{name}_{combined_hash}.pkl'

        feature_table = []
        for signal_id in tqdm(range(data_in.shape[0]), desc="Extracting features"):
            feature_vector = {}
            for feature_name, func in raw_features.items():
                feature_data = func(signal_id)

                for acc_name, acc_func in accumulation_types.items():
                    try:
                        acc_value = acc_func(feature_data)
                        if acc_value is None or (
                                isinstance(acc_value, float) and (np.isnan(acc_value) or np.isinf(acc_value))):
                            print(
                                f"Warning: Feature {feature_name} for signal {signal_id} produced invalid value {acc_value} with accumulation {acc_name}"
                            )
                    except Exception as e:
                        acc_value = np.nan
                        print(f"Error computing {acc_name} for feature {feature_name} on signal {signal_id}: {e}")

                    feature_vector[f"{feature_name}_{acc_name}"] = acc_value

            for feature_name, func in raw_features_no_accum.items():
                feature_data = func(signal_id)
                if isinstance(feature_data, (list, np.ndarray)):
                    for i, value in enumerate(feature_data):
                        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                            print(
                                f"Warning: Feature {feature_name} for signal {signal_id} produced invalid value {value} at index {i}"
                            )
                        feature_vector[f"{feature_name}_{i}"] = value
                else:
                    if feature_data is None or (
                            isinstance(feature_data, float) and (np.isnan(feature_data) or np.isinf(feature_data))):
                        print(
                            f"Warning: Feature {feature_name} for signal {signal_id} produced invalid value {feature_data}"
                        )
                    feature_vector[feature_name] = feature_data
            feature_table.append(feature_vector)

        df = pd.DataFrame(feature_table)
        return df

    feature_table = build_feature_table(data_in=x, name=name)

    feature_table = feature_table.replace(np.nan, 0.0)
    feature_table = feature_table.replace(np.inf, 0.0)
    feature_table = feature_table.replace(-np.inf, 0.0)
    feature_table = feature_table.fillna(0.0)

    selected_features_v0 = ['abp_syst_peaks_min', 'abp_syst_peaks_max', 'abp_syst_peaks_kurtosis', 'abp_syst_peaks_sum',
                            'abp_syst_peaks_var', 'abp_syst_peaks_var_coeff', 'abp_ditr_peaks_mean',
                            'abp_ditr_peaks_min', 'abp_ditr_peaks_max', 'abp_ditr_peaks_kurtosis',
                            'abp_ditr_peaks_median', 'abp_ditr_peaks_var', 'abp_dist_peaks_min',
                            'abp_dist_peaks_kurtosis', 'abp_dist_peaks_skewness', 'abp_dist_peaks_sum',
                            'abp_dist_peaks_var', 'pulse_pressure_median', 'pulse_pressure_var',
                            'pulse_pressure_var_coeff', 'pulse_transit_time_var_coeff', 'syst_dist_time_diff_max',
                            'syst_dist_time_diff_kurtosis', 'syst_dist_time_diff_sum', 'syst_dist_time_diff_var',
                            'abp_hr_sum', 'raw_ecgii_mean', 'raw_ecgii_kurtosis', 'raw_ecgii_skewness', 'raw_ecgii_sum',
                            'raw_ecgii_var_coeff', 'raw_abp_mean', 'raw_abp_max', 'raw_abp_median', 'raw_abp_sum',
                            'raw_ppg_min', 'raw_ppg_skewness', 'raw_ppg_var', 'raw_ppg_var_coeff', 'raw_ecgv_mean',
                            'raw_ecgv_median', 'wavelet_entropies_ecgii_0', 'wavelet_entropies_ecgii_2',
                            'wavelet_entropies_ecgii_5', 'wavelet_entropies_ecgii_6', 'wavelet_entropies_ecgii_7',
                            'wavelet_entropies_ecgii_8', 'wavelet_entropies_ecgii_10',
                            'wavelet_entropies_ecgii_filtered_0', 'wavelet_entropies_ecgii_filtered_2',
                            'wavelet_entropies_ecgii_filtered_3', 'wavelet_entropies_ecgii_filtered_5',
                            'wavelet_entropies_ecgii_filtered_7', 'wavelet_entropies_ecgii_filtered_8',
                            'wavelet_entropies_abp_0', 'wavelet_entropies_abp_2', 'wavelet_entropies_abp_3',
                            'wavelet_entropies_abp_7', 'wavelet_entropies_abp_9', 'wavelet_entropies_abp_11',
                            'wavelet_entropies_ppg_0', 'wavelet_entropies_ppg_1', 'wavelet_entropies_ppg_4',
                            'wavelet_entropies_ppg_5', 'wavelet_entropies_ppg_7', 'wavelet_entropies_ppg_10']
    selected_features_v1 = ['abp_syst_peaks_mean', 'abp_syst_peaks_std', 'abp_syst_peaks_skewness',
                            'abp_syst_peaks_median', 'abp_syst_peaks_sum', 'abp_syst_peaks_var', 'abp_ditr_peaks_mean',
                            'abp_ditr_peaks_kurtosis', 'abp_ditr_peaks_median', 'abp_ditr_peaks_var',
                            'abp_ditr_peaks_var_coeff', 'abp_dist_peaks_max', 'abp_dist_peaks_kurtosis',
                            'abp_dist_peaks_sum', 'pulse_pressure_mean', 'pulse_pressure_std', 'pulse_pressure_min',
                            'pulse_pressure_skewness', 'pulse_pressure_var', 'pulse_transit_time_mean',
                            'pulse_transit_time_kurtosis', 'pulse_transit_time_var', 'syst_dist_time_diff_mean',
                            'syst_dist_time_diff_std', 'syst_dist_time_diff_max', 'syst_dist_time_diff_kurtosis',
                            'syst_dist_time_diff_median', 'syst_dist_time_diff_sum', 'syst_dist_time_diff_var_coeff',
                            'abp_hr_std', 'abp_hr_skewness', 'abp_hr_median', 'abp_hr_sum', 'raw_ecgii_mean',
                            'raw_ecgii_kurtosis', 'raw_ecgii_sum', 'raw_abp_mean', 'raw_abp_skewness', 'raw_abp_sum',
                            'raw_ppg_mean', 'raw_ppg_min', 'raw_ppg_max', 'raw_ppg_median', 'raw_ppg_var_coeff',
                            'raw_ecgv_std', 'raw_ecgv_max', 'raw_ecgv_skewness', 'raw_ecgv_sum',
                            'wavelet_entropies_ecgii_1', 'wavelet_entropies_ecgii_2', 'wavelet_entropies_ecgii_3',
                            'wavelet_entropies_ecgii_4', 'wavelet_entropies_ecgii_8',
                            'wavelet_entropies_ecgii_filtered_0', 'wavelet_entropies_ecgii_filtered_2',
                            'wavelet_entropies_ecgii_filtered_4', 'wavelet_entropies_ecgii_filtered_7',
                            'wavelet_entropies_ecgii_filtered_8', 'wavelet_entropies_ecgii_filtered_9',
                            'wavelet_entropies_abp_7', 'wavelet_entropies_abp_10', 'wavelet_entropies_abp_11',
                            'wavelet_entropies_ppg_0', 'wavelet_entropies_ppg_1', 'wavelet_entropies_ppg_2',
                            'wavelet_entropies_ppg_3', 'wavelet_entropies_ppg_4', 'wavelet_entropies_ppg_5',
                            'wavelet_entropies_ppg_7', 'wavelet_entropies_ppg_8', 'wavelet_entropies_ppg_9',
                            'wavelet_entropies_ppg_10', 'wavelet_entropies_ppg_11']
    # Optimized for AUPCR
    selected_features_v2 = ['abp_syst_peaks_sum', 'abp_syst_peaks_var', 'abp_ditr_peaks_std', 'abp_ditr_peaks_max',
                            'abp_ditr_peaks_kurtosis', 'abp_ditr_peaks_skewness', 'abp_ditr_peaks_var_coeff',
                            'abp_dist_peaks_std', 'abp_dist_peaks_min', 'abp_dist_peaks_max', 'abp_dist_peaks_kurtosis',
                            'abp_dist_peaks_skewness', 'abp_dist_peaks_median', 'abp_dist_peaks_sum',
                            'abp_dist_peaks_var', 'pulse_pressure_max', 'pulse_pressure_kurtosis',
                            'pulse_pressure_median', 'pulse_pressure_sum', 'pulse_pressure_var',
                            'pulse_pressure_var_coeff', 'pulse_transit_time_skewness', 'pulse_transit_time_var',
                            'syst_dist_time_diff_mean', 'syst_dist_time_diff_max', 'syst_dist_time_diff_kurtosis',
                            'syst_dist_time_diff_var_coeff', 'abp_hr_min', 'abp_hr_var_coeff', 'raw_ecgii_mean',
                            'raw_ecgii_kurtosis', 'raw_ecgii_sum', 'raw_abp_mean', 'raw_abp_std', 'raw_abp_min',
                            'raw_abp_max', 'raw_abp_kurtosis', 'raw_abp_sum', 'raw_ppg_mean', 'raw_ppg_max',
                            'raw_ppg_skewness', 'raw_ppg_sum', 'raw_ppg_var', 'raw_ppg_var_coeff', 'raw_ecgv_mean',
                            'raw_ecgv_kurtosis', 'raw_ecgv_sum', 'raw_ecgv_var_coeff', 'wavelet_entropies_ecgii_0',
                            'wavelet_entropies_ecgii_5', 'wavelet_entropies_ecgii_8',
                            'wavelet_entropies_ecgii_filtered_0', 'wavelet_entropies_ecgii_filtered_2',
                            'wavelet_entropies_ecgii_filtered_3', 'wavelet_entropies_ecgii_filtered_4',
                            'wavelet_entropies_ecgii_filtered_5', 'wavelet_entropies_ecgii_filtered_8',
                            'wavelet_entropies_abp_0', 'wavelet_entropies_abp_1', 'wavelet_entropies_abp_3',
                            'wavelet_entropies_abp_4', 'wavelet_entropies_abp_5', 'wavelet_entropies_abp_9',
                            'wavelet_entropies_ppg_0', 'wavelet_entropies_ppg_1', 'wavelet_entropies_ppg_5',
                            'wavelet_entropies_ppg_6', 'wavelet_entropies_ppg_8', 'wavelet_entropies_ppg_9',
                            'wavelet_entropies_ppg_11']

    # 25 max Features, optimized for the same as main
    selected_features_v3 = ['abp_ditr_peaks_mean', 'abp_ditr_peaks_sum', 'abp_dist_peaks_max',
                            'abp_dist_peaks_kurtosis', 'pulse_pressure_mean', 'pulse_pressure_min',
                            'pulse_pressure_skewness', 'pulse_transit_time_mean', 'syst_dist_time_diff_max',
                            'syst_dist_time_diff_skewness', 'syst_dist_time_diff_sum', 'raw_ecgii_var_coeff',
                            'raw_abp_mean', 'raw_ecgv_skewness', 'wavelet_entropies_ecgii_5',
                            'wavelet_entropies_ecgv_9', 'wavelet_entropies_abp_9', 'wavelet_entropies_ppg_4',
                            'wavelet_entropies_ppg_6']

    # 50 max Features, but 100 estimators instead of 200
    selected_features_v4 = ['abp_syst_peaks_kurtosis', 'abp_syst_peaks_skewness', 'abp_syst_peaks_median',
                            'abp_syst_peaks_var', 'abp_dist_peaks_min', 'abp_dist_peaks_sum',
                            'abp_dist_peaks_var_coeff', 'pulse_pressure_std', 'pulse_pressure_min',
                            'pulse_pressure_kurtosis', 'pulse_pressure_var', 'pulse_transit_time_std',
                            'pulse_transit_time_min', 'pulse_transit_time_skewness', 'pulse_transit_time_var',
                            'pulse_transit_time_var_coeff', 'syst_dist_time_diff_var', 'abp_hr_max', 'abp_hr_kurtosis',
                            'abp_hr_skewness', 'ecg_hr_std', 'raw_ecgii_std', 'raw_ecgii_sum', 'raw_abp_mean',
                            'raw_abp_sum', 'raw_abp_var', 'raw_ppg_min', 'raw_ppg_var_coeff',
                            'wavelet_entropies_ecgii_3', 'wavelet_entropies_ecgii_5', 'wavelet_entropies_ecgii_8',
                            'wavelet_entropies_ecgii_filtered_6', 'wavelet_entropies_ecgv_3',
                            'wavelet_entropies_ecgv_7', 'wavelet_entropies_ecgv_11', 'wavelet_entropies_abp_0',
                            'wavelet_entropies_abp_2', 'wavelet_entropies_abp_9', 'wavelet_entropies_ppg_2',
                            'wavelet_entropies_ppg_4', 'wavelet_entropies_ppg_8', 'wavelet_entropies_ppg_9',
                            'ecg_lf_power', 'ecg_hf_power', 'ecg_lf_hf_ratio', 'ecg_lf_hf_ratio_v5']

    selected_features_v5 = ['abp_ditr_peaks_mean', 'abp_ditr_peaks_max', 'abp_ditr_peaks_kurtosis',
                            'abp_ditr_peaks_var', 'abp_dist_peaks_mean', 'abp_dist_peaks_std', 'abp_dist_peaks_max',
                            'abp_dist_peaks_kurtosis', 'abp_dist_peaks_var', 'abp_dist_peaks_var_coeff',
                            'pulse_pressure_var_coeff', 'pulse_transit_time_min', 'pulse_transit_time_kurtosis',
                            'pulse_transit_time_median', 'pulse_transit_time_var_coeff', 'abp_hr_mean', 'abp_hr_max',
                            'abp_hr_kurtosis', 'abp_hr_var', 'ecg_hr_var_coeff', 'raw_ecgii_std', 'raw_ecgii_var_coeff',
                            'raw_abp_mean', 'raw_ppg_mean', 'raw_ecgv_min', 'raw_ecgv_var', 'wavelet_entropies_ecgii_1',
                            'wavelet_entropies_ecgii_filtered_0', 'wavelet_entropies_ecgii_filtered_7',
                            'wavelet_entropies_abp_11', 'wavelet_entropies_ppg_4', 'wavelet_entropies_ppg_6',
                            'wavelet_entropies_ppg_7', 'ecg_hf_power', 'ecg_lf_hf_ratio', 'ecg_lf_hf_ratio_v5',
                            'abp_spectral_centroid']

    selected_features_select = selected_features_v5
    # take only selected features and return numpy array
    feature_table = feature_table[selected_features_select]

    # Remove outliers
    # if filter_outliers:
    #    x_ready = zero_outliers_iqr(feature_table, factor=20.0).to_numpy(dtype=np.float32)
    # else:
    x_ready = feature_table.to_numpy(dtype=np.float32)

    # Remove false values
    # this does not work
    # x_ready = x_ready.fillna(0.0)
    x_ready = np.nan_to_num(x_ready, nan=0.0, posinf=0.0, neginf=0.0)

    return x_ready


"""
# OLD
def handle_ecg(x):
    # Handle ECG specific processing
    lead_ii = x[:, 1]
    lead_v5 = x[:, 2]

    coeffs = pywt.wavedec(lead_ii, 'bior3.3', level=5)
    features = []
    for subband in coeffs:
        # Sample entropy
        sampen = ant.sample_entropy(subband)
        # Wavelet entropy (Shannon entropy of normalized coefficient energy)
        e = subband ** 2
        p = e / np.sum(e)
        wavelet_entropy = -np.sum(p * np.log2(p + 1e-12))
        features.extend([sampen, wavelet_entropy])
    # Wavelet Decomposition
    # features.extend(compute_stats(x, 0))
    # features.extend(compute_stats(x, 2))
    # return features
    # return extract_abp_features(lead_ii, fs=250,) + extract_abp_features(lead_v5, fs=250,)


def handle_bp(x):
    # Handle PPG and ABP Speciic things
    abp = x[:, 0]
    ppg = x[:, 3]

    # features = []
    # features.extend(compute_stats(x, 4))
    # features.extend(compute_stats(x, 3))
    # xreturn features
    features = extract_abp_features(ppg, fs=250, )
    features.extend(extract_abp_features(abp, fs=250, ))
    # return features

    return None


def handle_capnogram(x):
    # Handle Capnogram specific processing
    capnogram = x[:, 4]
    etco2_list = []
    # Detect maxima
    # peaks, _ = find_peaks(capnogram, distance=int(0.7*fs), height=30)

    # for peak in peaks:
    #    etco2_list.append(capnogram[peak])

    # Compute statistics
    # mean = np.mean(etco2_list)
    # std = np.std(etco2_list)

    if mean < 30:
        print(" Warning : Low EtCO2 mean value ", mean)
        from matplotlib import pyplot as plt
        plt.plot(capnogram)
        plt.show()

        for peak in peaks:
            print(capnogram[peak])

        print(" Peaks : ", peaks)
        print(" EtCO2 list : ", etco2_list)
        raise Exception()

    # return [mean, std]
    # return compute_stats(x, 4)
    return None


def compute_stationary_statistics(signal):
    # return np.mean(signal), np.std(signal)/(np.mean(signal)+1e-6), np.max(signal), np.min(signal), skew(signal), kurtosis(signal)
    return np.mean(signal), (np.std(signal) + 1e-6 / np.mean(signal)) * 100,


def compute_variance(signal):
    return []


def detect_systolic_diastolic_peaks(signal):
    peaks, _ = find_peaks(signal, distance=int(0.3 * fs), height=signal.mean() + 5)
    peaks_inverse, _ = find_peaks(-signal, distance=int(0.3 * fs), height=-signal.mean() + 5, )
    peaks_all = peaks_inverse

    # Filter Diastolic peaks by removing all but the first one after the peak foot
    for i in range(len(peaks) - 1):
        mask = (peaks_all > peaks[i]) & (peaks_all < peaks[i + 1])
        if np.sum(mask) > 1:
            to_remove = peaks_all[mask][1:]
            peaks_all = peaks_all[~np.isin(peaks_all, to_remove)]

    abp_diastolic_new = []
    for i in range(len(peaks) - 1):
        mask = (peaks_inverse > peaks[i]) & (peaks_inverse < peaks[i + 1])
        if np.sum(mask) > 0:
            to_keep = peaks_inverse[mask]
            closest = to_keep[np.argmin(peaks[i + 1] - to_keep)]
            abp_diastolic_new.append(closest)

    return peaks, peaks_inverse, abp_diastolic_new


def handle_fresh_abp_features(x):
    abp = x[:, 0]
    ppg = x[:, 3]

    syst_peaks, dicrotic_peaks, diastolic_peaks = detect_systolic_diastolic_peaks(abp)
    ppg_peaks = find_peaks(ppg, distance=int(0.3 * fs), height=np.percentile(ppg, 75))[0]

    features = []
    peaks_values = abp[syst_peaks]
    diastolic_values = abp[diastolic_peaks]

    # To calculate the pulse pressure, we need to match systolic and diastolic peaks
    pp_values = []
    time_diffs = []
    diastolic_peaks = np.array(diastolic_peaks)
    for i in range(len(syst_peaks) - 1):
        mask = (diastolic_peaks > syst_peaks[i]) & (diastolic_peaks < syst_peaks[i + 1])
        if np.sum(mask) > 0:
            diastolic_peak = diastolic_peaks[mask][0]
            pp = abp[syst_peaks[i]] - abp[diastolic_peak]
            time_diff = (diastolic_peak - syst_peaks[i]) / fs
            time_diffs.append(time_diff)
            pp_values.append(pp)

    map = diastolic_values + [v / 3 for v in pp_values]

    # Calculate Heart Rate with sliding window of 3 beats
    hr_values = []
    for i in range(1, len(syst_peaks)):
        left_time = (syst_peaks[i] - syst_peaks[i - 1]) / fs
        hr = 60 / left_time
        hr_values.append(hr)

    # Calculate transit time by finding the peak in the PPG coming after the ABP Syst peak
    ptt_values = []
    for sp in syst_peaks:
        ppg_candidates = ppg_peaks[ppg_peaks > sp]
        if len(ppg_candidates) > 0:
            ptt = (ppg_candidates[0] - sp) / fs
            ptt_values.append(ptt)

    features.extend(compute_stationary_statistics(peaks_values))
    features.extend(compute_stationary_statistics(diastolic_values))
    features.extend(compute_stationary_statistics(pp_values))
    features.extend(compute_stationary_statistics(hr_values))
    features.extend(compute_stationary_statistics(time_diffs))
    features.extend(compute_stationary_statistics(map))
    features.extend(compute_variance(ptt_values))

    # Wavelet features
    coeffs = pywt.wavedec(ppg, 'bior3.3', level=5)
    wavelet_features = []
    for subband in coeffs:
        # Sample entropy
        sampen = ant.sample_entropy(subband)
        # Wavelet entropy (Shannon entropy of normalized coefficient energy)
        e = subband ** 2
        p = e / np.sum(e)
        wavelet_entropy = -np.sum(p * np.log2(p + 1e-12))
        wavelet_features.extend([sampen, wavelet_entropy])

    features.extend(wavelet_features)
    return features


def handle_transit_time(x):
    # Handle Transit time specific processing

    ecg = x[:, 1]
    abp = x[:, 0]
    ppg = x[:, 3]

    # --- Step 1: R-peak detection in ECG ---
    if np.isnan(ecg).sum() > 0:
        imputer = KNNImputer(n_neighbors=3, weights="uniform")
        ecg = imputer.fit_transform(ecg.reshape(-1, 1)).flatten()

    try:
        ecg_processed, ecg_info = 0, 0  # nk.ecg_process(ecg, sampling_rate=fs)
        r_peaks = ecg_info["ECG_R_Peaks"]
    except ValueError:
        r_peaks = find_peaks(ecg, distance=int(0.6 * fs), height=ecg.mean() + 0.5 * ecg.std())[0]

    if len(r_peaks) < 2:
        r_peaks = find_peaks(ecg, distance=int(0.6 * fs), height=ecg.mean() + 0.5 * ecg.std())[0]

    # --- Step 2: Find ABP and PPG "foot" (pulse onset) ---
    # Simplest method: find minima before systolic upstroke
    # More robust: use derivative-based method
    # abp_diff = np.diff(abp)
    # ppg_diff = np.diff(ppg)

    abp_foot, _ = find_peaks(abp, distance=int(0.3 * fs), height=abp.mean() + 5)
    ppg_foot, _ = find_peaks(ppg, distance=int(0.3 * fs), height=ppg.mean() + 3)

    # --- Step 3: Match beats ---
    ptt_values = []
    pat_values = []
    for r in r_peaks:
        # Find closest ABP foot after R-peak
        abp_candidates = abp_foot[abp_foot > r]
        ppg_candidates = ppg_foot[ppg_foot > r]

        if len(abp_candidates) > 0:
            ptt = (abp_candidates[0] - r) / fs
            ptt_values.append(ptt)
        if len(ppg_candidates) > 0:
            pat = (ppg_candidates[0] - r) / fs
            pat_values.append(pat)

    ptts = np.array(ptt_values)
    pats = np.array(pat_values)
    return [np.mean(ptts), np.std(ptts), np.mean(pats), np.std(pats)]


if __name__ == '__main__':
    print('running features_comp.py')
"""

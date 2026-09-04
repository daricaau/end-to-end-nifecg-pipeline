"""
fecg_common.py
================
Shared pipeline for fetal-ECG extraction from simulated abdominal recordings.

Every one of the 4 thesis scenarios (3ch/OLS, 3ch/RLS, 5ch/OLS, 5ch/RLS)
imports ONLY from this file for:
    - loading + filtering + normalising the data
    - Pan-Tompkins maternal R-peak detection & consensus timing
    - windowed PCA (fit on QRS windows, projected onto full recording)
    - OLS regression   (one consistent implementation, used by both channel counts)
    - warm started RLS regression   (one consistent implementation, used by both channel counts)
    - fetal QRS detection + peak matching (for sensitivity / PPV / F1)
    - RMSE / PRD / SNR / Pearson r
    - bootstrap & permutation significance testing
    - every plot function, so all 4 scenarios produce the same figure set

This means the ONLY things that differ between scenario scripts are:
    1) which/how-many reference leads go into the PCA (3 vs 5), and
    2) which regression function is called (OLS vs RLS).
Everything else -- filtering, peak detection, windowing, stats, plots --
is identical code, guaranteeing a fair, controlled comparison.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from matplotlib.ticker import FormatStrFormatter

ALL_COLUMNS = ['time_ms', 'F1', 'F2', 'F3', 'V2', 'V5', 'V8', 'RA', 'LA', 'GND', 'LL']
ABDO_LEADS  = ['F1', 'F2', 'F3']

W_PRE_MS  = 200.0   # window before consensus R-peak
W_POST_MS = 300.0   # window after consensus R-peak
PEAK_MATCH_TOL_MS = 50.0   # PhysioNet/CinC 2013 fetal-QRS evaluation convention


# ============================================================
# 1. LOAD + PREPROCESS
# ============================================================
def load_and_preprocess(filepath, ref_leads):
    """Bandpass -> notch -> joint z-score. Identical for every scenario;
    only `ref_leads` (3 or 5 thoracic/limb leads) changes."""
    df = pd.read_csv(filepath, sep=r'\s+', header=None)
    df.columns = ALL_COLUMNS

    time_ms = df['time_ms'].values
    dt_ms = time_ms[1] - time_ms[0]
    fs = 1000.0 / dt_ms
    nyquist = fs / 2.0

    all_leads = ABDO_LEADS + ref_leads
    raw = df[all_leads].values
    N_samples, N_leads = raw.shape

    if 80.0 >= nyquist:
        raise ValueError(f"80 Hz cutoff invalid, Nyquist={nyquist:.2f} Hz")
    b_bp, a_bp = butter(2, [3.0 / nyquist, 80.0 / nyquist], btype='bandpass')
    bp = np.zeros_like(raw, dtype=float)
    for i in range(N_leads):
        bp[:, i] = filtfilt(b_bp, a_bp, raw[:, i])

    b_notch, a_notch = iirnotch(50.0 / nyquist, Q=30.0)
    notch = np.zeros_like(bp)
    for i in range(N_leads):
        notch[:, i] = filtfilt(b_notch, a_notch, bp[:, i])

    scaler = StandardScaler()
    norm = scaler.fit_transform(notch)
    norm_df = pd.DataFrame(norm, columns=all_leads)

    norm_abd = norm_df[ABDO_LEADS].values
    norm_ref = norm_df[ref_leads].values
    abd_std  = scaler.scale_[[all_leads.index(l) for l in ABDO_LEADS]]

    return dict(
        time_ms=time_ms, dt_ms=dt_ms, fs=fs, nyquist=nyquist,
        all_leads=all_leads, ref_leads=ref_leads,
        notch_signals=notch, norm_df=norm_df,
        norm_abd=norm_abd, norm_ref=norm_ref, abd_std=abd_std,
        scaler=scaler, N_samples=N_samples
    )


# ============================================================
# 2. PAN-TOMPKINS MATERNAL R-PEAK DETECTOR (used for windowing/PCA)
# ============================================================
def pan_tompkins_r_peaks(sig, fs):
    """Pan & Tompkins (1985): QRS-emphasis bandpass -> derivative -> squaring
    -> moving-window integration -> adaptive-height peak picking -> refine
    onto the true local maximum. Tuned for MATERNAL heart rate (<=100 bpm)."""
    low, high = 5.0 / (fs / 2), 20.0 / (fs / 2)
    b_qrs, a_qrs = butter(2, [low, high], btype='bandpass')
    qrs_bp = filtfilt(b_qrs, a_qrs, sig)

    diff = np.diff(qrs_bp, prepend=qrs_bp[0])
    squared = diff ** 2

    win_len = max(1, int(round(0.15 * fs)))
    integrated = np.convolve(squared, np.ones(win_len) / win_len, mode='same')

    min_distance = int(round(0.6 * fs))     # caps at 100 bpm
    threshold = 0.35 * np.max(integrated)
    peaks_int, _ = find_peaks(integrated, height=threshold, distance=min_distance)

    search_radius = int(round(0.075 * fs))
    r_peaks = []
    for p in peaks_int:
        lo = max(0, p - search_radius)
        hi = min(len(sig), p + search_radius)
        r_peaks.append(lo + np.argmax(sig[lo:hi]))
    return np.array(sorted(set(r_peaks)))


def print_cross_lead_r_peak_agreement(beat_times_matrix, lead_names, label):
    """Prints per-QRS-complex timing agreement across leads, in ms: for
    every detected beat, the time registered by each lead, the cross-lead
    mean, standard deviation ("error"), and range, followed by summary
    statistics across all beats. NaN entries mean that lead did not
    register that particular beat (relevant for the fetal case, where
    per-lead detections can disagree in count before consensus)."""
    beat_times_matrix = np.asarray(beat_times_matrix, dtype=float)
    n_beats, n_leads = beat_times_matrix.shape
    mean_t = np.nanmean(beat_times_matrix, axis=1)
    std_t = np.nanstd(beat_times_matrix, axis=1)
    range_t = np.nanmax(beat_times_matrix, axis=1) - np.nanmin(beat_times_matrix, axis=1)

    print(f"\n{'='*90}")
    print(f"CROSS-LEAD R-PEAK TIMING AGREEMENT — {label}")
    print(f"{'='*90}")
    header = f"{'Beat':<6}" + "".join(f"{l + ' (ms)':>12}" for l in lead_names) \
             + f"{'Mean(ms)':>12}{'Error/Std(ms)':>15}{'Range(ms)':>11}"
    print(header)
    print('-' * len(header))
    for k in range(n_beats):
        row = f"{k + 1:<6}"
        for j in range(n_leads):
            v = beat_times_matrix[k, j]
            row += f"{v:>12.1f}" if not np.isnan(v) else f"{'--':>12}"
        row += f"{mean_t[k]:>12.1f}{std_t[k]:>15.2f}{range_t[k]:>11.1f}"
        print(row)

    print(f"\n  Overall cross-lead QRS timing error ({label}), across all {n_beats} beats:")
    print(f"    Mean std across beats   : {np.nanmean(std_t):.2f} ms")
    print(f"    Max std (any beat)      : {np.nanmax(std_t):.2f} ms")
    print(f"    Mean range across beats : {np.nanmean(range_t):.2f} ms")
    print(f"    Max range (any beat)    : {np.nanmax(range_t):.2f} ms")

    return dict(mean_std_ms=float(np.nanmean(std_t)), max_std_ms=float(np.nanmax(std_t)),
                mean_range_ms=float(np.nanmean(range_t)), max_range_ms=float(np.nanmax(range_t)))


def detect_consensus_peaks(P, leads_to_check=('V2', 'V5', 'V8')):
    """Runs Pan-Tompkins independently on V2/V5/V8, pairs beats by index,
    and returns the per-beat MEDIAN time across leads (robust to one noisy lead).
    Also prints the cross-lead QRS timing agreement (error in ms) across
    V2/V5/V8 for every detected maternal beat."""
    time_ms, notch_signals, all_leads, fs = P['time_ms'], P['notch_signals'], P['all_leads'], P['fs']

    r_peaks_results = {}
    for L in leads_to_check:
        sig_clean = notch_signals[:, all_leads.index(L)]
        r_peaks_results[L] = pan_tompkins_r_peaks(sig_clean, fs)

    n_beats_per_lead = [len(r_peaks_results[L]) for L in leads_to_check]
    if len(set(n_beats_per_lead)) != 1:
        print("  WARNING: leads disagree on number of detected beats:",
              dict(zip(leads_to_check, n_beats_per_lead)))

    n_beats = min(n_beats_per_lead)
    beat_times_ms = np.column_stack([time_ms[r_peaks_results[L][:n_beats]] for L in leads_to_check])
    consensus_r_peak_times = np.median(beat_times_ms, axis=1)
    beat_std = beat_times_ms.std(axis=1)

    print_cross_lead_r_peak_agreement(beat_times_ms, list(leads_to_check),
                                       "Maternal R-Peaks (Pan-Tompkins, V2/V5/V8)")

    return dict(r_peaks_results=r_peaks_results, beat_times_ms=beat_times_ms,
                consensus_r_peak_times=consensus_r_peak_times, beat_std=beat_std,
                leads_to_check=list(leads_to_check))


# ============================================================
# 3. QRS-CENTRED WINDOWS FOR PCA (edge beats zero-padded, never dropped)
# ============================================================
def extract_qrs_windows(P, consensus_r_peak_times):
    time_ms, dt_ms, N_samples, norm_ref = P['time_ms'], P['dt_ms'], P['N_samples'], P['norm_ref']
    n_ref = norm_ref.shape[1]

    W_pre = int(round(W_PRE_MS / dt_ms))
    W_post = int(round(W_POST_MS / dt_ms))
    window_length = W_pre + W_post

    window_indices, window_segments = [], []
    for t_peak in consensus_r_peak_times:
        n_k = int(round((t_peak - time_ms[0]) / dt_ms))
        lo, hi = n_k - W_pre, n_k + W_post

        segment = np.zeros((window_length, n_ref))
        valid_lo, valid_hi = max(0, lo), min(N_samples, hi)
        ins_start = valid_lo - lo
        ins_end = ins_start + (valid_hi - valid_lo)
        segment[ins_start:ins_end, :] = norm_ref[valid_lo:valid_hi, :]

        window_segments.append(segment)
        window_indices.append((valid_lo, valid_hi))   

    T_windowed = np.vstack(window_segments)
    return dict(window_indices=window_indices, window_segments=window_segments,
                T_windowed=T_windowed, W_pre=W_pre, W_post=W_post)


# ============================================================
# 4. WINDOWED-FIT PCA, PROJECTED ONTO FULL RECORDING
# ============================================================
def run_pca_windowed(P, W, n_retain=2):
    """PCA basis is fit ONLY on pooled QRS windows (maternal-dominated
    segments) then the whole recording is projected onto that basis.
    Eigenvector sign is fixed (sum of loadings > 0) so PC polarity is
    reproducible run to run / scenario to scenario."""
    T = P['norm_ref']
    pca = PCA(n_components=T.shape[1])
    pca.fit(W['T_windowed'])

    eigenvectors = pca.components_.copy()
    T_pca = pca.transform(T)
    for k in range(eigenvectors.shape[0]):
        if np.sum(eigenvectors[k]) < 0:
            eigenvectors[k] = -eigenvectors[k]
            T_pca[:, k] = -T_pca[:, k]

    explained_ratio = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_
    cumulative_variance = np.cumsum(explained_ratio)

    return dict(pca=pca, eigenvectors=eigenvectors, eigenvalues=eigenvalues,
                explained_ratio=explained_ratio, cumulative_variance=cumulative_variance,
                T_pca=T_pca, M1=T_pca[:, 0], M2=T_pca[:, 1], n_retain=n_retain)


# ============================================================
# 5a. OLS REGRESSION  (one global batch fit -- used by BOTH 3ch & 5ch scripts)
# ============================================================
def fit_maternal_template_ols(P, PC):
    """Single least-squares fit per abdominal lead over the WHOLE recording:
    F_lead = w1*PC1 + w2*PC2 + error.  This is the one and only OLS
    implementation used across scenarios -- deliberately simple (no
    windowing/tapering) so it is a clean, directly-comparable baseline
    against the RLS filter, which is likewise applied globally/continuously."""
    X = np.column_stack([PC['M1'], PC['M2']])
    norm_abd = P['norm_abd']

    results = {}
    for i, lead in enumerate(ABDO_LEADS):
        y = norm_abd[:, i]
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ w
        residual = y - y_hat
        results[lead] = dict(y_hat=y_hat, residual=residual, weights=w)
    return dict(method='OLS', X=X, results=results)


# ============================================================
# 5b. RLS REGRESSION (one tuned implementation -- used by BOTH 3ch & 5ch scripts)
# ============================================================
RLS_LAMBDA = 0.999   # memory horizon = 1/(1-lambda) samples -> spans whole recording
RLS_DELTA  = 10.0    # small initial P (=0.1*I) so the OLS warm-start is trusted


def _rls_core(X, y, w_init, lam=RLS_LAMBDA, delta=RLS_DELTA):
    N, p = X.shape
    w = w_init.copy()
    Pmat = (1.0 / delta) * np.eye(p)
    y_hat = np.zeros(N)
    weights = np.zeros((N, p))
    for n in range(N):
        x_n = X[n, :]
        y_hat_n = x_n @ w
        e_n = y[n] - y_hat_n
        denom = lam + x_n @ Pmat @ x_n
        k_n = (Pmat @ x_n) / denom
        w = w + k_n * e_n
        Pmat = (Pmat - np.outer(k_n, x_n @ Pmat)) / lam
        y_hat[n] = y_hat_n
        weights[n, :] = w
    return y_hat, y - y_hat, weights


def fit_maternal_template_rls(P, PC):
    """Recursive Least Squares, applied CONTINUOUSLY across the whole
    recording (not per-beat), warm-started from the OLS solution so RLS
    isn't spending the first beats re-learning what OLS already knows.
    lambda=0.999 gives a ~5 s memory horizon at fs~200Hz, i.e. it can
    track slow (respiration-driven) drift in the maternal-to-abdominal
    coupling without forgetting each beat before the next one arrives."""
    X = np.column_stack([PC['M1'], PC['M2']])
    norm_abd = P['norm_abd']

    results = {}
    for i, lead in enumerate(ABDO_LEADS):
        y = norm_abd[:, i]
        w_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat, residual, weights = _rls_core(X, y, w_init=w_ols)
        results[lead] = dict(y_hat=y_hat, residual=residual, weights=weights, w_ols=w_ols)
    return dict(method='RLS', X=X, results=results, lam=RLS_LAMBDA, delta=RLS_DELTA)


def to_mV(P, REG):
    """Convert z-scored template/residual back into mV using the abdominal
    lead standard deviations captured before normalisation."""
    abd_std = P['abd_std']
    maternal_mV = np.column_stack([REG['results'][l]['y_hat'] * abd_std[i] for i, l in enumerate(ABDO_LEADS)])
    fetal_mV = np.column_stack([REG['results'][l]['residual'] * abd_std[i] for i, l in enumerate(ABDO_LEADS)])
    abd_mV = P['norm_abd'] * abd_std
    return maternal_mV, fetal_mV, abd_mV


# ============================================================
# 6. FETAL QRS DETECTOR (tighter than the maternal one -- higher HR, narrower QRS)
# ============================================================
def fetal_qrs_detect(sig, fs, edge_guard_ms=50.0):
    """Detects fetal QRS peaks. A peak sitting within `edge_guard_ms` of
    the very start or end of the recording is discarded: the detector's
    derivative/integration/refinement steps all need genuine signal on
    BOTH sides of a peak to confirm it's a true local maximum, and a peak
    right at the recording boundary (e.g. the last sample) cannot be
    verified this way -- it represents a truncated, incomplete beat
    rather than a real, complete QRS complex, and would otherwise be
    silently counted as a genuine detection."""
    low, high = 10.0 / (fs / 2), min(25.0 / (fs / 2), 0.99)
    b_q, a_q = butter(2, [low, high], btype='bandpass')
    qrs = filtfilt(b_q, a_q, sig)
    diff = np.diff(qrs, prepend=qrs[0])
    sq = diff ** 2
    win = max(1, int(round(0.05 * fs)))
    integ = np.convolve(sq, np.ones(win) / win, mode='same')
    min_distance = int(round(0.3 * fs))       # caps at 200 bpm
    threshold = 0.25 * np.max(integ)
    peaks_int, _ = find_peaks(integ, height=threshold, distance=min_distance)
    search_radius = int(round(0.04 * fs))
    r_peaks = []
    for p in peaks_int:
        lo = max(0, p - search_radius)
        hi = min(len(sig), p + search_radius)
        r_peaks.append(lo + np.argmax(np.abs(sig[lo:hi])))
    r_peaks = sorted(set(r_peaks))

    guard_samples = int(round(edge_guard_ms / 1000.0 * fs))
    n = len(sig)
    r_peaks = [p for p in r_peaks if guard_samples <= p <= (n - 1 - guard_samples)]
    return np.array(r_peaks)


def _match_peaks_indices(ext_times_ms, gt_times_ms, tol_ms=PEAK_MATCH_TOL_MS):
    """Nearest-neighbour peak matching within tol_ms (the same +/-50 ms
    tolerance convention used by PhysioNet. Returns
    matched (gt_index, ext_index) pairs plus the unmatched FP/FN indices."""
    gt_used = np.zeros(len(gt_times_ms), dtype=bool)
    ext_used = np.zeros(len(ext_times_ms), dtype=bool)
    matches = []
    for ei, et in enumerate(ext_times_ms):
        diffs = np.abs(gt_times_ms - et)
        diffs[gt_used] = np.inf
        if len(diffs) and diffs.min() <= tol_ms:
            gi = int(np.argmin(diffs))
            gt_used[gi] = True
            ext_used[ei] = True
            matches.append((gi, ei))
    fp_idx = [i for i in range(len(ext_times_ms)) if not ext_used[i]]
    fn_idx = [i for i in range(len(gt_times_ms)) if not gt_used[i]]
    return matches, fp_idx, fn_idx


def match_peaks(ext_times_ms, gt_times_ms, tol_ms=PEAK_MATCH_TOL_MS):
    """TP/FP/FN counts (+ SE/PPV/F1 as fractions 0-1) from nearest-neighbour
    matching"""
    matches, fp_idx, fn_idx = _match_peaks_indices(ext_times_ms, gt_times_ms, tol_ms)
    tp, fp, fn = len(matches), len(fp_idx), len(fn_idx)
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
    return dict(tp=tp, fp=fp, fn=fn, sensitivity=sens, ppv=ppv, f1=f1, matches=matches)


# ============================================================
# 7. WAVEFORM-FIDELITY METRICS
# ============================================================
def rmse_fn(e, g):
    return np.sqrt(np.mean((e - g) ** 2))


def prd_fn(e, g):
    return np.sqrt(np.sum((e - g) ** 2) / np.sum(g ** 2)) * 100


def snr_fn(e, g):
    """SNR (dB) of `e` relative to reference `g`, defined as
    10*log10( power(g) / power(e-g) )."""
    err_power = np.sum((e - g) ** 2)
    if err_power <= 0:
        return np.inf
    return 10 * np.log10(np.sum(g ** 2) / err_power)


# ============================================================
# 8. CONFIDENCE INTERVALS
# ============================================================
def bootstrap_ci_metric(a, b, window_indices, metric_fn, n_boot=2000, seed=0, block_len=None):
    """Moving-block bootstrap over the WHOLE signal (not just the
    pre-extracted beat windows). Blocks of `block_len` contiguous samples
    are drawn with replacement from anywhere in the full recording and
    concatenated back to the original length, so each bootstrap replicate
    preserves the same quiet:active (baseline:QRS) sample ratio that the
    point estimate itself is computed over. `block_len` defaults to the
    median beat-window length, giving blocks on the same time-scale as a
    single QRS complex while still being drawn from anywhere in the
    recording (not restricted to be beat-centred)."""
    rng = np.random.default_rng(seed)
    n_total = len(a)
    if block_len is None:
        block_len = int(np.median([hi - lo for lo, hi in window_indices]))
    block_len = max(1, min(block_len, n_total))
    n_blocks = int(np.ceil(n_total / block_len))
    max_start = n_total - block_len
    vals = np.zeros(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        a_cat = np.concatenate([a[s:s + block_len] for s in starts])[:n_total]
        b_cat = np.concatenate([b[s:s + block_len] for s in starts])[:n_total]
        vals[k] = metric_fn(a_cat, b_cat)
    return vals.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def bootstrap_dsnr(before, after, gt, window_indices, n_boot=2000, seed=0, block_len=None):
    """Bootstrap distribution of dSNR = SNR_after - SNR_before, via a
    moving-block bootstrap over the WHOLE signal (see bootstrap_ci_metric)
    -> mean, 95% CI, and two-sided p-value vs H0: dSNR=0. before/after/gt
    are resampled with the SAME block start positions each iteration so
    the three stay aligned sample-for-sample within every replicate."""
    rng = np.random.default_rng(seed)
    n_total = len(before)
    if block_len is None:
        block_len = int(np.median([hi - lo for lo, hi in window_indices]))
    block_len = max(1, min(block_len, n_total))
    n_blocks = int(np.ceil(n_total / block_len))
    max_start = n_total - block_len
    vals = np.zeros(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        bef = np.concatenate([before[s:s + block_len] for s in starts])[:n_total]
        aft = np.concatenate([after[s:s + block_len] for s in starts])[:n_total]
        g = np.concatenate([gt[s:s + block_len] for s in starts])[:n_total]
        vals[k] = snr_fn(aft, g) - snr_fn(bef, g)
    p = 2 * min(np.mean(vals <= 0), np.mean(vals >= 0))
    p = min(p, 1.0)
    return vals.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5), p


def wilson_ci(x, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion x/n (Wilson, 1927).
    Standard choice for diagnostic-accuracy proportions (sensitivity, PPV)
    with small n, where the normal approximation is unreliable."""
    if n == 0:
        return (np.nan, np.nan)
    p = x / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def pearson_r_ci(r, n, z=1.96):
    """95% CI for Pearson r via the Fisher z-transformation (Fisher, 1921)."""
    if n <= 3 or abs(r) >= 1.0:
        return (np.nan, np.nan)
    zt = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    lo, hi = zt - z * se, zt + z * se
    return (float(np.tanh(lo)), float(np.tanh(hi)))


def bootstrap_detection_ci(ext_times_ms, gt_times_ms, tol_ms=PEAK_MATCH_TOL_MS,
                            n_boot=2000, seed=0):
    matches, fp_idx, fn_idx = _match_peaks_indices(ext_times_ms, gt_times_ms, tol_ms)
    n_gt = len(gt_times_ms)
    n_ext = len(ext_times_ms)
    gt_matched = np.zeros(n_gt, dtype=bool)
    ext_matched = np.zeros(n_ext, dtype=bool)
    for gi, ei in matches:
        gt_matched[gi] = True
        ext_matched[ei] = True

    rng = np.random.default_rng(seed)
    sens_v = np.full(n_boot, np.nan)
    ppv_v = np.full(n_boot, np.nan)
    f1_v = np.full(n_boot, np.nan)
    for k in range(n_boot):
        se_k = gt_matched[rng.integers(0, n_gt, size=n_gt)].mean() if n_gt > 0 else np.nan
        ppv_k = ext_matched[rng.integers(0, n_ext, size=n_ext)].mean() if n_ext > 0 else np.nan
        sens_v[k] = se_k
        ppv_v[k] = ppv_k
        f1_v[k] = 2 * se_k * ppv_k / (se_k + ppv_k) if (se_k + ppv_k) > 0 else np.nan

    def _ci(v):
        v = v[~np.isnan(v)]
        if len(v) == 0:
            return (np.nan, np.nan)
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    return dict(sensitivity_ci=_ci(sens_v), ppv_ci=_ci(ppv_v), f1_ci=_ci(f1_v))


# ============================================================
# 9. FETAL HEART RATE AGREEMENT -- BLAND-ALTMAN
# ============================================================
def paired_fhr_from_matches(ext_times_ms, gt_times_ms, matches):
    """Builds paired (extracted fHR, ground-truth fHR) arrays from
    consecutive ground-truth beats that are both matched to an extracted
    beat (i.e. two consecutive true positives)."""
    match_dict = {gi: ei for gi, ei in matches}
    gt_order = sorted(match_dict.keys())
    fhr_ext, fhr_gt = [], []
    for a, b in zip(gt_order[:-1], gt_order[1:]):
        if b - a != 1:
            continue   
        rr_gt = gt_times_ms[b] - gt_times_ms[a]
        rr_ext = ext_times_ms[match_dict[b]] - ext_times_ms[match_dict[a]]
        if rr_gt > 0 and rr_ext > 0:
            fhr_gt.append(60000.0 / rr_gt)
            fhr_ext.append(60000.0 / rr_ext)
    return np.array(fhr_ext), np.array(fhr_gt)


def bland_altman_stats(fhr_ext, fhr_gt):
    if len(fhr_ext) < 2:
        return dict(n_pairs=len(fhr_ext), bias=np.nan, sd=np.nan,
                    loa_lower=np.nan, loa_upper=np.nan,
                    avg=np.array([]), diff=np.array([]))
    diff = fhr_ext - fhr_gt
    avg = (fhr_ext + fhr_gt) / 2.0
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    return dict(n_pairs=len(diff), bias=bias, sd=sd,
                loa_lower=bias - 1.96 * sd, loa_upper=bias + 1.96 * sd,
                avg=avg, diff=diff)


# ============================================================
# 10. FULL STATISTICS TABLE FOR ONE SCENARIO
# ============================================================
def detect_consensus_fetal_peaks(gt_F, time_ms, fs, tol_ms=PEAK_MATCH_TOL_MS, min_leads=None):
    n_leads = gt_F.shape[1]
    if min_leads is None:
        min_leads = n_leads // 2 + 1   # majority, e.g. 2 of 3
    lead_names = ABDO_LEADS[:n_leads]

    all_times_labeled = []
    for i in range(n_leads):
        idx = fetal_qrs_detect(gt_F[:, i], fs)
        for t in time_ms[idx]:
            all_times_labeled.append((t, i))
    all_times_labeled.sort(key=lambda x: x[0])
    if len(all_times_labeled) == 0:
        return np.array([])

    clusters, current = [], [all_times_labeled[0]]
    for item in all_times_labeled[1:]:
        if item[0] - current[-1][0] <= tol_ms:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    clusters.append(current)

    consensus_times = []
    cluster_lead_times = []   
    for c in clusters:
        if len(c) >= min_leads:
            times = [t for t, _ in c]
            consensus_times.append(np.median(times))
            cluster_lead_times.append({lead_names[li]: t for t, li in c})
    consensus_times = np.array(consensus_times)


    # contribute to that consensus beat, for the agreement printout.
    agreement_matrix = np.full((len(cluster_lead_times), n_leads), np.nan)
    for k, cd in enumerate(cluster_lead_times):
        for j, lead in enumerate(lead_names):
            if lead in cd:
                agreement_matrix[k, j] = cd[lead]
    if len(cluster_lead_times) > 0:
        print_cross_lead_r_peak_agreement(agreement_matrix, lead_names,
                                           "Fetal R-Peaks (ground truth, F1/F2/F3)")

    return np.sort(consensus_times)


def compute_full_stats(P, W, fetal_mV, gt_F, tag, outdir):
    time_ms, fs, abd_std = P['time_ms'], P['fs'], P['abd_std']
    total_duration_ms = time_ms[-1] - time_ms[0]
    window_indices = W['window_indices']
    abd_mV_before = P['norm_abd'] * abd_std     # BEFORE maternal cancellation, in mV

    gt_times_consensus = detect_consensus_fetal_peaks(gt_F, time_ms, fs)
    print(f"\n  Consensus fetal beats in ground truth (F1+F2+F3, majority vote): "
          f"{len(gt_times_consensus)}")

    print(f"\n{'='*100}")
    print(f"HEART RATE DETECTION & AGREEMENT — {tag}")
    print(f"{'='*100}")
    hdr1 = (f"{'Lead':<5}{'TP':>5}{'FP':>5}{'FN':>5}{'SE(%)':>8}{'PPV(%)':>8}"
            f"{'F1(%)':>8}{'bias(bpm)':>11}{'LoA(bpm)':>20}{'n_pairs':>9}")
    print(hdr1)
    print('-' * len(hdr1))

    all_stats = {}
    ba_data = {}
    for i, lead in enumerate(ABDO_LEADS):
        ext = fetal_mV[:, i]
        gt = gt_F[:, i]

        ext_idx = fetal_qrs_detect(ext, fs)
        ext_times = time_ms[ext_idx]
        gt_times = gt_times_consensus   

        det = match_peaks(ext_times, gt_times)
        se_ci = wilson_ci(det['tp'], det['tp'] + det['fn'])
        ppv_ci = wilson_ci(det['tp'], det['tp'] + det['fp'])
        det_ci = bootstrap_detection_ci(ext_times, gt_times)

        fhr_ext, fhr_gt = paired_fhr_from_matches(ext_times, gt_times, det['matches'])
        ba = bland_altman_stats(fhr_ext, fhr_gt)
        ba_data[lead] = ba

        loa_str = f"[{ba['loa_lower']:.2f}, {ba['loa_upper']:.2f}]" if ba['n_pairs'] >= 2 else "n/a"
        print(f"{lead:<5}{det['tp']:>5}{det['fp']:>5}{det['fn']:>5}"
              f"{det['sensitivity']*100:>8.2f}{det['ppv']*100:>8.2f}{det['f1']*100:>8.2f}"
              f"{ba['bias']:>11.2f}{loa_str:>20}{ba['n_pairs']:>9}")

        all_stats[lead] = dict(
            tp=det['tp'], fp=det['fp'], fn=det['fn'],
            sensitivity_pct=det['sensitivity'] * 100,
            sensitivity_ci_pct=(se_ci[0] * 100, se_ci[1] * 100),
            ppv_pct=det['ppv'] * 100, ppv_ci_pct=(ppv_ci[0] * 100, ppv_ci[1] * 100),
            f1_pct=det['f1'] * 100,
            f1_ci_pct=(det_ci['f1_ci'][0] * 100, det_ci['f1_ci'][1] * 100),
            n_ext_peaks=len(ext_idx), n_gt_peaks_consensus=len(gt_times_consensus),
            ba_bias_bpm=ba['bias'], ba_sd_bpm=ba['sd'],
            ba_loa_lower_bpm=ba['loa_lower'], ba_loa_upper_bpm=ba['loa_upper'],
            ba_n_pairs=ba['n_pairs'],
        )

    print(f"\n  All 3 leads scored against the SAME {len(gt_times_consensus)} consensus fetal "
          f"beats (so TP+FN = {len(gt_times_consensus)} for every lead by construction); "
          f"remaining per-lead TP/FN differences reflect that lead's own extraction quality.")
    print(f"  TP/FP/FN via nearest-neighbour peak matching, tolerance = {PEAK_MATCH_TOL_MS:.0f} ms.")
    print(f"  SE/PPV 95% CI: Wilson score interval. F1 95% CI: moving-block bootstrap (2000 resamples, whole signal).")
    print(f"  Bland-Altman bias = mean(extracted fHR - reference fHR) over consecutive TP-TP beat pairs; "
          f"LoA = bias +/- 1.96*SD (Bland & Altman, 1986), following Barnova et al. (2021).")
    if any(ba_data[l]['n_pairs'] < 5 for l in ABDO_LEADS):
        print(f"  NOTE: this is a {total_duration_ms/1000:.0f} s recording with very few matched beat "
              f"pairs -- Bland-Altman bias/LoA and CIs should be read as indicative, not conclusive.")

    print(f"\n{'='*100}")
    print(f"MORPHOLOGY / SIGNAL FIDELITY — {tag}")
    print(f"{'='*100}")
    hdr2 = (f"{'Lead':<5}{'r':>8}{'r 95% CI':>18}{'r p-val':>10}{'RMSE(mV)':>11}"
            f"{'PRD(%)':>9}{'SNR_bef':>9}{'SNR_aft':>9}{'dSNR':>8}{'dSNR p':>8}")
    print(hdr2)
    print('-' * len(hdr2))

    for i, lead in enumerate(ABDO_LEADS):
        ext = fetal_mV[:, i]
        gt = gt_F[:, i]
        before = abd_mV_before[:, i]

        r, p_r = pearsonr(ext, gt)
        r_ci = pearson_r_ci(r, len(ext))

        rmse = rmse_fn(ext, gt)
        prd = prd_fn(ext, gt)
        _, rmse_lo, rmse_hi = bootstrap_ci_metric(ext, gt, window_indices, rmse_fn)
        _, prd_lo, prd_hi = bootstrap_ci_metric(ext, gt, window_indices, prd_fn)

        snr_before = snr_fn(before, gt)
        snr_after = snr_fn(ext, gt)
        dsnr = snr_after - snr_before
        _, snr_b_lo, snr_b_hi = bootstrap_ci_metric(before, gt, window_indices, snr_fn)
        _, snr_a_lo, snr_a_hi = bootstrap_ci_metric(ext, gt, window_indices, snr_fn)
        dsnr_mean, dsnr_lo, dsnr_hi, dsnr_p = bootstrap_dsnr(before, ext, gt, window_indices)

        r_ci_str = f"[{r_ci[0]:.3f}, {r_ci[1]:.3f}]"
        print(f"{lead:<5}{r:>8.3f}{r_ci_str:>18}{p_r:>10.2e}{rmse:>11.5f}"
              f"{prd:>9.2f}{snr_before:>9.2f}{snr_after:>9.2f}{dsnr:>8.2f}{dsnr_p:>8.3f}")

        all_stats[lead].update(dict(
            pearson_r=r, pearson_p=p_r, pearson_r_ci=r_ci,
            rmse=rmse, rmse_ci=(rmse_lo, rmse_hi),
            prd=prd, prd_ci=(prd_lo, prd_hi),
            snr_before=snr_before, snr_before_ci=(snr_b_lo, snr_b_hi),
            snr_after=snr_after, snr_after_ci=(snr_a_lo, snr_a_hi),
            dsnr=dsnr, dsnr_ci=(dsnr_lo, dsnr_hi), dsnr_p=dsnr_p,
        ))

    print(f"\n  RMSE/PRD/SNR/dSNR 95% CI: moving-block bootstrap (2000 resamples, whole signal).")
    print(f"  Pearson r 95% CI: Fisher z-transformation. dSNR p-value tests H0: no change in SNR.")

    ba_path = plot_bland_altman(ba_data, tag, outdir)
    print(f"\n  Saved: {ba_path}")

    return all_stats


# ============================================================
# 11. PLOTTING FIGURES
# ============================================================
TITLE_FS = 16
SUBTITLE_FS = 14
LABEL_FS = 13
TICK_FS = 11
SPINE_LW = 1.6
DATA_LW = 1.8

PC_COLOURS = ['blue', 'orange', 'green']


def _style_axes(ax, title=None, xlabel=None, ylabel=None,
                 title_fs=SUBTITLE_FS, label_fs=LABEL_FS, tick_fs=TICK_FS):
    if title is not None:
        ax.set_title(title, fontsize=title_fs, fontweight='bold')
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fs, fontweight='bold')
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fs, fontweight='bold')
    ax.tick_params(axis='both', labelsize=tick_fs)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
    ax.grid(False)


def plot_bland_altman(ba_data, tag, outdir):
    OVERLAP_TOL_BPM = 0.05   

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for i, lead in enumerate(ABDO_LEADS):
        ax = axes[i]
        ba = ba_data[lead]
        if ba['n_pairs'] < 2:
            ax.text(0.5, 0.5, 'Insufficient matched\nbeat pairs', ha='center', va='center',
                    transform=ax.transAxes, fontsize=TICK_FS, fontweight='bold')
            _style_axes(ax, title=lead)
            continue

        ax.scatter(ba['avg'], ba['diff'], color='steelblue', s=40, alpha=0.85,
                   edgecolor='black', linewidth=0.7, zorder=3)
        ax.axhline(ba['bias'], color='black', linewidth=DATA_LW, zorder=2)
        ax.axhline(ba['loa_upper'], color='red', linestyle='--', linewidth=DATA_LW * 0.85, zorder=2)
        ax.axhline(ba['loa_lower'], color='red', linestyle='--', linewidth=DATA_LW * 0.85, zorder=2)

        _style_axes(ax, title=lead,
                    xlabel='Average of Residual and Ground Truth (bpm)',
                    ylabel='Residual \u2212 Ground Truth (bpm)')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        upper_overlaps_bias = abs(ba['loa_upper'] - ba['bias']) < OVERLAP_TOL_BPM
        lower_overlaps_bias = abs(ba['bias'] - ba['loa_lower']) < OVERLAP_TOL_BPM
        lines_overlap = upper_overlaps_bias or lower_overlaps_bias

        if not lines_overlap:
            x0, x1 = ax.get_xlim()
            x_txt = x0 + 0.02 * (x1 - x0)
            ax.text(x_txt, ba['loa_upper'], f"+1.96 SD = {ba['loa_upper']:+.2f}", color='red',
                    fontsize=TICK_FS, fontweight='bold', va='bottom', zorder=4)
            ax.text(x_txt, ba['loa_lower'], f"-1.96 SD = {ba['loa_lower']:+.2f}", color='red',
                    fontsize=TICK_FS, fontweight='bold', va='top', zorder=4)
            ax.text(x_txt, ba['bias'], f"Bias = {ba['bias']:+.2f}", color='black',
                    fontsize=TICK_FS, fontweight='bold', va='bottom', zorder=4)

    fig.suptitle('Bland-Altman Plot', fontsize=TITLE_FS, fontweight='bold')
    plt.tight_layout()
    fpath = f"{outdir}/{tag}_bland_altman.png"
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


def plot_pan_tompkins(P, CR, W, tag, outdir):
    time_ms = P['time_ms']
    notch_signals, all_leads = P['notch_signals'], P['all_leads']
    leads_to_check = CR['leads_to_check']

    fig, axes = plt.subplots(len(leads_to_check), 1, figsize=(15, 10), sharex=True)
    for i, (ax, L) in enumerate(zip(axes, leads_to_check)):
        sig_clean = notch_signals[:, all_leads.index(L)]
        r_idx = CR['r_peaks_results'][L]
        ax.plot(time_ms, sig_clean, color='steelblue', linewidth=DATA_LW * 0.7)
        r_line, = ax.plot(time_ms[r_idx], sig_clean[r_idx], 'ro', markersize=8, label='R-peak')
        for lo, hi in W['window_indices']:
            ax.axvspan(time_ms[lo], time_ms[hi - 1], color='green', alpha=0.15)
        _style_axes(ax, title=L, ylabel='mV')
        if i == 0:
            ax.legend(handles=[r_line], loc='upper right', fontsize=LABEL_FS,
                      prop={'weight': 'bold'})
    axes[-1].set_xlabel('Time (ms)', fontsize=LABEL_FS, fontweight='bold')
    fig.suptitle('Maternal QRS Detection', fontsize=TITLE_FS, fontweight='bold')
    plt.tight_layout()
    fpath = f"{outdir}/{tag}_01_pan_tompkins.png"
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


def plot_scree(PC, tag, outdir, ref_leads):
    explained_ratio = PC['explained_ratio']
    n = len(explained_ratio)
    x_ticks = [str(k + 1) for k in range(n)]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(x_ticks, explained_ratio * 100, '-o', color='steelblue',
            linewidth=DATA_LW, markersize=9)
    for k, val in enumerate(explained_ratio * 100):
        ax.annotate(f'{val:.1f}%', xy=(k, val), xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=TICK_FS, fontweight='bold')
    ax.set_ylim(0, 110)
    _style_axes(ax, title='Principal Component Analysis: Scree Plot',
                xlabel='Principal Component', ylabel='Explained Variance (%)')
    plt.tight_layout()
    fpath = f"{outdir}/{tag}_02_scree.png"
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


def plot_cumulative_variance(PC, tag, outdir, ref_leads):
    cumulative_variance = PC['cumulative_variance']
    n = len(cumulative_variance)
    x_ticks = [str(k + 1) for k in range(n)]
    palette = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'gray']
    bar_colours = [palette[k % len(palette)] for k in range(n)]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    bars = ax.bar(x_ticks, cumulative_variance * 100, color=bar_colours,
                   edgecolor='black', linewidth=1.2, width=0.55)
    for bar, val in zip(bars, cumulative_variance * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=TICK_FS, fontweight='bold')
    ax.set_ylim(0, 112)
    _style_axes(ax, title='Principal Component Analysis: Cumulative Variance',
                xlabel='Principal Component', ylabel='Cumulative Variance (%)')
    plt.tight_layout()
    fpath = f"{outdir}/{tag}_03_cumulative_variance.png"
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


def plot_pca_3d(P, PC, tag, outdir):
    """Principal Component Analysis: 3D Signal Space"""
    norm_df, ref_leads = P['norm_df'], P['ref_leads']
    axes3 = ref_leads[:3]
    idx3 = [ref_leads.index(l) for l in axes3]
    eigenvectors = PC['eigenvectors']

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')

    T3 = norm_df[axes3].values
    data_range = np.percentile(np.abs(T3), 95)

    step = 2
    x_p, y_p, z_p = T3[::step, 0], T3[::step, 1], T3[::step, 2]
    time_colour = np.linspace(0, 1, len(x_p))
    sc = ax.scatter(x_p, y_p, z_p, c=time_colour, cmap='viridis', s=28, alpha=0.75, linewidths=0)

    scales = [1.5, 1.0, 0.6]
    for pc_idx in range(min(3, eigenvectors.shape[0])):
        direction = eigenvectors[pc_idx][idx3] * data_range * scales[pc_idx]
        lbl = f'Principal Component {pc_idx + 1}'
        ax.quiver(0, 0, 0, direction[0], direction[1], direction[2],
                  color=PC_COLOURS[pc_idx], linewidth=4.5, arrow_length_ratio=0.2, label=lbl)
        ax.quiver(0, 0, 0, -direction[0], -direction[1], -direction[2],
                  color=PC_COLOURS[pc_idx], linewidth=4.5, arrow_length_ratio=0.0, alpha=0.3)

    ax.scatter([0], [0], [0], color='black', s=130, marker='+', linewidths=3, zorder=10)
    ax.grid(False)

    
    ax.view_init(elev=20, azim=225)
    ax.set_xlabel(axes3[0], fontsize=LABEL_FS, fontweight='bold', labelpad=18)
    ax.set_ylabel(axes3[1], fontsize=LABEL_FS, fontweight='bold', labelpad=18)
    ax.set_zlabel(axes3[2], fontsize=LABEL_FS, fontweight='bold', labelpad=22)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.zaxis.set_tick_params(pad=10)
    ax.legend(fontsize=LABEL_FS - 1, loc='upper left', prop={'weight': 'bold'})

    fig.suptitle('Principal Component Analysis: 3D Signal Space', fontsize=TITLE_FS, fontweight='bold')
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.12)
    cbar.set_label('Time progression', fontsize=LABEL_FS, fontweight='bold')

    
    fig.subplots_adjust(left=0.08, right=0.90, top=0.90, bottom=0.05)
    fig.canvas.draw()
    fpath = f"{outdir}/{tag}_04_pca3d.png"
    plt.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath


def plot_lead_loadings(PC, tag, outdir, ref_leads):
    eigenvectors = PC['eigenvectors']
    n_pc_show = min(3, eigenvectors.shape[0])
    squared_loadings = eigenvectors ** 2
    palette = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'gray']
    bar_colours = [palette[k % len(palette)] for k in range(len(ref_leads))]

    fig, axes = plt.subplots(1, n_pc_show, figsize=(5.5 * n_pc_show, 5.5))
    if n_pc_show == 1:
        axes = [axes]
    for k in range(n_pc_show):
        ax = axes[k]
        sq_pct = squared_loadings[k] * 100
        bars = ax.bar(ref_leads, sq_pct, color=bar_colours, edgecolor='black', linewidth=1.0, width=0.55)
        for bar, val in zip(bars, sq_pct):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=TICK_FS, fontweight='bold')
        ax.set_ylim(0, 115)
        _style_axes(ax, title=f'Principal Component {k + 1}', ylabel='Squared Loading (%)')
    fig.suptitle('Lead Loadings per Principal Component', fontsize=TITLE_FS, fontweight='bold')
    plt.tight_layout()
    fpath = f"{outdir}/{tag}_05_lead_loadings.png"
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


def plot_maternal_template_mV(P, maternal_mV, abd_mV, tag, outdir, method):
    """Constructed Maternal ECG Template vs
    Original Recording."""
    time_ms = P['time_ms']
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for i, lead in enumerate(ABDO_LEADS):
        orig_line, = axes[i].plot(time_ms, abd_mV[:, i], color='black', linewidth=DATA_LW * 0.7,
                                   alpha=0.75, label='Original')
        template_line, = axes[i].plot(time_ms, maternal_mV[:, i], color='red', linewidth=DATA_LW,
                                       label='Constructed Maternal Template')
        _style_axes(axes[i], title=lead, ylabel='Voltage (mV)')
        if i == 0:
            axes[i].legend(handles=[orig_line, template_line], loc='upper right',
                           fontsize=LABEL_FS - 1, prop={'weight': 'bold'})
    axes[-1].set_xlabel('Time (ms)', fontsize=LABEL_FS, fontweight='bold')
    fig.suptitle('Constructed Maternal ECG Template vs Original Recording',
                 fontsize=TITLE_FS, fontweight='bold')
    plt.tight_layout()
    fpath = f"{outdir}/{tag}_06_maternal_template.png"
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


def plot_fetal_residual_mV(P, fetal_mV, tag, outdir, method):
    """Residual Foetal ECG."""
    time_ms = P['time_ms']
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for i, lead in enumerate(ABDO_LEADS):
        axes[i].plot(time_ms, fetal_mV[:, i], color='steelblue', linewidth=DATA_LW)
        _style_axes(axes[i], title=lead, ylabel='Voltage (mV)')
    axes[-1].set_xlabel('Time (ms)', fontsize=LABEL_FS, fontweight='bold')
    fig.suptitle('Residual Foetal ECG', fontsize=TITLE_FS, fontweight='bold')
    plt.tight_layout()
    fpath = f"{outdir}/{tag}_07_residual_fetal_ecg.png"
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


def plot_extracted_vs_groundtruth(P, fetal_mV, gt_F, tag, outdir, method):
    time_ms = P['time_ms']
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for i, lead in enumerate(ABDO_LEADS):
        res_line, = axes[i].plot(time_ms, fetal_mV[:, i], color='red', linewidth=DATA_LW,
                                  label='Residual Foetal ECG')
        gt_line, = axes[i].plot(time_ms, gt_F[:, i], color='black', linewidth=DATA_LW * 0.8,
                                 alpha=0.85, label='Ground Truth Foetal ECG')
        _style_axes(axes[i], title=lead, ylabel='Voltage (mV)')
        if i == 0:
            axes[i].legend(handles=[res_line, gt_line], loc='upper right',
                           fontsize=LABEL_FS - 1, prop={'weight': 'bold'})
    axes[-1].set_xlabel('Time (ms)', fontsize=LABEL_FS, fontweight='bold')
    fig.suptitle('Residual Foetal ECG vs Ground Truth', fontsize=TITLE_FS, fontweight='bold')
    plt.tight_layout()
    fpath = f"{outdir}/{tag}_08_residual_vs_groundtruth.png"
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


def plot_residual_over_original(P, fetal_mV, abd_mV, tag, outdir, method):
    time_ms = P['time_ms']
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for i, lead in enumerate(ABDO_LEADS):
        orig_line, = axes[i].plot(time_ms, abd_mV[:, i], color='black', linewidth=DATA_LW * 0.7,
                                   label='Original')
        res_line, = axes[i].plot(time_ms, fetal_mV[:, i], color='red', linewidth=DATA_LW,
                                  label='Residual Foetal ECG')
        _style_axes(axes[i], title=lead, ylabel='Voltage (mV)')
        if i == 0:
            axes[i].legend(handles=[orig_line, res_line], loc='upper right',
                           fontsize=LABEL_FS - 1, prop={'weight': 'bold'})
    axes[-1].set_xlabel('Time (ms)', fontsize=LABEL_FS, fontweight='bold')
    fig.suptitle('Residual Foetal ECG vs Original Recording', fontsize=TITLE_FS, fontweight='bold')
    plt.tight_layout()
    fpath = f"{outdir}/{tag}_09_residual_vs_original.png"
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


def make_all_plots(P, CR, W, PC, REG, fetal_mV, maternal_mV, abd_mV, gt_F, tag, outdir):
    paths = []
    paths.append(plot_pan_tompkins(P, CR, W, tag, outdir))
    paths.append(plot_scree(PC, tag, outdir, P['ref_leads']))
    paths.append(plot_cumulative_variance(PC, tag, outdir, P['ref_leads']))
    paths.append(plot_pca_3d(P, PC, tag, outdir))
    paths.append(plot_lead_loadings(PC, tag, outdir, P['ref_leads']))
    paths.append(plot_maternal_template_mV(P, maternal_mV, abd_mV, tag, outdir, REG['method']))
    paths.append(plot_fetal_residual_mV(P, fetal_mV, tag, outdir, REG['method']))
    paths.append(plot_extracted_vs_groundtruth(P, fetal_mV, gt_F, tag, outdir, REG['method']))
    paths.append(plot_residual_over_original(P, fetal_mV, abd_mV, tag, outdir, REG['method']))
    return paths


# ============================================================
# 11. GROUND TRUTH LOADER
# ============================================================
def load_ground_truth(filepath):
    gt = np.loadtxt(filepath)
    return gt[:, [1, 2, 3]]   # F1, F2, F3 in mV

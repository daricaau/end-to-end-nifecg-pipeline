"""
O5 -- Fetal ECG Extraction and CTG-Style Fetal Heart Rate Assessment
====================================================================
This module implements the signal-processing and clinical-assessment
pipeline used to convert a mixed (maternal + fetal) abdominal ECG
recording into a set of CTG-style fetal monitoring outputs.

Maternal ECG cancellation uses a five-channel reference lead
configuration (V2, V5, V8, RA, LA): a time-matched R-peak consensus
locates maternal beats across the reference leads, principal component
analysis is applied to the aligned QRS windows to build a maternal
ECG template, and a cold-start recursive least squares (RLS) adaptive
filter (forgetting factor lambda = 0.9995) regresses this template out
of each abdominal lead, leaving the fetal ECG residual.

An auto-detecting loader accepts either of two input layouts: the
column-ordered .xlsx recordings produced by the abdominal ECG hardware
(e.g. Pos1_lying.xlsx), or whitespace-delimited .txt recordings
(e.g. phie_fetal_only.txt / phie_with_maternal_interference.txt).

For any input mixed-ECG file this script produces:

  1) Fetal ECG             -- extracted residual fetal ECG (F1/F2/F3),
                               saved as .png + per-sample values in the
                               JSON report.
  2) FHR concordance       -- per-lead (F1/F2/F3) mean fetal heart
                               rate + the standard deviation across
                               the three lead-level estimates (SD_FHR).
  3) Signal confidence     -- Confidence(%) = 100 * (1 - SD_FHR / 10)
  4) Accelerations         -- yes/no: HR rise > 10 bpm within the
                               trailing 15 s, evaluated continuously.
  5) Heart-rate variability-- beat-to-beat FHR variation, in bpm.
  6) Decelerations         -- yes/no: HR fall > 10 bpm within the
                               trailing 15 s, evaluated continuously.

...and then grades baseline FHR, baseline variability and
decelerations as Reassuring / Non-reassuring / Abnormal using
NICE-style intrapartum banding (see the classify_* functions below
for the exact thresholds).

Known limitations:
  - NICE's variability/deceleration bands are duration-based (15-50+
    minutes of continuous monitoring). Short recordings will correctly
    report "Indeterminate (insufficient duration)" for those two
    categories rather than a false Reassuring/Non-reassuring/Abnormal
    call. A longer recording is needed to obtain a duration-qualified
    grade.
  - There is no uterine-contraction / tocograph channel in these
    recordings, so decelerations cannot be classed as "variable" vs
    "late", and "% of contractions" cannot be computed. Every
    deceleration is conservatively treated as if it were a variable
    deceleration (the more common, less specific category) and
    graded on duration plus the two documented "concerning
    characteristics" (>60 s long, or reduced variability within the
    dip) instead.
  - Sinusoidal-pattern detection is not implemented (it requires
    dedicated spectral analysis) and is always reported as "not
    assessed".
  - This is a research/engineering pipeline evaluated on simulated and
    short real recordings, not a validated clinical device.

Usage:
    python O5.py [path/to/mixed_ecg_file] [--outdir DIR]

    path/to/mixed_ecg_file : .xlsx (Pos1_lying-style, no header) or
                              .txt (phie_*-style, whitespace-delimited,
                              no header). Defaults to the bundled
                              Pos1_lying.xlsx demo file (expected in the same
                              folder as this script).
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fecg_common as fc

ABDO_LEADS = fc.ABDO_LEADS                       # ['F1', 'F2', 'F3']
REF_LEADS  = ['V2', 'V5', 'V8', 'RA', 'LA']       # 5-channel reference set

COLD_LAMBDA = 0.9995
COLD_DELTA  = fc.RLS_DELTA                        # 10.0, unchanged default

ACCEL_DECEL_WINDOW_MS = 15000.0    # "past 15 seconds"
ACCEL_DECEL_THRESH_BPM = 10.0      # "> 10 bpm"
VARIABILITY_WINDOW_MS = 60000.0    # rolling 1-minute bandwidth window


# ============================================================
# 0. Axis styling helper (no gridlines, bold labels)
# ============================================================
def _style_axes_no_grid(ax, title=None, xlabel=None, ylabel=None,
                         title_fs=fc.SUBTITLE_FS, label_fs=fc.LABEL_FS, tick_fs=fc.TICK_FS):
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
        spine.set_linewidth(fc.SPINE_LW)
    ax.grid(False)


fc._style_axes = _style_axes_no_grid


# ============================================================
# 1. Auto-detecting file loader
#    xlsx  -> Pos1_lying-style: no header, cols = time,RA,LA,avF,V2,V5,V8,F1,F2,F3
#    other -> phie_*-style: whitespace-delimited, no header, cols =
#             time,F1,F2,F3,V2,V5,V8,RA,LA,GND,LL  (fc.load_and_preprocess)
# ============================================================
def load_and_preprocess_xlsx(filepath, abdo_leads, ref_leads):
    col_names = ['time_ms', 'RA', 'LA', 'avF', 'V2', 'V5', 'V8', 'F1', 'F2', 'F3']
    df = pd.read_excel(filepath, header=None)
    df.columns = col_names
    time_ms = df['time_ms'].values.astype(float)
    dt_ms = time_ms[1] - time_ms[0]
    fs = 1000.0 / dt_ms
    nyquist = fs / 2.0
    all_leads = abdo_leads + ref_leads
    raw = df[all_leads].values.astype(float)
    N_samples, N_leads = raw.shape
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
    norm_abd = norm_df[abdo_leads].values
    norm_ref = norm_df[ref_leads].values
    abd_std = scaler.scale_[[all_leads.index(l) for l in abdo_leads]]
    return dict(time_ms=time_ms, dt_ms=dt_ms, fs=fs, nyquist=nyquist,
                all_leads=all_leads, ref_leads=ref_leads, notch_signals=notch,
                norm_df=norm_df, norm_abd=norm_abd, norm_ref=norm_ref,
                abd_std=abd_std, scaler=scaler, N_samples=N_samples)


def load_mixed_ecg(filepath):
    """Auto-detects the input file format and loads + filters + normalises
    it with the appropriate loader. Returns (P, format_label)."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.xlsx', '.xls'):
        P = load_and_preprocess_xlsx(filepath, ABDO_LEADS, REF_LEADS)
        fmt = 'xlsx (Pos1_lying-style: time,RA,LA,avF,V2,V5,V8,F1,F2,F3)'
    else:
        P = fc.load_and_preprocess(filepath, REF_LEADS)
        fmt = 'whitespace-delimited txt (phie_*-style: time,F1,F2,F3,V2,V5,V8,RA,LA,GND,LL)'
    return P, fmt


# ============================================================
# 2. Time-matched maternal R-peak consensus
# ============================================================
def detect_consensus_peaks_time_matched(P, anchor_leads=('V5', 'V8'), check_lead='V2',
                                         tol_ms=150.0, display_order=('V2', 'V5', 'V8')):
    time_ms, notch_signals, all_leads, fs = P['time_ms'], P['notch_signals'], P['all_leads'], P['fs']
    r_peaks_results = {}
    for L in set(list(anchor_leads) + [check_lead]):
        sig_clean = notch_signals[:, all_leads.index(L)]
        r_peaks_results[L] = fc.pan_tompkins_r_peaks(sig_clean, fs)
    n_anchor = min(len(r_peaks_results[L]) for L in anchor_leads)
    anchor_stack = np.column_stack([time_ms[r_peaks_results[L][:n_anchor]] for L in anchor_leads])
    anchor_consensus = np.median(anchor_stack, axis=1)
    check_times = time_ms[r_peaks_results[check_lead]]
    matched_check = np.full(n_anchor, np.nan)
    used = np.zeros(len(check_times), dtype=bool)
    for k, t in enumerate(anchor_consensus):
        diffs = np.abs(check_times - t)
        diffs[used] = np.inf
        if len(diffs) and diffs.min() <= tol_ms:
            j = int(np.argmin(diffs))
            matched_check[k] = check_times[j]
            used[j] = True
    col_of = {**{L: anchor_stack[:, i] for i, L in enumerate(anchor_leads)}, check_lead: matched_check}
    beat_times_ms = np.column_stack([col_of[L] for L in display_order])
    consensus_r_peak_times = np.nanmedian(beat_times_ms, axis=1)
    return dict(r_peaks_results=r_peaks_results, beat_times_ms=beat_times_ms,
                consensus_r_peak_times=consensus_r_peak_times,
                beat_std=np.nanstd(beat_times_ms, axis=1), leads_to_check=list(display_order))


# ============================================================
# 3. Cold-start recursive least squares (RLS) maternal template fit
# ============================================================
def _rls_core_cold(X, y, lam, delta):
    N, p = X.shape
    w = np.zeros(p)                       # COLD START -- no OLS warm start
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


def fit_maternal_template_rls_cold(P, PC, lam=COLD_LAMBDA, delta=COLD_DELTA):
    X = np.column_stack([PC['M1'], PC['M2']])
    norm_abd = P['norm_abd']
    results = {}
    for i, lead in enumerate(ABDO_LEADS):
        y = norm_abd[:, i]
        y_hat, residual, weights = _rls_core_cold(X, y, lam, delta)
        results[lead] = dict(y_hat=y_hat, residual=residual, weights=weights, w_ols=None)
    return dict(method='RLS', X=X, results=results, lam=lam, delta=delta)


# ============================================================
# 4. Fetal R-peak detection per lead
# ============================================================
def detect_fetal_rpeaks_per_lead(fetal_mV, time_ms, fs):
    fetal_results = {}
    for i, lead in enumerate(ABDO_LEADS):
        residual = fetal_mV[:, i]
        r_idx = fc.fetal_qrs_detect(residual, fs)
        r_times_ms = time_ms[r_idx]
        n_peaks = int(len(r_idx))
        if n_peaks >= 2:
            rr_ms = np.diff(r_times_ms)
            mean_hr = float(np.mean(60000.0 / rr_ms))
            median_hr = float(np.median(60000.0 / rr_ms))
        else:
            mean_hr = median_hr = float('nan')
        fetal_results[lead] = dict(n_r_peaks_detected=n_peaks, mean_hr_bpm=mean_hr,
                                    median_hr_bpm=median_hr, r_peak_times_ms=r_times_ms.tolist())
    return fetal_results


# ============================================================
# 5. Cross-lead FHR concordance and signal confidence
# ============================================================
def compute_concordance_confidence(fetal_results, leads=ABDO_LEADS):
    """SD_FHR = standard deviation (bpm) across the three leads' mean
    foetal heart-rate estimates. Confidence(%) = 100*(1 - SD_FHR/10),
    clipped to [0, 100] for display (the raw, unclipped value is kept
    too, since a very discordant reading is itself informative)."""
    lead_means = {l: fetal_results[l]['mean_hr_bpm'] for l in leads}
    valid = [v for v in lead_means.values() if not np.isnan(v)]
    if len(valid) < 2:
        return dict(lead_means_bpm=lead_means, sd_fhr_bpm=float('nan'),
                     confidence_pct_raw=float('nan'), confidence_pct=float('nan'),
                     n_leads_with_valid_hr=len(valid))
    sd_fhr = float(np.std(valid, ddof=0))
    conf_raw = 100.0 * (1.0 - sd_fhr / 10.0)
    conf_clipped = float(np.clip(conf_raw, 0.0, 100.0))
    return dict(lead_means_bpm=lead_means, sd_fhr_bpm=sd_fhr,
                confidence_pct_raw=float(conf_raw), confidence_pct=conf_clipped,
                n_leads_with_valid_hr=len(valid))


# ============================================================
# 6. Time-matched fetal beat consensus (F1/F2/F3 -> one beat train)
# ============================================================
def consensus_fetal_beats(fetal_results, leads=ABDO_LEADS, tol_ms=50.0):
    """Time-matches independently-detected F1/F2/F3 foetal R-peaks into
    one consensus beat train, robust to any single lead missing or
    falsely detecting a beat. Anchor = the lead with the MEDIAN peak
    count (avoids anchoring on whichever lead over- or under-detects
    the most). tol_ms=50 matches PEAK_MATCH_TOL_MS, the fetal-QRS
    matching convention used for fetal QRS detection elsewhere in
    this pipeline."""
    times = {l: np.asarray(fetal_results[l]['r_peak_times_ms'], dtype=float) for l in leads}
    counts = {l: len(times[l]) for l in leads}
    if all(c == 0 for c in counts.values()):
        return dict(beat_times_ms=np.array([]), n_leads_agree=np.array([]), anchor_lead=None)
    anchor = sorted(leads, key=lambda l: counts[l])[1]   # median-count lead
    others = [l for l in leads if l != anchor]
    used = {l: np.zeros(len(times[l]), dtype=bool) for l in others}
    consensus_times, n_agree = [], []
    for t in times[anchor]:
        matched = [t]
        for l in others:
            if len(times[l]) == 0:
                continue
            diffs = np.abs(times[l] - t)
            diffs = np.where(used[l], np.inf, diffs)
            j = int(np.argmin(diffs))
            if diffs[j] <= tol_ms:
                matched.append(times[l][j])
                used[l][j] = True
        consensus_times.append(float(np.median(matched)))
        n_agree.append(len(matched))
    order = np.argsort(consensus_times)
    return dict(beat_times_ms=np.asarray(consensus_times)[order],
                n_leads_agree=np.asarray(n_agree)[order], anchor_lead=anchor)


def fhr_trace_from_beats(beat_times_ms):
    """Instantaneous beat-to-beat HR (bpm) from a consensus beat train.
    Each HR value is assigned to the time of the LATER beat in its RR
    pair (standard convention)."""
    beat_times_ms = np.asarray(beat_times_ms, dtype=float)
    if len(beat_times_ms) < 2:
        return np.array([]), np.array([])
    rr_ms = np.diff(beat_times_ms)
    hr_bpm = 60000.0 / rr_ms
    t_hr_ms = beat_times_ms[1:]
    return t_hr_ms, hr_bpm


# ============================================================
# 7. Heart rate variability (beat-to-beat)
# ============================================================
def compute_hrv(hr_bpm):
    """HRV(bpm) = |change in FHR from one beat to the next|."""
    hr_bpm = np.asarray(hr_bpm, dtype=float)
    if len(hr_bpm) < 2:
        return dict(beat_to_beat_bpm=[], mean_hrv_bpm=float('nan'), median_hrv_bpm=float('nan'))
    btb = np.abs(np.diff(hr_bpm))
    return dict(beat_to_beat_bpm=btb.tolist(), mean_hrv_bpm=float(np.mean(btb)),
                median_hrv_bpm=float(np.median(btb)))


# ============================================================
# 8. Accelerations and decelerations
#    change in FHR vs. ~15 s ago, evaluated at every beat once >=15 s
#    of history is available.
# ============================================================
def detect_accel_decel(t_hr_ms, hr_bpm, window_ms=ACCEL_DECEL_WINDOW_MS, thresh_bpm=ACCEL_DECEL_THRESH_BPM):
    t_hr_ms = np.asarray(t_hr_ms, dtype=float)
    hr_bpm = np.asarray(hr_bpm, dtype=float)
    n = len(hr_bpm)
    delta_bpm = np.full(n, np.nan)
    accel_flag = np.zeros(n, dtype=bool)
    decel_flag = np.zeros(n, dtype=bool)
    if n == 0:
        return dict(delta_bpm=delta_bpm, accel_flag=accel_flag, decel_flag=decel_flag)
    for i in range(n):
        target = t_hr_ms[i] - window_ms
        if target < t_hr_ms[0]:
            continue   # not yet 15 s of history
        j = int(np.searchsorted(t_hr_ms, target))
        j = min(j, i)
        if j > 0 and abs(t_hr_ms[j - 1] - target) < abs(t_hr_ms[j] - target):
            j -= 1
        d = hr_bpm[i] - hr_bpm[j]
        delta_bpm[i] = d
        if d > thresh_bpm:
            accel_flag[i] = True
        elif d < -thresh_bpm:
            decel_flag[i] = True
    return dict(delta_bpm=delta_bpm, accel_flag=accel_flag, decel_flag=decel_flag)


def sustained_deviation_episodes(t_hr_ms, hr_bpm, baseline_bpm, thresh_bpm=ACCEL_DECEL_THRESH_BPM):
    """Episodes where FHR stays more than `thresh_bpm` away from the
    baseline (up = acceleration-type, down = deceleration-type).

    This is different from detect_accel_decel()'s moment-of-change flag:
    that one flags the ~15 s rate of change used for the yes/no
    accelerations/decelerations read-out. This function instead measures how
    long FHR actually stays displaced from baseline once it gets there
    -- which is what NICE's duration-based bands (e.g. "single prolonged
    deceleration lasting 3 minutes or more") are actually defined on. A
    slow drop that settles onto a new, lower plateau only trips the
    15-s rate flag once, at the moment of the drop, but would correctly
    show up here as a long sustained-deviation episode."""
    hr_bpm = np.asarray(hr_bpm, dtype=float)
    if len(hr_bpm) == 0 or np.isnan(baseline_bpm):
        return [], []
    up = hr_bpm > (baseline_bpm + thresh_bpm)
    down = hr_bpm < (baseline_bpm - thresh_bpm)
    return find_episodes(up, t_hr_ms), find_episodes(down, t_hr_ms)


def find_episodes(flag, t_hr_ms):
    """Contiguous True runs in `flag` -> list of (start_ms, end_ms, duration_sec)."""
    flag = np.asarray(flag, dtype=bool)
    t_hr_ms = np.asarray(t_hr_ms, dtype=float)
    episodes = []
    in_run = False
    start_idx = None
    for i, f in enumerate(flag):
        if f and not in_run:
            in_run, start_idx = True, i
        elif not f and in_run:
            in_run = False
            episodes.append((float(t_hr_ms[start_idx]), float(t_hr_ms[i - 1]),
                              float((t_hr_ms[i - 1] - t_hr_ms[start_idx]) / 1000.0)))
    if in_run:
        episodes.append((float(t_hr_ms[start_idx]), float(t_hr_ms[-1]),
                          float((t_hr_ms[-1] - t_hr_ms[start_idx]) / 1000.0)))
    return episodes


# ============================================================
# 9. Baseline FHR and rolling variability bandwidth
# ============================================================
def compute_baseline_fhr(hr_bpm, accel_flag, decel_flag):
    """Median FHR, excluding beats flagged as accelerating/decelerating
    (NICE convention: baseline excludes periodic/episodic changes),
    rounded to the nearest 5 bpm. Falls back to the unfiltered median
    if filtering leaves too little data."""
    hr_bpm = np.asarray(hr_bpm, dtype=float)
    if len(hr_bpm) == 0:
        return float('nan'), 0
    stable_mask = ~(np.asarray(accel_flag, dtype=bool) | np.asarray(decel_flag, dtype=bool))
    stable = hr_bpm[stable_mask] if stable_mask.sum() >= max(3, int(0.2 * len(hr_bpm))) else hr_bpm
    baseline = float(5 * np.round(np.median(stable) / 5.0))
    return baseline, int(stable_mask.sum())


def rolling_variability_bandwidth(t_hr_ms, hr_bpm, window_ms=VARIABILITY_WINDOW_MS):
    """Rolling bandwidth (max-min FHR) in a trailing `window_ms` window,
    approximating NICE's baseline-variability amplitude measure. Only
    evaluated once at least half a window of history is available."""
    t_hr_ms = np.asarray(t_hr_ms, dtype=float)
    hr_bpm = np.asarray(hr_bpm, dtype=float)
    n = len(hr_bpm)
    bandwidth = np.full(n, np.nan)
    for i in range(n):
        lo = t_hr_ms[i] - window_ms
        idx = np.where((t_hr_ms >= lo) & (t_hr_ms <= t_hr_ms[i]))[0]
        if len(idx) >= 2 and (t_hr_ms[idx[-1]] - t_hr_ms[idx[0]]) >= 0.5 * window_ms:
            bandwidth[i] = hr_bpm[idx].max() - hr_bpm[idx].min()
    return bandwidth


def _masked_duration_sec(mask, t_hr_ms):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0.0
    total = 0.0
    run_start = t_hr_ms[idx[0]]
    prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
            continue
        total += (t_hr_ms[prev] - run_start) / 1000.0
        run_start = t_hr_ms[k]
        prev = k
    total += (t_hr_ms[prev] - run_start) / 1000.0
    return total


# ============================================================
# 10. NICE-style classification
# ============================================================
def classify_baseline_hr(baseline_bpm):
    if np.isnan(baseline_bpm):
        return "Indeterminate", "Baseline FHR could not be determined (too few consensus beats)."
    if 110 <= baseline_bpm <= 160:
        return "Reassuring", f"Baseline {baseline_bpm:.0f} bpm is within 110-160 bpm."
    if (100 <= baseline_bpm < 110) or (160 < baseline_bpm <= 180):
        return "Non-reassuring", f"Baseline {baseline_bpm:.0f} bpm falls in the 100-109 or 161-180 bpm range."
    return "Abnormal", f"Baseline {baseline_bpm:.0f} bpm is below 100 or above 180 bpm."


def classify_variability(bandwidth, t_hr_ms, recording_duration_sec):
    valid = ~np.isnan(bandwidth)
    if valid.sum() == 0:
        return ("Indeterminate",
                "Not enough consensus beats / recording duration to compute a rolling "
                "variability bandwidth (needs >= ~30 s of continuous beats).")
    reduced_mask = valid & (bandwidth < 5)
    increased_mask = valid & (bandwidth > 25)
    reduced_min = _masked_duration_sec(reduced_mask, t_hr_ms) / 60.0
    increased_min = _masked_duration_sec(increased_mask, t_hr_ms) / 60.0
    rec_min = recording_duration_sec / 60.0
    note = (f"Reduced variability (<5 bpm) for {reduced_min:.2f} min; "
            f"increased variability (>25 bpm) for {increased_min:.2f} min; "
            f"recording length {rec_min:.2f} min.")

    if reduced_min > 50:
        return "Abnormal", note + " Reduced variability sustained >50 min -> abnormal."
    if increased_min > 25:
        return "Abnormal", note + " Increased variability sustained >25 min -> abnormal."
    if 30 <= reduced_min <= 50:
        return "Non-reassuring", note + " Reduced variability sustained 30-50 min -> non-reassuring."
    if 15 <= increased_min <= 25:
        return "Non-reassuring", note + " Increased variability sustained 15-25 min -> non-reassuring."
    if reduced_min > 0 or increased_min > 0:
        if rec_min < 15:
            return ("Indeterminate",
                    note + f" Recording is only {rec_min:.2f} min -- too short to confirm "
                    "NICE's duration-based non-reassuring/abnormal thresholds (15-50+ min).")
        return "Reassuring", note + " Out-of-band variability did not persist long enough to meet non-reassuring/abnormal duration thresholds."
    return "Reassuring", note + " Variability remained within 5-25 bpm throughout."


def classify_decelerations(decel_episodes, bandwidth, t_hr_ms, recording_duration_sec):
    """decel_episodes: list of (start_ms, end_ms, duration_sec).
    No contraction/tocograph channel is available, so every deceleration
    is conservatively treated as a variable deceleration (rather than
    late), and 'concerning characteristics' are limited to the two that
    are directly computable from the FHR trace alone: duration > 60 s,
    and reduced (<5 bpm) variability within the dip."""
    if len(decel_episodes) == 0:
        return "Reassuring", "No decelerations (FHR fall > 10 bpm within a trailing 15 s window) detected."

    durations = [e[2] for e in decel_episodes]
    max_dur = max(durations)
    if max_dur >= 180.0:
        return ("Abnormal",
                f"A single deceleration episode lasted {max_dur:.0f} s (>= 3 min): meets the "
                "'acute bradycardia / single prolonged deceleration' abnormal criterion.")

    concerning = []
    for (s_ms, e_ms, dur_s) in decel_episodes:
        idx = np.where((t_hr_ms >= s_ms) & (t_hr_ms <= e_ms))[0]
        bw_in_decel = np.nanmin(bandwidth[idx]) if len(idx) and np.any(~np.isnan(bandwidth[idx])) else np.nan
        is_concerning = (dur_s > 60.0) or (not np.isnan(bw_in_decel) and bw_in_decel < 5.0)
        concerning.append(is_concerning)
    n_concerning = int(sum(concerning))

    total_decel_min = sum(durations) / 60.0
    rec_min = recording_duration_sec / 60.0
    note = (f"{len(decel_episodes)} deceleration episode(s) totalling {total_decel_min:.2f} min "
            f"({n_concerning} with a concerning characteristic: >60 s duration and/or reduced "
            f"variability within the dip), out of a {rec_min:.2f} min recording. No contraction/"
            f"tocograph channel is available, so 'variable vs late' and '% of contractions' "
            f"could not be assessed directly.")

    if rec_min < 30:
        return ("Indeterminate",
                note + f" Recording is only {rec_min:.2f} min -- too short to confirm NICE's "
                "30-50 min duration-based non-reassuring/abnormal thresholds for decelerations.")
    if n_concerning > 0:
        frac = total_decel_min / rec_min if rec_min > 0 else 0.0
        if frac > 0.5:
            return "Abnormal", note + " Concerning decelerations occupy >50% of a recording >=30 min -> abnormal."
        return "Non-reassuring", note + " Concerning decelerations present in a recording long enough to assess -> non-reassuring."
    return "Reassuring", note + " None of the decelerations met the concerning-characteristics criteria."


_SEVERITY = {"Reassuring": 1, "Indeterminate": 1, "Non-reassuring": 2, "Abnormal": 3}


def overall_classification(*labels):
    if "Abnormal" in labels:
        return "Abnormal"
    if "Non-reassuring" in labels:
        return "Non-reassuring"
    if "Indeterminate" in labels:
        return "Indeterminate (insufficient duration for a full grading)"
    return "Reassuring"


# ============================================================
# 11. Plotting functions
# ============================================================
def plot_ctg_trend(t_hr_ms, hr_bpm, baseline_bpm, accel_flag, decel_flag, tag, outdir):
    fig, ax = plt.subplots(figsize=(14, 5))
    t_sec = t_hr_ms / 1000.0
    ax.plot(t_sec, hr_bpm, color='steelblue', linewidth=fc.DATA_LW,
            label='Consensus foetal HR (F1/F2/F3)')
    if not np.isnan(baseline_bpm):
        ax.axhline(baseline_bpm, color='black', linestyle='--', linewidth=fc.DATA_LW * 0.8,
                    label=f'Baseline ({baseline_bpm:.0f} bpm)')

    def shade(flag, color, label):
        idx = np.where(flag)[0]
        if len(idx) == 0:
            return
        runs, start, prev = [], idx[0], idx[0]
        for k in idx[1:]:
            if k == prev + 1:
                prev = k
                continue
            runs.append((start, prev))
            start, prev = k, k
        runs.append((start, prev))
        first = True
        for (s, e) in runs:
            ax.axvspan(t_sec[s], t_sec[e], color=color, alpha=0.25, label=(label if first else None))
            first = False

    shade(accel_flag, 'green', 'Acceleration (>10 bpm rise / 15 s)')
    shade(decel_flag, 'red', 'Deceleration (>10 bpm fall / 15 s)')
    _style_axes_no_grid(ax, title='Fetal Heart Rate Trend (CTG-style)', xlabel='Time (s)', ylabel='FHR (bpm)')
    ax.legend(loc='upper right', fontsize=fc.LABEL_FS - 1, prop={'weight': 'bold'})
    fig.suptitle('Fetal Heart Rate Trend', fontsize=fc.TITLE_FS, fontweight='bold')
    plt.tight_layout()
    fpath = f"{outdir}/{tag}_ctg_trend.png"
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


def plot_ecg_strip(fetal_mV, lead_index=0, fpath="/tmp/ecg_strip.png",
                    line_color="#21de78", n_samples_max=1500):
    """Renders a slim, transparent-background foetal-ECG waveform strip
    sized to match the dashboard's confidence-card__ecg-image widget
    (aspect-ratio 810/55.1386, no axes/ticks/spines) -- NOT the full
    3-lead diagnostic figure from plot_residual_over_original(), which
    is far too tall for that widget. Shows the most recent
    `n_samples_max` samples of one fetal lead (default: F1)."""
    sig = fetal_mV[-n_samples_max:, lead_index]
    fig = plt.figure(figsize=(8.1, 0.551), dpi=100)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('none')
    ax.plot(sig, color=line_color, linewidth=1.3)
    ax.axis('off')
    plt.savefig(fpath, dpi=100, transparent=True)
    plt.close(fig)
    return fpath


# ============================================================
# 12. Pipeline entry points
# ============================================================
def analyze_file(input_file, outdir=None, verbose=True):
    """Callable entry point (no argparse/CLI involved) -- runs the full
    O5 pipeline on `input_file` and returns the JSON-able report dict.
    This is what a server/backend should import and call directly;
    main() below is just a thin CLI wrapper around this same function,
    so the command-line tool and any backend stay in lock-step."""
    base = os.path.splitext(os.path.basename(input_file))[0]
    TAG = f"O5_{base}"
    OUTDIR = outdir or os.path.join(os.path.dirname(os.path.abspath(input_file)), "O5_outputs", base)
    os.makedirs(OUTDIR, exist_ok=True)

    def _p(*a):
        if verbose:
            print(*a)

    _p("=" * 70)
    _p(f"O5 -- {TAG}")
    _p("=" * 70)

    # ---- 1) load + extract foetal ECG (cold-start RLS, 5ch) ----
    P, fmt = load_mixed_ecg(input_file)
    _p(f"Input file : {input_file}")
    _p(f"Format     : {fmt}")
    _p(f"fs = {P['fs']:.2f} Hz, N samples = {P['N_samples']}, "
          f"duration = {P['N_samples'] / P['fs']:.2f} s")

    CR = detect_consensus_peaks_time_matched(P)
    W = fc.extract_qrs_windows(P, CR['consensus_r_peak_times'])
    PC = fc.run_pca_windowed(P, W, n_retain=2)
    REG = fit_maternal_template_rls_cold(P, PC)
    maternal_mV, fetal_mV, abd_mV = fc.to_mV(P, REG)
    time_ms, fs = P['time_ms'], P['fs']
    recording_duration_sec = P['N_samples'] / fs

    # ---- (1) FOETAL ECG figure(s) ----
    fig_path_fecg = fc.plot_residual_over_original(P, fetal_mV, abd_mV, TAG, OUTDIR, REG['method'])
    _p(f"\nSaved foetal ECG figure : {fig_path_fecg}")
    fig_path_strip = plot_ecg_strip(fetal_mV, lead_index=0, fpath=f"{OUTDIR}/{TAG}_ecg_strip.png")
    _p(f"Saved foetal ECG strip  : {fig_path_strip}")

    # ---- per-lead foetal R-peaks / HR ----
    fetal_results = detect_fetal_rpeaks_per_lead(fetal_mV, time_ms, fs)
    hdr = f"{'Lead':<6}{'N R-peaks':>12}{'Mean HR':>12}{'Median HR':>12}"
    _p(f"\n{hdr}\n{'-' * len(hdr)}")
    for lead in ABDO_LEADS:
        r = fetal_results[lead]
        m = f"{r['mean_hr_bpm']:.1f}" if r['n_r_peaks_detected'] >= 2 else "n/a"
        md = f"{r['median_hr_bpm']:.1f}" if r['n_r_peaks_detected'] >= 2 else "n/a"
        _p(f"{lead:<6}{r['n_r_peaks_detected']:>12d}{m:>12}{md:>12}")

    # ---- (2)+(3) concordance + confidence ----
    conc = compute_concordance_confidence(fetal_results)
    _p(f"\nFHR concordance (F1/F2/F3 mean HR): {conc['lead_means_bpm']}")
    _p(f"SD_FHR = {conc['sd_fhr_bpm']:.3f} bpm" if not np.isnan(conc['sd_fhr_bpm']) else "SD_FHR = n/a")
    _p(f"Signal confidence = {conc['confidence_pct']:.1f}%"
          if not np.isnan(conc['confidence_pct']) else "Signal confidence = n/a")

    # ---- consensus beat train + FHR trace ----
    CFB = consensus_fetal_beats(fetal_results)
    t_hr_ms, hr_bpm = fhr_trace_from_beats(CFB['beat_times_ms'])
    _p(f"\nConsensus foetal beats: {len(CFB['beat_times_ms'])} "
          f"(anchor lead: {CFB['anchor_lead']})")

    # ---- (5) HRV ----
    hrv = compute_hrv(hr_bpm)
    _p(f"HRV (mean beat-to-beat |change|) = {hrv['mean_hrv_bpm']:.2f} bpm"
          if not np.isnan(hrv['mean_hrv_bpm']) else "HRV = n/a")

    # ---- (4)+(6) accelerations / decelerations ----
    AD = detect_accel_decel(t_hr_ms, hr_bpm)
    accel_present = bool(np.any(AD['accel_flag']))
    decel_present = bool(np.any(AD['decel_flag']))
    accel_episodes = find_episodes(AD['accel_flag'], t_hr_ms)
    decel_episodes = find_episodes(AD['decel_flag'], t_hr_ms)
    _p(f"Accelerations: {'Yes' if accel_present else 'No'} ({len(accel_episodes)} episode(s))")
    _p(f"Decelerations: {'Yes' if decel_present else 'No'} ({len(decel_episodes)} episode(s))")

    # ---- baseline + variability + classification ----
    baseline_bpm, n_stable = compute_baseline_fhr(hr_bpm, AD['accel_flag'], AD['decel_flag'])
    bandwidth = rolling_variability_bandwidth(t_hr_ms, hr_bpm)

    # Sustained-deviation episodes (how long FHR actually stays >10 bpm
    # away from baseline) drive duration-based classification -- see
    # sustained_deviation_episodes() docstring for why this differs from
    # the moment-of-change accel/decel flags used for the yes/no read-out.
    accel_dev_episodes, decel_dev_episodes = sustained_deviation_episodes(t_hr_ms, hr_bpm, baseline_bpm)

    hr_label, hr_reason = classify_baseline_hr(baseline_bpm)
    var_label, var_reason = classify_variability(bandwidth, t_hr_ms, recording_duration_sec)
    decel_label, decel_reason = classify_decelerations(decel_dev_episodes, bandwidth, t_hr_ms, recording_duration_sec)
    overall = overall_classification(hr_label, var_label, decel_label)

    _p(f"\n{'='*70}\nCLASSIFICATION\n{'='*70}")
    _p(f"Baseline FHR        : {baseline_bpm:.0f} bpm -> {hr_label}\n  {hr_reason}")
    _p(f"Baseline variability: {var_label}\n  {var_reason}")
    _p(f"Decelerations        : {decel_label}\n  {decel_reason}")
    _p(f"\nOVERALL: {overall}")
    _p("(Sinusoidal pattern: not assessed -- requires dedicated spectral analysis.)")

    # ---- (2) CTG trend figure ----
    fig_path_ctg = None
    if len(t_hr_ms) > 0:
        fig_path_ctg = plot_ctg_trend(t_hr_ms, hr_bpm, baseline_bpm, AD['accel_flag'], AD['decel_flag'], TAG, OUTDIR)
        _p(f"\nSaved CTG trend figure  : {fig_path_ctg}")
    else:
        _p("\nNo consensus foetal beats detected -- CTG trend figure skipped.")

    # ---- JSON report ----
    report = dict(
        input_file=input_file,
        format=fmt,
        fs_hz=fs,
        n_samples=int(P['N_samples']),
        recording_duration_sec=recording_duration_sec,
        per_lead_fetal_rpeaks=fetal_results,
        concordance=conc,
        consensus_beats=dict(n_beats=int(len(CFB['beat_times_ms'])), anchor_lead=CFB['anchor_lead']),
        hrv=hrv,
        accelerations=dict(present=accel_present,
                            change_events_sec=[dict(start_ms=s, end_ms=e, duration_sec=d) for s, e, d in accel_episodes],
                            sustained_above_baseline_episodes_sec=[dict(start_ms=s, end_ms=e, duration_sec=d) for s, e, d in accel_dev_episodes]),
        decelerations=dict(present=decel_present,
                            change_events_sec=[dict(start_ms=s, end_ms=e, duration_sec=d) for s, e, d in decel_episodes],
                            sustained_below_baseline_episodes_sec=[dict(start_ms=s, end_ms=e, duration_sec=d) for s, e, d in decel_dev_episodes]),
        baseline_fhr_bpm=baseline_bpm,
        classification=dict(
            baseline_heart_rate=dict(label=hr_label, reason=hr_reason),
            baseline_variability=dict(label=var_label, reason=var_reason),
            decelerations=dict(label=decel_label, reason=decel_reason),
            sinusoidal_pattern="not assessed",
            overall=overall,
        ),
        figures=dict(fetal_ecg=fig_path_fecg, ecg_strip=fig_path_strip, ctg_trend=fig_path_ctg),
    )
    report_path = f"{OUTDIR}/{TAG}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    _p(f"\nSaved JSON report       : {report_path}")
    _p(f"\n{TAG} COMPLETE")
    return report


def main():
    """Thin CLI wrapper around analyze_file() -- parses argv, then calls
    the exact same function a backend/server would call directly."""
    parser = argparse.ArgumentParser(
        description="O5: Fetal ECG extraction + CTG-style FHR assessment for mixed-ECG recordings.")
    default_demo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pos1_lying.xlsx")
    parser.add_argument("input_file", nargs="?", default=default_demo,
                         help="Path to a mixed-ECG file: .xlsx (Pos1_lying-style) or "
                              ".txt (phie_*-style, whitespace-delimited). "
                              "Default: Pos1_lying.xlsx next to this script.")
    parser.add_argument("--outdir", default=None, help="Output directory.")
    args = parser.parse_args()
    analyze_file(args.input_file, outdir=args.outdir, verbose=True)


if __name__ == '__main__':
    main()

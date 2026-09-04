"""Objective 4: cold-start 5-channel RLS foetal ECG extraction on a real recording (Pos1_lying.xlsx)."""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  
import fecg_common as fc

DATA_PATH = "Pos1_lying.xlsx"                 # real recording, columns: time_ms,RA,LA,avF,V2,V5,V8,F1,F2,F3
OUTDIR = "outputs"
TAG = "Pos1_lying_5ch_RLS_COLDSTART"
ABDO_LEADS = fc.ABDO_LEADS                    # foetal abdominal leads: F1, F2, F3
REF_LEADS = ['V2', 'V5', 'V8', 'RA', 'LA']    # 5-channel maternal reference 
COLD_LAMBDA = 0.9995                          # RLS forgetting factor
COLD_DELTA = fc.RLS_DELTA                     

os.makedirs(OUTDIR, exist_ok=True)


def _style_axes_no_grid(ax, title=None, xlabel=None, ylabel=None,
                         title_fs=fc.SUBTITLE_FS, label_fs=fc.LABEL_FS, tick_fs=fc.TICK_FS):
    """Reapply fecg_common's bold axis styling but with background gridlines switched off."""
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


def load_and_preprocess_xlsx(filepath, abdo_leads, ref_leads):
    """Load the real recording and run the same bandpass -> notch -> z-score pipeline as fecg_common.py."""
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

    b_bp, a_bp = butter(2, [3.0 / nyquist, 80.0 / nyquist], btype='bandpass')  # 3-80 Hz bandpass, all leads
    bp = np.zeros_like(raw, dtype=float)
    for i in range(N_leads):
        bp[:, i] = filtfilt(b_bp, a_bp, raw[:, i])

    b_notch, a_notch = iirnotch(50.0 / nyquist, Q=30.0)  # 50 Hz mains notch, all leads
    notch = np.zeros_like(bp)
    for i in range(N_leads):
        notch[:, i] = filtfilt(b_notch, a_notch, bp[:, i])

    scaler = StandardScaler()
    norm = scaler.fit_transform(notch)  # joint z-score normalisation across all leads
    norm_df = pd.DataFrame(norm, columns=all_leads)

    abd_std = scaler.scale_[[all_leads.index(l) for l in abdo_leads]]  # per-lead std, for rescaling back to mV later
    return dict(time_ms=time_ms, dt_ms=dt_ms, fs=fs, nyquist=nyquist,
                all_leads=all_leads, ref_leads=ref_leads, notch_signals=notch,
                norm_df=norm_df, norm_abd=norm_df[abdo_leads].values, norm_ref=norm_df[ref_leads].values,
                abd_std=abd_std, scaler=scaler, N_samples=N_samples)


def detect_consensus_peaks_time_matched(P, anchor_leads=('V5', 'V8'), check_lead='V2',
                                         tol_ms=150.0, display_order=('V2', 'V5', 'V8')):
    """Build maternal R-peak consensus by time-matching check_lead onto the reliable anchor_leads (fixes index-shift from missed beats)."""
    time_ms, notch_signals, all_leads, fs = P['time_ms'], P['notch_signals'], P['all_leads'], P['fs']

    r_peaks_results = {L: fc.pan_tompkins_r_peaks(notch_signals[:, all_leads.index(L)], fs)
                        for L in set(list(anchor_leads) + [check_lead])}  

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
    return dict(r_peaks_results=r_peaks_results, beat_times_ms=beat_times_ms,
                consensus_r_peak_times=np.nanmedian(beat_times_ms, axis=1),
                beat_std=np.nanstd(beat_times_ms, axis=1), leads_to_check=list(display_order))


def _rls_core_cold(X, y, lam, delta):
    """Recursive least squares, cold-started from w=0 (mirrors fecg_common._rls_core minus the OLS warm start)."""
    N, p = X.shape
    w = np.zeros(p)
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
    """Fit the cold-start RLS maternal template (PC1, PC2 -> each abdominal lead) independently per lead."""
    X = np.column_stack([PC['M1'], PC['M2']])
    results = {}
    for i, lead in enumerate(ABDO_LEADS):
        y_hat, residual, weights = _rls_core_cold(X, P['norm_abd'][:, i], lam, delta)
        results[lead] = dict(y_hat=y_hat, residual=residual, weights=weights, w_ols=None)
    return dict(method='RLS', X=X, results=results, lam=lam, delta=delta)  


def detect_foetal_rpeaks(fetal_mV, time_ms, fs):
    """Run fecg_common's foetal Pan-Tompkins detector on each residual lead and derive HR from consecutive R-peaks."""
    results = {}
    for i, lead in enumerate(ABDO_LEADS):
        r_idx = fc.fetal_qrs_detect(fetal_mV[:, i], fs)  
        r_times_ms = time_ms[r_idx]
        n_peaks = int(len(r_idx))
        if n_peaks >= 2:
            rr_ms = np.diff(r_times_ms)
            mean_hr, median_hr = float(np.mean(60000.0 / rr_ms)), float(np.median(60000.0 / rr_ms))
        else:
            mean_hr = median_hr = float('nan')
        results[lead] = dict(n_r_peaks_detected=n_peaks, mean_hr_bpm=mean_hr,
                              median_hr_bpm=median_hr, r_peak_times_ms=r_times_ms.tolist())
    return results


def print_results_table(fetal_results):
    """Print a plain-text summary of R-peak counts and heart rate per lead."""
    hdr = f"{'Lead':<6}{'N foetal R-peaks':>18}{'Mean HR (bpm)':>16}{'Median HR (bpm)':>18}"
    print(hdr)
    print('-' * len(hdr))
    for lead, r in fetal_results.items():
        mean_str = f"{r['mean_hr_bpm']:.1f}" if r['n_r_peaks_detected'] >= 2 else "n/a"
        med_str = f"{r['median_hr_bpm']:.1f}" if r['n_r_peaks_detected'] >= 2 else "n/a"
        print(f"{lead:<6}{r['n_r_peaks_detected']:>18d}{mean_str:>16}{med_str:>18}")


def main():
    print(f"{TAG}  (cold-start: w_init=[0,0], lambda={COLD_LAMBDA}, delta={COLD_DELTA})")

    P = load_and_preprocess_xlsx(DATA_PATH, ABDO_LEADS, REF_LEADS)                 # 1. load + preprocess
    CR = detect_consensus_peaks_time_matched(P)                                    # 2. maternal R-peak consensus
    W = fc.extract_qrs_windows(P, CR['consensus_r_peak_times'])                    # 3. QRS-centred PCA windows
    PC = fc.run_pca_windowed(P, W, n_retain=2)                                     # 4. windowed PCA basis (PC1, PC2)
    REG = fit_maternal_template_rls_cold(P, PC)                                    # 5. cold-start RLS regression
    maternal_mV, fetal_mV, abd_mV = fc.to_mV(P, REG)                               # 6. rescale to mV, get residual
    fetal_results = detect_foetal_rpeaks(fetal_mV, P['time_ms'], P['fs'])          # 7. foetal Pan-Tompkins detection

    print_results_table(fetal_results)

    results_path = os.path.join(OUTDIR, f"{TAG}_foetal_rpeak_results.json")
    with open(results_path, "w") as f:
        json.dump(fetal_results, f, indent=2, default=float)                       # 8. save results
    print(f"Saved: {results_path}")

    fig_path = fc.plot_residual_over_original(P, fetal_mV, abd_mV, TAG, OUTDIR, REG['method'])  # 9. save figure
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()

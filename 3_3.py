"""

======
Starting from the same two source recordings used throughout this
project -- one with only the foetal heartbeat, one with the maternal and
foetal heartbeats mixed together -- this script:

  PART A: extends the recording to 200 maternal heartbeats (same real,
          un-warped template-tiling method as 3_1.py), keeping the
          maternal and foetal components SEPARATE (not yet summed), so
          that physiological/instrumentation disturbances below can be
          applied to the correct source before the two are re-mixed.

  PART B: implements four noise mechanisms that can
          disturb a real abdominal recording 

  PART C: combines these into 5 injected-noise recordings (one per
          mechanism, plus one with all four together) alongside the
          clean baseline, giving 6 recordings in total.

  PART D: for each of the 6 recordings, extracts the foetal ECG with
          OLS and with cold-started RLS (7 forgetting factors), scoring
          every extraction against the known foetal signal.

  PART E: statistics table

  PART F: regression coefficient figure


REQUIRED FILES 
----------------------------------
  phie_fetal_only.txt                 (source recording, foetal only)
  phie_with_maternal_interference.txt (source recording, both mixed)
  fecg_common.py                      
"""

# ============================================================
# 0. IMPORTS
# ============================================================
import os
import csv
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import fecg_common as fc   # shared preprocessing / OLS / RLS / statistics engine


# ============================================================
# 1. CONFIGURE
# ============================================================
OUTDIR = 'outputs_3_3'
FETAL_FILE = 'phie_fetal_only.txt'
MIXED_FILE = 'phie_with_maternal_interference.txt'

REF_LEADS = ['V2', 'V5', 'V8', 'RA', 'LA']    # 5-channel reference set
LEADS = fc.ABDO_LEADS                          # ['F1', 'F2', 'F3'], the leads being scored
LAMBDAS = [0.9970, 0.9975, 0.9980, 0.9985, 0.9990, 0.9995, 0.9999]   # RLS forgetting factors swept
TARGET_N_MAT_BEATS = 200

# ---- Real-template extraction parameters (identical to 3_1.py / 3_2.py) ----
COLUMNS = ['time_ms', 'F1', 'F2', 'F3', 'V2', 'V5', 'V8', 'RA', 'LA', 'GND', 'LL']
ALL_LEADS = [c for c in COLUMNS if c != 'time_ms']
ABD_IDX = [ALL_LEADS.index(l) for l in LEADS]   # F1/F2/F3's column positions within ALL_LEADS
DT_MS = 5.0                # source files' 200 Hz sample grid
ENV_THRESH_FRAC = 0.02     # 2% of local peak = "signal back to baseline"
DEBOUNCE_MAT_SAMP = 6      # 30 ms below threshold -> declare maternal beat edge
DEBOUNCE_FET_SAMP = 4      # 20 ms below threshold -> declare foetal beat edge
REF_PEAK_MAT_MS = 1095     # timestamp of one clean maternal beat to cut out
REF_PEAK_FET_MS = 1860     # timestamp of one clean foetal beat to cut out

# ---- Mechanism 1: respiratory sinusoidal transfer-gain modulation ----
SIN_F_HZ = 0.25            # maternal respiration rate (Hz)
SIN_A = 0.20                # modulation depth (dimensionless)
SIN_PHASES = {ABD_IDX[0]: 0.0,               # F1: phi = 0 deg
               ABD_IDX[1]: 2 * np.pi / 3,     # F2: phi = 120 deg  (each electrode's own
               ABD_IDX[2]: 4 * np.pi / 3}     # F3: phi = 240 deg  geometric phase lag)

# ---- Mechanism 2: electrode motion (a single, lasting gain step) ----
EM_JUMP_TIME_S = 100.0      # physiologically plausible mid-recording event
EM_ALPHAS = [1.00, 1.25]    # gain jumps from 1.00 to 1.25 (+25%) and stays there

# ---- Mechanism 3: heteroscedastic (EMG-like) burst noise ----
HET_SIGMA0_FRAC = 0.04      # baseline noise std, as a fraction of each lead's own signal std
HET_BURST_MULT = 10.0       # burst-time noise std = 10x baseline
HET_BURST_WIDTH_S = 1.0
HET_N_BURSTS = 15
HET_SEED = 501

# ---- Mechanism 4: maternal ectopic beats (Behar et al. 2014, Sec. 2.11) ----
ECTOPIC_SEED = 701


# ############################################################
# PART A -- EXTENDED RECORDING
# ############################################################
def load(path):
    """Read one source recording into a labelled DataFrame."""
    df = pd.read_csv(path, sep=r'\s+', header=None)
    df.columns = COLUMNS
    return df


def envelope(df):
    """Per-sample max(|signal|) across all leads -> one activity trace,
    used to find where a heartbeat starts/ends."""
    return np.max(np.abs(df[ALL_LEADS].values), axis=1)


def find_complex_bounds(env, peak_idx, thresh, debounce_samples):
    """Walk outward from a known beat's peak until the signal has stayed
    below `thresh` for `debounce_samples` in a row (so a brief dip
    mid-beat isn't mistaken for the true edge)."""
    n = len(env)
    i, below = peak_idx, 0
    while i > 0:
        below = below + 1 if env[i - 1] < thresh else 0
        if below >= debounce_samples:
            break
        i -= 1
    onset = i
    i, below = peak_idx, 0
    while i < n - 1:
        below = below + 1 if env[i + 1] < thresh else 0
        if below >= debounce_samples:
            break
        i += 1
    offset = i
    return onset, offset


def extract_real_complex(df, time_ms, ref_peak_ms, debounce_samples):
    """Cut out one real heartbeat (all leads, untouched samples),
    centred on the peak nearest ref_peak_ms."""
    env = envelope(df)
    peak_idx_guess = int(round(ref_peak_ms / DT_MS))
    search = slice(max(0, peak_idx_guess - 30), min(len(env), peak_idx_guess + 30))
    peak_idx = search.start + int(np.argmax(env[search]))
    thresh = ENV_THRESH_FRAC * env[peak_idx]
    onset, offset = find_complex_bounds(env, peak_idx, thresh, debounce_samples)
    template = df[ALL_LEADS].values[onset:offset + 1, :].copy()
    return template, time_ms[onset]


def measure_period_ms(sig, time_ms, fs, distance_frac):
    """Median inter-peak interval -> signal's period."""
    dist = int(distance_frac * fs)
    peaks, _ = find_peaks(np.abs(sig), height=0.3 * np.max(np.abs(sig)), distance=max(2, dist))
    return float(np.median(np.diff(time_ms[peaks])))


def tile_aligned(template, onset_ms, period_ms, n_samples, dt_ms=DT_MS):
    """Repeats one real template at every multiple of its own natural
    period, with no stretching or interpolation -- the baseline tiling
    method used everywhere in this project."""
    out = np.zeros((n_samples, template.shape[1]))
    template_len = template.shape[0]
    k = int(np.floor((0 - onset_ms) / period_ms)) - 1
    while True:
        start_idx = int(round((onset_ms + k * period_ms) / dt_ms))
        if start_idx >= n_samples:
            break
        end_idx = start_idx + template_len
        if end_idx > 0:
            seg_start = max(0, -start_idx)
            seg_end = template_len - max(0, end_idx - n_samples)
            out_start, out_end = max(0, start_idx), min(n_samples, end_idx)
            out[out_start:out_end, :] += template[seg_start:seg_end, :]
        k += 1
    return out


def build_base_200beat():
    """Builds the isolated 200-maternal-beat maternal and foetal strips"""
    fetal = load(FETAL_FILE)
    mixed = load(MIXED_FILE)
    time_ms_src = mixed['time_ms'].values
    fs = 1000.0 / DT_MS

    # Maternal-only signal = mixed - fetal (the two source files share an
    # identical fetal component, so this recovers it exactly)
    maternal_recovered = mixed.copy()
    for c in ALL_LEADS:
        maternal_recovered[c] = mixed[c] - fetal[c]

    mat_period_ms = measure_period_ms(maternal_recovered['V5'].values, time_ms_src, fs, 0.4)
    fet_period_ms = measure_period_ms(fetal['V5'].values, time_ms_src, fs, 0.2)
    print(f"  Maternal rate: {60000/mat_period_ms:.2f} bpm | Fetal rate: {60000/fet_period_ms:.2f} bpm")

    mat_template, mat_onset_ms = extract_real_complex(
        maternal_recovered, time_ms_src, REF_PEAK_MAT_MS, DEBOUNCE_MAT_SAMP)
    fet_template, fet_onset_ms = extract_real_complex(
        fetal, time_ms_src, REF_PEAK_FET_MS, DEBOUNCE_FET_SAMP)

    total_duration_ms = TARGET_N_MAT_BEATS * mat_period_ms
    n_samples = int(round(total_duration_ms / DT_MS)) + 1
    t_ms = np.arange(n_samples) * DT_MS

    mat_strip = tile_aligned(mat_template, mat_onset_ms, mat_period_ms, n_samples)
    fet_strip = tile_aligned(fet_template, fet_onset_ms, fet_period_ms, n_samples)
    print(f"  Built {total_duration_ms/1000:.1f} s recording, {n_samples} samples "
          f"({TARGET_N_MAT_BEATS} maternal beats).")

    return dict(t_ms=t_ms, fs=fs, mat_strip=mat_strip, fet_strip=fet_strip,
                mat_template=mat_template, mat_onset_ms=mat_onset_ms, mat_period_ms=mat_period_ms)


# ############################################################
# PART B -- FOUR NON-STATIONARITY ("NOISE") MECHANISMS
# ############################################################


def apply_transfer_gain_modulation_phased(strip, t_s, f_resp, A, abd_lead_idx, phases):
    """MECHANISM 1 -- respiration"""
    out = strip.copy()
    for i in abd_lead_idx:
        h_k = 1.0 + A * np.sin(2 * np.pi * f_resp * t_s + phases[i])
        out[:, i] = strip[:, i] * h_k
    return out


def apply_piecewise_gain(strip, t_s, jump_time_s, alphas, abd_lead_idx):
    """MECHANISM 2 -- electrode motion"""
    S = np.where(t_s < jump_time_s, alphas[0], alphas[1])
    out = strip.copy()
    for i in abd_lead_idx:
        out[:, i] = strip[:, i] * S
    return out


def heteroscedastic_burst_noise(n_samples, fs, sigma0_per_lead, burst_mult,
                                  burst_width_s, n_bursts, n_leads, seed, min_gap_s=5.0):
    """MECHANISM 3 -- EMG-like bursts"""
    rng = np.random.default_rng(seed)
    t_s = np.arange(n_samples) / fs
    sigma = np.ones(n_samples)     # scalar envelope; scaled per-lead below
    w_samp = int(round(burst_width_s * fs))
    placed, attempts = [], 0
    while len(placed) < n_bursts and attempts < n_bursts * 50:
        attempts += 1
        c = rng.uniform(min_gap_s, t_s[-1] - min_gap_s)
        if all(abs(c - p) > (burst_width_s + min_gap_s) for p in placed):
            placed.append(c)
    for c in placed:
        c_idx = int(round(c * fs))
        lo, hi = max(0, c_idx - w_samp // 2), min(n_samples, c_idx + w_samp // 2)
        sigma[lo:hi] = burst_mult
    eta = rng.standard_normal((n_samples, n_leads))
    return sigma[:, None] * sigma0_per_lead[None, :] * eta, np.array(sorted(placed))


def ectopic_beat_states(n_beats, rng):
    """MECHANISM 4a -- decides WHICH beats are ectopic"""
    state, states = 0, np.zeros(n_beats, dtype=int)
    for k in range(n_beats):
        states[k] = state
        rn = 0.7 + 0.1 * rng.uniform()    # P(next=Normal  | current=Normal)
        re = 0.2 + 0.1 * rng.uniform()    # P(next=Ectopic | current=Ectopic)
        state = (0 if rng.uniform() < rn else 1) if state == 0 else (1 if rng.uniform() < re else 0)
    return states


def distort_pvc_template(template, stretch=1.4, amp_scale=1.3):
    """MECHANISM 4b -- decides WHAT an ectopic beat looks like"""
    n = template.shape[0]
    x_new = np.linspace(0, n - 1, int(round(n * stretch)))
    out = np.zeros((len(x_new), template.shape[1]))
    for c in range(template.shape[1]):
        out[:, c] = CubicSpline(np.arange(n), template[:, c])(x_new) * amp_scale
    return out


def tile_maternal_with_ectopics(template, onset_ms, period_ms, n_samples, dt_ms,
                                  compensatory_pause_frac=1.3, seed=ECTOPIC_SEED):
    """MECHANISM 4c -- re-tiles the maternal strip beat-by-beat, swapping
    in the distorted template (+ a lengthened following gap, the
    physiological compensatory pause after a true PVC) wherever the
    Markov chain flagged that beat as ectopic."""
    rng = np.random.default_rng(seed)
    n_beats_est = int(np.ceil(n_samples * dt_ms / period_ms)) + 4
    states = ectopic_beat_states(n_beats_est, rng)
    pvc_template = distort_pvc_template(template)

    out = np.zeros((n_samples, template.shape[1]))
    t_cursor_ms = onset_ms + (int(np.floor((0 - onset_ms) / period_ms)) - 1) * period_ms
    beat_idx = 0
    while t_cursor_ms < n_samples * dt_ms and beat_idx < n_beats_est:
        is_ectopic = bool(states[beat_idx])
        beat_template = pvc_template if is_ectopic else template
        start_idx = int(round(t_cursor_ms / dt_ms))
        tlen = beat_template.shape[0]
        end_idx = start_idx + tlen
        if end_idx > 0 and start_idx < n_samples:
            seg_start = max(0, -start_idx)
            seg_end = tlen - max(0, end_idx - n_samples)
            out_start, out_end = max(0, start_idx), min(n_samples, end_idx)
            out[out_start:out_end, :] += beat_template[seg_start:seg_end, :]
        t_cursor_ms += period_ms * (compensatory_pause_frac if is_ectopic else 1.0)
        beat_idx += 1
    return out


# ############################################################
# PART C -- ASSEMBLE THE 6 RECORDINGS (5 noise-injection + clean baseline)
# ############################################################
def build_scenarios(base):
    t_ms, t_s, fs = base['t_ms'], base['t_ms'] / 1000.0, base['fs']
    mat_strip, fet_strip = base['mat_strip'], base['fet_strip']
    n_samples = len(t_ms)

    mat_sin = apply_transfer_gain_modulation_phased(mat_strip, t_s, SIN_F_HZ, SIN_A, ABD_IDX, SIN_PHASES)
    mat_em = apply_piecewise_gain(mat_strip, t_s, EM_JUMP_TIME_S, EM_ALPHAS, ABD_IDX)
    sigma0 = HET_SIGMA0_FRAC * np.std(mat_strip, axis=0)
    het_noise, _ = heteroscedastic_burst_noise(n_samples, fs, np.where(sigma0 > 0, sigma0, 1e-6),
                                                 HET_BURST_MULT, HET_BURST_WIDTH_S, HET_N_BURSTS,
                                                 len(ALL_LEADS), HET_SEED)
    mat_ectopic = tile_maternal_with_ectopics(base['mat_template'], base['mat_onset_ms'],
                                                base['mat_period_ms'], n_samples, DT_MS)

   
    mat_all = apply_transfer_gain_modulation_phased(mat_ectopic, t_s, SIN_F_HZ, SIN_A, ABD_IDX, SIN_PHASES)
    mat_all = apply_piecewise_gain(mat_all, t_s, EM_JUMP_TIME_S, EM_ALPHAS, ABD_IDX)

    scenarios = {
        'S0_baseline':          dict(label='Baseline (clean)',        mixed=mat_strip + fet_strip,          gt=fet_strip),
        'S1_respiratory':       dict(label='Respiratory modulation',   mixed=mat_sin + fet_strip,            gt=fet_strip),
        'S2_electrode_motion':  dict(label='Electrode motion',         mixed=mat_em + fet_strip,             gt=fet_strip),
        'S3_emg_bursts':        dict(label='EMG bursts',               mixed=mat_strip + fet_strip + het_noise, gt=fet_strip),
        'S4_ectopic':           dict(label='Maternal ectopic beats',   mixed=mat_ectopic + fet_strip,        gt=fet_strip),
        'S5_combined':          dict(label='Combined',                 mixed=mat_all + fet_strip + het_noise, gt=fet_strip),
    }
    for tag, sc in scenarios.items():
        np.savetxt(f"{tag}_mixed.txt", np.column_stack([t_ms, sc['mixed']]), fmt='%.6g', delimiter=' ')
        np.savetxt(f"{tag}_gt.txt", np.column_stack([t_ms, sc['gt']]), fmt='%.6g', delimiter=' ')
    print(f"  Saved {len(scenarios)} scenario recordings "
          f"({len(scenarios)-1} noise-injection + 1 clean baseline).")
    return scenarios


# ############################################################
# PART D -- OLS vs COLD-STARTED RLS, EVERY SCENARIO x FORGETTING FACTOR
# ############################################################
def fit_rls_cold(P, PC, lam, delta=fc.RLS_DELTA):
    """Cold-started RLS"""
    X = np.column_stack([PC['M1'], PC['M2']])
    results = {}
    for i, lead in enumerate(LEADS):
        y = P['norm_abd'][:, i]
        y_hat, residual, weights = fc._rls_core(X, y, w_init=np.zeros(2), lam=lam, delta=delta)
        results[lead] = dict(y_hat=y_hat, residual=residual, weights=weights)
    return dict(method='RLS', X=X, results=results, lam=lam, delta=delta)


def average_across_leads(all_stats, keys_scalar, keys_ci):
    """Collapses fecg_common's per-lead statistics to the mean
    across F1/F2/F3."""
    out = {k: float(np.mean([all_stats[l][k] for l in LEADS])) for k in keys_scalar}
    for k in keys_ci:
        out[k] = (float(np.mean([all_stats[l][k][0] for l in LEADS])),
                  float(np.mean([all_stats[l][k][1] for l in LEADS])))
    return out


KEYS_SCALAR = ['sensitivity_pct', 'ppv_pct', 'f1_pct', 'pearson_r', 'rmse', 'prd',
               'snr_before', 'snr_after', 'dsnr']
KEYS_CI = ['sensitivity_ci_pct', 'ppv_ci_pct', 'f1_ci_pct', 'pearson_r_ci',
           'rmse_ci', 'prd_ci', 'snr_before_ci', 'snr_after_ci', 'dsnr_ci']


def run_scenario(tag, label):
    """Runs OLS + all 7 RLS forgetting factors on one scenario recording,
    scores each against that scenario's own ground truth, and returns
    {method_label: mean-across-leads stats dict}."""
    print(f"\n  --- {tag} ({label}) ---")
    P = fc.load_and_preprocess(f"{tag}_mixed.txt", REF_LEADS)
    CR = fc.detect_consensus_peaks(P)
    W = fc.extract_qrs_windows(P, CR['consensus_r_peak_times'])
    PC = fc.run_pca_windowed(P, W, n_retain=2)
    gt_F = fc.load_ground_truth(f"{tag}_gt.txt")

    results = {}
    REG_ols = fc.fit_maternal_template_ols(P, PC)
    _, fetal_mV, _ = fc.to_mV(P, REG_ols)
    stats = fc.compute_full_stats(P, W, fetal_mV, gt_F, f"{tag}_OLS", OUTDIR)
    results['OLS'] = average_across_leads(stats, KEYS_SCALAR, KEYS_CI)
    print(f"    OLS: F1={results['OLS']['f1_pct']:.1f}%  RMSE={results['OLS']['rmse']:.5f}  "
          f"dSNR={results['OLS']['dsnr']:.2f}dB")

    for lam in LAMBDAS:
        REG = fit_rls_cold(P, PC, lam)
        _, fetal_mV, _ = fc.to_mV(P, REG)
        stats = fc.compute_full_stats(P, W, fetal_mV, gt_F, f"{tag}_RLS_{lam}", OUTDIR)
        results[f'RLS_lam={lam}'] = average_across_leads(stats, KEYS_SCALAR, KEYS_CI)

    best_lam = min(LAMBDAS, key=lambda l: results[f'RLS_lam={l}']['rmse'])
    results['best_lambda'] = best_lam
    print(f"    Best RLS (\u03bb={best_lam}): F1={results[f'RLS_lam={best_lam}']['f1_pct']:.1f}%  "
          f"RMSE={results[f'RLS_lam={best_lam}']['rmse']:.5f}  "
          f"dSNR={results[f'RLS_lam={best_lam}']['dsnr']:.2f}dB")
    return results


# ############################################################
# PART E -- TABLE 7: FULL RESULTS WITH 95% CONFIDENCE INTERVALS
# ############################################################
def write_table7(all_results, scenarios):
    fieldnames = ['scenario', 'method', 'forgetting_factor',
                  'sensitivity_pct', 'sensitivity_ci_lo', 'sensitivity_ci_hi',
                  'ppv_pct', 'ppv_ci_lo', 'ppv_ci_hi',
                  'f1_pct', 'f1_ci_lo', 'f1_ci_hi',
                  'pearson_r', 'pearson_r_ci_lo', 'pearson_r_ci_hi',
                  'rmse_mV', 'rmse_ci_lo', 'rmse_ci_hi',
                  'prd_pct', 'prd_ci_lo', 'prd_ci_hi',
                  'snr_before_dB', 'snr_after_dB',
                  'dsnr_dB', 'dsnr_ci_lo', 'dsnr_ci_hi']

    def row(scenario, method, lam, d):
        return dict(scenario=scenario, method=method, forgetting_factor=lam if lam else '',
                    sensitivity_pct=round(d['sensitivity_pct'], 2), sensitivity_ci_lo=round(d['sensitivity_ci_pct'][0], 2),
                    sensitivity_ci_hi=round(d['sensitivity_ci_pct'][1], 2),
                    ppv_pct=round(d['ppv_pct'], 2), ppv_ci_lo=round(d['ppv_ci_pct'][0], 2), ppv_ci_hi=round(d['ppv_ci_pct'][1], 2),
                    f1_pct=round(d['f1_pct'], 2), f1_ci_lo=round(d['f1_ci_pct'][0], 2), f1_ci_hi=round(d['f1_ci_pct'][1], 2),
                    pearson_r=round(d['pearson_r'], 4), pearson_r_ci_lo=round(d['pearson_r_ci'][0], 4),
                    pearson_r_ci_hi=round(d['pearson_r_ci'][1], 4),
                    rmse_mV=round(d['rmse'], 6), rmse_ci_lo=round(d['rmse_ci'][0], 6), rmse_ci_hi=round(d['rmse_ci'][1], 6),
                    prd_pct=round(d['prd'], 2), prd_ci_lo=round(d['prd_ci'][0], 2), prd_ci_hi=round(d['prd_ci'][1], 2),
                    snr_before_dB=round(d['snr_before'], 2), snr_after_dB=round(d['snr_after'], 2),
                    dsnr_dB=round(d['dsnr'], 2), dsnr_ci_lo=round(d['dsnr_ci'][0], 2), dsnr_ci_hi=round(d['dsnr_ci'][1], 2))

    full_path = f"{OUTDIR}/table7_full_results.csv"
    with open(full_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tag, res in all_results.items():
            label = scenarios[tag]['label']
            writer.writerow(row(label, 'OLS', None, res['OLS']))
            for lam in LAMBDAS:
                writer.writerow(row(label, 'RLS', lam, res[f'RLS_lam={lam}']))

    headline_path = f"{OUTDIR}/table7_headline_OLS_vs_bestRLS.csv"
    with open(headline_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tag, res in all_results.items():
            label = scenarios[tag]['label']
            writer.writerow(row(label, 'OLS', None, res['OLS']))
            bl = res['best_lambda']
            writer.writerow(row(label, f'RLS (best)', bl, res[f'RLS_lam={bl}']))
    print(f"\n  Saved {full_path}")
    print(f"  Saved {headline_path}  (-> Table 7)")

  
    print("\n" + "=" * 155)
    print("TABLE 7 -- OLS vs. BEST-PERFORMING RLS BY SCENARIO (mean across F1/F2/F3, 95% CI in brackets)")
    print("=" * 155)
    hdr = f"{'Scenario':<24}{'Method':<16}{'SE(%)':>20}{'PPV(%)':>20}{'F1(%)':>20}{'r':>18}{'RMSE(mV)':>22}{'PRD(%)':>18}{'dSNR(dB)':>18}"
    print(hdr); print('-' * len(hdr))
    fmt = lambda v, ci, p: f"{v:.{p}f} [{ci[0]:.{p}f},{ci[1]:.{p}f}]"
    for tag, res in all_results.items():
        label = scenarios[tag]['label']
        for mname, d in [('OLS', res['OLS']), (f"RLS (\u03bb={res['best_lambda']})", res[f"RLS_lam={res['best_lambda']}"])]:
            print(f"{label:<24}{mname:<16}{fmt(d['sensitivity_pct'], d['sensitivity_ci_pct'], 1):>20}"
                  f"{fmt(d['ppv_pct'], d['ppv_ci_pct'], 1):>20}{fmt(d['f1_pct'], d['f1_ci_pct'], 1):>20}"
                  f"{fmt(d['pearson_r'], d['pearson_r_ci'], 3):>18}{fmt(d['rmse'], d['rmse_ci'], 5):>22}"
                  f"{fmt(d['prd'], d['prd_ci'], 1):>18}{fmt(d['dsnr'], d['dsnr_ci'], 2):>18}")
    return headline_path


# ############################################################
# PART F -- FIGURE 12: COEFFICIENT TRACKING (Combined scenario, lead F1)
# ############################################################
def build_figure12(base):
    t_ms, t_s = base['t_ms'], base['t_ms'] / 1000.0

    # w0: the NOMINAL (undisturbed) regression coefficient for F1, fit on
    # the clean baseline recording -- this is what a fixed OLS fit would
    # find if the maternal-to-abdominal coupling never changed.
    P0 = fc.load_and_preprocess('S0_baseline_mixed.txt', REF_LEADS)
    CR0 = fc.detect_consensus_peaks(P0)
    W0 = fc.extract_qrs_windows(P0, CR0['consensus_r_peak_times'])
    PC0 = fc.run_pca_windowed(P0, W0, n_retain=2)
    w0 = fc.fit_maternal_template_ols(P0, PC0)['results']['F1']['weights']

    # True coefficient: the SAME sinusoidal x step transfer-gain
    # function injected into the Combined maternal strip in Part B/C,
    # applied to w0 -- this is known exactly because we generated it.
    h_sin = 1.0 + SIN_A * np.sin(2 * np.pi * SIN_F_HZ * t_s + SIN_PHASES[ABD_IDX[0]])
    h_step = np.where(t_s < EM_JUMP_TIME_S, EM_ALPHAS[0], EM_ALPHAS[1])
    true_w1 = h_sin * h_step * w0[0]

    # OLS and RLS estimates, both fit on the actual Combined recording.
    P5 = fc.load_and_preprocess('S5_combined_mixed.txt', REF_LEADS)
    CR5 = fc.detect_consensus_peaks(P5)
    W5 = fc.extract_qrs_windows(P5, CR5['consensus_r_peak_times'])
    PC5 = fc.run_pca_windowed(P5, W5, n_retain=2)
    X5 = np.column_stack([PC5['M1'], PC5['M2']])
    y5 = P5['norm_abd'][:, 0]     # F1

    w_ols_F1 = fc.fit_maternal_template_ols(P5, PC5)['results']['F1']['weights']
    ols_w1 = np.full(len(y5), w_ols_F1[0])
    _, _, rls_weights = fc._rls_core(X5, y5, w_init=np.zeros(2), lam=0.9995, delta=fc.RLS_DELTA)
    rls_w1 = rls_weights[:, 0]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(t_s, true_w1, color='black', linewidth=2.5, label='True')
    ax.plot(t_s, ols_w1, color='blue', linewidth=2.5, linestyle='--', label='OLS')
    ax.plot(t_s, rls_w1, color='red', linewidth=1.6, alpha=0.9, label='RLS')
    ax.set_ylim(-0.40, -0.10)
    ax.set_title('True vs OLS vs RLS estimate of maternal transfer gain\nunder Combined Scenario in Lead F1',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Regression coefficient', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right', prop={'weight': 'bold'})
    for s in ax.spines.values():
        s.set_linewidth(1.2)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    plt.tight_layout()

    fpath = f"{OUTDIR}/figure12_coefficient_tracking.png"
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fpath}  (-> Figure 12)")
    return fpath


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("### PART A: building the 200-maternal-beat base recording ###")
    base = build_base_200beat()

    print("\n### PART B/C: injecting 4 mechanisms -> 5 scenarios + baseline ###")
    scenarios = build_scenarios(base)

    print("\n### PART D: OLS vs. cold-started RLS, every scenario x forgetting factor ###")
    all_results = {}
    for tag, sc in scenarios.items():
        all_results[tag] = run_scenario(tag, sc['label'])
    with open(f"{OUTDIR}/all_results_raw.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    print("\n### PART E: writing Table 7 ###")
    write_table7(all_results, scenarios)

    print("\n### PART F: building Figure 12 ###")
    build_figure12(base)

    print("\nDONE. See", OUTDIR)


if __name__ == '__main__':
    main()

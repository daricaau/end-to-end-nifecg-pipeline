"""
Starting from two recordings of the same simulated pregnancy -- one with
only the foetal heartbeat, one with both the maternal and foetal
heartbeats mixed together -- this script:

  PART A: builds 5 synthetic test recordings, each with the maternal and
          foetal heartbeats at a different combination of speeds (e.g.
          89 maternal : 113 foetal beats per minute).

  PART B: for each of those 5 recordings, tries to computationally
          "subtract out" the mother's heartbeat and recover the baby's,
          using two competing methods (OLS and RLS), then scores how
          accurate the recovered baby heartbeat is against the known
          answer.

  PART C: Outputs table and figure for Section 3.2.

REQUIRED FILES IN THE SAME FOLDER
----------------------------------
  phie_fetal_only.txt                 (source recording, foetal only)
  phie_with_maternal_interference.txt (source recording, both mixed)
  fecg_common.py                (shared extraction/statistics
                                        library)
"""

# ============================================================
# 0. IMPORTS
# ============================================================
import os
import io
import csv
import json
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fecg_common as fc


# ============================================================
# 1. SHARED CONFIGURATION -- the 5 maternal:foetal heart-rate scenarios
#    (single source of truth: used by both Part A and Part B)
# ============================================================
SCENARIOS = [
    dict(n=1, mat_hr=89, fet_hr=113),
    dict(n=2, mat_hr=83, fet_hr=137),
    dict(n=3, mat_hr=73, fet_hr=149),
    dict(n=4, mat_hr=71, fet_hr=157),
    dict(n=5, mat_hr=71, fet_hr=160),
]


# ############################################################
# PART A -- BUILD THE 5 SCENARIOS FROM THE 2 SOURCE RECORDINGS
# ############################################################
# Method: cut ONE real maternal heartbeat and ONE real foetal heartbeat
# out of the source recordings (exact samples, never stretched/warped),
# then lay copies of each back-to-back at the spacing each scenario's
# target heart rate implies. Only the GAP between beats changes.

# ---- 1a. Data-generation constants ----
COLUMNS    = ['time_ms', 'F1', 'F2', 'F3', 'V2', 'V5', 'V8', 'RA', 'LA', 'GND', 'LL']
ALL_LEADS  = [c for c in COLUMNS if c != 'time_ms']
PLOT_LEADS = ['F1', 'F2', 'F3', 'V2', 'V5', 'V8', 'RA', 'LA', 'LL']   # GND omitted (always 0)

FETAL_FILE = 'phie_fetal_only.txt'
MIXED_FILE = 'phie_with_maternal_interference.txt'

DT_MS             = 5.0     # source files' 200 Hz sample grid
ENV_THRESH_FRAC   = 0.02    # 2% of local peak = "signal back to baseline"
DEBOUNCE_MAT_SAMP = 6       # 30 ms below threshold -> declare maternal beat edge
DEBOUNCE_FET_SAMP = 4       # 20 ms below threshold -> declare foetal beat edge
REF_PEAK_MAT_MS   = 1095    # timestamp of one clean maternal beat to cut out
REF_PEAK_FET_MS   = 1860    # timestamp of one clean foetal beat to cut out

TARGET_N_MAT_BEATS = 205    # recording length: >=200 maternal beats required
N_ZOOM_COMPLEXES   = 6      # diagnostic "zoom" plots show first 6 maternal beats

NORMAL_LINEWIDTH, ZOOM_LINEWIDTH = 0.4, 2.0
NORMAL_LABEL_FONTSIZE = 9
BOLD_LABEL_FONTSIZE, BOLD_TICK_FONTSIZE = 14, 12
BOLD_TITLE_FONTSIZE, BOLD_XLABEL_FONTSIZE = 15, 14


# ---- 1b. Load source files & recover the pure maternal signal ----
def load(path):
    """Read one source recording into a labelled DataFrame."""
    df = pd.read_csv(path, sep=r'\s+', header=None)
    df.columns = COLUMNS
    return df


def envelope(df):
    """Per-sample max(|signal|) across all leads -> one activity trace,
    used to find where a heartbeat starts/ends."""
    return np.max(np.abs(df[ALL_LEADS].values), axis=1)


# ---- 1c. Cut ONE real, un-warped heartbeat out of a recording ----
def find_complex_bounds(env, peak_idx, thresh, debounce_samples):
    """Walk outward from a known beat's peak until the signal has stayed
    below `thresh` for `debounce_samples` in a row (so a brief dip
    mid-beat isn't mistaken for the true edge)."""
    n = len(env)
    i, below = peak_idx, 0
    while i > 0:
        if env[i - 1] < thresh:
            below += 1
            if below >= debounce_samples:
                break
        else:
            below = 0
        i -= 1
    onset = i
    i, below = peak_idx, 0
    while i < n - 1:
        if env[i + 1] < thresh:
            below += 1
            if below >= debounce_samples:
                break
        else:
            below = 0
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
    return template, time_ms[onset], time_ms[offset]


# ---- 1d. Re-space that one heartbeat at each scenario's target rate ----
def tile_real_beats(template, period_ms, n_samples):
    """Lay copies of `template` back-to-back every `period_ms`; if two
    copies would ever overlap, sum them (as real overlapping bioelectric
    sources would)."""
    n_leads = template.shape[1]
    out = np.zeros((n_samples, n_leads))
    template_len = template.shape[0]
    k, n_beats = 0, 0
    while True:
        start_idx = int(round(k * period_ms / DT_MS))
        if start_idx >= n_samples:
            break
        end_idx = start_idx + template_len
        seg = template
        if end_idx > n_samples:
            seg = template[:n_samples - start_idx, :]
            end_idx = n_samples
        out[start_idx:end_idx, :] += seg
        n_beats += 1
        k += 1
    return out, n_beats


def save_txt(path, time_ms, data_all_leads):
    """Write one scenario's recording in the same 10-column text format
    as the original source files."""
    out = np.column_stack([time_ms, data_all_leads])
    np.savetxt(path, out, fmt='%.6g', delimiter=' ')


# ---- 1e. Diagnostic strip plots (one full-length overview + one bold zoom, per signal) ----
def plot_full_strip(t_s, data, fname, title, color):
    """Thin-line, full-duration, all-leads overview strip."""
    fig, axes = plt.subplots(len(PLOT_LEADS), 1, figsize=(22, 1.8 * len(PLOT_LEADS)), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    for ax, lead in zip(axes, PLOT_LEADS):
        li = ALL_LEADS.index(lead)
        ax.plot(t_s, data[:, li], color=color, linewidth=NORMAL_LINEWIDTH)
        ax.set_ylabel(lead, fontsize=NORMAL_LABEL_FONTSIZE)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_xlim(t_s[0], t_s[-1])
    plt.tight_layout()
    plt.savefig(fname, dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_bold_zoom(t_s, data, fname, title, color):
    """Thick-line, large bold-font, zoomed-in (first 6 beats), all-leads strip."""
    fig, axes = plt.subplots(len(PLOT_LEADS), 1, figsize=(16, 2.1 * len(PLOT_LEADS)), sharex=True)
    fig.suptitle(title, fontsize=BOLD_TITLE_FONTSIZE, fontweight='bold')
    for ax, lead in zip(axes, PLOT_LEADS):
        li = ALL_LEADS.index(lead)
        ax.plot(t_s, data[:, li], color=color, linewidth=ZOOM_LINEWIDTH)
        ax.set_ylabel(lead, fontsize=BOLD_LABEL_FONTSIZE, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=BOLD_TICK_FONTSIZE)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight('bold')
    axes[-1].set_xlabel('Time (s)', fontsize=BOLD_XLABEL_FONTSIZE, fontweight='bold')
    axes[-1].set_xlim(t_s[0], t_s[-1])
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---- 1f. Part A entry point ----
def generate_scenario_datasets():
    """Build all 5 scenarios' recordings (.txt) and diagnostic strips
    (.png) from the 2 source files. Returns nothing; writes files to
    the working directory, ready for Part B to read."""
    fetal = load(FETAL_FILE)
    mixed = load(MIXED_FILE)
    time_ms_src = mixed['time_ms'].values

    # maternal-only = mixed - foetal (exact, since the sources are a
    # linear superposition from the same forward simulation)
    maternal_recovered = mixed.copy()
    for c in ALL_LEADS:
        maternal_recovered[c] = mixed[c] - fetal[c]

    # cut the two real, un-warped template beats ONCE; reused for every scenario
    mat_template, mat_onset_ms, mat_offset_ms = extract_real_complex(
        maternal_recovered, time_ms_src, REF_PEAK_MAT_MS, DEBOUNCE_MAT_SAMP)
    fet_template, fet_onset_ms, fet_offset_ms = extract_real_complex(
        fetal, time_ms_src, REF_PEAK_FET_MS, DEBOUNCE_FET_SAMP)

    print(f"Maternal template: {mat_offset_ms - mat_onset_ms:.0f} ms wide "
          f"({mat_template.shape[0]} samples, reused for all scenarios)")
    print(f"Foetal template:   {fet_offset_ms - fet_onset_ms:.0f} ms wide "
          f"({fet_template.shape[0]} samples, reused for all scenarios)\n")

    for sc in SCENARIOS:
        n, mat_hr, fet_hr = sc['n'], sc['mat_hr'], sc['fet_hr']
        tag = f"scenario{n}_mat{mat_hr}_fet{fet_hr}"
        print(f"=== Scenario {n}: {mat_hr} maternal : {fet_hr} foetal bpm ===")

        mat_period_ms = 60000.0 / mat_hr
        fet_period_ms = 60000.0 / fet_hr
        total_duration_ms = TARGET_N_MAT_BEATS * mat_period_ms
        n_samples = int(round(total_duration_ms / DT_MS)) + 1
        t_ms = np.arange(n_samples) * DT_MS
        t_s = t_ms / 1000.0

        mat_strip, n_mat_beats = tile_real_beats(mat_template, mat_period_ms, n_samples)
        fet_strip, n_fet_beats = tile_real_beats(fet_template, fet_period_ms, n_samples)
        mixed_strip = mat_strip + fet_strip
        print(f"  {total_duration_ms/1000:.1f}s -> {n_mat_beats} maternal beats, {n_fet_beats} foetal beats")

        # .txt outputs -- these are what Part B reads
        save_txt(f'{tag}_mixed_strip.txt', t_ms, mixed_strip)
        save_txt(f'{tag}_pure_fetal_strip.txt', t_ms, fet_strip)

        # diagnostic .png strips (full-length overview + bold 6-beat zoom)
        plot_full_strip(t_s, mixed_strip, f'{tag}_full_mixed_all_leads.png',
                         f'Scenario {n}: Mixed Maternal+Foetal ECG — {mat_hr}:{fet_hr} bpm', color='black')
        plot_full_strip(t_s, fet_strip, f'{tag}_pure_fetal_all_leads.png',
                         f'Scenario {n}: Pure Foetal ECG — {fet_hr} bpm', color='steelblue')
        plot_full_strip(t_s, mat_strip, f'{tag}_pure_maternal_all_leads.png',
                         f'Scenario {n}: Pure Maternal ECG — {mat_hr} bpm', color='darkred')

        zoom_mask = t_s <= N_ZOOM_COMPLEXES * mat_period_ms / 1000.0
        plot_bold_zoom(t_s[zoom_mask], mixed_strip[zoom_mask], f'{tag}_zoom6_mixed_all_leads.png',
                        f'Mixed Maternal and Foetal ECG Strip at a ratio of {mat_hr}:{fet_hr} (Maternal:Foetal)',
                        color='black')
        plot_bold_zoom(t_s[zoom_mask], fet_strip[zoom_mask], f'{tag}_zoom6equivalent_pure_fetal_all_leads.png',
                        f'Pure Foetal ECG Strip for the ratio of {mat_hr}:{fet_hr} (Maternal:Foetal)',
                        color='steelblue')
        print(f"  saved: {tag}_mixed_strip.txt, {tag}_pure_fetal_strip.txt (+5 diagnostic .png)\n")


# ############################################################
# PART B -- 5-CHANNEL OLS vs RLS EXTRACTION & STATISTICS
# ############################################################
# For each of the 5 scenarios, tries to recover the foetal ECG from the
# mixed recording using two competing regression methods, then scores
# accuracy against the known (ground-truth) foetal-only recording.
#
#   OLS = batch fit
#   RLS = adaptive, COLD-STARTED (starts from zero, not warm-started
#         from the OLS answer) at forgetting factor lambda=0.999

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

REF_LEADS_5CH = ['V2', 'V5', 'V8', 'RA', 'LA']    # reference channels used to build the maternal template
RLS_COLD_LAMBDA = 0.999                           # Forgetting factor 
METHODS = ['OLS', 'RLS']
RLS_TAG = 'RLScold' + str(RLS_COLD_LAMBDA).replace('0.', '')   


# ---- 2a. Cold-started RLS  ----
def fit_rls_cold(P, PC, lam=RLS_COLD_LAMBDA, delta=fc.RLS_DELTA):
    """RLS fit starting from zero weights (w_init=zeros), not warm-started
    from the OLS solution -- calls fc._rls_core directly, since
    fc.fit_maternal_template_rls is hard-wired to warm-start."""
    X = np.column_stack([PC['M1'], PC['M2']])
    norm_abd = P['norm_abd']
    results = {}
    for i, lead in enumerate(fc.ABDO_LEADS):
        y = norm_abd[:, i]
        y_hat, residual, weights = fc._rls_core(X, y, w_init=np.zeros(2), lam=lam, delta=delta)
        results[lead] = dict(y_hat=y_hat, residual=residual, weights=weights)
    return dict(method='RLS', X=X, results=results, lam=lam, delta=delta)


# ---- 2b. Run one (scenario, method) through the full extraction pipeline ----
def run_one(sc, method):
    """1. preprocess -> 2. find maternal beats -> 3. window them ->
    4. PCA maternal template -> 5. OLS or cold-RLS fit -> 6. subtract to
    recover foetal ECG -> 7. score against ground truth. Saves full
    per-lead stats (+CIs) to outputs/{tag}_stats.json."""
    n, mat_hr, fet_hr = sc['n'], sc['mat_hr'], sc['fet_hr']
    method_tag = RLS_TAG if method == 'RLS' else method
    tag = f"S{n}_5ch_{method_tag}_mat{mat_hr}_fet{fet_hr}"
    mixed_file = f"scenario{n}_mat{mat_hr}_fet{fet_hr}_mixed_strip.txt"
    fetal_file = f"scenario{n}_mat{mat_hr}_fet{fet_hr}_pure_fetal_strip.txt"

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):   # suppress fecg_common's per-run print firehose
        P = fc.load_and_preprocess(mixed_file, REF_LEADS_5CH)
        CR = fc.detect_consensus_peaks(P)
        W = fc.extract_qrs_windows(P, CR['consensus_r_peak_times'])
        PC = fc.run_pca_windowed(P, W, n_retain=2)
        REG = fc.fit_maternal_template_ols(P, PC) if method == 'OLS' else fit_rls_cold(P, PC)
        maternal_mV, fetal_mV, abd_mV = fc.to_mV(P, REG)
        gt_F = fc.load_ground_truth(fetal_file)
        stats = fc.compute_full_stats(P, W, fetal_mV, gt_F, tag, OUTDIR)

    with open(f"{OUTDIR}/{tag}_run_log.txt", "w") as f:
        f.write(buf.getvalue())
    with open(f"{OUTDIR}/{tag}_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=float)
    return stats


# ---- 2c. Average each scenario/method's 3 leads (F1/F2/F3) into one summary ----
AVG_KEYS = ['sensitivity_pct', 'ppv_pct', 'f1_pct',
            'ba_bias_bpm', 'ba_sd_bpm', 'ba_loa_lower_bpm', 'ba_loa_upper_bpm',
            'pearson_r', 'rmse', 'prd', 'snr_before', 'snr_after', 'dsnr']
SUM_KEYS = ['tp', 'fp', 'fn']


def average_across_leads(stats):
    """Collapse compute_full_stats' F1/F2/F3 dicts into one mean dict."""
    avg = {k: float(np.mean([stats[lead][k] for lead in fc.ABDO_LEADS])) for k in AVG_KEYS}
    for k in SUM_KEYS:
        avg[k + '_total'] = int(np.sum([stats[lead][k] for lead in fc.ABDO_LEADS]))
        avg[k + '_mean'] = float(np.mean([stats[lead][k] for lead in fc.ABDO_LEADS]))
    avg['n_gt_peaks_consensus'] = stats['F1']['n_gt_peaks_consensus']
    return avg


# ############################################################
# PART C -- OUTPUT RESULTS 
# ############################################################

# ---- 3a. Full statistics  ----
CSV_FIELDNAMES = [
    "scenario", "ratio_mat_fet", "method", "lead",
    "tp", "fp", "fn",
    "sensitivity_pct", "sensitivity_ci_lo", "sensitivity_ci_hi",
    "ppv_pct", "ppv_ci_lo", "ppv_ci_hi",
    "f1_pct", "f1_ci_lo", "f1_ci_hi",
    "n_ext_peaks", "n_gt_peaks_consensus",
    "ba_bias_bpm", "ba_sd_bpm", "ba_loa_lower_bpm", "ba_loa_upper_bpm", "ba_n_pairs",
    "pearson_r", "pearson_r_ci_lo", "pearson_r_ci_hi", "pearson_p",
    "rmse_mV", "rmse_ci_lo", "rmse_ci_hi",
    "prd_pct", "prd_ci_lo", "prd_ci_hi",
    "snr_before_dB", "snr_before_ci_lo", "snr_before_ci_hi",
    "snr_after_dB", "snr_after_ci_lo", "snr_after_ci_hi",
    "dsnr_dB", "dsnr_ci_lo", "dsnr_ci_hi", "dsnr_p",
]


def _lead_row(scenario_n, ratio, method, lead, s):
    """One CSV row: full stats + 95% CI for a single F1/F2/F3 lead."""
    return dict(
        scenario=scenario_n, ratio_mat_fet=ratio, method=method, lead=lead,
        tp=s['tp'], fp=s['fp'], fn=s['fn'],
        sensitivity_pct=round(s['sensitivity_pct'], 4),
        sensitivity_ci_lo=round(s['sensitivity_ci_pct'][0], 4), sensitivity_ci_hi=round(s['sensitivity_ci_pct'][1], 4),
        ppv_pct=round(s['ppv_pct'], 4),
        ppv_ci_lo=round(s['ppv_ci_pct'][0], 4), ppv_ci_hi=round(s['ppv_ci_pct'][1], 4),
        f1_pct=round(s['f1_pct'], 4),
        f1_ci_lo=round(s['f1_ci_pct'][0], 4), f1_ci_hi=round(s['f1_ci_pct'][1], 4),
        n_ext_peaks=s['n_ext_peaks'], n_gt_peaks_consensus=s['n_gt_peaks_consensus'],
        ba_bias_bpm=round(s['ba_bias_bpm'], 4), ba_sd_bpm=round(s['ba_sd_bpm'], 4),
        ba_loa_lower_bpm=round(s['ba_loa_lower_bpm'], 4), ba_loa_upper_bpm=round(s['ba_loa_upper_bpm'], 4),
        ba_n_pairs=s['ba_n_pairs'],
        pearson_r=round(s['pearson_r'], 5),
        pearson_r_ci_lo=round(s['pearson_r_ci'][0], 5), pearson_r_ci_hi=round(s['pearson_r_ci'][1], 5),
        pearson_p=s['pearson_p'],
        rmse_mV=round(s['rmse'], 6),
        rmse_ci_lo=round(s['rmse_ci'][0], 6), rmse_ci_hi=round(s['rmse_ci'][1], 6),
        prd_pct=round(s['prd'], 4),
        prd_ci_lo=round(s['prd_ci'][0], 4), prd_ci_hi=round(s['prd_ci'][1], 4),
        snr_before_dB=round(s['snr_before'], 4),
        snr_before_ci_lo=round(s['snr_before_ci'][0], 4), snr_before_ci_hi=round(s['snr_before_ci'][1], 4),
        snr_after_dB=round(s['snr_after'], 4),
        snr_after_ci_lo=round(s['snr_after_ci'][0], 4), snr_after_ci_hi=round(s['snr_after_ci'][1], 4),
        dsnr_dB=round(s['dsnr'], 4),
        dsnr_ci_lo=round(s['dsnr_ci'][0], 4), dsnr_ci_hi=round(s['dsnr_ci'][1], 4),
        dsnr_p=s['dsnr_p'],
    )


def _mean_row(scenario_n, ratio, method, stats):
    """One CSV row: F1/F2/F3 averaged (point estimates AND CI bounds
    each averaged across leads; TP/FP/FN summed) -- this is the row
    that populates Table S5."""
    def m(key):
        return float(np.mean([stats[l][key] for l in fc.ABDO_LEADS]))

    def m_ci(key):
        return (float(np.mean([stats[l][key][0] for l in fc.ABDO_LEADS])),
                float(np.mean([stats[l][key][1] for l in fc.ABDO_LEADS])))

    se_ci, ppv_ci, f1_ci = m_ci('sensitivity_ci_pct'), m_ci('ppv_ci_pct'), m_ci('f1_ci_pct')
    r_ci, rmse_ci, prd_ci = m_ci('pearson_r_ci'), m_ci('rmse_ci'), m_ci('prd_ci')
    snrb_ci, snra_ci, dsnr_ci = m_ci('snr_before_ci'), m_ci('snr_after_ci'), m_ci('dsnr_ci')

    return dict(
        scenario=scenario_n, ratio_mat_fet=ratio, method=method, lead='Mean',
        tp=int(np.sum([stats[l]['tp'] for l in fc.ABDO_LEADS])),
        fp=int(np.sum([stats[l]['fp'] for l in fc.ABDO_LEADS])),
        fn=int(np.sum([stats[l]['fn'] for l in fc.ABDO_LEADS])),
        sensitivity_pct=round(m('sensitivity_pct'), 4),
        sensitivity_ci_lo=round(se_ci[0], 4), sensitivity_ci_hi=round(se_ci[1], 4),
        ppv_pct=round(m('ppv_pct'), 4),
        ppv_ci_lo=round(ppv_ci[0], 4), ppv_ci_hi=round(ppv_ci[1], 4),
        f1_pct=round(m('f1_pct'), 4),
        f1_ci_lo=round(f1_ci[0], 4), f1_ci_hi=round(f1_ci[1], 4),
        n_ext_peaks=round(m('n_ext_peaks'), 2), n_gt_peaks_consensus=stats['F1']['n_gt_peaks_consensus'],
        ba_bias_bpm=round(m('ba_bias_bpm'), 4), ba_sd_bpm=round(m('ba_sd_bpm'), 4),
        ba_loa_lower_bpm=round(m('ba_loa_lower_bpm'), 4), ba_loa_upper_bpm=round(m('ba_loa_upper_bpm'), 4),
        ba_n_pairs=round(m('ba_n_pairs'), 2),
        pearson_r=round(m('pearson_r'), 5),
        pearson_r_ci_lo=round(r_ci[0], 5), pearson_r_ci_hi=round(r_ci[1], 5),
        pearson_p=m('pearson_p'),
        rmse_mV=round(m('rmse'), 6),
        rmse_ci_lo=round(rmse_ci[0], 6), rmse_ci_hi=round(rmse_ci[1], 6),
        prd_pct=round(m('prd'), 4),
        prd_ci_lo=round(prd_ci[0], 4), prd_ci_hi=round(prd_ci[1], 4),
        snr_before_dB=round(m('snr_before'), 4),
        snr_before_ci_lo=round(snrb_ci[0], 4), snr_before_ci_hi=round(snrb_ci[1], 4),
        snr_after_dB=round(m('snr_after'), 4),
        snr_after_ci_lo=round(snra_ci[0], 4), snr_after_ci_hi=round(snra_ci[1], 4),
        dsnr_dB=round(m('dsnr'), 4),
        dsnr_ci_lo=round(dsnr_ci[0], 4), dsnr_ci_hi=round(dsnr_ci[1], 4),
        dsnr_p=m('dsnr_p'),
    )


def write_full_stats_csv(all_results, outdir):
    """Write the full-precision CSV (5 scenarios x 2 methods x [F1,F2,F3,Mean])
    that Table S5's values are read from."""
    csv_path = f"{outdir}/FULL_STATS_5scenarios_5ch_OLS_vs_{RLS_TAG}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for sc in SCENARIOS:
            n = sc['n']
            ratio = f"{sc['mat_hr']}:{sc['fet_hr']}"
            for method in METHODS:
                stats = all_results[n][method]['per_lead']
                for lead in fc.ABDO_LEADS:
                    writer.writerow(_lead_row(n, ratio, method, lead, stats[lead]))
                writer.writerow(_mean_row(n, ratio, method, stats))
    return csv_path


# ---- 3b. Figure 10: trend plot ----
def plot_trends_vs_ratio(all_results, outdir):
    """RMSE / Pearson's r / dSNR / PRD vs foetal:maternal ratio, RLS only.
    Layout: top-left RMSE (blue), top-right Pearson's r (green),
    bottom-left dSNR (orange), bottom-right PRD (red)."""
    order = sorted(all_results.keys(),
                    key=lambda n: all_results[n]['RLS']['fet_hr'] / all_results[n]['RLS']['mat_hr'])
    ratios, r_vals, rmse_vals, prd_vals, dsnr_vals = [], [], [], [], []
    for n in order:
        rec = all_results[n]['RLS']
        ratios.append(rec['fet_hr'] / rec['mat_hr'])
        a = rec['avg']
        r_vals.append(a['pearson_r']); rmse_vals.append(a['rmse'])
        prd_vals.append(a['prd']); dsnr_vals.append(a['dsnr'])
    ratios = np.array(ratios)

    LW, MARKER_SIZE, TICK_FONTSIZE = 3.2, 9, 13
    panels = [
        ('RMSE', rmse_vals, '#1f77b4', 'RMSE (mV)'),
        ("Pearson's r", r_vals, '#2ca02c', "Pearson's r"),
        ('dSNR', dsnr_vals, '#ff7f0e', 'dSNR (dB)'),
        ('PRD', prd_vals, '#d62728', 'PRD (%)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Trends with Varying Foetal:Maternal Ratio', fontsize=18, fontweight='bold')
    for (title, vals, colour, ylabel), ax in zip(panels, axes.flat):
        ax.plot(ratios, vals, color=colour, linewidth=LW, marker='o', markersize=MARKER_SIZE,
                markeredgecolor='white', markeredgewidth=1.2, solid_capstyle='round')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Foetal:Maternal Ratio', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.grid(False)
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)
        ax.tick_params(axis='both', labelsize=TICK_FONTSIZE, width=1.2)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
            lbl.set_fontsize(TICK_FONTSIZE)
    fig.patch.set_facecolor('white')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fpath = f"{outdir}/trends_vs_fetal_maternal_ratio.png"
    fig.savefig(fpath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return fpath


# ---- 3c. Supplementary figure: OLS vs RLS overlaid on the same axes ----
def _lighten(hex_colour, amount=0.55):
    """Blend a hex colour toward white by `amount` (0=no change, 1=white)."""
    hex_colour = hex_colour.lstrip('#')
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(v + (255 - v) * amount) for v in (r, g, b))
    return f'#{r:02x}{g:02x}{b:02x}'


def plot_ols_vs_rls_vs_ratio(all_results, outdir):
    """Same 4-panel layout/colours as Figure 10, but OLS (solid, full
    colour) and RLS (dashed, lightened tint) are overlaid per panel, so
    the reader can see whether OLS follows the same ratio-driven trend."""
    order = sorted(all_results.keys(),
                    key=lambda n: all_results[n]['OLS']['fet_hr'] / all_results[n]['OLS']['mat_hr'])
    ratios = []
    vals = {'OLS': {'rmse': [], 'r': [], 'dsnr': [], 'prd': []},
            'RLS': {'rmse': [], 'r': [], 'dsnr': [], 'prd': []}}
    for n in order:
        ratios.append(all_results[n]['OLS']['fet_hr'] / all_results[n]['OLS']['mat_hr'])
        for method in ('OLS', 'RLS'):
            a = all_results[n][method]['avg']
            vals[method]['rmse'].append(a['rmse']); vals[method]['r'].append(a['pearson_r'])
            vals[method]['dsnr'].append(a['dsnr']); vals[method]['prd'].append(a['prd'])
    ratios = np.array(ratios)

    LW, MARKER_SIZE = 3.0, 8
    panels = [('RMSE', 'rmse', '#1f77b4'), ("Pearson's r", 'r', '#2ca02c'),
              ('dSNR', 'dsnr', '#ff7f0e'), ('PRD', 'prd', '#d62728')]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Trends with Varying Foetal:Maternal Ratio\nOLS vs RLS (cold-start, \u03bb=0.999)",
                 fontsize=18, fontweight='bold')
    for (title, key, base_colour), ax in zip(panels, axes.flat):
        style = {'OLS': dict(color=base_colour, linestyle='-'),
                 'RLS': dict(color=_lighten(base_colour, 0.55), linestyle='--')}
        for method in ('OLS', 'RLS'):
            st = style[method]
            ax.plot(ratios, vals[method][key], color=st['color'], linewidth=LW, linestyle=st['linestyle'],
                     marker='o', markersize=MARKER_SIZE, markeredgecolor='white', markeredgewidth=1.0, label=method)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Foetal:Maternal Ratio', fontsize=11, fontweight='bold')
        ax.grid(False)
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)
        ax.tick_params(labelsize=10, width=1.2)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
        ax.legend(fontsize=10, prop={'weight': 'bold'}, frameon=False)
    fig.patch.set_facecolor('white')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fpath = f"{outdir}/trends_ols_vs_rls_vs_ratio.png"
    fig.savefig(fpath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return fpath


def run_analysis_pipeline():
    """For each of the 5 scenarios x {OLS, RLS}: run the extraction
    pipeline, score it, then build the combined JSON/CSV/figures."""
    all_results = {}
    for sc in SCENARIOS:
        n = sc['n']
        all_results[n] = {}
        for method in METHODS:
            print(f"Running scenario {n} ({sc['mat_hr']}:{sc['fet_hr']}) -- 5ch {method} ...", flush=True)
            stats = run_one(sc, method)
            all_results[n][method] = dict(per_lead=stats, avg=average_across_leads(stats),
                                           mat_hr=sc['mat_hr'], fet_hr=sc['fet_hr'])

    with open(f"{OUTDIR}/ALL_SCENARIOS_5CH_OLS_vs_RLS_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    
    print("\n" + "=" * 130)
    print("5-CHANNEL OLS vs RLS -- AVERAGED ACROSS F1/F2/F3 -- ALL 5 MATERNAL:FOETAL RATIO SCENARIOS")
    print("=" * 130)
    hdr = (f"{'Scen':<6}{'Ratio(M:F)':<12}{'Method':<7}{'SE%':>7}{'PPV%':>7}{'F1%':>7}"
           f"{'BAbias':>9}{'BAsd':>8}{'r':>8}{'RMSE':>9}{'PRD%':>8}"
           f"{'SNRb':>8}{'SNRa':>8}{'dSNR':>8}{'TP':>5}{'FP':>5}{'FN':>5}")
    print(hdr)
    print('-' * len(hdr))
    for sc in SCENARIOS:
        n = sc['n']
        ratio = f"{sc['mat_hr']}:{sc['fet_hr']}"
        for method in METHODS:
            a = all_results[n][method]['avg']
            print(f"{n:<6}{ratio:<12}{method:<7}"
                  f"{a['sensitivity_pct']:>7.2f}{a['ppv_pct']:>7.2f}{a['f1_pct']:>7.2f}"
                  f"{a['ba_bias_bpm']:>9.3f}{a['ba_sd_bpm']:>8.3f}{a['pearson_r']:>8.4f}"
                  f"{a['rmse']:>9.5f}{a['prd']:>8.2f}"
                  f"{a['snr_before']:>8.2f}{a['snr_after']:>8.2f}{a['dsnr']:>8.2f}"
                  f"{a['tp_total']:>5}{a['fp_total']:>5}{a['fn_total']:>5}")
    print("=" * 130)
    print("SE/PPV/F1 = sensitivity/positive-predictive-value/F1 (%), foetal-QRS detection vs consensus GT")
    print("BAbias/BAsd = Bland-Altman fHR bias/SD (bpm)   r/RMSE(mV)/PRD(%) = waveform-morphology accuracy")
    print("SNRb/SNRa/dSNR (dB) = signal-to-noise before/after cancellation, and the improvement")

    csv_path = write_full_stats_csv(all_results, OUTDIR)
    trend_path = plot_trends_vs_ratio(all_results, OUTDIR)
    ols_vs_rls_path = plot_ols_vs_rls_vs_ratio(all_results, OUTDIR)
    print(f"\nSaved: {csv_path}   (-> Table S5)")
    print(f"Saved: {trend_path}   (-> Figure 10)")
    print(f"Saved: {ols_vs_rls_path}   (supplementary OLS-vs-RLS overlay)")


# ############################################################
# ENTRY POINT -- run Part A, then Part B/C, in order
# ############################################################
def main():
    print("### PART A: building the 5 scenario datasets ###\n")
    generate_scenario_datasets()
    print("### PART B/C: 5-channel OLS vs RLS extraction, statistics, Table S5 & Figure 10 ###\n")
    run_analysis_pipeline()


if __name__ == '__main__':
    main()

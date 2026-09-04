"""
Compares three ways of estimating and removing the maternal ECG from a
synthetic recording: OLS, warm-started (RLS) i.e. started from that OLS solution 
or cold-started RLS i.e. started from zero.
RLS is also run at seven different "forgetting
factors" (how quickly it prioritises recent data over old data).

WHY: the original recording is 5 maternal heartbeats, which is
too short to compare these methods reliably. This script first extends
it to 200 maternal heartbeats using a real, unwarped heartbeat template
tiled at the recording's own natural rate, then runs every method on 
the extended recording and reports how well each one recovers 
the known fetal ECG.

LEADS ON FROM: fecg_common.py, scenario3_5ch_OLS.py (Scenario 3 -- OLS)
and scenario4_5ch_RLS.py (Scenario 4 -- RLS, warm-started only, single
fixed lambda=0.999). This script reuses fecg_common.py's OLS fit and RLS
core routine (fc._rls_core) unchanged, extended in two ways:
    1) RLS is swept across seven forgetting factors instead of one, and
    2) RLS is run BOTH warm-started (as in Scenario 4) AND cold-started
"""
import csv
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fecg_common as fc

# ----------------------------------------------------------------
# Configure
# ----------------------------------------------------------------
OUTDIR = "outputs_warm_vs_cold"
ORIG_FETAL_FILE = "data/phie_fetal_only.txt"
ORIG_MIXED_FILE = "data/phie_with_maternal_interference.txt"
PROLONGED_FETAL_FILE = "data/phie_fetal_only_200beat.txt"
PROLONGED_MIXED_FILE = "data/phie_with_maternal_interference_200beat.txt"

REF_LEADS = ["V2", "V5", "V8", "RA", "LA"]   # 5-channel reference set
LEADS = fc.ABDO_LEADS                         # ['F1', 'F2', 'F3']
LAMBDAS = [0.997, 0.9975, 0.998, 0.9985, 0.999, 0.9995, 0.9999]
TARGET_N_MAT_BEATS = 200

# Real-template extraction parameters (see build_prolonged_dataset)
COLUMNS = ["time_ms", "F1", "F2", "F3", "V2", "V5", "V8", "RA", "LA", "GND", "LL"]
ALL_LEADS = [c for c in COLUMNS if c != "time_ms"]
DT_MS = 5.0
ENV_THRESH_FRAC = 0.02
DEBOUNCE_MAT_SAMP = 6
DEBOUNCE_FET_SAMP = 4
REF_PEAK_MAT_MS = 1095
REF_PEAK_FET_MS = 1860


# ================================================================
# 1. BUILD THE EXTENDED RECORDING
# ================================================================
def _load_leads(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = COLUMNS
    return df


def _envelope(df):
    """Finds where a heartbeat starts/ends."""
    return np.max(np.abs(df[ALL_LEADS].values), axis=1)


def _find_complex_bounds(env, peak_idx, thresh, debounce_samples):
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


def _extract_real_complex(df, time_ms, ref_peak_ms, debounce_samples):
    """Pulls one real, unaveraged heartbeat out of the source signal."""
    env = _envelope(df)
    guess = int(round(ref_peak_ms / DT_MS))
    search = slice(max(0, guess - 30), min(len(env), guess + 30))
    peak_idx = search.start + int(np.argmax(env[search]))
    thresh = ENV_THRESH_FRAC * env[peak_idx]
    onset, offset = _find_complex_bounds(env, peak_idx, thresh, debounce_samples)
    template = df[ALL_LEADS].values[onset:offset + 1, :].copy()
    return template, time_ms[onset]


def _measure_period_ms(sig, time_ms, fs, distance_frac):
    """Median inter-peak interval -> signal's period."""
    dist = int(distance_frac * fs)
    peaks, _ = find_peaks(np.abs(sig), height=0.3 * np.max(np.abs(sig)), distance=max(2, dist))
    return float(np.median(np.diff(time_ms[peaks])))


def _tile_aligned(template, onset_ms, period_ms, n_samples, dt_ms=DT_MS):
    """Repeats template at every multiple of period without any interpolation or stretching."""
    out = np.zeros((n_samples, template.shape[1]))
    template_len = template.shape[0]
    k = int(np.floor((0 - onset_ms) / period_ms)) - 1
    while True:
        start_idx = int(round((onset_ms + k * period_ms) / dt_ms))
        end_idx = start_idx + template_len
        if start_idx >= n_samples:
            break
        if end_idx > 0:
            seg_start = max(0, -start_idx)
            seg_end = template_len - max(0, end_idx - n_samples)
            out_start, out_end = max(0, start_idx), min(n_samples, end_idx)
            out[out_start:out_end, :] += template[seg_start:seg_end, :]
        k += 1
    return out


def build_prolonged_dataset_if_needed():
    """Builds the 200-beat mixed + pure-fetal recordings"""
    if os.path.exists(PROLONGED_FETAL_FILE) and os.path.exists(PROLONGED_MIXED_FILE):
        print(f"Found existing {PROLONGED_FETAL_FILE} / {PROLONGED_MIXED_FILE} -- skipping rebuild.")
        return

    print(f"Building {TARGET_N_MAT_BEATS}-maternal-beat recording from "
          f"{ORIG_FETAL_FILE} and {ORIG_MIXED_FILE} ...")
    fetal = _load_leads(ORIG_FETAL_FILE)
    mixed = _load_leads(ORIG_MIXED_FILE)
    time_ms_src = mixed["time_ms"].values
    fs_src = 1000.0 / DT_MS

    # Maternal-only signal = mixed - fetal (the two source files share
    # an identical fetal component, so this recovers it exactly)
    maternal_recovered = mixed.copy()
    for c in ALL_LEADS:
        maternal_recovered[c] = mixed[c] - fetal[c]

    mat_period_ms = _measure_period_ms(maternal_recovered["V5"].values, time_ms_src, fs_src, 0.4)
    fet_period_ms = _measure_period_ms(fetal["V5"].values, time_ms_src, fs_src, 0.2)
    print(f"  Maternal rate: {60000/mat_period_ms:.2f} bpm | Fetal rate: {60000/fet_period_ms:.2f} bpm")

    mat_template, mat_onset_ms = _extract_real_complex(
        maternal_recovered, time_ms_src, REF_PEAK_MAT_MS, DEBOUNCE_MAT_SAMP)
    fet_template, fet_onset_ms = _extract_real_complex(
        fetal, time_ms_src, REF_PEAK_FET_MS, DEBOUNCE_FET_SAMP)

    total_duration_ms = TARGET_N_MAT_BEATS * mat_period_ms
    n_samples = int(round(total_duration_ms / DT_MS)) + 1
    t_ms = np.arange(n_samples) * DT_MS

    mat_strip = _tile_aligned(mat_template, mat_onset_ms, mat_period_ms, n_samples)
    fet_strip = _tile_aligned(fet_template, fet_onset_ms, fet_period_ms, n_samples)

    np.savetxt(PROLONGED_FETAL_FILE, np.column_stack([t_ms, fet_strip]), fmt="%.6g", delimiter=" ")
    np.savetxt(PROLONGED_MIXED_FILE, np.column_stack([t_ms, mat_strip + fet_strip]), fmt="%.6g", delimiter=" ")
    print(f"  Saved {PROLONGED_FETAL_FILE} and {PROLONGED_MIXED_FILE} "
          f"({total_duration_ms/1000:.1f} s, {n_samples} samples)")


# ================================================================
# 2. RLS REGRESSION -- warm-started (as in Scenario 4) AND
#    cold-started (i.e. no OLS prior)
# ================================================================
def fit_rls_lambda(P, PC, lam, cold=False, delta=fc.RLS_DELTA):
    #                    ^^^^^^^^^^ COLD-START RLS switch (default False = warm-started)
    X = np.column_stack([PC["M1"], PC["M2"]])
    results = {}
    for i, lead in enumerate(LEADS):
        y = P["norm_abd"][:, i]
        w_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        # ---- COLD-START RLS: initial weights = zero instead of the OLS ----
        # ---- solution (Scenario 4 / warm-start always uses w_ols here) ----
        w_init = np.zeros(2) if cold else w_ols
        y_hat, residual, weights = fc._rls_core(X, y, w_init=w_init, lam=lam, delta=delta)
        results[lead] = dict(y_hat=y_hat, residual=residual, weights=weights, w_ols=w_ols)
    return dict(method="RLS", X=X, results=results, lam=lam, delta=delta)


# ================================================================
# 3. STATISTICS 
# ================================================================
def compute_or_load_stats(P, W, fetal_mV, gt_F, tag, outdir):
    cache_path = f"{outdir}/{tag}_stats.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        for lead in LEADS:
            for k, v in raw[lead].items():
                if k.endswith("_ci") or k.endswith("_ci_pct"):
                    raw[lead][k] = tuple(v)
        print(f"  [cached] {cache_path}")
        return raw
    stats = fc.compute_full_stats(P, W, fetal_mV, gt_F, tag, outdir)
    with open(cache_path, "w") as f:
        json.dump(stats, f, indent=2, default=float)
    return stats


def print_mean_rows(all_stats, tag):
    """Prints one extra 'Mean' row (across leads F1/F2/F3) beneath
    fecg_common's own per-lead table, and returns those cross-lead
    means for use in the summary table / figure / CSV below."""
    hdr1 = (f"{'Lead':<5}{'TP':>5}{'FP':>5}{'FN':>5}{'SE(%)':>8}{'PPV(%)':>8}"
            f"{'F1(%)':>8}{'bias(bpm)':>11}{'LoA(bpm)':>20}{'n_pairs':>9}")
    hdr2 = (f"{'Lead':<5}{'r':>8}{'r 95% CI':>18}{'r p-val':>10}{'RMSE(mV)':>11}"
            f"{'RMSE 95% CI':>18}{'PRD(%)':>9}{'PRD 95% CI':>16}"
            f"{'SNR_bef':>9}{'SNR_aft':>9}{'dSNR':>8}{'dSNR 95% CI':>16}{'dSNR p':>8}")

    def m(key):
        return np.mean([all_stats[l][key] for l in LEADS])

    def m_ci(key):
        return (np.mean([all_stats[l][key][0] for l in LEADS]),
                np.mean([all_stats[l][key][1] for l in LEADS]))

    loa_str = f"[{m('ba_loa_lower_bpm'):.2f}, {m('ba_loa_upper_bpm'):.2f}]"
    print(f"\n  Mean across F1/F2/F3 -- {tag}")
    print("-" * len(hdr1))
    print(hdr1)
    print(f"{'Mean':<5}{m('tp'):>5.1f}{m('fp'):>5.1f}{m('fn'):>5.1f}"
          f"{m('sensitivity_pct'):>8.2f}{m('ppv_pct'):>8.2f}{m('f1_pct'):>8.2f}"
          f"{m('ba_bias_bpm'):>11.2f}{loa_str:>20}{m('ba_n_pairs'):>9.1f}")

    r_ci, rmse_ci, prd_ci, dsnr_ci = m_ci("pearson_r_ci"), m_ci("rmse_ci"), m_ci("prd_ci"), m_ci("dsnr_ci")
    print()
    print(hdr2)
    print(f"{'Mean':<5}{m('pearson_r'):>8.3f}{f'[{r_ci[0]:.3f}, {r_ci[1]:.3f}]':>18}"
          f"{m('pearson_p'):>10.2e}{m('rmse'):>11.5f}{f'[{rmse_ci[0]:.5f}, {rmse_ci[1]:.5f}]':>18}"
          f"{m('prd'):>9.2f}{f'[{prd_ci[0]:.2f}, {prd_ci[1]:.2f}]':>16}"
          f"{m('snr_before'):>9.2f}{m('snr_after'):>9.2f}{m('dsnr'):>8.2f}"
          f"{f'[{dsnr_ci[0]:.2f}, {dsnr_ci[1]:.2f}]':>16}{m('dsnr_p'):>8.3f}")

    return dict(dsnr=m("dsnr"), dsnr_ci=dsnr_ci, rmse=m("rmse"), rmse_ci=rmse_ci,
                prd=m("prd"), prd_ci=prd_ci, pearson_r=m("pearson_r"), pearson_r_ci=r_ci,
                snr_after=m("snr_after"), f1_pct=m("f1_pct"))


# ================================================================
# 4. FIGURE
# ================================================================
def make_figure(warm, cold, mean_ols, fs, outdir):
    mem_horizon_s = [(1.0 / (1.0 - l)) / fs for l in LAMBDAS]

    def panel(ax, key, ylabel, color):
        ax.plot(mem_horizon_s, [warm[l][key] for l in LAMBDAS], linestyle=":", color=color,
                alpha=0.45, linewidth=2.5, marker="o", markersize=5, label="Warm-Started RLS")
        # ---- COLD-START RLS series plotted here (solid line) ----
        ax.plot(mem_horizon_s, [cold[l][key] for l in LAMBDAS], linestyle="-", color=color,
                alpha=1.0, linewidth=2, marker="s", markersize=5, label="Cold-Started RLS")
        ax.axhline(mean_ols[key], color=color, linestyle=":", linewidth=2.2, label="OLS")
        ax.set_xscale("log")
        ax.set_xticks(mem_horizon_s)
        ax.set_xticklabels([str(l) for l in LAMBDAS], rotation=45, ha="right")
        ax.tick_params(axis="both", which="major", labelsize=13)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        ax.set_xlabel("Forgetting Factor \u03bb", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    panel(axes[0, 0], "rmse", "Mean RMSE (mV)", "tab:blue")
    panel(axes[0, 1], "pearson_r", "Mean Pearson r", "tab:green")
    panel(axes[1, 0], "dsnr", "Mean dSNR (dB)", "tab:orange")
    panel(axes[1, 1], "prd", "Mean PRD (%)", "tab:red")
    fig.suptitle("Cold-started versus Warm-started RLS", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fpath = f"{outdir}/warmstart_vs_coldstart_4panel.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {fpath}")


# ================================================================
# 5. CSV EXPORT 
# ================================================================
def write_csv(mean_ols, stats_ols, warm, warm_full, cold, cold_full, outdir):
    fieldnames = [
        "scenario", "start_type", "forgetting_factor",
        "sensitivity_pct", "sensitivity_ci_lo", "sensitivity_ci_hi",
        "ppv_pct", "ppv_ci_lo", "ppv_ci_hi",
        "f1_pct", "f1_ci_lo", "f1_ci_hi",
        "pearson_r", "pearson_r_ci_lo", "pearson_r_ci_hi",
        "rmse_mV", "rmse_ci_lo", "rmse_ci_hi",
        "prd_pct", "prd_ci_lo", "prd_ci_hi",
        "dsnr_dB", "dsnr_ci_lo", "dsnr_ci_hi",
    ]
    csv_path = f"{outdir}/warm_vs_cold_vs_ols_full_stats.csv"

    def row_from(writer, name, start_type, lam, d, stats_full):
        def m(key):
            return np.mean([stats_full[l][key] for l in LEADS])

        def m_ci(key):
            return (np.mean([stats_full[l][key][0] for l in LEADS]),
                    np.mean([stats_full[l][key][1] for l in LEADS]))

        se_ci, ppv_ci, f1_ci = m_ci("sensitivity_ci_pct"), m_ci("ppv_ci_pct"), m_ci("f1_ci_pct")
        writer.writerow(dict(
            scenario=name, start_type=start_type, forgetting_factor=lam if lam is not None else "",
            sensitivity_pct=round(m("sensitivity_pct"), 4), sensitivity_ci_lo=round(se_ci[0], 4),
            sensitivity_ci_hi=round(se_ci[1], 4), ppv_pct=round(m("ppv_pct"), 4),
            ppv_ci_lo=round(ppv_ci[0], 4), ppv_ci_hi=round(ppv_ci[1], 4),
            f1_pct=round(d["f1_pct"], 4), f1_ci_lo=round(f1_ci[0], 4), f1_ci_hi=round(f1_ci[1], 4),
            pearson_r=round(d["pearson_r"], 5), pearson_r_ci_lo=round(d["pearson_r_ci"][0], 5),
            pearson_r_ci_hi=round(d["pearson_r_ci"][1], 5), rmse_mV=round(d["rmse"], 6),
            rmse_ci_lo=round(d["rmse_ci"][0], 6), rmse_ci_hi=round(d["rmse_ci"][1], 6),
            prd_pct=round(d["prd"], 4), prd_ci_lo=round(d["prd_ci"][0], 4),
            prd_ci_hi=round(d["prd_ci"][1], 4), dsnr_dB=round(d["dsnr"], 4),
            dsnr_ci_lo=round(d["dsnr_ci"][0], 4), dsnr_ci_hi=round(d["dsnr_ci"][1], 4),
        ))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        row_from(writer, "OLS", "n/a", None, mean_ols, stats_ols)
        for lam in LAMBDAS:
            row_from(writer, "RLS", "warm", lam, warm[lam], warm_full[lam])
        # ---- COLD-START RLS rows written here ----
        for lam in LAMBDAS:
            row_from(writer, "RLS", "cold", lam, cold[lam], cold_full[lam])
    print(f"Saved {csv_path}")


# ================================================================
# MAIN
# ================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    build_prolonged_dataset_if_needed()

    print("\n" + "=" * 100)
    print("DATA PROVENANCE")
    print("=" * 100)
    print(f"  {PROLONGED_MIXED_FILE}  <-  built from {ORIG_MIXED_FILE}")
    print(f"  {PROLONGED_FETAL_FILE}  <-  built from {ORIG_FETAL_FILE}")

    
    P = fc.load_and_preprocess(PROLONGED_MIXED_FILE, REF_LEADS)
    fs = P["fs"]
    CR = fc.detect_consensus_peaks(P)
    W = fc.extract_qrs_windows(P, CR["consensus_r_peak_times"])
    PC = fc.run_pca_windowed(P, W, n_retain=2)
    gt_F = fc.load_ground_truth(PROLONGED_FETAL_FILE)
    print(f"\nfs={fs:.1f} Hz  N_samples={P['N_samples']}  "
          f"consensus maternal beats={len(CR['consensus_r_peak_times'])}  "
          f"PCA var(PC1+PC2)={PC['cumulative_variance'][1]*100:.2f}%")

    # ---- OLS: single reference, used on every panel of the figure ----
    print("\n" + "#" * 100 + "\n# OLS (5-channel)\n" + "#" * 100)
    REG_OLS = fc.fit_maternal_template_ols(P, PC)
    _, fetal_mV, _ = fc.to_mV(P, REG_OLS)
    stats_ols = compute_or_load_stats(P, W, fetal_mV, gt_F, "OLS_200beat", OUTDIR)
    mean_ols = print_mean_rows(stats_ols, "OLS_200beat")

    # ---- RLS: warm-started and cold-started, 7 forgetting factors each ----
    warm, cold, warm_full, cold_full = {}, {}, {}, {}
    for cold_flag, store, store_full, label in [(False, warm, warm_full, "warm-started"),
                                                 (True, cold, cold_full, "cold-started")]:
                                                 # ^^^^ COLD-START RLS branch (cold_flag=True)
        for lam in LAMBDAS:
            tag = f"RLS_200beat_{'cold' if cold_flag else 'warm'}_lam{lam}"
            print("\n" + "#" * 100 + f"\n# RLS (5-channel), {label}, lambda={lam}\n" + "#" * 100)
            REG = fit_rls_lambda(P, PC, lam, cold=cold_flag)
            #                                  ^^^^^^^^^^^^^^^^^ COLD-START RLS triggered here
            _, fetal_mV, _ = fc.to_mV(P, REG)
            stats = compute_or_load_stats(P, W, fetal_mV, gt_F, tag, OUTDIR)
            store[lam] = print_mean_rows(stats, tag)
            store_full[lam] = stats

    # ---- consolidated summary table ----
    print("\n" + "=" * 100)
    print("CONSOLIDATED SUMMARY -- mean across F1/F2/F3 (point [95% CI])")
    print("=" * 100)
    fmt = lambda v, ci, p: f"{v:.{p}f} [{ci[0]:.{p}f},{ci[1]:.{p}f}]"
    print(f"{'Scenario':<22}{'r':>20}{'RMSE(mV)':>24}{'PRD(%)':>20}{'dSNR(dB)':>20}")
    print(f"{'OLS':<22}{fmt(mean_ols['pearson_r'], mean_ols['pearson_r_ci'], 4):>20}"
          f"{fmt(mean_ols['rmse'], mean_ols['rmse_ci'], 5):>24}"
          f"{fmt(mean_ols['prd'], mean_ols['prd_ci'], 2):>20}"
          f"{fmt(mean_ols['dsnr'], mean_ols['dsnr_ci'], 2):>20}")
    for lam in LAMBDAS:
        w = warm[lam]
        print(f"{'RLS warm lam=' + str(lam):<22}{fmt(w['pearson_r'], w['pearson_r_ci'], 4):>20}"
              f"{fmt(w['rmse'], w['rmse_ci'], 5):>24}{fmt(w['prd'], w['prd_ci'], 2):>20}"
              f"{fmt(w['dsnr'], w['dsnr_ci'], 2):>20}")
    # ---- COLD-START RLS rows printed here ----
    for lam in LAMBDAS:
        c = cold[lam]
        print(f"{'RLS cold lam=' + str(lam):<22}{fmt(c['pearson_r'], c['pearson_r_ci'], 4):>20}"
              f"{fmt(c['rmse'], c['rmse_ci'], 5):>24}{fmt(c['prd'], c['prd_ci'], 2):>20}"
              f"{fmt(c['dsnr'], c['dsnr_ci'], 2):>20}")

    make_figure(warm, cold, mean_ols, fs, OUTDIR)
    write_csv(mean_ols, stats_ols, warm, warm_full, cold, cold_full, OUTDIR)
    print("\nDONE")


if __name__ == "__main__":
    main()

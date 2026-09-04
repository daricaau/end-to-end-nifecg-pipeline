"""
SCENARIO 1 -- 3-channel reference (V2, V5, V8), OLS regression.

Runs the shared pipeline in fecg_common.py end-to-end.
"""
import os
import json
import fecg_common as fc

TAG = "S1_3ch_OLS"
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

REF_LEADS = ['V2', 'V5', 'V8']

print("=" * 70)
print(f"{TAG}: 3-channel reference, OLS maternal-template regression")
print("=" * 70)

# 1. Load & preprocess (bandpass -> notch -> joint z-score)
P = fc.load_and_preprocess("phie_with_maternal_interference.txt", REF_LEADS)
print(f"fs={P['fs']:.1f} Hz  N_samples={P['N_samples']}  ref_leads={REF_LEADS}")

# 2. Maternal R-peak consensus (Pan-Tompkins on V2/V5/V8)
CR = fc.detect_consensus_peaks(P)
print(f"Consensus maternal beats detected: {len(CR['consensus_r_peak_times'])}")
print(f"Cross-lead timing std (mean over beats): {CR['beat_std'].mean():.2f} ms")

# 3. QRS-centred windows for PCA fitting
W = fc.extract_qrs_windows(P, CR['consensus_r_peak_times'])
print(f"PCA windows: {len(W['window_indices'])}  window length: {W['W_pre']+W['W_post']} samples")

# 4. Windowed-fit PCA -> maternal components M1, M2
PC = fc.run_pca_windowed(P, W, n_retain=2)
print(f"Explained variance PC1+PC2: {PC['cumulative_variance'][1]*100:.2f}%")

# 5. OLS maternal-template regression  <-- scenario-specific step
REG = fc.fit_maternal_template_ols(P, PC)
for lead in fc.ABDO_LEADS:
    w = REG['results'][lead]['weights']
    print(f"  {lead}: w_PC1={w[0]:+.4f}  w_PC2={w[1]:+.4f}")

# 6. Convert template/residual back to mV; load ground truth
maternal_mV, fetal_mV, abd_mV = fc.to_mV(P, REG)
gt_F = fc.load_ground_truth("phie_fetal_only.txt")

# 7. Full statistics (detection, heart-rate agreement, waveform fidelity)
stats = fc.compute_full_stats(P, W, fetal_mV, gt_F, TAG, OUTDIR)

# 8. Plots -- the same 9-figure set produced by every scenario
#    (the Bland-Altman figure, a 10th figure, is saved by compute_full_stats)
paths = fc.make_all_plots(P, CR, W, PC, REG, fetal_mV, maternal_mV, abd_mV, gt_F, TAG, OUTDIR)
print("\nSaved figures:")
for p in paths:
    print(f"  {p}")

with open(f"{OUTDIR}/{TAG}_stats.json", "w") as f:
    json.dump(stats, f, indent=2, default=float)
print(f"\nStats saved to {OUTDIR}/{TAG}_stats.json")
print(f"\n{TAG} COMPLETE")

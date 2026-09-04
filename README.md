# End to End NIFECG Project

**MSc Thesis Project**

This repository contains the full codebase for MSc Thesis on extracting the
fetal ECG, evaluating robustness, and turning the result into a prototype dashboard

## Table of contents

1. [Repository structure](#repository-structure)
2. [Data files](#data-files)
3. [The core pipeline — `fecg_common.py`](#the-core-pipeline--fecg_commonpy)
4. [How the thesis objectives map to the code](#how-the-thesis-objectives-map-to-the-code)
   - [Objectives 1 & 2 — the four extraction scenarios](#objectives-1--2--the-four-extraction-scenarios)
   - [Objective 3 — robustness testing (3.1 / 3.2 / 3.3)](#objective-3--robustness-testing)
   - [Objective 4 — validation on a real recording](#objective-4--validation-on-a-real-recording)
   - [Objective 5 — clinical dashboard app](#objective-5--the-clinical-dashboard-app)
5. [Setup](#setup)
6. [Running each part of the project](#running-each-part-of-the-project)
7. [Outputs](#outputs)
8. [Limitations & disclaimer](#limitations--disclaimer)

---

## Repository structure

```
.
├── fecg_common.py                  # Core shared pipeline — used by every other script
│
├── phie_fetal_only.txt             # Simulated recording: fetal ECG only (ground truth)
├── phie_with_maternal_interference.txt   # Simulated recording: fetal + maternal ECG mixed
├── Pos1_lying.xlsx                 # Real NIFECG recording from pregnant individual
│
├── scenario1_3ch_OLS.py            # Objective 1/2 — 3-channel reference, OLS
├── scenario2_3ch_RLS.py            # Objective 1/2 — 3-channel reference, RLS
├── scenario3_5ch_OLS.py            # Objective 1/2 — 5-channel reference, OLS
├── scenario4_5ch_RLS.py            # Objective 1/2 — 5-channel reference, RLS
│
├── 3_1.py                          # Objective 3.1 — warm-start vs cold-start RLS
├── 3_2.py                          # Objective 3.2 — varying maternal:fetal heart-rate ratio
├── 3_3.py                          # Objective 3.3 — robustness to injected noise
│
├── 4.py                            # Objective 4 — cold-start extraction on a real recording
│
├── O5.py                           # Objective 5 — extraction +  FHR assessment engine
├── server.py                       # Objective 5 — Flask backend serving the dashboard
├── index.html                      # Objective 5 — clinical dashboard front end
│
└── outputs/ , outputs_*/           # Generated automatically when scripts are run
```

## Data files- Available on request.

| File | Type | Description |
|---|---|---|
| `phie_fetal_only.txt` | Simulated | Only the fetal ECG component. This is the **ground truth** used to score how accurately each method recovers the fetal signal. |
| `phie_with_maternal_interference.txt` | Simulated | The same recording, but mixed with maternal input. This is the input every scenario script tries to "clean". |
| `Pos1_lying.xlsx` | Real recording |  No fetal-only ground truth exists for this file — it is used in Objective 4 and Objective 5 to test the pipeline on real, noisier data. |

Both simulated text files share the same file layout (whitespace-separated columns:
`time_ms, F1, F2, F3, V2, V5, V8, RA, LA, GND, LL`), and because they were generated
from the same underlying simulation, subtracting one from the other exactly
recovers the pure maternal signal — used in Objective 3 to build longer
test recordings.

## The core pipeline — `fecg_common.py`

Every scenario script and both apps import from `fecg_common.py` for the
actual signal-processing and statistics. 

`fecg_common.py` performs the following tasks:
1. **Load & preprocess** — reads the recording, applies a 3–80 Hz band-pass filter
   and a 50 Hz mains-notch filter, then z-score normalises every channel.
2. **Maternal R-peak detection** — Pan–Tompkins QRS detector runs independently on the thoracic/limb reference leads, combined
   into a single consensus beat time per heartbeat (the median across leads, so one
   noisy lead cannot throw off the timing).
3. **Windowed PCA** — Principal Component Analysis fitted on short windows
   centred on each maternal heartbeat then projected across the whole recording to give 
   "maternal template" components.
4. **Maternal template regression** — two interchangeable regression engines that
   both predict each abdominal lead from principal components and treat the
   leftover residual as the extracted fetal ECG:
   - **OLS** (Ordinary Least Squares) — a single, batch fit.
   - **RLS** (Recursive Least Squares) — an adaptive filter that continuously
     updates its fit sample-by-sample, so it can track slow changes in the
     maternal signal. It can be **warm-started** from the OLS solution or
     **cold-started** from zero.
6. **Fetal QRS detection** — Pan-Tompkins-style detector tuned
   for the fetal heart rate range, run on the extracted residual.
7. **Statistics** — sensitivity/PPV/F1 for beat detection, Bland–Altman bias
   and RMSE / PRD / SNR / Pearson's r for waveform-shape accuracy — all with
   moving-block bootstrap confidence intervals appropriate for time-series data.
9. **Plotting** — the same set of diagnostic figures (QRS detection, PCA scree
   plot, 3D PCA signal space, lead loadings, maternal template vs original,
   residual fetal ECG, residual vs ground truth, etc.) for every scenario.

## How the thesis objectives map to the code

### Objectives 1 & 2 — the four extraction scenarios

**Files:** `scenario1_3ch_OLS.py`, `scenario2_3ch_RLS.py`, `scenario3_5ch_OLS.py`,
`scenario4_5ch_RLS.py`

These four scripts implement and compare the core extraction method under a 2×2
design: **3-channel vs 5-channel** reference leads, analysed with **OLS vs RLS**
regression. Each script runs the full `fecg_common.py` pipeline end-to-end on the
simulated mixed recording, scores the result against the fetal-only ground truth,
and saves the full statistics (JSON) and figure set for that scenario. Together,
these four scripts constitute Objectives 1 and 2: establishing and
comparing baseline extraction performance.

| Script | Reference leads | Regression |
|---|---|---|
| `scenario1_3ch_OLS.py` | V2, V5, V8 | OLS |
| `scenario2_3ch_RLS.py` | V2, V5, V8 | RLS (warm-started) |
| `scenario3_5ch_OLS.py` | V2, V5, V8, RA, LA | OLS |
| `scenario4_5ch_RLS.py` | V2, V5, V8, RA, LA | RLS (warm-started) |

### Objective 3 — robustness testing

Objective 3 stress-tests the method beyond the short, clean simulated recording
used in Objectives 1–2. It is split into three sub-scripts, matching thesis
sections 3.1, 3.2 and 3.3.

**`3_1.py` — Warm-start vs. cold-start RLS, across forgetting factors**
The original recording is only a few heartbeats long — too short to properly judge
RLS. This script first extends it to 200 maternal heartbeats by
cutting one real, un-warped heartbeat template out of the source recording and
tiling it at the recording's own natural rate (no stretching or averaging), then
compares OLS against RLS run both **warm-started** (from the OLS solution, as in
Objective 2) and **cold-started** (from zero), swept across seven RLS forgetting
factors (0.9970 to 0.9999, intervals in 0.0005).

**`3_2.py` — Altering fetal:maternal heart rate ratio**
Builds five new test recordings, each re-spacing the same real maternal and fetal
heartbeat templates to hit a specific maternal:fetal heart-rate ratios, then runs OLS 
and cold-started RLS extraction on each and reports how
accuracy changes as the two heart rates converge or diverge.

**`3_3.py` — Robustness to realistic noise/interference**
Extends the recording to 200 beats (as in 3.1), then injects four physiologically
motivated disturbances individually and in combination, producing six test
recordings in total (five noise conditions + a clean baseline):

1. **Respiration** 
2. **Electrode motion** 
3. **EMG-like bursts** 
4. **Ectopic maternal beats** 

Each of the six recordings is then run through OLS and cold-started RLS (again
swept across seven forgetting factors), producing a full results table and a
figure tracking how the regression coefficients respond to the disturbances.

### Objective 4 — validation on a real recording

**File:** `4.py`

Runs the same 5-channel, cold-started RLS extraction on a **real recording** (`Pos1_lying.xlsx`)
instead of the simulated files. Because there is no fetal-only ground truth for a
real recording, this script cannot score RMSE/PRD/SNR against a reference — instead
it reports what can be checked on real data: how many fetal R-peaks are detected
per fetal lead, and the resulting mean/median fetal heart rate, alongside the
extracted-ECG-over-original figure.

### Objective 5 — the clinical dashboard app

**Files:** `O5.py`, `server.py`, `index.html` — **these three files must be kept in
the same folder** (along with `fecg_common.py` and, for the bundled demo, `Pos1_lying.xlsx`)
for the app to run.

Objective 5 translate the extraction pipeline into a clinically usable tool:

- **`O5.py`** is the analysis engine. It runs the same cold-start, 5-channel RLS
  extraction as Objective 4, then goes further and derives fetal
  monitoring outputs from the extracted signal:
  - fetal heart rate (FHR) per lead, and cross-lead concordance/confidence
  - accelerations and decelerations (heart-rate rises/falls of >10 bpm within a
    trailing 15-second window)
  - beat-to-beat heart-rate variability (HRV)
  - a baseline FHR, baseline variability, and deceleration classification using
    NICE-style intrapartum banding (Reassuring / Non-reassuring / Abnormal /
    Indeterminate)

  It accepts either the real `.xlsx` recording format or the simulated `.txt`
  format automatically— **this is an engineering prototype, not a
  validated clinical device.**

- **`server.py`** is a small Flask backend that runs `O5.analyze_file()` on a
  recording, caches the result, and exposes it as JSON over a few endpoints
  (`/api/latest`, `/api/analyze`, `/api/history`, `/api/patient`, `/api/ecg.png`)
  for the dashboard's JavaScript to consume. It also accepts an uploaded file to
  re-run the analysis on, and keeps a simple history log of past analyses.

- **`index.html`** is the front-end: a single-page clinical monitoring dashboard
  (the "NIFECG Monitoring Dashboard") that displays the live fetal ECG strip,
  heart-rate badge, variability, accelerations/decelerations, and an editable
  patient header, all populated from `server.py`'s API.

## Setup

**Requirements:** Python 3.9+ recommended.

```bash
pip install numpy pandas scipy scikit-learn matplotlib openpyxl flask
```

- `openpyxl` is required for `pandas` to read `Pos1_lying.xlsx`.
- `flask` is only required to run the Objective 5 dashboard (`server.py`); the
  analysis scripts (Objectives 1–4) do not need it.

Clone the repository and make sure the data files (`phie_fetal_only.txt`,
`phie_with_maternal_interference.txt`, `Pos1_lying.xlsx`) are in the same folder as
the scripts you want to run — each script expects them by relative filename.

## Running each part of the project

**Objectives 1 & 2** (run each independently; each is self-contained):
```bash
python scenario1_3ch_OLS.py
python scenario2_3ch_RLS.py
python scenario3_5ch_OLS.py
python scenario4_5ch_RLS.py
```

**Objective 3:**
```bash
python 3_1.py   # warm vs cold-start RLS
python 3_2.py   # maternal:fetal heart-rate ratio 
python 3_3.py   # noise-injection for 5 scenarios
```
`3_1.py` and `3_3.py` expect the source data under a `data/` subfolder
(`data/phie_fetal_only.txt`, `data/phie_with_maternal_interference.txt`) and will
automatically build the extended 200-beat recordings from those on first run.
`3_2.py` reads the source files directly from the working folder.

**Objective 4:**
```bash
python 4.py
```

**Objective 5 — the dashboard app.** `O5.py`, `server.py`, `index.html` and
`fecg_common.py` must all be in the same folder, along with a recording file
(`Pos1_lying.xlsx` by default):
```bash
python server.py
```
Then open **http://localhost:5000** in a browser. To point the server at a
different recording, set the `O5_RECORDING` environment variable before running,
or use the dashboard's upload/re-analyze control. `O5.py` can also be run directly
from the command line, independent of the server, for a single-recording analysis:
```bash
python O5.py path/to/recording.xlsx --outdir results/
```

## Outputs

Every script creates its own output folder (e.g. `outputs/`, `outputs_warm_vs_cold/`,
`outputs_3_3/`, `O5_outputs/`) the first time it is run.

## Limitations & disclaimer

This is an engineering research pipeline developed for an MSc thesis. It has
been evaluated on simulated recordings with known ground truth, and a real recording without ground truth. The Objective 5 dashboard's
clinical classifications follow NICE-style intrapartum banding conventions.

**This tool is not a validated medical device and must not be used for clinical
decision-making.** It is a demonstration of the underlying signal-processing and
data pipeline built for this MSc thesis.

"""
server.py -- backend service connecting O5.py's analysis pipeline to
the index.html clinical dashboard.
=================================================================
Run:
    pip install flask
    python server.py
    -> open http://localhost:5000

Overview
--------
- Serves index.html (and any other files next to it) as static content.
- Runs O5.analyze_file() on a recording and caches the result in memory.
- Exposes JSON endpoints that the dashboard's JavaScript calls to
  populate the interface with live values:

    GET  /api/latest      -> most recent result, shaped for the dashboard
                              (runs an initial analysis on first call)
    POST /api/analyze     -> re-run analysis, optionally on an uploaded
                              file (multipart/form-data, field "file");
                              falls back to O5_RECORDING if none is posted
    GET  /api/history     -> accumulated {date, fhr_bpm, status} log,
                              one row per analysis run so far
    GET  /api/ecg.png     -> the most recently generated fetal-ECG
                              waveform image (used by the live ECG panel)

This implementation favours simplicity over scale: one recording is
processed synchronously into a JSON report, cached in memory, with
reading history persisted to a flat JSON file rather than a database.
For continuous deployment, RECORDING_PATH would be replaced with a
live acquisition feed and history.json with a proper data store.
"""
import os
import json
import datetime
import sys

from flask import Flask, jsonify, send_from_directory, request, send_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import O5

APP_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDING_PATH = os.environ.get("O5_RECORDING", os.path.join(APP_DIR, "Pos1_lying.xlsx"))
HISTORY_PATH = os.path.join(APP_DIR, "history.json")
PATIENT_PATH = os.path.join(APP_DIR, "patient.json")
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")

DEFAULT_PATIENT = {"name": "Jane Doe", "hospitalNo": "123-456-789", "gestation": "38 weeks 0 days"}
PATIENT_FIELDS = set(DEFAULT_PATIENT.keys())

LOW_CONFIDENCE_THRESHOLD_PCT = 80.0   # below this, the frontend shows a caution box

STATE = {"report": None}

app = Flask(__name__, static_folder=APP_DIR, static_url_path="")


# ============================================================
# Mapping O5 analysis reports onto dashboard JSON
# ============================================================
def _badge(label):
    """Maps O5's Reassuring / Non-reassuring / Abnormal / Indeterminate
    labels onto one consistent {text, tone} pair. `tone` is either
    "good" or "warning" -- the frontend uses this single tone to set a
    badge's background, text colour, and status dot together, so the
    three can never fall out of sync with one another."""
    label = label or "Indeterminate"
    key = label.split(" ")[0]
    text = {"Reassuring": "Reassuring", "Non-reassuring": "Non-Reassuring",
            "Abnormal": "Abnormal", "Indeterminate": "Indeterminate"}.get(key, key)
    tone = "good" if key == "Reassuring" else "warning"
    return {"text": text, "tone": tone}


def _confidence_status(pct):
    if pct is None or pct != pct:
        return "Unknown"
    if pct >= 80:
        return "Excellent"
    if pct >= 60:
        return "Good"
    if pct >= 40:
        return "Fair"
    return "Poor"


def _num(x, digits=0):
    if x is None or x != x:  # NaN
        return None
    return round(x, digits) if digits else round(x)


def report_to_dashboard_json(report):
    conc = report["concordance"]
    cls = report["classification"]
    hrv = report["hrv"]
    confidence_pct = _num(conc["confidence_pct"], 1)
    return {
        "generated_at": report.get("generated_at"),
        "input_file": os.path.basename(report["input_file"]),
        "confidence_pct": confidence_pct,
        "confidence_status": _confidence_status(conc["confidence_pct"]),
        "low_confidence": confidence_pct is not None and confidence_pct < LOW_CONFIDENCE_THRESHOLD_PCT,
        "fhr_bpm": _num(report["baseline_fhr_bpm"]),
        "fhr_badge": _badge(cls["baseline_heart_rate"]["label"]),
        "rr_variability_bpm": _num(hrv["mean_hrv_bpm"], 1),
        "rr_variability_badge": _badge(cls["baseline_variability"]["label"]),
        "accelerations_present": report["accelerations"]["present"],
        "decelerations_present": report["decelerations"]["present"],
        "decelerations_badge": _badge(cls["decelerations"]["label"]),
        "overall": cls["overall"],
    }


# ============================================================
# Analysis execution and history logging
# ============================================================
def run_analysis(path):
    outdir = os.path.join(APP_DIR, "latest_analysis")
    report = O5.analyze_file(path, outdir=outdir, verbose=False)
    report["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    STATE["report"] = report
    _append_history(report)
    return report


def _append_history(report):
    history = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            history = json.load(f)
    history.append({
        "date": report["generated_at"],
        "fhr_bpm": _num(report["baseline_fhr_bpm"]),
        "status": report["classification"]["overall"],
    })
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


# ============================================================
# HTTP routes
# ============================================================
@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.route("/api/latest")
def latest():
    if STATE["report"] is None:
        try:
            run_analysis(RECORDING_PATH)
        except Exception as exc:
            return jsonify({"error": str(exc), "recording_path": RECORDING_PATH}), 500
    return jsonify(report_to_dashboard_json(STATE["report"]))


@app.route("/api/analyze", methods=["POST"])
def analyze():
    path = RECORDING_PATH
    if "file" in request.files and request.files["file"].filename:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        f = request.files["file"]
        path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(path)
    try:
        report = run_analysis(path)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(report_to_dashboard_json(report))


@app.route("/api/history")
def history():
    if not os.path.exists(HISTORY_PATH):
        return jsonify([])
    with open(HISTORY_PATH) as f:
        return jsonify(json.load(f))


@app.route("/api/patient", methods=["GET"])
def get_patient():
    """Returns the patient header fields (name, hospital number,
    gestation) shown in both the Monitor and History view headers."""
    if os.path.exists(PATIENT_PATH):
        with open(PATIENT_PATH) as f:
            saved = json.load(f)
        return jsonify({**DEFAULT_PATIENT, **saved})
    return jsonify(DEFAULT_PATIENT)


@app.route("/api/patient", methods=["POST"])
def set_patient():
    """Updates one patient field at a time (matches the frontend's
    save-on-blur behaviour: {"field": "name", "value": "..."}), or
    accepts a full {"name":..., "hospitalNo":..., "gestation":...}
    object. Values are capped at 200 chars and written to patient.json."""
    body = request.get_json(force=True, silent=True) or {}
    current = DEFAULT_PATIENT.copy()
    if os.path.exists(PATIENT_PATH):
        with open(PATIENT_PATH) as f:
            current.update(json.load(f))

    if "field" in body and body["field"] in PATIENT_FIELDS:
        current[body["field"]] = str(body.get("value", ""))[:200]
    else:
        for k in PATIENT_FIELDS:
            if k in body:
                current[k] = str(body[k])[:200]

    with open(PATIENT_PATH, "w") as f:
        json.dump(current, f, indent=2)
    return jsonify(current)


@app.route("/api/ecg.png")
def ecg_png():
    """Slim transparent strip -- sized for the dashboard's ECG widget."""
    if STATE["report"] is None:
        try:
            run_analysis(RECORDING_PATH)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
    fig_path = STATE["report"]["figures"]["ecg_strip"]
    return send_file(fig_path, mimetype="image/png")


@app.route("/api/ecg_full.png")
def ecg_full_png():
    """Full 3-lead residual-vs-original diagnostic figure (F1/F2/F3)."""
    if STATE["report"] is None:
        run_analysis(RECORDING_PATH)
    fig_path = STATE["report"]["figures"]["fetal_ecg"]
    return send_file(fig_path, mimetype="image/png")


if __name__ == "__main__":
    print(f"O5 recording source: {RECORDING_PATH}")
    if not os.path.exists(RECORDING_PATH):
        print(f"\nWARNING: '{RECORDING_PATH}' does not exist.")
        print("Put a recording file (e.g. Pos1_lying.xlsx) next to server.py,")
        print("or set O5_RECORDING=/path/to/your/file before running this script.\n")
    print("Serving on http://localhost:5000  (Ctrl+C to stop)")
    app.run(debug=False, port=5000)

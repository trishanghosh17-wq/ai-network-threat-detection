"""DDoS Detection Service - FastAPI backend.

Run:  DDOS_API_KEY=change-me uvicorn backend.main:app --port 8000
Docs: http://localhost:8000/docs   Console: http://localhost:8000/
"""
import csv, io, json, logging, os, secrets, sqlite3, threading, time
from contextlib import contextmanager
from typing import Optional

import joblib, numpy as np, pandas as pd, xgboost as xgb
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_KEY = os.getenv("DDOS_API_KEY", "")
ALLOW_ANON = os.getenv("DDOS_ALLOW_ANON") == "1"          # local dev only
THRESHOLD = float(os.getenv("DDOS_THRESHOLD", "0.5"))
HIGH = float(os.getenv("DDOS_HIGH_SEVERITY", "0.9"))
MAX_ROWS = int(os.getenv("DDOS_MAX_BATCH_ROWS", "100000"))
DB_PATH = os.getenv("DDOS_DB", os.path.join(ROOT, "data", "audit.db"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ddos")
if not API_KEY and not ALLOW_ANON:
    log.warning("DDOS_API_KEY is not set: all /api routes will reject requests.")

def _json(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return json.load(f)

MODEL = joblib.load(os.path.join(ROOT, "models", "ddos_xgboost_final.joblib"))
try:
    MODEL.named_steps["model"].set_params(n_jobs=-1)
except Exception:
    pass
# Enforce multicore XGBoost at inference when the loaded estimator supports it.
try:
    MODEL.named_steps["model"].set_params(n_jobs=-1)
except Exception:
    pass
FEATURES = [x.strip() for x in open(os.path.join(ROOT, "models", "features.txt")) if x.strip()]
STATS = _json("app", "feature_stats.json")
METRICS = _json("results", "final_metrics.json")
PRESETS = _json("results", "scenario_presets.json")
BOOSTER = MODEL.named_steps["model"].get_booster()
IMPUTER = MODEL[:-1]
MODEL_VERSION = os.getenv("DDOS_MODEL_VERSION", "xgboost-final-1")

# ---------------- audit store ----------------
_lock = threading.Lock()

@contextmanager
def db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _lock:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

with db() as c:
    c.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, source TEXT, label TEXT NOT NULL,
        prob REAL NOT NULL, severity TEXT NOT NULL, model_version TEXT, factors TEXT, features TEXT)""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_ts ON predictions(ts)")

# ---------------- scoring ----------------
def severity(p):
    return "high" if p >= HIGH else "medium" if p >= THRESHOLD else "none"

def score(X: pd.DataFrame):
    """Returns probabilities and exact per-feature contributions (XGBoost TreeSHAP, log-odds)."""
    prob = MODEL.predict_proba(X)[:, 1]
    contrib = BOOSTER.predict(xgb.DMatrix(IMPUTER.transform(X)), pred_contribs=True)[:, :-1]
    return prob, contrib

def top_factors(row, n=8):
    idx = np.argsort(-np.abs(row))[:n]
    return [{"feature": FEATURES[i], "contribution": round(float(row[i]), 4)} for i in idx]

def range_warnings(vals):
    out = []
    for f, v in vals.items():
        s = STATS.get(f)
        if s is not None and v is not None and (v < s["min"] or v > s["max"]):
            out.append(f"{f}={v:g} is outside the range seen in training ({s['min']:g} to {s['max']:g})")
    return out

def save(rows):
    with db() as c:
        c.executemany("INSERT INTO predictions(ts,source,label,prob,severity,model_version,factors,features) VALUES(?,?,?,?,?,?,?,?)", rows)

def record(source, label, p, factors, feats):
    return (time.time(), source, label, float(p), severity(p), MODEL_VERSION, json.dumps(factors), json.dumps(feats))

# ---------------- app ----------------
app = FastAPI(title="DDoS Detection Service", version="1.0.0")

def auth(x_api_key: Optional[str] = Header(default=None)):
    if ALLOW_ANON:
        return
    if not API_KEY or not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "Invalid or missing API key")

class FlowIn(BaseModel):
    features: dict[str, Optional[float]]
    source: str = Field(default="manual", max_length=64)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/model", dependencies=[Depends(auth)])
def model_info():
    return {"version": MODEL_VERSION, "threshold": THRESHOLD, "high_severity": HIGH, "features": FEATURES,
            "feature_stats": STATS, "holdout_metrics": METRICS,
            "limitations": "Trained on one public dataset (ISCX DDoS). No timestamp/session ID exists, so a time-based "
                           "holdout could not be verified. Validate on your own traffic before relying on it."}

@app.get("/api/presets", dependencies=[Depends(auth)])
def presets():
    return PRESETS

@app.post("/api/predict", dependencies=[Depends(auth)])
def predict(flow: FlowIn):
    missing = [f for f in FEATURES if f not in flow.features]
    if missing:
        raise HTTPException(422, {"message": "Missing features", "missing": missing})
    vals = {f: flow.features[f] for f in FEATURES}
    prob, contrib = score(pd.DataFrame([vals], columns=FEATURES, dtype=float))
    p = float(prob[0]); label = "DDoS" if p >= THRESHOLD else "BENIGN"
    factors = top_factors(contrib[0])
    save([record(flow.source, label, p, factors, vals)])
    return {"label": label, "prob_ddos": p, "severity": severity(p), "threshold": THRESHOLD,
            "factors": factors, "warnings": range_warnings(vals), "model_version": MODEL_VERSION}

@app.post("/api/predict/batch", dependencies=[Depends(auth)])
async def predict_batch(file: UploadFile = File(...), source: str = Query("batch", max_length=64)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "Upload a .csv file")
    df = pd.read_csv(io.BytesIO(await file.read()), low_memory=False)
    df.columns = df.columns.str.strip()
    if len(df) > MAX_ROWS:
        raise HTTPException(413, f"Too many rows ({len(df)}); limit is {MAX_ROWS}")
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise HTTPException(422, {"message": "Missing columns", "missing": missing})
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    prob, contrib = score(X)
    flagged = np.where(prob >= THRESHOLD)[0]
    save([record(source, "DDoS", prob[i], top_factors(contrib[i], 5), X.iloc[i].where(X.iloc[i].notna(), None).to_dict())
          for i in flagged[:5000]])
    out = {"rows": len(df), "ddos": int(len(flagged)), "benign": int(len(df) - len(flagged)),
           "high_severity": int((prob >= HIGH).sum()), "threshold": THRESHOLD,
           "invalid_cells": int(X.isna().sum().sum()),
           "top_flagged": [{"row": int(i), "prob_ddos": float(prob[i])} for i in flagged[np.argsort(-prob[flagged])][:100]]}
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col:
        y = df[label_col].astype(str).str.strip().str.upper().eq("DDOS").to_numpy()
        pr = prob >= THRESHOLD
        out["vs_labels"] = {"accuracy": float((y == pr).mean()), "tp": int((y & pr).sum()), "fp": int((~y & pr).sum()),
                            "fn": int((y & ~pr).sum()), "tn": int((~y & ~pr).sum())}
    return out

@app.get("/api/summary", dependencies=[Depends(auth)])
def summary(hours: int = Query(24, ge=1, le=720)):
    since = time.time() - hours * 3600
    bucket = 3600 if hours <= 48 else 86400
    with db() as c:
        tot = c.execute("SELECT label, severity, COUNT(*) n FROM predictions WHERE ts>=? GROUP BY label, severity", (since,)).fetchall()
        series = c.execute("SELECT CAST(ts/? AS INTEGER)*? t, label, COUNT(*) n FROM predictions WHERE ts>=? GROUP BY t, label ORDER BY t",
                           (bucket, bucket, since)).fetchall()
    return {"hours": hours, "totals": [dict(r) for r in tot], "series": [dict(r) for r in series]}

def _alert_query(limit, severity_f, label_f):
    q, a = "SELECT id,ts,source,label,prob,severity,model_version,factors FROM predictions WHERE 1=1", []
    if severity_f: q += " AND severity=?"; a.append(severity_f)
    if label_f: q += " AND label=?"; a.append(label_f)
    with db() as c:
        return c.execute(q + " ORDER BY id DESC LIMIT ?", (*a, limit)).fetchall()

@app.get("/api/alerts", dependencies=[Depends(auth)])
def alerts(limit: int = Query(100, ge=1, le=1000), severity: Optional[str] = Query(None, pattern="^(none|medium|high)$"),
           label: Optional[str] = Query(None, pattern="^(DDoS|BENIGN)$")):
    return [{**dict(r), "factors": json.loads(r["factors"] or "[]")} for r in _alert_query(limit, severity, label)]

@app.get("/api/alerts.csv", dependencies=[Depends(auth)])
def alerts_csv(limit: int = Query(1000, ge=1, le=10000)):
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["id", "time_utc", "source", "label", "prob_ddos", "severity", "model_version"])
    for r in _alert_query(limit, None, None):
        w.writerow([r["id"], time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r["ts"])), r["source"], r["label"],
                    round(r["prob"], 6), r["severity"], r["model_version"]])
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=alerts.csv"})

app.mount("/assets", StaticFiles(directory=os.path.join(ROOT, "frontend")), name="assets")

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(ROOT, "frontend", "index.html"))

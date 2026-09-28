"""
Build the lightweight reference assets used by the dashboard.

Why this script exists
----------------------
The raw dataset is ~77 MB, which is too heavy to load inside a dashboard on
every page view. This script distils it into small files under ``results/``:

  * holdout_predictions.csv  - true label + predicted DDoS probability for the
                               strict group-aware holdout fold (real,
                               out-of-sample scores for ROC / threshold views)
  * sample_flows.csv         - stratified sample of cleaned flows (EDA views)
  * feature_profile.csv      - per-feature ranges and class medians (UI limits)
  * scenario_presets.json    - real example flows for one-click demos

It mirrors the cleaning and split logic in ``src/train.py`` exactly so the
holdout scores reproduce the reported metrics in ``results/final_metrics.json``.

Run:  python src/build_app_assets.py
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "DDos.pcap_ISCX.csv")
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
SAMPLE_PER_CLASS = 3000
SEED = 42


def load_clean():
    """Same cleaning as train.py: strip headers, drop exact duplicates, drop constants."""
    df = pd.read_csv(DATA, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()
    clean = df.loc[~df.duplicated()].copy()
    X = (
        clean.drop(columns=["Label"])
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    y = (clean["Label"].str.upper() == "DDOS").astype(int)
    constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    return X.drop(columns=constant), y


def reproduce_holdout(X, y):
    """Re-run the fold-0 group-aware split and score it with the tuned parameters."""
    groups = pd.util.hash_pandas_object(X.fillna(-999999999999), index=False).values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, te = next(cv.split(X, y, groups))
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    # Feature selection on training data only (leakage-safe), as in train.py.
    idx = np.random.RandomState(SEED).choice(len(Xtr), min(50000, len(Xtr)), replace=False)
    Xi = SimpleImputer(strategy="median").fit_transform(Xtr.iloc[idx])
    mi = mutual_info_classif(Xi, ytr.iloc[idx], random_state=SEED)
    feats = pd.Series(mi, index=Xtr.columns).sort_values(ascending=False).head(30).index.tolist()

    with open(os.path.join(RESULTS, "tuned_parameters.json")) as f:
        params = {k.replace("model__", ""): v for k, v in json.load(f).items()}
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(**params, objective="binary:logistic",
                                eval_metric="logloss", random_state=SEED, n_jobs=-1)),
    ])
    pipe.fit(Xtr[feats], ytr)
    prob = pipe.predict_proba(Xte[feats])[:, 1]
    out = pd.DataFrame({"y_true": yte.values, "prob_ddos": prob})
    tn, fp, fn, tp = confusion_matrix(yte, (prob >= 0.5).astype(int)).ravel()
    return out, dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)), feats


def nearest_to_median(frame, features, medians, scale):
    """Row whose standardised distance to the given medians is smallest."""
    d = ((frame[features].fillna(0) - medians[features]) / scale[features]).abs().sum(axis=1)
    return frame.loc[d.idxmin()]


def main():
    X, y = load_clean()
    features = [x.strip() for x in open(os.path.join(MODELS, "features.txt")) if x.strip()]

    # 1) Holdout predictions -------------------------------------------------
    hold, cm, hold_feats = reproduce_holdout(X, y)
    hold.round(6).to_csv(os.path.join(RESULTS, "holdout_predictions.csv"), index=False)
    ref = json.load(open(os.path.join(RESULTS, "final_metrics.json")))
    print("Reproduced holdout confusion matrix:", cm)
    print("Reported  holdout confusion matrix :",
          {k: ref[k] for k in ("TN", "FP", "FN", "TP")})

    # 2) Stratified EDA sample (deployment feature set) ----------------------
    Xf = X[features]
    parts = []
    for label, name in ((0, "BENIGN"), (1, "DDoS")):
        idx = Xf.index[y == label]
        pick = np.random.RandomState(SEED).choice(idx, min(SAMPLE_PER_CLASS, len(idx)), replace=False)
        part = Xf.loc[pick].copy()
        part["Label"] = name
        parts.append(part)
    sample = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample.round(4).to_csv(os.path.join(RESULTS, "sample_flows.csv"), index=False)

    # 3) Feature profile used to size sliders and describe distributions -----
    prof = pd.DataFrame({
        "min": Xf.min(),
        "p01": Xf.quantile(0.01),
        "median_benign": Xf[y == 0].median(),
        "median_ddos": Xf[y == 1].median(),
        "median_all": Xf.median(),
        "p99": Xf.quantile(0.99),
        "max": Xf.max(),
        "std": Xf.std(),
    })
    prof.index.name = "feature"
    prof.round(4).to_csv(os.path.join(RESULTS, "feature_profile.csv"))

    # 4) Scenario presets: real flows closest to a narrative archetype -------
    med = Xf.median()
    scale = (Xf.quantile(0.9) - Xf.quantile(0.1)).replace(0, 1)
    labelled = Xf.assign(_y=y)
    benign = labelled[labelled._y == 0].drop(columns="_y")
    ddos = labelled[labelled._y == 1].drop(columns="_y")

    picks = {
        "Typical HTTPS browsing": (
            "BENIGN",
            nearest_to_median(benign[benign["Destination Port"] == 443], features,
                              benign[benign["Destination Port"] == 443].median(), scale)),
        "DNS lookup": (
            "BENIGN",
            nearest_to_median(benign[benign["Destination Port"] == 53], features,
                              benign[benign["Destination Port"] == 53].median(), scale)),
        "HTTP flood on port 80": (
            "DDoS",
            nearest_to_median(ddos[ddos["Destination Port"] == 80], features,
                              ddos[ddos["Destination Port"] == 80].median(), scale)),
    }
    presets = {}
    for name, (truth, row) in picks.items():
        presets[name] = {"truth": truth, "values": {f: float(row[f]) for f in features}}

    # The two DDoS flows the holdout model missed (evasive, tiny single-packet probes).
    mis = pd.read_csv(os.path.join(RESULTS, "misclassified_flows.csv"))
    presets["Evasive single-packet probe (missed in holdout)"] = {
        "truth": "DDoS",
        "values": {f: float(mis.iloc[0][f]) for f in features},
    }
    with open(os.path.join(RESULTS, "scenario_presets.json"), "w") as f:
        json.dump(presets, f, indent=2)
    print("Assets written to", RESULTS)


if __name__ == "__main__":
    main()

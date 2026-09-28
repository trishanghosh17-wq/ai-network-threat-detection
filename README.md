# Network Traffic Classification & Attack Detection

## Project objective
Build a machine-learning network traffic classifier. The **bundled trained model is a binary BENIGN-vs-DDoS benchmark** because the supplied raw file is the DDoS-specific CICIDS2017 capture slice. The project documentation is now explicit about that scope and includes a multi-attack extension path for the full CICIDS2017 dataset. The supplied dataset description lists features covering destination port, flow duration, forward/backward packet counts and lengths, bytes/packets per second, IAT statistics, TCP flags, subflows, TCP window values, active/idle statistics, and a `Label` target. fileciteturn3file0L10-L40

## Completed
- Data audit and cleaning
- Duplicate and constant-feature handling
- Missing/infinite-value handling
- EDA outputs
- Mutual-information feature selection
- Decision Tree, Random Forest, Extra Trees, XGBoost
- Soft-voting ensemble
- Group-aware leakage/generalization audit
- Randomized XGBoost tuning with grouped CV
- Confusion matrix, ROC curve, feature importance
- Error analysis
- Final deployment model
- Streamlit prediction application
- PDF report and PPT presentation

## Key data audit
Raw: 225,745 rows × 79 columns.
After exact duplicate removal: 223,112 rows.
Exact duplicates removed: 2,633.
Class distribution after cleaning: DDoS 128,016; BENIGN 95,096.
10 constant features removed; infinite values converted to missing and median-imputed inside model pipelines.

## Final evaluation
The final reported holdout is an **internal benchmark**, not an external production test. It uses a **StratifiedGroupKFold** split where identical cleaned numeric feature vectors are kept within one group, preventing exact feature-vector duplicates from crossing train/test. Feature selection is fitted on training data only.

Final tuned XGBoost holdout:
- Accuracy: 99.9955%
- Precision: 100.0000%
- Recall: 99.9922%
- F1: 99.9961%
- ROC-AUC: 99.9980%
- Confusion matrix: TN=19,019, FP=0, FN=2, TP=25,601

The arithmetic is internally consistent with the confusion matrix. However, the score comes from one holdout within the supplied DDoS capture family; there is no timestamp/session/flow-ID field for a true temporal/session holdout. It must therefore be described as a benchmark, not as guaranteed deployment accuracy.

## Important limitation
The CSV does not expose an explicit timestamp/session/flow-ID field in the supplied column description. Therefore, a true time-based or capture-session holdout cannot be verified from this dataset alone. The group-aware evaluation reduces one important leakage pathway but does not prove deployment-level generalization to a completely different network environment.

## Run locally
1. Create environment: `python -m venv .venv`
2. Activate: `source .venv/bin/activate`
3. Install: `pip install -r requirements.txt`
4. Train: `python src/train.py`
5. Launch: `streamlit run app/streamlit_app.py`

## Main files
- `data/DDos.pcap_ISCX.csv` — supplied dataset
- `data/Description.docx` — supplied dataset description
- `src/train.py` — reproducible training pipeline
- `models/ddos_xgboost_final.joblib` — trained deployment model
- `models/features.txt` — model feature list
- `app/streamlit_app.py` — interactive application
- `results/` — metrics and tables
- `visualizations/` — final plots
- `reports/` — PDF report and PPT


## Company deployment (API + web console)

```bash
pip install -r requirements-serve.txt
export DDOS_API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
uvicorn backend.main:app --port 8000     # console: http://localhost:8000  API docs: /docs
pip install pytest httpx && pytest tests # run the API tests

# or with Docker
docker build -t ddos-console . && docker run -p 8000:8000 -e DDOS_API_KEY=... -v ddos-data:/srv/data ddos-console
```

Backend: `backend/main.py` (API-key auth, validated inputs, SQLite audit log, TreeSHAP explanations, batch CSV scan, CSV export). Frontend: `frontend/index.html` (dashboard, single-flow analysis, batch scan, alert log, model card). Settings are environment variables; see `.env.example`.

## Data Quality & Drift + multicore
The Streamlit app includes a robust **Data Quality & Drift** tab. It checks missing/non-finite values and computes PSI using a stable reference/current split of the bundled EDA sample. Constant/insufficient-variation features are skipped safely. PSI is a monitoring signal, not an attack label.

Multicore execution is enabled for the tree ensemble, XGBoost, soft-voting ensemble, XGBoost tuning search, and backend inference (`n_jobs=-1` where supported). The dashboard reports detected CPU cores.

### One-click Streamlit launch (macOS/Linux)
`bash RUN_STREAMLIT.command`

## Reviewer clarifications

### Multiple attacks
The current bundled model is **BENIGN vs DDoS only**. The complete CICIDS2017 release is distributed across multiple daily capture files containing other attack families. To classify multiple attack types, combine those files, preserve the original labels, and retrain a multi-class classifier. The present project materials explicitly distinguish the current binary benchmark from that extension.

### Feature importance
The model uses 30 selected features even though two features dominate the current XGBoost native gain. CICFlowMeter includes redundant derived variables (for example, Subflow Fwd Bytes and Total Length of Fwd Packets). Native tree importance can therefore concentrate on one member of a redundant pair. See `results/exact_duplicate_feature_pairs_in_sample.csv` and `results/REVIEWER_CHECKS.txt`.

### Validation
`results/reviewer_sanity_check_6000_sample.csv` is a sample-level 5-fold sanity check. It is deliberately not described as external validation.

"""
DDoS Traffic Intelligence Dashboard
-----------------------------------
Interactive Streamlit app for the Network Traffic Classification
(BENIGN vs DDoS) project. Every number shown here is read live from the
project's own results/ files, the trained model, and a stratified
sample of the source dataset -- nothing is hard-coded or invented.

Run: streamlit run app/streamlit_app.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os as _os

# --------------------------------------------------------------------------
# Paths & page config
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "ddos_xgboost_final.joblib")
FEATURES_PATH = os.path.join(ROOT, "models", "features.txt")
STATS_PATH = os.path.join(ROOT, "app", "feature_stats.json")
SAMPLE_PATH = os.path.join(ROOT, "data", "eda_sample.csv")
RESULTS = os.path.join(ROOT, "results")
VIZ = os.path.join(ROOT, "visualizations")
CPU_CORES = _os.cpu_count() or 1

st.set_page_config(page_title="DDoS Traffic Intelligence", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
.section-note { color: #8a94a6; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_features():
    return [x.strip() for x in open(FEATURES_PATH) if x.strip()]

@st.cache_data(show_spinner=False)
def load_feature_stats():
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            return json.load(f)
    return {}

@st.cache_data(show_spinner=False)
def load_sample():
    if os.path.exists(SAMPLE_PATH):
        return pd.read_csv(SAMPLE_PATH)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_json(name):
    p = os.path.join(RESULTS, name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}

@st.cache_data(show_spinner=False)
def load_csv(name, results=True):
    p = os.path.join(RESULTS if results else VIZ, name)
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()

model = load_model()
features = load_features()
feature_stats = load_feature_stats()
sample_df = load_sample()
metrics = load_json("final_metrics.json")
tuned_params = load_json("tuned_parameters.json")
model_comp = load_csv("final_model_comparison_strict.csv")
feat_imp = load_csv("final_feature_importance.csv").rename(columns={"Unnamed: 0": "Feature"})
mi_train = load_csv("feature_selection_mutual_information.csv").rename(columns={"Unnamed: 0": "Feature"})
class_dist = load_csv("class_distribution.csv")
dataset_audit = load_csv("dataset_audit.csv")
misclassified = load_csv("misclassified_flows.csv")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🛡️ Network Traffic Classification & Attack Detection")
st.caption("Current bundled model: BENIGN vs DDoS | Multi-attack extension documented in the project")
st.info("**Scope note:** The bundled trained model is binary (BENIGN vs DDoS) because the supplied raw dataset is the DDoS-specific CICIDS2017 slice. True multi-attack classification requires merging the other CICIDS2017 attack files and retraining.")

if model is None:
    st.error("Model file not found at models/ddos_xgboost_final.joblib. Run `python src/train.py` first.")
    st.stop()

with st.sidebar:
    st.header("Project snapshot")
    if metrics:
        st.metric("Holdout Accuracy", f"{metrics.get('Accuracy', 0):.4%}")
        st.metric("Holdout ROC-AUC", f"{metrics.get('ROC_AUC', 0):.4%}")
        st.caption(f"Model: {metrics.get('Model', '—')}")
        st.caption(metrics.get("Evaluation", ""))
    st.divider()
    if not dataset_audit.empty:
        row = dataset_audit.iloc[0]
        st.write("**Dataset**")
        st.caption(
            f"{int(row['raw_rows']):,} raw rows → {int(row['clean_rows']):,} after de-duplication\n\n"
            f"{int(row['selected_features'])} features selected via mutual information"
        )
    st.divider()
    st.warning(
        "⚠️ **Known limitation**: no timestamp/session ID is available in the source "
        "data, so a true time-based holdout can't be verified. The group-aware split "
        "reduces exact-duplicate leakage but doesn't guarantee generalization to a "
        "different network environment.",
        icon="⚠️",
    )

tab_overview, tab_explore, tab_single, tab_batch, tab_insights, tab_quality = st.tabs(
    ["📊 Overview", "🔍 Explore the Data", "🎯 Test a Single Flow", "📁 Batch Prediction", "🧠 Model Insights", "🧪 Data Quality & Drift"]
)

# --------------------------------------------------------------------------
# TAB 1 — Overview
# --------------------------------------------------------------------------
with tab_overview:
    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{metrics['Accuracy']:.4%}")
        c2.metric("Precision", f"{metrics['Precision']:.4%}")
        c3.metric("Recall", f"{metrics['Recall']:.4%}")
        c4.metric("F1 Score", f"{metrics['F1']:.4%}")
        c5.metric("ROC-AUC", f"{metrics['ROC_AUC']:.4%}")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Confusion matrix (holdout)")
        if metrics:
            cm = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
            norm = st.toggle("Show as %", value=False, key="cm_norm")
            z = (cm / cm.sum(axis=1, keepdims=True) * 100) if norm else cm
            text = np.array([[f"{v:.1f}%" if norm else f"{v:,}" for v in row] for row in z])
            fig = go.Figure(go.Heatmap(
                z=z, x=["Pred: BENIGN", "Pred: DDoS"], y=["Actual: BENIGN", "Actual: DDoS"],
                text=text, texttemplate="%{text}", colorscale="Blues", showscale=False,
            ))
            fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Class distribution")
        if not class_dist.empty:
            fig = px.pie(class_dist, names="Label", values="Count", hole=0.55,
                         color="Label", color_discrete_map={"BENIGN": "#2E86AB", "DDoS": "#E63946"})
            fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model comparison")
    if not model_comp.empty:
        metric_choice = st.selectbox("Metric", ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"], index=4)
        sorted_comp = model_comp.sort_values(metric_choice, ascending=True)
        fig = px.bar(sorted_comp, x=metric_choice, y="Model", orientation="h",
                     color=metric_choice, color_continuous_scale="Blues", text_auto=".4%")
        fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
        fig.update_xaxes(tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("View full comparison table"):
            st.dataframe(model_comp, use_container_width=True)

    with st.expander("📎 Original report figures (ROC curve, confusion matrix, feature importance, mutual information)"):
        imgs = ["final_roc_curve.png", "final_confusion_matrix.png",
                "top15_feature_importance.png", "top20_mutual_information.png"]
        cols = st.columns(2)
        for i, name in enumerate(imgs):
            p = os.path.join(VIZ, name)
            if os.path.exists(p):
                cols[i % 2].image(p, caption=name, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 2 — Explore the data
# --------------------------------------------------------------------------
with tab_explore:
    st.caption("Computed live from a bundled stratified sample of 6,000 flows (3,000 BENIGN / 3,000 DDoS). This supports exploration, not independent validation.")
    if sample_df.empty:
        st.info("Sample data not found (data/eda_sample.csv).")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            feat_choice = st.selectbox("Feature to inspect", features, index=0)
        with col2:
            chart_type = st.radio("Chart", ["Histogram", "Box plot"], horizontal=True)
        log_scale = st.checkbox("Log scale (helps with heavily skewed features)", value=True)

        plot_df = sample_df[[feat_choice, "Label"]].copy()
        if log_scale:
            plot_df[feat_choice] = np.log1p(plot_df[feat_choice].clip(lower=0))

        if chart_type == "Histogram":
            fig = px.histogram(plot_df, x=feat_choice, color="Label", barmode="overlay", nbins=60,
                                color_discrete_map={"BENIGN": "#2E86AB", "DDOS": "#E63946"}, opacity=0.65)
        else:
            fig = px.box(plot_df, x="Label", y=feat_choice, color="Label",
                         color_discrete_map={"BENIGN": "#2E86AB", "DDOS": "#E63946"})
        fig.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation between top features")
        n_top = st.slider("Number of top (by mutual information) features to correlate", 5, 20, 10)
        top_feats = mi_train.sort_values("mutual_information", ascending=False)["Feature"].head(n_top).tolist()
        top_feats = [f for f in top_feats if f in sample_df.columns]
        if top_feats:
            corr = sample_df[top_feats].corr()
            fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(height=500, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 3 — Single flow prediction
# --------------------------------------------------------------------------
with tab_single:
    st.subheader("Enter (or load) network-flow feature values")

    def apply_preset(kind):
        for f in features:
            s = feature_stats.get(f, {})
            key = "median_benign" if kind == "benign" else "median_ddos"
            st.session_state[f"flow_{f}"] = float(s.get(key, 0.0))

    pcol1, pcol2, pcol3 = st.columns([1, 1, 3])
    pcol1.button("Load typical BENIGN example", on_click=apply_preset, args=("benign",), use_container_width=True)
    pcol2.button("Load typical DDoS example", on_click=apply_preset, args=("ddos",), use_container_width=True)

    vals = {}
    cols = st.columns(3)
    for i, f in enumerate(features):
        s = feature_stats.get(f, {})
        lo = float(s.get("min", 0.0))
        hi = float(s.get("max", lo + 1.0))
        if hi <= lo:
            hi = lo + 1.0
        default = st.session_state.get(f"flow_{f}", float(s.get("median_overall", lo)))
        vals[f] = cols[i % 3].number_input(
            f, min_value=lo, max_value=hi, value=min(max(default, lo), hi),
            format="%.4f", key=f"flow_{f}",
            help=f"Observed range in training data: {lo:,.2f} – {hi:,.2f}",
        )

    if st.button("Predict Traffic", type="primary"):
        X = pd.DataFrame([vals])[features]
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0, 1])

        r1, r2 = st.columns([1, 1])
        with r1:
            if pred == 1:
                st.error(f"⚠️ DDoS detected — probability {prob:.2%}")
            else:
                st.success(f"✅ BENIGN traffic — DDoS probability {prob:.2%}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#E63946" if prob >= 0.5 else "#2E86AB"},
                    "steps": [
                        {"range": [0, 50], "color": "#eaf3fb"},
                        {"range": [50, 100], "color": "#fbe9ea"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
                },
            ))
            fig.update_layout(height=280, margin=dict(t=30, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            st.caption(
                "Approximate drivers of this prediction — model feature importance weighted by "
                "how far each value sits from the typical BENIGN flow. This is a simple heuristic, "
                "not a SHAP/exact attribution."
            )
            if not feat_imp.empty:
                imp_map = dict(zip(feat_imp["Feature"], feat_imp["importance"]))
                contrib = []
                for f in features:
                    s = feature_stats.get(f, {})
                    spread = max(abs(s.get("max", 1) - s.get("min", 0)), 1e-9)
                    dev = abs(vals[f] - s.get("median_benign", 0)) / spread
                    contrib.append({"Feature": f, "Signal": dev * imp_map.get(f, 0.0)})
                contrib_df = pd.DataFrame(contrib).sort_values("Signal", ascending=True).tail(10)
                fig = px.bar(contrib_df, x="Signal", y="Feature", orientation="h")
                fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 4 — Batch prediction
# --------------------------------------------------------------------------
with tab_batch:
    st.subheader("Upload a CSV of network flows")
    st.caption("Must contain the same feature columns the model was trained on. A `Label` column is optional — include it to see live accuracy.")
    file = st.file_uploader("CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        missing = [f for f in features if f not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        else:
            X = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            pred = model.predict(X)
            prob = model.predict_proba(X)[:, 1]
            out = df.copy()
            out["Prediction"] = np.where(pred == 1, "DDoS", "BENIGN")
            out["DDoS_Probability"] = prob

            c1, c2, c3 = st.columns(3)
            c1.metric("Total flows", f"{len(out):,}")
            c2.metric("Predicted DDoS", f"{int((pred == 1).sum()):,}")
            c3.metric("Predicted BENIGN", f"{int((pred == 0).sum()):,}")

            thresh = st.slider("Decision threshold (DDoS probability ≥)", 0.0, 1.0, 0.5, 0.01)
            out["Prediction"] = np.where(prob >= thresh, "DDoS", "BENIGN")

            colA, colB = st.columns(2)
            with colA:
                st.write("**Probability distribution**")
                fig = px.histogram(out, x="DDoS_Probability", nbins=40, color="Prediction",
                                    color_discrete_map={"BENIGN": "#2E86AB", "DDoS": "#E63946"})
                fig.add_vline(x=thresh, line_dash="dash", line_color="black")
                fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)

            with colB:
                label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
                if label_col:
                    st.write("**Live accuracy vs. uploaded labels**")
                    y_true = df[label_col].astype(str).str.strip().str.upper().eq("DDOS").astype(int)
                    y_pred = (prob >= thresh).astype(int)
                    cm = pd.crosstab(y_true, y_pred, rownames=["Actual"], colnames=["Predicted"])
                    acc = (y_true == y_pred).mean()
                    st.metric("Accuracy on this file", f"{acc:.2%}")
                    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                     x=["Pred BENIGN", "Pred DDoS"], y=["Actual BENIGN", "Actual DDoS"])
                    fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No `Label` column found — upload one to see live accuracy on this file.")

            st.write("**Results table**")
            st.dataframe(out[["Prediction", "DDoS_Probability"]].join(df), use_container_width=True)
            st.download_button("⬇️ Download predictions as CSV", out.to_csv(index=False),
                                "ddos_predictions.csv", "text/csv")

# --------------------------------------------------------------------------
# TAB 5 — Model insights
# --------------------------------------------------------------------------
with tab_insights:
    st.subheader("Why do two features dominate?")
    if not feat_imp.empty:
        top2_share = feat_imp.sort_values("importance", ascending=False).head(2)["importance"].sum()
        st.info(
            f"The model still uses {len(features)} selected features. The top two account for {top2_share:.2%} of the current XGBoost native importance. "
            "That concentration is partly explained by redundant CICFlowMeter-derived variables; a tree model can favor one member of a duplicate pair. "
            "The mutual-information ranking shows many additional features carrying signal, so the chart should not be read as saying only two measurements matter."
        )
        if os.path.exists(os.path.join(RESULTS, "exact_duplicate_feature_pairs_in_sample.csv")):
            dup = load_csv("exact_duplicate_feature_pairs_in_sample.csv")
            if not dup.empty:
                with st.expander("Show redundant feature pairs found in the bundled sample"):
                    st.dataframe(dup, use_container_width=True)

    st.subheader("Feature importance (final deployment model)")
    if not feat_imp.empty:
        n = st.slider("Show top N features", 5, len(feat_imp), 15, key="imp_n")
        top = feat_imp.sort_values("importance", ascending=True).tail(n)
        fig = px.bar(top, x="importance", y="Feature", orientation="h", color="importance",
                     color_continuous_scale="Blues")
        fig.update_layout(height=max(320, 22 * n), margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mutual information ranking (training-only, leakage-safe)")
    if not mi_train.empty:
        n2 = st.slider("Show top N features", 5, len(mi_train), 20, key="mi_n")
        top_mi = mi_train.sort_values("mutual_information", ascending=True).tail(n2)
        fig = px.bar(top_mi, x="mutual_information", y="Feature", orientation="h", color="mutual_information",
                     color_continuous_scale="Teal")
        fig.update_layout(height=max(320, 22 * n2), margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Tuned XGBoost hyperparameters")
        if tuned_params:
            st.json(tuned_params)
    with c2:
        st.subheader("Dataset audit")
        if not dataset_audit.empty:
            st.dataframe(dataset_audit.T.rename(columns={0: "value"}), use_container_width=True)

    if not misclassified.empty:
        with st.expander(f"⚠️ Misclassified holdout flows ({len(misclassified)})"):
            st.dataframe(misclassified, use_container_width=True)


# --------------------------------------------------------------------------
# TAB 6 — Data Quality & Drift
# --------------------------------------------------------------------------
with tab_quality:
    st.subheader("🧪 Data Quality & Drift")
    st.caption("Robust checks on the bundled EDA sample. Drift is a monitoring signal, not proof of an attack.")

    if sample_df.empty:
        st.warning("Data quality checks cannot run because data/eda_sample.csv is missing.")
    else:
        dfq = sample_df.copy()
        numeric = dfq.select_dtypes(include=np.number).columns.tolist()
        if not numeric:
            st.warning("No numeric features are available for quality/drift analysis.")
        else:
            # Stable split: first half as reference and second half as current.
            # This avoids random re-sampling on every Streamlit rerun.
            mid = max(1, len(dfq) // 2)
            ref = dfq.iloc[:mid].copy()
            cur = dfq.iloc[mid:].copy()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows checked", f"{len(dfq):,}")
            c2.metric("Numeric features", f"{len(numeric):,}")
            c3.metric("Missing cells", f"{int(dfq.isna().sum().sum()):,}")
            c4.metric("Infinite cells", f"{int(np.isinf(dfq[numeric].to_numpy(dtype=float, copy=True)).sum()):,}")

            # Missingness / non-finite quality checks
            quality = pd.DataFrame({
                "Feature": numeric,
                "Missing %": [100 * dfq[f].isna().mean() for f in numeric],
                "Infinite": [int(np.isinf(pd.to_numeric(dfq[f], errors="coerce").to_numpy(dtype=float, copy=True)).sum()) for f in numeric],
                "Unique": [int(dfq[f].nunique(dropna=True)) for f in numeric],
            }).sort_values(["Missing %", "Infinite"], ascending=False)

            st.markdown("### Data quality")
            st.dataframe(quality.style.format({"Missing %": "{:.3f}%"}), use_container_width=True, height=300)

            # PSI is calculated with reference quantile bins. Constant/empty columns are skipped safely.
            def psi_score(a, b, bins=10):
                a = pd.to_numeric(a, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                b = pd.to_numeric(b, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                if len(a) < 2 or len(b) < 2 or np.nanmin(a) == np.nanmax(a):
                    return np.nan
                qs = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1)))
                if len(qs) < 3:
                    return np.nan
                lo, hi = qs[0], qs[-1]
                edges = np.concatenate(([-np.inf], qs[1:-1], [np.inf]))
                pa = np.histogram(a, bins=edges)[0].astype(float)
                pb = np.histogram(b, bins=edges)[0].astype(float)
                pa /= max(pa.sum(), 1.0); pb /= max(pb.sum(), 1.0)
                pa = np.clip(pa, 1e-6, None); pb = np.clip(pb, 1e-6, None)
                return float(np.sum((pb - pa) * np.log(pb / pa)))

            drift_rows=[]
            for f in numeric:
                drift_rows.append({"Feature": f, "PSI": psi_score(ref[f], cur[f]),
                                   "Reference missing %": 100*ref[f].isna().mean(),
                                   "Current missing %": 100*cur[f].isna().mean()})
            drift = pd.DataFrame(drift_rows).dropna(subset=["PSI"]).sort_values("PSI", ascending=False)

            st.markdown("### Distribution drift")
            if drift.empty:
                st.info("No numeric feature had enough variation/data for a stable PSI calculation.")
            else:
                def drift_band(x):
                    if x < 0.10: return "Low"
                    if x < 0.25: return "Moderate"
                    return "High"
                drift["Status"] = drift["PSI"].map(drift_band)
                st.dataframe(drift.style.format({"PSI":"{:.4f}", "Reference missing %":"{:.2f}%", "Current missing %":"{:.2f}%"}), use_container_width=True, height=360)
                top = drift.head(15).sort_values("PSI")
                fig = px.bar(top, x="PSI", y="Feature", orientation="h", color="PSI", color_continuous_scale="Blues")
                fig.add_vline(x=0.10, line_dash="dash", annotation_text="0.10")
                fig.add_vline(x=0.25, line_dash="dash", annotation_text="0.25")
                fig.update_layout(height=max(420, 24*len(top)), margin=dict(t=20,b=20,l=10,r=10), coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("PSI interpretation used here: <0.10 low, 0.10–0.25 moderate, ≥0.25 high. Thresholds are monitoring conventions, not attack labels.")

            st.markdown("### Runtime / multicore status")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("CPU cores detected", CPU_CORES)
            mc2.metric("XGBoost threads", "All cores (n_jobs=-1)")
            mc3.metric("Tree ensemble threads", "All cores (n_jobs=-1)")
            st.success("Multicore configuration is enabled for the trained tree/XGBoost models. Streamlit itself remains a single UI process; model operations use the configured worker threads.")

"""
Nexus Fraud Defense
Digital Twin-Enabled Framework for Forecasting and Mitigating Fraud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

from config import (DATASET_META, MODEL_NAMES, DASHBOARD_CONFIG,
                    FRAUD_THRESHOLD, DEFAULT_FRAUD_INPUTS,
                    FRAUD_EXAMPLES, NORMAL_EXAMPLES)


# ================================================================
#  CSS
# ================================================================

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.block-container {
    padding: 1.6rem 2rem;
    max-width: 1200px;
}

/* Header */
.app-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: -webkit-linear-gradient(45deg, #0ea5e9, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.3px;
    margin-top: 10px;
    margin-bottom: 4px;
    padding-top: 5px;
    line-height: 1.25;
}
.app-subtitle {
    font-size: 1rem;
    opacity: 0.8;
    margin-top: 0;
    margin-bottom: 1rem;
    font-weight: 500;
}

/* Section headings */
.section-title {
    font-size: 1.4rem;
    font-weight: 800;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 8px;
    margin: 1.25rem 0 0.75rem 0;
}

/* Cards */
.stat-card {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.stat-card .label {
    font-size: 0.85rem;
    opacity: 0.8;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.stat-card .value {
    font-size: 1.6rem;
    font-weight: 800;
    margin-top: 6px;
}
.stat-card .extra {
    font-size: 0.82rem;
    opacity: 0.7;
    margin-top: 4px;
}

/* Tables: improved padding and header contrast */
.stDataFrame table {
    background-color: transparent;
    border-collapse: separate;
    border-spacing: 0 6px;
}
.stDataFrame thead th {
    background: var(--secondary-background-color);
    font-weight: 700;
    padding: 10px 12px !important;
}
.stDataFrame tbody tr {
    background: var(--secondary-background-color);
    border-radius: 8px;
}
.stDataFrame tbody tr td {
    padding: 10px 12px !important;
}

/* Best badge */
.best-tag {
    background: var(--primary-color);
    color: white;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 700;
    display: inline-block;
    margin-top: 6px;
}

/* Insight cards */
.insight {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.insight strong { font-weight: 800; color: var(--primary-color); }

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 700;
    padding: 0.6rem 1.2rem;
    font-size: 0.95rem;
    transition: all 0.15s ease-in-out;
}
.stButton > button:hover { transform: translateY(-2px); }

/* Tabs */
.stTabs [data-baseweb="tab"] { font-weight: 700; font-size: 0.98rem; }

/* Footer */
.app-footer {
    text-align: center;
    opacity: 0.7;
    font-size: 0.9rem;
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border-color);
}
</style>
"""

def _card(label, value, extra=""):
    extra_html = f'<div class="extra">{extra}</div>' if extra else ""
    st.markdown(
        f'<div class="stat-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'{extra_html}</div>',
        unsafe_allow_html=True)


# ================================================================
#  STATE
# ================================================================

import json
import joblib

@st.cache_data(show_spinner=False)
def load_cached_metrics():
    try:
        with open("models/evaluation_metrics.json", "r") as f:
            metrics = json.load(f)
        with open("models/dataset_stats.json", "r") as f:
            stats = json.load(f)
        return metrics, stats
    except FileNotFoundError:
        return {}, {}

@st.cache_resource(show_spinner=False)
def load_all_cached_models():
    from modules.model_trainer import load_all_models
    import os
    models = {}
    pipelines = {}
    for ds in DATASET_META:
        models[ds] = load_all_models(ds)
        pipe_path = f"models/pipeline_{ds}.pkl"
        if os.path.exists(pipe_path):
            pipelines[ds] = joblib.load(pipe_path)
    return models, pipelines

def _init():
    pass

# ================================================================
#  HEADER
# ================================================================

def _header():
    st.markdown(APP_CSS, unsafe_allow_html=True)
    c1, c2 = st.columns([0.78, 0.22])
    with c1:
        st.markdown('<div class="app-title">Nexus Fraud Defense</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="app-subtitle">Digital Twin-Enabled Framework for Forecasting '
            'and Mitigating Fraud</div>',
            unsafe_allow_html=True)
    with c2:
        metrics, _ = load_cached_metrics()
        models, _ = load_all_cached_models()
        loaded = sum(1 for ds in DATASET_META if ds in metrics)
        total_m = sum(len(m_dict) for m_dict in models.values())
        st.metric("Datasets", f"{loaded}/4")
        st.metric("Models", total_m)


# ================================================================
#  1. DATA MANAGEMENT
# ================================================================

def _page_data():
    metrics, stats = load_cached_metrics()
    
    if not metrics:
        st.error("Offline training not found. Please run `python offline_trainer.py` to precompute models.")
        return
        
    st.markdown('<div class="section-title">Data Management</div>', unsafe_allow_html=True)
    st.success("Models loaded successfully (Instant Load)")

    tabs = st.tabs([m["short_name"] for m in DATASET_META.values()])
    for i, (key, meta) in enumerate(DATASET_META.items()):
        with tabs[i]:
            _dataset_tab(key, meta, metrics.get(key, {}), stats.get(key, {}))


def _dataset_tab(key, meta, ds_metrics, stats):
    st.markdown(f"**{meta['display_name']}** -- {meta['domain']}")
    st.caption(meta["description"])

    if not ds_metrics:
        st.warning(f"No offline data found for {key}.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1: _card("Total Dataset Rows", f"{stats.get('total_rows_available', 0):,}")
    with c2: _card("Rows Used for ML", f"{stats.get('rows_used', 0):,}",
                    f"{stats.get('rows_used', 0)/max(1, stats.get('total_rows_available', 1))*100:.1f}% limit")
    with c3: _card("Fraud Cases", f"{stats.get('sampled_fraud_count', 0):,}",
                    f"{stats.get('sampled_fraud_ratio', 0):.2%} ratio")
    with c4: _card("Dataset Columns", str(stats.get("total_features", "N/A")), "Input Features")

    st.write("") # Spacing
    t1, t2, t3, t4 = st.columns(4)
    test_pct = 20 # Static configured via DATA_CONFIG
    test_rows = int(stats.get('rows_used', 0) * (test_pct / 100))
    train_rows = stats.get('rows_used', 0) - test_rows
    
    with t1: _card("Training Split", f"{train_rows:,}", "80% (Learned Data)")
    with t2: _card("Testing Split (Unseen)", f"{test_rows:,}", "20% (Validation Data)")
    with t3: _card("Models Trained", str(len(ds_metrics.get("results", {}))), "Pipeline Count")
    with t4:
        if stats.get('sampling_used'):
            _card("Memory Status", "Downsampled", "For stability")
        else:
            _card("Memory Status", "Full Dataset", "100% Loaded")

    results = ds_metrics.get("results", {})
    best = ds_metrics.get("best_model")
    if results:
        _results_table(results, best)


def _results_table(results, best=None):
    rows = []
    for name, met in results.items():
        # normalize display names
        row = {"Model": name}
        row.update(met)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    if "ROC_CURVE" in df.columns:
        df = df.drop(columns=["ROC_CURVE"])
    if "CM" in df.columns:
        df = df.drop(columns=["CM"])

    fmt = {c: "{:.4f}" for c in df.columns if df[c].dtype in (float, np.float64)}
    st.dataframe(df.style.format(fmt))

    if best:
        f1 = results.get(best, {}).get("F1-Score", 0)
        st.markdown(f'<span class="best-tag">Best: {best} (F1 = {f1:.4f})</span>',
                    unsafe_allow_html=True)


# ================================================================
#  2. MODEL EVALUATION
# ================================================================

def _page_eval():
    metrics_master, _ = load_cached_metrics()
    
    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown(f"All metrics computed offline securely. Fraud threshold = **{FRAUD_THRESHOLD}**.")

    if not metrics_master:
        st.error("No metrics found. Please execute `python offline_trainer.py` in the CLI.")
        return

    # Build a consolidated metrics DataFrame for all datasets/models
    try:
        rows = []
        for ds_key, ds_val in metrics_master.items():
            res = ds_val.get('results', {})
            for mname, mm in res.items():
                rows.append({
                    'dataset': ds_key,
                    'model': mname,
                    'accuracy': mm.get('Accuracy', None),
                    'precision': mm.get('Precision', None),
                    'recall': mm.get('Recall', None),
                    'f1': mm.get('F1-Score', None),
                    'best_threshold': mm.get('Best_Threshold', None)
                })
        if rows:
            df_summary = pd.DataFrame(rows)
            st.markdown('**Metrics Summary (all datasets & models)**')
            st.dataframe(df_summary)
            csv = df_summary.to_csv(index=False).encode('utf-8')
            st.download_button('Download metrics CSV', csv, file_name='metrics_summary.csv', mime='text/csv')
    except Exception:
        pass

    tabs = st.tabs([m["short_name"] for m in DATASET_META.values()])
    for i, (key, meta) in enumerate(DATASET_META.items()):
        with tabs[i]:
            ds_metrics = metrics_master.get(key)
            if not ds_metrics:
                st.warning(f"No offline data found for {meta['short_name']}.")
                continue

            results = ds_metrics.get("results", {})
            best = ds_metrics.get("best_model")

            if not results:
                continue

            _results_table(results, best)

            timings = ds_metrics.get("timings", {})
            if timings:
                cols = st.columns(min(4, len(timings)))
                for j, (n, t) in enumerate(timings.items()):
                    with cols[j % len(cols)]:
                        _card(n, f"{t:.1f}s")

            sel = st.selectbox("Detailed view:", list(results.keys()), key=f"ev_{key}")

            if sel and sel in results:
                mets = results[sel]
                cm = mets.get("CM", {"TP":0, "TN":0, "FP":0, "FN":0})
                roc = mets.get("ROC_CURVE", {"fpr": [0], "tpr": [0]})

                c1, c2 = st.columns(2)
                with c1:
                    cm_matrix = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
                    fig = go.Figure(data=go.Heatmap(
                        z=cm_matrix, x=["Legitimate","Fraud"], y=["Legitimate","Fraud"],
                        text=cm_matrix, texttemplate="%{text}", colorscale="Blues", showscale=False))
                    fig.update_layout(title=f"Confusion Matrix  -  {sel}",
                                      yaxis_title="Actual", xaxis_title="Predicted",
                                      height=380, margin=dict(t=50,b=40))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"TP={cm['TP']:,}  TN={cm['TN']:,}  FP={cm['FP']:,}  FN={cm['FN']:,}")

                with c2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], name=f"AUC={mets.get('ROC-AUC', 0):.4f}",
                                             fill='tozeroy', line=dict(color="#0ea5e9", width=2.5)))
                    fig.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
                    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                                      yaxis_title="True Positive Rate", height=380, margin=dict(t=50,b=40),
                                      template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                # Comparison bar chart for all models in this dataset
                try:
                    import plotly.express as px
                    # build DataFrame from results
                    rows = []
                    for mname, mm in results.items():
                        rows.append({
                            "Model": mname,
                            "Accuracy": mm.get("Accuracy", 0),
                            "Precision": mm.get("Precision", 0),
                            "Recall": mm.get("Recall", 0),
                            "F1-Score": mm.get("F1-Score", 0),
                        })
                    if rows:
                        dfm = pd.DataFrame(rows)
                        dfm_melt = dfm.melt(id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
                                           var_name="Metric", value_name="Value")
                        fig_metrics = px.bar(dfm_melt, x="Model", y="Value", color="Metric", barmode='group',
                                             title="Model Metrics Comparison", height=420)
                        st.plotly_chart(fig_metrics, use_container_width=True)
                except Exception:
                    pass


# ================================================================
#  3. DIGITAL TWIN
# ================================================================

def _page_twin():
    st.markdown('<div class="section-title">Digital Twin Simulator</div>', unsafe_allow_html=True)
    st.markdown("Forecast entity trajectories using Hidden Markov Model with DRL intervention.")

    c1, c2 = st.columns(2)
    with c1:
        domain = st.selectbox("Domain:", list(DATASET_META.keys()),
                               format_func=lambda k: DATASET_META[k]["display_name"], key="dt_d")
    with c2:
        steps = st.slider("Forecast steps:", 10, 40, 20, key="dt_s")

    if st.button("Generate Forecast", type="primary", key="dt_go"):
        from modules.digital_twin import run_drl_forecast, run_mitigation_comparison
        with st.spinner("Simulating..."):
            base, threat = run_drl_forecast(domain, steps)
            noact, drl = run_mitigation_comparison(domain, steps)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=base["Forecast Time"], y=base["Fraud Probability Score"],
                                  name="Normal Entity", line=dict(color="#22c55e", width=2.5)))
        fig.add_trace(go.Scatter(x=threat["Forecast Time"], y=threat["Fraud Probability Score"],
                                  name="Threat (DRL Active)", line=dict(color="#ef4444", width=2.5, dash="dot")))
        fig.update_layout(title="Threat Trajectory", template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=noact["Forecast Time"], y=noact["Fraud Probability Score"],
                                   name="No Intervention", line=dict(color="#f59e0b", width=2.5)))
        fig2.add_trace(go.Scatter(x=drl["Forecast Time"], y=drl["Fraud Probability Score"],
                                   name="DRL Active", line=dict(color="#3b82f6", width=2.5)))
        fig2.update_layout(title="Mitigation Comparison", template="plotly_white", height=400)
        st.plotly_chart(fig2, use_container_width=True)


# ================================================================
#  4. LIVE PREDICTION
# ================================================================

def _page_predict():
    st.markdown('<div class="section-title">Live Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        "Test trained models with custom inputs. "
        "Uses the exact saved ML pipeline from training for inference. "
        f"Threshold = **{FRAUD_THRESHOLD}**.")

    metrics, _ = load_cached_metrics()
    all_models, pipelines = load_all_cached_models()

    if not metrics:
        st.warning("Offline metrics missing. Please train models offline via CLI.")
        return

    ds = st.selectbox("Dataset:", list(metrics.keys()),
                       format_func=lambda k: DATASET_META[k]["display_name"], key="lp_ds")

    models = all_models.get(ds, {})
    best = metrics.get(ds, {}).get("best_model")
    fn = metrics.get(ds, {}).get("feature_names", [])

    if not models or not fn:
        st.warning("No models or active pipeline features available. Run offline trainer!")
        return

    names = list(models.keys())
    idx = names.index(best) if best in names else 0
    sel = st.selectbox("Model:", names, index=idx, key=f"lp_m_{ds}")

    st.markdown("---")
    st.markdown("**Transaction Details** (defaults are pre-filled with FRAUD examples for testing)")

    # 13. DEFAULT BEHAVIOR: Default input -> FRAUD (for all datasets)
    defaults = FRAUD_EXAMPLES.get(ds, {})
    raw_fn = list(defaults.keys()) 
    tx = _input_form(ds, raw_fn, defaults)

    with st.expander("Show Example Inputs"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**👉 Fraud Example**")
            fraud_ex = FRAUD_EXAMPLES.get(ds, {})
            for k, v in fraud_ex.items():
                st.markdown(f"- {k} = {v}")
        with col2:
            st.markdown("**👉 Normal Example**")
            norm_ex = NORMAL_EXAMPLES.get(ds, {})
            for k, v in norm_ex.items():
                st.markdown(f"- {k} = {v}")

    if st.button("Analyze Transaction", type="primary", key=f"lp_go_{ds}"):
        from modules.model_trainer import predict_fraud
        
        # 3. ENSURE FEATURE CONSISTENCY
        # The variables map directly to DataFrame bounds mapping extraction dependencies strictly
        # We wrap in a DataFrame for the pipeline which expect a DF
        X_raw = pd.DataFrame([tx])
        
        # Note: The pipeline (loaded in all_models[sel]) contains the DomainFeatureExtractor
        # which will handle feature engineering and alignment internally.
        # We just need to ensure the columns match the EXPECTED input of the first step.
        
        # Actually, to be extra safe and follow the 'assert' rule:
        try:
            pred, prob, details = predict_fraud(models[sel], X_raw, sel, raw_tx=tx, ds_key=ds)

            # --- RESULT HEADER / ALERT BOX ---
            alert_color = '#ef4444' if pred == 1 else '#10b981'
            status_text = 'ATTACK DETECTED' if pred == 1 else 'Legitimate'
            st.markdown(
                f"<div style='border-radius:10px;padding:18px;margin-bottom:8px;background-color:var(--secondary-background-color);border:1px solid {alert_color};'>"
                f"<h3 style='margin:0 0 6px 0;color:{alert_color};'>{status_text}</h3>"
                f"<div style='color:var(--text-color);'>Confidence: <strong>{prob:.2%}</strong> | Dataset: <strong>{DATASET_META[ds]['display_name']}</strong> | Features: <strong>{len(tx)}</strong></div>"
                f"</div>", unsafe_allow_html=True
            )

            # --- Gauge + Probability breakdown ---
            c1, c2 = st.columns([0.45, 0.55])
            with c1:
                try:
                    st.markdown("<div style='font-weight:700; font-size:0.85rem; padding-top:10px; margin-bottom:0px;'>Model Probability</div>", unsafe_allow_html=True)
                    fig_g = go.Figure(go.Indicator(
                        mode='gauge+number', value=prob,
                        number={'valueformat':'.2f'},
                        gauge={'axis':{'range':[0,1]},
                               'bar':{'color':'#ef4444' if prob>0.6 else '#10b981'},
                               'steps':[{'range':[0,0.5],'color':'#10b981'},{'range':[0.5,0.8],'color':'#f59e0b'},{'range':[0.8,1],'color':'#ef4444'}]}))
                    fig_g.update_layout(height=180, margin=dict(t=20,b=10,l=10,r=10))
                    st.plotly_chart(fig_g, use_container_width=True)
                except Exception:
                    st.metric('Confidence Score', f"{prob:.1%}")

            # collect probabilities from all models for breakdown chart
            model_probs = []
            for mname, mobj in models.items():
                try:
                    # prefer predict_fraud to ensure alignment and threshold-awareness
                    _, mprob, _ = predict_fraud(mobj, X_raw, mname, raw_tx=tx, ds_key=ds)
                except Exception:
                    try:
                        from modules.model_trainer import get_probabilities
                        p = get_probabilities(mobj, X_raw, mname)
                        mprob = float(p[0]) if len(p) else 0.0
                    except Exception:
                        mprob = 0.0
                model_probs.append((mname, mprob))

            with c2:
                try:
                    st.markdown("<div style='font-weight:700; font-size:0.85rem; padding-top:10px; margin-bottom:0px;'>Model Probability Breakdown</div>", unsafe_allow_html=True)
                    df_mp = pd.DataFrame({'Model':[n for n,_ in model_probs], 'Probability':[p for _,p in model_probs]})
                    df_mp['label'] = (df_mp['Probability']*100).round(2).astype(str) + '%'
                    fig_bar = px.bar(df_mp, x='Model', y='Probability', color='Model', text='label', height=180)
                    fig_bar.update_traces(textposition='outside')
                    fig_bar.update_yaxes(range=[0,1])
                    fig_bar.update_layout(showlegend=False, margin=dict(t=20,b=10,l=10,r=10))
                    st.plotly_chart(fig_bar, use_container_width=True)
                except Exception:
                    pass

            # show ML raw output and threshold
            st.markdown('---')
            # Render a small colored risk-level box for ALL datasets
            rl_text = details.get('risk_level', 'UNKNOWN')
            rl_lower = rl_text.lower()
            # simplified label and default colors
            simple_label = 'Unknown'
            box_border = '#64748b'
            box_bg = 'transparent'
            text_color = 'var(--text-color)'

            if 'low' in rl_lower or 'legitimate' in rl_lower:
                simple_label = 'Low Risk'
                box_border = '#10b981'
                box_bg = 'rgba(16,185,129,0.08)'
                text_color = '#065f46'
            elif 'elevated' in rl_lower or 'medium' in rl_lower or 'elev' in rl_lower:
                simple_label = 'Medium Risk'
                box_border = '#f59e0b'
                box_bg = 'rgba(245,158,11,0.08)'
                text_color = '#92400e'
            elif 'high' in rl_lower or 'critical' in rl_lower or 'fraud' in rl_lower:
                simple_label = 'High Risk'
                box_border = '#ef4444'
                box_bg = 'rgba(239,68,68,0.12)'
                text_color = '#7f1d1d'

            # compact box (smaller length and breadth)
            st.markdown(
                f"<div style='display:inline-block;border-radius:8px;padding:8px 12px;margin-bottom:8px;background-color:{box_bg};border:1px solid {box_border};max-width:420px;'>"
                f"<div style='font-weight:700;font-size:0.95rem;margin:0;color:{text_color};'>{simple_label}</div>"
                f"</div>", unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                if pred == 1:
                    st.error('FRAUD DETECTED')
                else:
                    st.success('LEGITIMATE')
            with c2:
                st.metric('Confidence Score', f"{prob:.1%}")
            with c3:
                st.metric('ML Output (Raw)', f"{details['ml_probability']:.3f}")
            
            thresh = details['threshold']
            if pred == 0 and prob >= 0.5:
                st.info(f"**Why is this Legitimate?** The model evaluated a **{prob:.1%} chance of fraud**, but this falls *below* our strict decision threshold of **{thresh:.1%}**. The threshold is elevated intentionally to prevent **false alarms** (blocking innocent users) for this specific dataset.")
            elif pred == 0:
                st.caption(f"**Note:** At {prob:.1%}, the threat level is safely below the {thresh:.1%} blocking threshold. The transaction is Legitimate.")
            else:
                st.warning(f"**Why is this an Attack?** The model's fraud probability of **{prob:.1%}** strictly exceeds the dataset's required threshold of **{thresh:.1%}**, triggering immediate defensive action.")

            if details.get('factors'):
                st.info('**Risk Analysis Factors:**\n' + '\n'.join([f'- {f}' for f in details['factors']]))

            from modules.fraud_explainer import get_simple_fraud_explanation, display_fraud_explanation
            expl = get_simple_fraud_explanation(tx, tx, pred, prob, ds)
            display_fraud_explanation(expl, ds)

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
            st.exception(e)


def _input_form(ds, feat_names, defaults):
    if ds == "paysim":
        return _form_paysim(defaults)
    elif ds == "banksim":
        return _form_banksim(defaults)
    elif ds == "cicids":
        return _form_cicids(defaults)
    elif ds == "ieee":
        return _form_ieee(defaults)
    return {}


def _form_paysim(d):
    st.caption("Only non-leakage features: type, amount, oldbalanceOrg, oldbalanceDest")
    c1, c2, c3 = st.columns(3)
    with c1:
        amount = st.number_input("Amount", 0.0, 1e7, float(d.get("amount", 500000)), 1000.0, key="fp_a")
        types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
        default_index = types.index(d.get("type", "TRANSFER")) if d.get("type", "TRANSFER") in types else 4
        tx = st.selectbox("Type", types, index=default_index, key="fp_t")
    with c2:
        obo = st.number_input("Old Balance (Source)", 0.0, 1e8, float(d.get("oldbalanceOrg", 0)), 100.0, key="fp_ob")
    with c3:
        obd = st.number_input("Old Balance (Dest)", 0.0, 1e8, float(d.get("oldbalanceDest", 0)), 100.0, key="fp_od")

    return {"type": tx, "amount": amount,
            "oldbalanceOrg": obo, "oldbalanceDest": obd}


def _form_banksim(d):
    c1, c2, c3 = st.columns(3)
    cats = ["es_transportation","es_health","es_otherservices","es_food",
            "es_hotelservices","es_tech","es_sportsandtoys","es_wellnessandbeauty",
            "es_hyper","es_fashion","es_barsandrestaurants","es_travel","es_leisure"]
    with c1:
        amount = st.number_input("Amount", 0.0, 50000.0, float(d.get("amount", 25000)), 100.0, key="fb_a")
        cat = st.selectbox("Category", cats, index=cats.index(d.get("category","es_travel")), key="fb_c")
    with c2:
        ages = ["0","1","2","3","4","5","6","U"]
        age = st.selectbox("Age Group", ages, index=ages.index(d.get("age","U")), key="fb_ag")
        genders = ["F","M","E","U"]
        gen = st.selectbox("Gender", genders, index=genders.index(d.get("gender","E")), key="fb_g")
    with c3:
        step = st.number_input("Day Step", 1, 180, int(d.get("step", 1)), key="fb_s")
    return {"step": step, "amount": amount, "category": cat, "age": age, "gender": gen}


def _form_cicids(d):
    c1, c2, c3 = st.columns(3)
    with c1:
        port = st.number_input("Dest Port", 0, 65535, int(d.get("Destination Port", 22)), key="fc_p")
        dur = st.number_input("Flow Duration", 0.0, 1e8, float(d.get("Flow Duration", 500000)), key="fc_d")
        fp = st.number_input("Fwd Packets", 0, 100000, int(d.get("Total Fwd Packets", 5000)), key="fc_fp")
        bp = st.number_input("Bwd Packets", 0, 100000, int(d.get("Total Backward Packets", 3000)), key="fc_bp")
    with c2:
        fb = st.number_input("Fwd Length", 0.0, 1e8, float(d.get("Total Length of Fwd Packets", 200000)), key="fc_fb")
        bb = st.number_input("Bwd Length", 0.0, 1e8, float(d.get("Total Length of Bwd Packets", 150000)), key="fc_bb")
        bps = st.number_input("Bytes/s", 0.0, 1e9, float(d.get("Flow Bytes/s", 50000)), key="fc_bps")
    with c3:
        pps = st.number_input("Packets/s", 0.0, 1e6, float(d.get("Flow Packets/s", 800)), key="fc_pps")
        plm = st.number_input("Pkt Len Mean", 0.0, 1e6, float(d.get("Packet Length Mean", 1200)), key="fc_plm")
        aps = st.number_input("Avg Pkt Size", 0.0, 1e6, float(d.get("Average Packet Size", 1300)), key="fc_aps")
        syn = st.number_input("SYN Flags", 0, 1000, int(d.get("SYN Flag Count", 50)), key="fc_syn")
    return {"Destination Port":port, "Flow Duration":dur,
            "Total Fwd Packets":fp, "Total Backward Packets":bp,
            "Total Length of Fwd Packets":fb, "Total Length of Bwd Packets":bb,
            "Flow Bytes/s":bps, "Flow Packets/s":pps,
            "Packet Length Mean":plm, "Average Packet Size":aps, "SYN Flag Count":syn}


def _form_ieee(d):
    c1, c2, c3 = st.columns(3)
    with c1:
        amt = st.number_input("Amount", 0.0, 50000.0, float(d.get("TransactionAmt", 800)), key="fi_a")
        prods = ["W","H","C","S","R"]
        prod = st.selectbox("ProductCD", prods, index=prods.index(d.get("ProductCD","C")), key="fi_p")
        c4s = ["visa","mastercard","american express","discover"]
        c4 = st.selectbox("Card Provider", c4s, index=c4s.index(d.get("card4","visa")), key="fi_c4")
    with c2:
        c6s = ["credit","debit"]
        c6 = st.selectbox("Card Type", c6s, index=c6s.index(d.get("card6","credit")), key="fi_c6")
        devs = ["desktop","mobile"]
        dev = st.selectbox("Device", devs, index=devs.index(d.get("DeviceType","mobile")), key="fi_dv")
        ems = ["gmail.com","yahoo.com","hotmail.com","anonymous.com"]
        em = st.selectbox("Email", ems, index=ems.index(d.get("P_emaildomain","anonymous.com")), key="fi_em")
    with c3:
        dt = st.number_input("Time Delta", 86400, 1000000, int(d.get("TransactionDT", 86400)), key="fi_dt")
        dist = st.number_input("Distance", 0.0, 1000.0, float(d.get("dist1", 200)), key="fi_d1")
        c1v = st.number_input("Card ID", 1000, 20000, int(d.get("card1", 10000)), key="fi_c1")
    return {"TransactionDT":dt, "TransactionAmt":amt, "ProductCD":prod,
            "card1":c1v, "card4":c4, "card6":c6,
            "P_emaildomain":em, "DeviceType":dev, "dist1":dist}


# ================================================================
#  5. EXECUTIVE INSIGHTS
# ================================================================

def _page_insights():
    st.markdown('<div class="section-title">Executive Insights</div>', unsafe_allow_html=True)

    metrics_master, stats_master = load_cached_metrics()
    
    if not metrics_master:
        st.warning("No metrics found. Ensure offline dataset processing completed.")
        return
        
    st.success("Models preloaded successfully")

    # Comparison table
    rows = []
    for key, meta in DATASET_META.items():
        if key not in metrics_master: continue
        stats = stats_master.get(key, {})
        results = metrics_master.get(key, {}).get("results", {})
        best = metrics_master.get(key, {}).get("best_model", "")
        bm = results.get(best, {})
        rows.append({
            "Dataset": meta["short_name"],
            "Domain": meta["domain"],
            "Total Rows": f"{stats.get('total_rows_available',0):,}",
            "Used": f"{stats.get('rows_used',0):,}",
            "Fraud %": f"{stats.get('sampled_fraud_ratio',0):.2%}",
            "Best Model": best,
            "F1": f"{bm.get('F1-Score',0):.4f}",
            "AUC": f"{bm.get('ROC-AUC',0):.4f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    # Best per dataset
    loaded_keys = [k for k in DATASET_META if k in metrics_master]
    cols = st.columns(len(loaded_keys))
    for i, key in enumerate(loaded_keys):
        with cols[i]:
            meta = DATASET_META[key]
            best = metrics_master.get(key, {}).get("best_model", "")
            f1 = metrics_master.get(key, {}).get("results", {}).get(best, {}).get("F1-Score", 0)
            _card(meta["short_name"], best or "-", f"F1: {f1:.4f}" if f1 else "")

    st.markdown("---")

    # Fraud distribution
    fd_cols = st.columns(len(loaded_keys))
    for i, key in enumerate(loaded_keys):
        with fd_cols[i]:
            stats = stats_master.get(key, {})
            fraud = stats.get("sampled_fraud_count", 0)
            legit = stats.get("rows_used", 0) - fraud
            fig = go.Figure(data=go.Pie(
                labels=["Legit", "Fraud"], values=[legit, fraud],
                marker_colors=["#0ea5e9","#ef4444"], hole=0.55,
                textinfo="label+percent"))
            fig.update_layout(title=DATASET_META[key]["short_name"],
                              height=280, margin=dict(t=40,b=10,l=10,r=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature importance
    for key in loaded_keys:
        models = st.session_state.get(f"{key}_models", {})
        best = st.session_state.get(f"{key}_best_model")
        fn = st.session_state.get(f"{key}_feature_names", [])
        if best and best in models:
            from modules.model_trainer import get_feature_importance
            imp = get_feature_importance(models[best], fn, best)
            if imp:
                top = list(imp.items())[:5]
                meta = DATASET_META[key]
                with st.expander(f"{meta['short_name']} -- Top 5 Features ({best})", expanded=True):
                    fig = go.Figure(go.Bar(
                        x=[v for _,v in top][::-1], y=[k for k,_ in top][::-1],
                        orientation="h", marker_color="#374151"))
                    fig.update_layout(height=200, margin=dict(l=150,t=10,b=10,r=10))
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Key Findings**")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="insight">'
            '<strong>High-value transactions carry elevated risk.</strong> '
            'In PaySim, TRANSFER and CASH_OUT operations above $200k have '
            'the highest fraud incidence. Full account drains are strong indicators.'
            '</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="insight">'
            '<strong>Network attacks cluster during business hours.</strong> '
            'CICIDS data shows DDoS and brute-force attacks peak Mon-Fri. '
            'High SYN flag counts and persistent connections are key markers.'
            '</div>', unsafe_allow_html=True)

    with c2:
        st.markdown(
            '<div class="insight">'
            '<strong>Tree-based models dominate consistently.</strong> '
            'Random Forest and XGBoost outperform Logistic Regression '
            'across all datasets by capturing non-linear feature interactions.'
            '</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="insight">'
            '<strong>Anonymous emails and address mismatches flag fraud.</strong> '
            'IEEE-CIS data reveals that unknown email providers and large '
            'billing-shipping distances are among the strongest predictors.'
            '</div>', unsafe_allow_html=True)


# ================================================================
#  MAIN
# ================================================================

def main():
    st.set_page_config(
        page_title=DASHBOARD_CONFIG["page_title"],
        page_icon=DASHBOARD_CONFIG.get("page_icon") or None,
        layout=DASHBOARD_CONFIG["layout"],
        initial_sidebar_state="expanded")

    _init()
    _header()

    with st.sidebar:
        st.markdown("## Navigation")
        st.markdown("---")
        page = st.radio("Page Selection", [
            "1. Data Management",
            "2. Model Evaluation",
            "3. Digital Twin Simulator",
            "4. Live Prediction",
            "5. Executive Insights",
        ], index=0, label_visibility="collapsed")

        st.markdown("---")
        st.markdown("### Status")
        metrics, _ = load_cached_metrics()
        models_c, _ = load_all_cached_models()
        
        for key, meta in DATASET_META.items():
            ok = key in metrics
            n = len(models_c.get(key, {}))
            mark = "●" if ok else "○"
            st.markdown(f"{mark} **{meta['short_name']}** -- {n} models")

        st.markdown("---")
        st.markdown(f"Threshold: {FRAUD_THRESHOLD}")
        st.caption("Nexus Fraud Defense v2.5")

    if "1." in page: _page_data()
    elif "2." in page: _page_eval()
    elif "3." in page: _page_twin()
    elif "4." in page: _page_predict()
    elif "5." in page: _page_insights()

    st.markdown(
        '<div class="app-footer">'
        'Digital Twin-Enabled Framework for Forecasting and Mitigating Fraud<br>'
        'Nexus Fraud Defense  |  PaySim  |  BankSim  |  CICIDS2017  |  IEEE-CIS'
        '</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

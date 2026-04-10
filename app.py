import streamlit as st
import pandas as pd
import numpy as np
import joblib
 
# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
try:
    st.set_page_config(
        page_title="HR AI Attrition Predictor",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass
 
# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Lato:wght@300;400;700&display=swap%22 rel="stylesheet">
<style>
:root {
  --bg:#f4f3ef; --surface:#ffffff; --surface2:#f0efe9;
  --border:#e2e0d8; --border-dark:#ccc9be;
  --text:#1a1916; --text-muted:#6b6860; --text-dim:#9c9a94;
  --red:#c0392b; --red-light:#fdf0ee; --red-mid:#f5c6c1;
  --amber:#b8620a; --amber-light:#fdf5e8; --amber-mid:#f5ddb0;
  --green:#1a6b3a; --green-light:#eaf5ee; --green-mid:#b5dfc3;
  --blue:#1a4a8a; --blue-light:#eef3fb;
  --r:10px; --r-sm:6px;
}
 
html,body,[class*="css"]{font-family:'Lato',sans-serif!important;background-color:var(--bg)!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
.stDeployButton{display:none!important;}
 
/* Topbar */
.topbar{position:fixed;top:0;left:0;right:0;height:52px;background:var(--text);color:#f4f3ef;
  display:flex;align-items:center;justify-content:space-between;padding:0 28px;z-index:9999;}
.topbar-brand{display:flex;align-items:center;gap:12px;}
.brand-shield{width:28px;height:28px;background:#f4f3ef;border-radius:5px;
  display:flex;align-items:center;justify-content:center;font-size:14px;}
.brand-name{font-family:'Syne',sans-serif;font-weight:700;font-size:15px;letter-spacing:-0.02em;color:#f4f3ef;}
.status-pill{display:flex;align-items:center;gap:6px;background:rgba(255,255,255,0.08);
  border:1px solid rgba(255,255,255,0.12);border-radius:20px;padding:4px 12px;
  font-size:11px;font-family:'IBM Plex Mono',monospace;color:rgba(255,255,255,0.7);}
.status-dot{width:6px;height:6px;border-radius:50%;background:#4ade80;animation:blink 2s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
 
/* Layout */
.block-container{padding-top:72px!important;max-width:1400px!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;top:52px!important;}
 
/* Elements */
.template-card{margin:12px 0;background:var(--surface2);border:1px solid var(--border);border-radius:var(--r);padding:16px;}
.col-chip{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:2px 6px;
  font-size:9px;font-family:'IBM Plex Mono',monospace;color:var(--text-muted);display:inline-block;margin:2px;}
.risk-badge{display:inline-flex;align-items:center;gap:6px;border-radius:20px;padding:4px 10px;font-size:11px;font-weight:500;}
.badge-critical{background:var(--red-light);color:var(--red);border:1px solid var(--red-mid);}
.badge-medium{background:var(--amber-light);color:var(--amber);border:1px solid var(--amber-mid);}
.badge-stable{background:var(--green-light);color:var(--green);border:1px solid var(--green-mid);}
 
/* Table */
.tbl-wrap{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;margin-top:20px;}
table{width:100%;border-collapse:collapse;font-size:13px;}
thead th{background:var(--surface2);padding:12px;text-align:left;font-family:'IBM Plex Mono';font-size:10px;text-transform:uppercase;color:var(--text-dim);border-bottom:1px solid var(--border);}
tbody td{padding:12px;border-bottom:1px solid var(--border);}
.rbar-track{width:60px;height:5px;background:var(--border);border-radius:3px;display:inline-block;margin-right:8px;vertical-align:middle;}
.rbar-fill{height:100%;border-radius:3px;}
</style>
""", unsafe_allow_html=True)
 
# ── TOPBAR ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
<div class="topbar-brand"><div class="brand-shield">🛡️</div><div class="brand-name">RetainIQ <span>/ HR Intelligence</span></div></div>
<div class="topbar-right"><div class="status-pill"><div class="status-dot"></div>Model Online</div></div>
</div>
""", unsafe_allow_html=True)
 
# ── ASSET LOADING ────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    try:
        m = joblib.load('attrition_xgb_model.pkl')
        sc = joblib.load('robust_scaler.pkl')
        cf = joblib.load('feature_columns.pkl')
        return m, sc, cf, True
    except:
        return None, None, None, False
 
model, scaler, core_features, model_loaded = load_assets()
 
# ── DATA PROCESSING ──────────────────────────────────────────────────────────
def get_sentiment(text):
    try:
        from textblob import TextBlob
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0.0
 
def process_data(df):
    dp = df.copy()
    # Feature Engineering
    dp['Comp_Ratio'] = dp['Base_Salary'] / (dp['Benchmark_Salary'] + 1)
    dp['Survey_Sentiment'] = dp['Feedback_Comments'].apply(get_sentiment)
    # Logic: If ML model isn't found, use a refined heuristic
    scores = []
    for _, r in dp.iterrows():
        s = 50.0 # Baseline
        s += 25 if r['Comp_Ratio'] < 0.85 else 10 if r['Comp_Ratio'] < 0.95 else -5
        s += (3 - r.get('Job_Satisfaction', 3)) * 8
        s -= r['Survey_Sentiment'] * 20
        scores.append(min(max(round(s, 1), 2), 99))
    dp['Risk_Score_%'] = scores
    dp['Risk_Tier'] = dp['Risk_Score_%'].apply(
        lambda x: 'High Risk (Critical)' if x >= 75 else ('Medium Risk (Monitor)' if x >= 40 else 'Low Risk (Stable)')
    )
    def get_action(row):
        if row['Risk_Score_%'] >= 75: return "Urgent Stay Interview / Salary Review"
        if row['Risk_Score_%'] >= 40: return "Manager 1-on-1 Mentorship"
        return "Regular Engagement"
    dp['Recommended_Action'] = dp.apply(get_action, axis=1)
    return dp
 
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown("### Navigation")
    st.info("📊 Risk Dashboard")
    st.markdown("""
<div class="template-card">
<div style="font-size:12px; font-weight:600; margin-bottom:8px;">📥 CSV Requirements</div>
<div class="col-chip">Employee_ID</div><div class="col-chip">Department</div>
<div class="col-chip">Base_Salary</div><div class="col-chip">Benchmark_Salary</div>
<div class="col-chip">Job_Satisfaction</div><div class="col-chip">Feedback_Comments</div>
</div>
    """, unsafe_allow_html=True)
 
# ── MAIN UI ──────────────────────────────────────────────────────────────────
st.markdown('<div style="font-size:28px; font-family:Syne; font-weight:700;">Employee Retention Risk Dashboard</div>', unsafe_allow_html=True)
st.write("Upload your workforce data to identify churn risk and view automated strategies.")
 
uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
 
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    processed = process_data(df)
    # Summary Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Headcount", len(processed))
    c2.metric("Critical Risks", len(processed[processed['Risk_Tier'] == 'High Risk (Critical)']))
    c3.metric("Avg Risk Score", f"{processed['Risk_Score_%'].mean():.1f}%")
    # Render Table
    rows_html = ""
    for _, row in processed.iterrows():
        sc = row['Risk_Score_%']
        color = "#c0392b" if sc >= 75 else "#b8620a" if sc >= 40 else "#1a6b3a"
        badge = "badge-critical" if sc >= 75 else "badge-medium" if sc >= 40 else "badge-stable"
        tier = "High" if sc >= 75 else "Medium" if sc >= 40 else "Low"
        rows_html += f"""
<tr>
<td style="font-family:'IBM Plex Mono';">{row['Employee_ID']}</td>
<td><span style="background:#eef3fb; padding:3px 8px; border-radius:4px; color:#1a4a8a;">{row['Department']}</span></td>
<td>{row['Comp_Ratio']:.2f}</td>
<td>
<div class="rbar-track"><div class="rbar-fill" style="width:{sc}%; background:{color};"></div></div>
<span style="font-family:'IBM Plex Mono'; font-weight:600; color:{color};">{sc}%</span>
</td>
<td><span class="risk-badge {badge}">{tier} Risk</span></td>
<td style="color:#6b6860;">{row['Recommended_Action']}</td>
</tr>
        """
    st.markdown(f"""
<div class="tbl-wrap">
<table>
<thead>
<tr>
<th>Emp ID</th><th>Department</th><th>Comp Ratio</th><th>Risk Level</th><th>Tier</th><th>Action</th>
</tr>
</thead>
<tbody>{rows_html}</tbody>
</table>
</div>
    """, unsafe_allow_html=True)
    st.download_button("Export Full Report", processed.to_csv(index=False), "RetainIQ_Report.csv", use_container_width=True)
else:
    st.markdown("""
<div style="text-align:center; padding:60px; border:1px dashed #ccc9be; border-radius:10px; margin-top:20px; color:#9c9a94;">
<div style="font-size:40px;">📂</div>
<div>No data loaded. Drag and drop your employee CSV to begin analysis.</div>
</div>
    """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR AI Attrition Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── INJECT CUSTOM CSS (mirrors hr_attrition_dashboard.html) ───────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Lato:wght@300;400;700&display=swap" rel="stylesheet">
<style>
/* ── CSS variables ── */
:root {
  --bg: #f4f3ef;
  --surface: #ffffff;
  --surface2: #f0efe9;
  --border: #e2e0d8;
  --border-dark: #ccc9be;
  --text: #1a1916;
  --text-muted: #6b6860;
  --text-dim: #9c9a94;
  --accent: #1a1916;
  --red: #c0392b;
  --red-light: #fdf0ee;
  --red-mid: #f5c6c1;
  --amber: #b8620a;
  --amber-light: #fdf5e8;
  --amber-mid: #f5ddb0;
  --green: #1a6b3a;
  --green-light: #eaf5ee;
  --green-mid: #b5dfc3;
  --blue: #1a4a8a;
  --blue-light: #eef3fb;
  --r: 10px;
  --r-sm: 6px;
}

/* ── Global resets ── */
html, body, [class*="css"] {
  font-family: 'Lato', sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }

/* ── App wrapper ── */
.stApp { background: var(--bg) !important; }

/* ── Topbar ── */
.topbar {
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 52px;
  background: var(--text);
  color: #f4f3ef;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 28px;
  z-index: 9999;
  box-shadow: 0 1px 0 rgba(255,255,255,0.05);
}
.topbar-brand { display: flex; align-items: center; gap: 12px; }
.brand-shield {
  width: 28px; height: 28px;
  background: #f4f3ef;
  border-radius: 5px;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px; flex-shrink: 0;
}
.brand-name {
  font-family: 'Syne', sans-serif;
  font-weight: 700; font-size: 15px;
  letter-spacing: -0.02em; color: #f4f3ef;
}
.brand-name span { font-weight: 400; opacity: 0.55; }
.topbar-right { display: flex; align-items: center; gap: 16px; }
.status-pill {
  display: flex; align-items: center; gap: 6px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 20px; padding: 4px 12px;
  font-size: 11px; font-family: 'IBM Plex Mono', monospace;
  color: rgba(255,255,255,0.7);
}
.status-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: #4ade80;
  animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
.topbar-user {
  width: 30px; height: 30px; border-radius: 50%;
  background: rgba(255,255,255,0.15);
  border: 1px solid rgba(255,255,255,0.2);
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700;
  color: #f4f3ef; font-family: 'Syne', sans-serif;
}

/* ── Push content below topbar ── */
.block-container {
  padding-top: 72px !important;
  padding-left: 24px !important;
  padding-right: 24px !important;
  max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
  top: 52px !important;
}
[data-testid="stSidebar"] > div { padding-top: 16px !important; }

/* ── Sidebar text ── */
.sb-section {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--text-dim);
  padding: 16px 10px 6px;
}
.sb-navitem {
  display: flex; align-items: center; gap: 10px;
  padding: 9px 12px; border-radius: var(--r-sm);
  color: var(--text-muted); font-size: 13px;
  border: 1px solid transparent; margin-bottom: 2px;
}
.sb-navitem.active {
  background: var(--text); color: #f4f3ef; font-weight: 500;
}
.sb-navitem:not(.active):hover { background: var(--surface2); color: var(--text); }

/* ── Template card in sidebar ── */
.template-card {
  margin: 12px 0;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 16px;
}
.template-title {
  font-family: 'Syne', sans-serif;
  font-size: 12px; font-weight: 600;
  color: var(--text); margin-bottom: 6px;
}
.template-desc { font-size: 11px; color: var(--text-muted); line-height: 1.5; margin-bottom: 10px; }
.template-cols { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; }
.col-chip {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px; padding: 2px 6px;
  font-size: 9px; font-family: 'IBM Plex Mono', monospace;
  color: var(--text-muted);
}

/* ── Sidebar model info ── */
.model-info {
  margin-top: 8px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 12px;
  font-size: 10px;
  font-family: 'IBM Plex Mono', monospace;
  color: var(--text-dim);
  line-height: 2;
}
.model-info strong { color: var(--text-muted); }

/* ── Page header ── */
.page-head {
  display: flex; justify-content: space-between;
  align-items: flex-start; margin-bottom: 28px;
}
.page-eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--text-dim);
  margin-bottom: 8px;
}
.page-title {
  font-family: 'Syne', sans-serif;
  font-size: 28px; font-weight: 700;
  letter-spacing: -0.03em; line-height: 1.15;
  color: var(--text); margin-bottom: 10px;
}
.page-sub {
  font-size: 13px; color: var(--text-muted);
  line-height: 1.6; max-width: 520px;
}

/* ── Upload zone ── */
.upload-zone {
  background: var(--surface);
  border: 1.5px dashed var(--border-dark);
  border-radius: var(--r);
  padding: 40px 24px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
  margin-bottom: 24px;
}
.upload-zone:hover { border-color: var(--text); background: var(--surface2); }
.upload-zone.loaded { border-style: solid; border-color: var(--green); background: var(--green-light); }
.upload-icon { font-size: 32px; margin-bottom: 10px; }
.upload-title {
  font-family: 'Syne', sans-serif;
  font-weight: 600; font-size: 14px;
  color: var(--text); margin-bottom: 4px;
}
.upload-hint { font-size: 12px; color: var(--text-muted); margin-bottom: 14px; }
.upload-tags { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; }
.upload-tag {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 4px; padding: 3px 8px;
  font-size: 10px; font-family: 'IBM Plex Mono', monospace;
  color: var(--text-muted);
}

/* ── Metric cards ── */
.metrics-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 24px; }
.metric-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 20px 20px 18px;
}
.metric-card.critical { border-left: 3px solid var(--red); }
.metric-card.avg { border-left: 3px solid var(--amber); }
.metric-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--text-dim);
  margin-bottom: 8px;
}
.metric-value {
  font-family: 'Syne', sans-serif;
  font-size: 36px; font-weight: 700;
  letter-spacing: -0.03em; line-height: 1;
  color: var(--text); margin-bottom: 6px;
}
.metric-card.critical .metric-value { color: var(--red); }
.metric-card.avg .metric-value { color: var(--amber); }
.metric-sub { font-size: 11px; color: var(--text-dim); }

/* ── Section row ── */
.section-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.section-head {
  font-family: 'Syne', sans-serif;
  font-size: 15px; font-weight: 600; color: var(--text);
}

/* ── Table wrapper ── */
.tbl-wrap {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  overflow: hidden; margin-bottom: 20px;
  overflow-x: auto;
}
table { width: 100%; border-collapse: collapse; font-size: 13px; }
thead th {
  background: var(--surface2);
  padding: 10px 14px;
  text-align: left;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--text-dim);
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
tbody tr { border-bottom: 1px solid var(--border); transition: background 0.1s; }
tbody tr:last-child { border-bottom: none; }
tbody tr:hover { background: var(--surface2); }
tbody td { padding: 11px 14px; vertical-align: middle; }

/* ── Risk bar ── */
.rbar-wrap { display: flex; align-items: center; gap: 8px; min-width: 120px; }
.rbar-track {
  flex: 1; height: 5px; border-radius: 3px;
  background: var(--border); overflow: hidden;
}
.rbar-fill { height: 100%; border-radius: 3px; transition: width 0.6s ease; }
.rbar-num { font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 500; min-width: 36px; }

/* ── Badges ── */
.risk-badge {
  display: inline-flex; align-items: center; gap: 6px;
  border-radius: 20px; padding: 4px 10px;
  font-size: 11px; font-weight: 500; white-space: nowrap;
}
.badge-critical { background: var(--red-light); color: var(--red); border: 1px solid var(--red-mid); }
.badge-medium   { background: var(--amber-light); color: var(--amber); border: 1px solid var(--amber-mid); }
.badge-stable   { background: var(--green-light); color: var(--green); border: 1px solid var(--green-mid); }
.badge-dot { width: 6px; height: 6px; border-radius: 50%; }

/* ── Dept tag ── */
.dept-tag {
  background: var(--blue-light);
  color: var(--blue);
  border-radius: 4px; padding: 3px 8px;
  font-size: 11px; font-weight: 500;
}

/* ── Action tag ── */
.action-tag { font-size: 12px; color: var(--text-muted); }

/* ── Emp mono ── */
.emp-mono { font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 500; }

/* ── Info bar ── */
.info-bar {
  background: var(--green-light);
  border: 1px solid var(--green-mid);
  border-radius: var(--r);
  padding: 14px 20px;
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 12px;
}
.info-text { font-size: 13px; color: var(--green); display: flex; align-items: center; gap: 8px; }

/* ── Footnote ── */
.footnote {
  font-size: 11px; color: var(--text-dim);
  display: flex; align-items: center; gap: 6px;
  padding: 0 4px; margin-bottom: 24px;
}

/* ── Buttons ── */
.btn-solid {
  padding: 9px 18px; border-radius: var(--r-sm);
  background: var(--text); color: #f4f3ef;
  border: none; font-family: 'Lato', sans-serif;
  font-size: 13px; font-weight: 700;
  cursor: pointer; letter-spacing: 0.02em;
  transition: opacity 0.15s;
}
.btn-solid:hover { opacity: 0.85; }
.btn-outline {
  padding: 9px 18px; border-radius: var(--r-sm);
  background: transparent; color: var(--text);
  border: 1px solid var(--border-dark);
  font-family: 'Lato', sans-serif; font-size: 13px;
  cursor: pointer; transition: background 0.12s;
}
.btn-outline:hover { background: var(--surface2); }

/* ── Override Streamlit native elements ── */
.stFileUploader { margin-bottom: 0 !important; }
.stFileUploader > label { display: none !important; }
div[data-testid="stFileUploader"] section {
  background: var(--surface) !important;
  border: 1.5px dashed var(--border-dark) !important;
  border-radius: var(--r) !important;
  padding: 32px 24px !important;
}
div[data-testid="stFileUploader"] section:hover {
  border-color: var(--text) !important;
  background: var(--surface2) !important;
}
.stSelectbox > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  font-size: 13px !important;
  color: var(--text) !important;
}
.stButton button {
  background: var(--text) !important;
  color: #f4f3ef !important;
  border: none !important;
  font-family: 'Lato', sans-serif !important;
  font-weight: 700 !important;
  border-radius: var(--r-sm) !important;
}
.stDownloadButton button {
  background: var(--text) !important;
  color: #f4f3ef !important;
  border: none !important;
  font-family: 'Lato', sans-serif !important;
  font-weight: 700 !important;
  border-radius: var(--r-sm) !important;
}
.stSpinner > div { color: var(--text) !important; }
div[data-testid="metric-container"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 16px 20px !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  font-size: 36px !important;
  font-weight: 700 !important;
  letter-spacing: -0.03em !important;
  color: var(--text) !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 10px !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--text-dim) !important;
}
.stDivider { border-color: var(--border) !important; }
h1,h2,h3 {
  font-family: 'Syne', sans-serif !important;
  color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ── TOPBAR ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">
    <div class="brand-shield">🛡️</div>
    <div class="brand-name">RetainIQ <span>/ HR Intelligence</span></div>
  </div>
  <div class="topbar-right">
    <div class="status-pill">
      <div class="status-dot"></div>
      Model Online
    </div>
    <div class="topbar-user">HR</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD ASSETS ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    try:
        import joblib
        model      = joblib.load('attrition_xgb_model.pkl')
        scaler     = joblib.load('robust_scaler.pkl')
        core_feats = joblib.load('feature_columns.pkl')
        return model, scaler, core_feats, True
    except Exception:
        return None, None, None, False

model, scaler, core_features, model_loaded = load_assets()

# ── TEMPLATE CSV ──────────────────────────────────────────────────────────────
template_df = pd.DataFrame({
    'Employee_ID':       ['EMP001'],
    'Department':        ['Sales'],
    'Role':              ['Manager'],
    'Work_Location':     ['Remote'],
    'Base_Salary':       [70000],
    'Benchmark_Salary':  [75000],
    'Job_Satisfaction':  [3],
    'Engagement_Level':  [3],
    'Work_Life_Balance': [3],
    'Management_Support':[3],
    'Career_Development':[3],
    'Tenure_Years':      [2.5],
    'Employment_Type':   ['Full-time'],
    'Feedback_Comments': ['Sample feedback here.'],
})
template_csv = template_df.to_csv(index=False).encode('utf-8')

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-navitem active">📊 &nbsp;Risk Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-navitem">📋 &nbsp;Triage List</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-navitem">📈 &nbsp;Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-navitem">⚙️ &nbsp;Settings</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section" style="margin-top:12px">Data Input</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="template-card">
      <div class="template-title">📥 CSV Template</div>
      <div class="template-desc">Download the required column structure before uploading your data.</div>
      <div class="template-cols">
        <span class="col-chip">Employee_ID</span>
        <span class="col-chip">Department</span>
        <span class="col-chip">Base_Salary</span>
        <span class="col-chip">Benchmark_Salary</span>
        <span class="col-chip">Tenure_Years</span>
        <span class="col-chip">Career_Development</span>
        <span class="col-chip">Management_Support</span>
        <span class="col-chip">Employment_Type</span>
        <span class="col-chip">Feedback_Comments</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.download_button(
        "↓  Download CSV Template",
        data=template_csv,
        file_name="hr_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown('<div class="sb-section" style="margin-top:16px">Model Info</div>', unsafe_allow_html=True)
    model_status = "✅ Loaded" if model_loaded else "⚠️  Demo mode (no .pkl)"
    st.markdown(f"""
    <div class="model-info">
      <strong>Algorithm:</strong> XGBoost<br>
      <strong>Scaler:</strong> RobustScaler<br>
      <strong>NLP:</strong> TextBlob Sentiment<br>
      <strong>Status:</strong> {model_status}
    </div>
    """, unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def risk_color(score):
    if score >= 75: return "#c0392b"
    if score >= 40: return "#b8620a"
    return "#1a6b3a"

def badge_class(score):
    if score >= 75: return "badge-critical"
    if score >= 40: return "badge-medium"
    return "badge-stable"

def tier_label(score):
    if score >= 75: return "High Risk (Critical)"
    if score >= 40: return "Medium Risk (Monitor)"
    return "Low Risk (Stable)"

def tier_short(score):
    if score >= 75: return "High Risk"
    if score >= 40: return "Medium Risk"
    return "Low Risk"

def sentiment_label(s):
    if s < -0.1: return f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#c0392b">{s:.2f} neg</span>'
    if s >  0.1: return f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#1a6b3a">{s:.2f} pos</span>'
    return f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#9c9a94">{s:.2f} neu</span>'

def assign_strategy(row):
    score = row['Risk_Score_%']
    if score >= 75:
        tier = 'High Risk (Critical)'
        if row.get('Comp_Ratio', 1) < 0.9:
            action = 'Urgent Salary Correction'
        elif row.get('Management_Support', 3) < 3:
            action = 'Skip-Level Meeting / Manager Review'
        else:
            action = 'Immediate Stay Interview'
    elif score >= 40:
        tier = 'Medium Risk (Monitor)'
        action = 'Engagement Project / Flex Work'
    else:
        tier = 'Low Risk (Stable)'
        action = 'Standard Engagement'
    return pd.Series([tier, action])

def process_dataframe(df):
    """Feature engineering + scoring (real model or heuristic demo)."""
    df_p = df.copy()

    # ── Feature engineering ──
    df_p['Comp_Ratio'] = df_p['Base_Salary'] / (df_p['Benchmark_Salary'] + 1)
    df_p['Stagnation_Index'] = df_p['Tenure_Years'] / (df_p['Career_Development'] + 0.1)
    df_p['Is_Contractor'] = np.where(df_p['Employment_Type'] == 'Contract', 1, 0)

    try:
        from textblob import TextBlob
        df_p['Survey_Sentiment'] = df_p['Feedback_Comments'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0.0
        )
    except ImportError:
        df_p['Survey_Sentiment'] = 0.0

    # ── Scoring ──
    if model_loaded and model is not None:
        try:
            X = df_p[core_features]
            X_s = scaler.transform(X)
            probs = model.predict_proba(X_s)[:, 1]
            df_p['Risk_Score_%'] = (probs * 100).round(1)
        except Exception:
            model_loaded_flag = False
            _heuristic_score(df_p)
    else:
        _heuristic_score(df_p)

    df_p[['Risk_Tier', 'Recommended_Action']] = df_p.apply(assign_strategy, axis=1)
    return df_p

def _heuristic_score(df_p):
    """Fallback scoring when no model is present."""
    scores = []
    for _, row in df_p.iterrows():
        s = 50.0
        comp   = row.get('Comp_Ratio', 1.0)
        sat    = row.get('Job_Satisfaction', 3)
        eng    = row.get('Engagement_Level', 3)
        wlb    = row.get('Work_Life_Balance', 3)
        mgmt   = row.get('Management_Support', 3)
        career = row.get('Career_Development', 3)
        sent   = row.get('Survey_Sentiment', 0.0)
        stag   = row.get('Stagnation_Index', 1.0)

        if comp < 0.85:  s += 25
        elif comp < 0.95: s += 10

        s -= (sat - 3) * 6
        s -= (eng - 3) * 5
        s -= (wlb - 3) * 4
        s -= (mgmt - 3) * 5
        s -= (career - 3) * 5
        s -= sent * 15
        s += min(stag * 3, 15)
        scores.append(min(max(round(s, 1), 0), 100))
    df_p['Risk_Score_%'] = scores

def render_table_html(df_p):
    rows_html = ""
    for _, row in df_p.iterrows():
        score   = row['Risk_Score_%']
        color   = risk_color(score)
        bc      = badge_class(score)
        tl      = tier_short(score)
        comp    = row.get('Comp_Ratio', 0)
        sent    = row.get('Survey_Sentiment', 0)
        dept    = row.get('Department', '')
        role    = row.get('Role', '')
        emp_id  = row.get('Employee_ID', '')
        action  = row.get('Recommended_Action', '')

        rows_html += f"""
        <tr>
          <td class="emp-mono">{emp_id}</td>
          <td><span class="dept-tag">{dept}</span></td>
          <td style="color:var(--text-muted);font-size:12px">{role}</td>
          <td style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--text-muted)">{comp:.2f}</td>
          <td>{sentiment_label(sent)}</td>
          <td>
            <div class="rbar-wrap">
              <div class="rbar-track">
                <div class="rbar-fill" style="width:{score}%;background:{color}"></div>
              </div>
              <span class="rbar-num" style="color:{color}">{score}%</span>
            </div>
          </td>
          <td>
            <span class="risk-badge {bc}">
              <span class="badge-dot" style="background:{color}"></span>
              {tl}
            </span>
          </td>
          <td><div class="action-tag">{action}</div></td>
        </tr>"""

    return f"""
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>Employee ID</th>
            <th>Department</th>
            <th>Role</th>
            <th>Comp Ratio</th>
            <th>Sentiment</th>
            <th>Risk Score</th>
            <th>Risk Tier</th>
            <th>Recommended Action</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""

# ── MAIN CONTENT ──────────────────────────────────────────────────────────────

# Page header
st.markdown("""
<div class="page-head">
  <div>
    <div class="page-eyebrow">AI-Driven Workforce Intelligence</div>
    <div class="page-title">Employee Retention<br>Risk Dashboard</div>
    <div class="page-sub">Upload employee survey and compensation data to identify flight risks and receive automated intervention strategies.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Upload zone label
st.markdown("""
<div class="upload-zone">
  <div class="upload-icon">📂</div>
  <div class="upload-title">Upload Employee Data (CSV Format)</div>
  <div class="upload-hint">Drag & drop your file, or click to browse</div>
  <div class="upload-tags">
    <span class="upload-tag">Employee_ID</span>
    <span class="upload-tag">Base_Salary</span>
    <span class="upload-tag">Benchmark_Salary</span>
    <span class="upload-tag">Tenure_Years</span>
    <span class="upload-tag">Career_Development</span>
    <span class="upload-tag">Employment_Type</span>
    <span class="upload-tag">Feedback_Comments</span>
    <span class="upload-tag">Management_Support</span>
  </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"],
    label_visibility="collapsed",
)

# ── RESULTS ───────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    with st.spinner("AI is analysing flight risk factors across all employees…"):
        raw_df = pd.read_csv(uploaded_file)
        df_proc = process_dataframe(raw_df)

    # ── Metrics ──
    total    = len(df_proc)
    critical = len(df_proc[df_proc['Risk_Tier'] == 'High Risk (Critical)'])
    avg_risk = df_proc['Risk_Score_%'].mean().round(1)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Employees Analysed", total)
        st.caption("From latest CSV upload")
    with c2:
        st.metric("Critical Risks Detected", critical)
        st.caption("Require immediate action")
    with c3:
        st.metric("Average Risk Score", f"{avg_risk}%")
        st.caption("Across all employees")

    st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

    # ── Filter row ──
    col_head, col_dept, col_risk, col_sort = st.columns([3, 1.2, 1.2, 1.2])
    with col_head:
        st.markdown('<div class="section-head" style="padding-top:8px">📋 Employee Triage List</div>', unsafe_allow_html=True)
    with col_dept:
        depts = ["All Departments"] + sorted(df_proc['Department'].dropna().unique().tolist())
        dept_filter = st.selectbox("Department", depts, label_visibility="collapsed")
    with col_risk:
        risk_opts = ["All Risk Tiers", "High Risk (Critical)", "Medium Risk (Monitor)", "Low Risk (Stable)"]
        risk_filter = st.selectbox("Risk Tier", risk_opts, label_visibility="collapsed")
    with col_sort:
        sort_opt = st.selectbox("Sort", ["Sort: Risk ↓", "Sort: Risk ↑"], label_visibility="collapsed")

    # Apply filters
    view = df_proc.copy()
    if dept_filter != "All Departments":
        view = view[view['Department'] == dept_filter]
    if risk_filter != "All Risk Tiers":
        view = view[view['Risk_Tier'] == risk_filter]
    ascending = sort_opt == "Sort: Risk ↑"
    view = view.sort_values('Risk_Score_%', ascending=ascending).reset_index(drop=True)

    # ── Render table ──
    st.markdown(render_table_html(view), unsafe_allow_html=True)

    # ── Info bar + export ──
    export_csv = df_proc.to_csv(index=False).encode('utf-8')

    st.markdown(f"""
    <div class="info-bar">
      <div class="info-text">
        <span>✓</span>
        Analysis complete — {total} employees scored. Actionable strategies assigned to all high and medium risk profiles.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        label="↓  Download Full Actionable Report (CSV)",
        data=export_csv,
        file_name="Actionable_HR_Attrition_Report.csv",
        mime="text/csv",
    )

    st.markdown("""
    <div class="footnote">
      <span>🔒</span>
      All data is processed locally · Pipeline: Comp_Ratio · Stagnation_Index · Is_Contractor · Survey_Sentiment (TextBlob NLP)
    </div>
    """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style="
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r);
      padding: 48px 32px;
      text-align: center;
      color: var(--text-dim);
      font-size: 13px;
      line-height: 1.7;
      margin-top: 8px;
    ">
      <div style="font-size:32px;margin-bottom:12px">📊</div>
      <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:600;color:var(--text);margin-bottom:8px">
        No data uploaded yet
      </div>
      Upload a CSV file above to begin the AI risk analysis.<br>
      Use the <strong>Download CSV Template</strong> button in the sidebar to get the correct column format.
    </div>
    """, unsafe_allow_html=True)

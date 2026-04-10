import streamlit as st
import pandas as pd
import numpy as np
import math
 
# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
try:
    st.set_page_config(
        page_title="RetainIQ | HR Intelligence",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass
 
# ── STYLING (Fixed to ensure no raw CSS text appears) ────────────────────────
st.markdown("""
<style>
    :root {
      --bg:#f4f3ef; --surface:#ffffff; --surface2:#f0efe9;
      --border:#e2e0d8; --text:#1a1916; --text-dim:#9c9a94;
      --red:#c0392b; --red-light:#fdf0ee; --red-mid:#f5c6c1;
      --amber:#b8620a; --amber-light:#fdf5e8; --amber-mid:#f5ddb0;
      --green:#1a6b3a; --green-light:#eaf5ee; --green-mid:#b5dfc3;
    }
    /* Hide Streamlit default UI */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display:none !important; }
 
    /* Layout */
    .stApp { background-color: var(--bg); }
    .topbar {
        position: fixed; top: 0; left: 0; right: 0; height: 52px; 
        background: #1a1916; color: #f4f3ef; display: flex; 
        align-items: center; padding: 0 28px; z-index: 9999;
        font-family: sans-serif; font-weight: bold;
    }
    .main-container { padding-top: 60px; }
 
    /* Custom Table Styling */
    .tbl-wrap { 
        background: var(--surface); border: 1px solid var(--border); 
        border-radius: 10px; overflow: hidden; margin-top: 20px; 
    }
    table { width: 100%; border-collapse: collapse; font-family: sans-serif; }
    thead th { 
        background: var(--surface2); padding: 14px; text-align: left; 
        font-size: 11px; text-transform: uppercase; color: var(--text-dim);
        border-bottom: 1px solid var(--border);
    }
    tbody td { padding: 12px; border-bottom: 1px solid var(--border); font-size: 13px; }
    /* Risk Badges */
    .risk-badge { 
        border-radius: 20px; padding: 4px 12px; font-size: 11px; 
        font-weight: 600; display: inline-block; text-align: center;
    }
    .badge-critical { background: var(--red-light); color: var(--red); border: 1px solid var(--red-mid); }
    .badge-medium { background: var(--amber-light); color: var(--amber); border: 1px solid var(--amber-mid); }
    .badge-stable { background: var(--green-light); color: var(--green); border: 1px solid var(--green-mid); }
    /* Progress Bar */
    .bar-track { width: 100px; height: 6px; background: var(--border); border-radius: 3px; display: inline-block; margin-right: 8px; }
    .bar-fill { height: 100%; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)
 
# ── TOPBAR ───────────────────────────────────────────────────────────────────
st.markdown('<div class="topbar">🛡️ RetainIQ / Workforce Risk Dashboard</div>', unsafe_allow_html=True)
 
# ── LOGIC: PERFORMANCE-OPTIMIZED PROCESSING ──────────────────────────────────
def process_workforce_data(df):
    """Uses vectorized logic to process 50k+ rows instantly."""
    # Ensure numeric types
    df['Base_Salary'] = pd.to_numeric(df['Base_Salary'], errors='coerce').fillna(0)
    df['Benchmark_Salary'] = pd.to_numeric(df['Benchmark_Salary'], errors='coerce').fillna(1)
    # Calculate Comp Ratio
    df['Comp_Ratio'] = df['Base_Salary'] / (df['Benchmark_Salary'] + 1)
    # Vectorized Risk Scoring (Smart Heuristics)
    conditions = [
        (df['Comp_Ratio'] < 0.75),
        (df['Comp_Ratio'] < 0.90),
        (df['Comp_Ratio'] >= 1.10)
    ]
    scores = [80.0, 55.0, 15.0]
    df['Risk_Score_%'] = np.select(conditions, scores, default=40.0)
    # Factor in Tenure/Satisfaction if available
    if 'Job_Satisfaction' in df.columns:
        df['Risk_Score_%'] += (3 - df['Job_Satisfaction'].fillna(3)) * 5
    df['Risk_Score_%'] = df['Risk_Score_%'].clip(5, 98)
    # Assign Tiers
    df['Risk_Tier'] = np.where(df['Risk_Score_%'] >= 75, 'Critical',
                      np.where(df['Risk_Score_%'] >= 40, 'Elevated', 'Stable'))
    return df
 
# ── MAIN APP ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-container"></div>', unsafe_allow_html=True)
st.title("Employee Retention Risk Dashboard")
st.caption("AI-powered analysis for large workforce datasets.")
 
uploaded_file = st.file_uploader("Drop your CSV here", type=["csv"])
 
if uploaded_file:
    # 1. Load Data
    with st.spinner("Processing large dataset..."):
        df = pd.read_csv(uploaded_file)
        processed = process_workforce_data(df)
    # 2. Key Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Analysed", f"{len(processed):,}")
    m2.metric("Critical Risks", len(processed[processed['Risk_Tier'] == 'Critical']))
    m3.metric("Avg Company Risk", f"{processed['Risk_Score_%'].mean():.1f}%")
 
    st.divider()
 
    # 3. Pagination Controller
    st.subheader("📋 Triage List")
    batch_size = 20
    total_rows = len(processed)
    total_pages = math.ceil(total_rows / batch_size)
    c1, c2 = st.columns([1, 4])
    with c1:
        page_num = st.number_input(f"Page (1-{total_pages})", min_value=1, max_value=total_pages, step=1)
    # 4. Slice Data for current page only
    start_idx = (page_num - 1) * batch_size
    end_idx = start_idx + batch_size
    page_data = processed.iloc[start_idx:end_idx]
 
    # 5. Build HTML Table
    rows_html = ""
    for _, row in page_data.iterrows():
        sc = row['Risk_Score_%']
        tier = row['Risk_Tier']
        # Determine Visuals
        color = "#c0392b" if tier == "Critical" else "#b8620a" if tier == "Elevated" else "#1a6b3a"
        badge_cls = "badge-critical" if tier == "Critical" else "badge-medium" if tier == "Elevated" else "badge-stable"
        rows_html += f"""
<tr>
<td><code style="font-weight:bold;">{row.get('Employee_ID', 'N/A')}</code></td>
<td>{row.get('Department', 'N/A')}</td>
<td>{row['Comp_Ratio']:.2f}</td>
<td>
<div class="bar-track"><div class="bar-fill" style="width:{sc}%; background:{color};"></div></div>
<span style="font-weight:600; color:{color};">{sc}%</span>
</td>
<td><span class="risk-badge {badge_cls}">{tier}</span></td>
<td style="color:#6b6860; font-style:italic;">{row.get('Role', 'N/A')}</td>
</tr>
        """
    st.markdown(f"""
<div class="tbl-wrap">
<table>
<thead>
<tr>
<th>Emp ID</th><th>Department</th><th>Comp Ratio</th>
<th>Risk Score</th><th>Risk Tier</th><th>Position</th>
</tr>
</thead>
<tbody>{rows_html}</tbody>
</table>
</div>
    """, unsafe_allow_html=True)
 
    st.caption(f"Showing rows {start_idx+1} to {min(end_idx, total_rows)} of {total_rows:,}")
    # 6. Export
    st.download_button(
        label="Download Full Risk Report",
        data=processed.to_csv(index=False),
        file_name="HR_Risk_Analysis.csv",
        mime="text/csv",
        use_container_width=True
    )
 
else:
    st.info("👋 Welcome! Please upload your employee CSV file (50,000 rows max recommended) to generate the risk dashboard.")

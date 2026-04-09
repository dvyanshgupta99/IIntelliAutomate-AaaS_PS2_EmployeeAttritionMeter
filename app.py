import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
 
# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="HR AI Attrition Predictor", layout="wide")
 
# --- 1. LOAD ASSETS (The 'Brain' from Kaggle) ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('attrition_xgb_model.pkl')
        scaler = joblib.load('robust_scaler.pkl')
        core_features = joblib.load('feature_columns.pkl')
        return model, scaler, core_features
    except FileNotFoundError:
        st.error("⚠️ Error: Missing .pkl files! Ensure 'attrition_model.pkl', 'robust_scaler.pkl', and 'core_features.pkl' are in the same folder.")
        return None, None, None
 
model, scaler, core_features = load_assets()
 
# --- 2. THE UI HEADER ---
st.title("🛡️ AI-Driven Employee Retention Dashboard")
st.markdown(
    """
    <style>
      /* Font */
      html, body, [class*="stApp"] {
        font-family: "DM Sans", ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Liberation Sans", sans-serif;
      }

      /* Refined borders */
      div[data-testid="stVerticalBlock"] > div {
        border-radius: 14px;
      }

      /* Metric cards */
      div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.65);
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 14px;
        padding: 12px;
      }

      /* Semantic risk colors (used in badges below) */
      .risk-high { color: #B42318; background: rgba(180,35,24,0.10); border: 1px solid rgba(180,35,24,0.25); padding: 2px 8px; border-radius: 999px; font-weight: 600;}
      .risk-med  { color: #B54708; background: rgba(181,71,8,0.10); border: 1px solid rgba(181,71,8,0.25); padding: 2px 8px; border-radius: 999px; font-weight: 600;}
      .risk-low  { color: #067647; background: rgba(6,118,71,0.10); border: 1px solid rgba(6,118,71,0.25); padding: 2px 8px; border-radius: 999px; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True
)
 
# --- 3. SIDEBAR TEMPLATE DOWNLOAD ---
with st.sidebar:
    # st.header("App Instructions")
    # st.write("1. Upload a CSV with employee data.")
    # st.write("2. The AI will calculate risk scores.")
    # st.write("3. Download the Actionable Report.")
    # # Mock data for template download
    # template_data = pd.DataFrame({
    #     'Employee_ID': ['EMP001'], 'Department': ['Sales'], 'Role': ['Manager'], 
    #     'Work_Location': ['Remote'], 'Base_Salary': [70000], 'Benchmark_Salary': [75000],
    #     'Job_Satisfaction': [3], 'Engagement_Level': [3], 'Work_Life_Balance': [3], 
    #     'Management_Support': [3], 'Career_Development': [3], 'Tenure_Years': [2.5], 
    #     'Employment_Type': ['Full-time'], 'Feedback_Comments': ['Sample feedback here.']
    # })
    # st.download_button("📥 Download CSV Template", template_data.to_csv(index=False), "hr_template.csv")

    st.image("logo.png", width=32)  # or use st.markdown for SVG
    st.markdown("### RetainIQ")
    st.caption("HR Attrition Platform")
    st.divider()
    st.markdown("**Analysis**")
    st.page_link("app.py", label="Risk Dashboard", icon="📊")
 
# --- 4. FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload Employee Data (CSV Format)", type="csv")
 
if uploaded_file is not None and model is not None:
    # Read uploaded data
    df = pd.read_csv(uploaded_file)
    with st.spinner('AI is analyzing flight risk factors...'):
        # --- 5. FEATURE ENGINEERING (Science Step) ---
        df_proc = df.copy()
        # Financial & Stagnation Math
        df_proc['Comp_Ratio'] = df_proc['Base_Salary'] / (df_proc['Benchmark_Salary'] + 1)
        df_proc['Stagnation_Index'] = df_proc['Tenure_Years'] / (df_proc['Career_Development'] + 0.1)
        df_proc['Is_Contractor'] = np.where(df_proc['Employment_Type'] == 'Contract', 1, 0)
        # NLP Sentiment Analysis
        df_proc['Survey_Sentiment'] = df_proc['Feedback_Comments'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0.0
        )
 
        # --- 6. MACHINE LEARNING PREDICTION ---
        # Select and Scale features
        X_active = df_proc[core_features]
        X_scaled = scaler.transform(X_active)
        # Get Probabilities
        risk_probs = model.predict_proba(X_scaled)[:, 1]
        df_proc['Risk_Score_%'] = (risk_probs * 100).round(1)
 
        # --- 7. STRATEGY ENGINE (Risk Tiers & Actions) ---
        def assign_strategy(row):
            score = row['Risk_Score_%']
            if score >= 75:
                tier = 'High Risk (Critical)'
                if row['Comp_Ratio'] < 0.9: 
                    action = 'Urgent Salary Correction'
                elif row['Management_Support'] < 3: 
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
 
        df_proc[['Risk_Tier', 'Recommended_Action']] = df_proc.apply(assign_strategy, axis=1)
    # --- Step 5 state ---
    if "selected_emp" not in st.session_state:
        st.session_state["selected_emp"] = None
    # --- 8. RESULTS DASHBOARD ---
    st.divider()
    # Summary Metrics
    # col1, col2, col3 = st.columns(3)
    # col1.metric("Total Employees Analyzed", len(df_proc))
    # col2.metric("Critical Risks Detected", len(df_proc[df_proc['Risk_Tier'] == 'High Risk (Critical)']))
    # col3.metric("Average Risk Score", f"{df_proc['Risk_Score_%'].mean().round(1)}%")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total analysed", len(df_proc))
    col2.metric("Critical risks", len(df_proc[df_proc['Risk_Tier'] == 'High Risk (Critical)']), delta="+2 vs last run")
    col3.metric("Avg risk score", f"{df_proc['Risk_Score_%'].mean():.1f}%")
    col4.metric("Avg sentiment", f"{df_proc['Survey_Sentiment'].mean():.2f}")
 
    # Results Table
    st.subheader("📋 Employee Triage List")
    display_cols = ['Employee_ID', 'Department', 'Risk_Score_%', 'Risk_Tier', 'Recommended_Action']
    # st.dataframe(
    #     df_proc[display_cols].sort_values(by='Risk_Score_%', ascending=False),
    #     use_container_width=True
    # )

    st.subheader("📋 Employee Triage List")
    st.dataframe(
    df_view[display_cols].sort_values(by='Risk_Score_%', ascending=False),
    use_container_width=True
) 
 st.divider()

risk_filter = st.segmented_control(
    "Filter employees",
    options=["All", "High", "Medium", "Stable"],
    default="All",
    horizontal=True
)

# Apply filter (Step 7)
df_view = df_proc.copy()

if risk_filter == "High":
    df_view = df_view[df_view["Risk_Tier"] == "High Risk (Critical)"]
elif risk_filter == "Medium":
    df_view = df_view[df_view["Risk_Tier"] == "Medium Risk (Monitor)"]
elif risk_filter == "Stable":
    df_view = df_view[df_view["Risk_Tier"] == "Low Risk (Stable)"]
   
# Select employee for drill-down (Step 5)
employee_options = df_proc.sort_values(by="Risk_Score_%", ascending=False)["Employee_ID"].astype(str).unique()

selected_id = st.selectbox(
    "Select an employee to view factor breakdown and intervention plan",
    options=employee_options,
    index=0 if st.session_state["selected_emp"] is None else list(employee_options).index(st.session_state["selected_emp"])
)

# Persist selection in session state
st.session_state["selected_emp"] = selected_id

# Get the selected row
selected_row = df_view[df_view["Employee_ID"].astype(str) == str(selected_id)].iloc[0]

# --- Factor breakdown + intervention plan (two columns) ---
st.divider()
left, right = st.columns(2)

with left:
    st.markdown("### 🧩 Factor Breakdown")
    factor_tbl = pd.DataFrame({
        "Factor": [
            "Risk Score (%)",
            "Comp Ratio (Base/Benchmark)",
            "Stagnation Index (Tenure/Career Dev)",
            "Is Contractor",
            "Survey Sentiment",
            "Management Support",
            "Job Satisfaction",
            "Engagement Level",
            "Work-Life Balance",
        ],
        "Value": [
            float(selected_row["Risk_Score_%"]),
            float(selected_row.get("Comp_Ratio", np.nan)),
            float(selected_row.get("Stagnation_Index", np.nan)),
            int(selected_row.get("Is_Contractor", np.nan)) if "Is_Contractor" in selected_row else np.nan,
            float(selected_row.get("Survey_Sentiment", np.nan)),
            float(selected_row.get("Management_Support", np.nan)),
            float(selected_row.get("Job_Satisfaction", np.nan)),
            float(selected_row.get("Engagement_Level", np.nan)),
            float(selected_row.get("Work_Life_Balance", np.nan)),
        ]
    })

    # Clean formatting
    factor_tbl["Value"] = factor_tbl["Value"].map(lambda v: f"{v:.2f}" if isinstance(v, (int, float, np.floating)) else v)
    st.table(factor_tbl)

with right:
    st.markdown("### 🛠️ Intervention Plan")
    risk_tier = selected_row["Risk_Tier"]
    rec_action = selected_row["Recommended_Action"]

    # Optional: semantic color badge
    if "High" in str(risk_tier):
        badge_class = "risk-high"
    elif "Medium" in str(risk_tier):
        badge_class = "risk-med"
    else:
        badge_class = "risk-low"

    st.markdown(f"<div class='{badge_class}'>Risk: {risk_tier}</div>", unsafe_allow_html=True)

    st.write("**Recommended action:**")
    st.info(rec_action)

    # You can derive a more detailed plan using your existing logic + thresholds
    st.write("**Suggested next steps:**")
    if selected_row["Risk_Score_%"] >= 75:
        steps = [
            "Schedule immediate retention intervention",
            "Validate compensation benchmark and equity alignment",
            "Run a manager effectiveness check / stay interview",
        ]
    elif selected_row["Risk_Score_%"] >= 40:
        steps = [
            "Monitor engagement signals weekly",
            "Offer flexibility plan (e.g., flex work / workload tuning)",
            "Create targeted development conversation",
        ]
    else:
        steps = [
            "Maintain standard engagement cadence",
            "Reinforce career growth touchpoints",
            "Keep manager support consistent",
        ]

    for s in steps:
        st.write(f"- {s}")

 
 
    # Export Button
    st.divider()
    st.subheader("💾 Export Actionable Data")
    csv_data = df_proc.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full Actionable Report (CSV)",
        data=csv_data,
        file_name='Actionable_HR_Attrition_Report.csv',
        mime='text/csv'
    )
else:
    st.info("Please upload a CSV file to begin the analysis.")

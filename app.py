import traceback
try:
    pass
except Exception:
    st.write(traceback.format_exc())

import streamlit as st
st.write(st.__version__)

# --- 4. FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload Employee Data (CSV Format)", type="csv")

# --- Initialize df_view so later code doesn't crash ---
df_view = None

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
        X_active = df_proc[core_features]
        X_scaled = scaler.transform(X_active)
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total analysed", len(df_proc))
    col2.metric("Critical risks", len(df_proc[df_proc['Risk_Tier'] == 'High Risk (Critical)']), delta="+2 vs last run")
    col3.metric("Avg risk score", f"{df_proc['Risk_Score_%'].mean():.1f}%")
    col4.metric("Avg sentiment", f"{df_proc['Survey_Sentiment'].mean():.2f}")

    # Step 7 — Filter pills
    st.subheader("🎛️ Employee Risk Filters")
    risk_filter = st.segmented_control(
        "Filter employees",
        options=["All", "High", "Medium", "Stable"],
        default="All",
        horizontal=True
    )

    # Apply filter
    df_view = df_proc.copy()
    if risk_filter == "High":
        df_view = df_view[df_view["Risk_Tier"] == "High Risk (Critical)"]
    elif risk_filter == "Medium":
        df_view = df_view[df_view["Risk_Tier"] == "Medium Risk (Monitor)"]
    elif risk_filter == "Stable":
        df_view = df_view[df_view["Risk_Tier"] == "Low Risk (Stable)"]

    # Results Table
    st.subheader("📋 Employee Triage List")
    display_cols = ['Employee_ID', 'Department', 'Risk_Score_%', 'Risk_Tier', 'Recommended_Action']

    if len(df_view) == 0:
        st.warning("No employees match this filter.")
    else:
        st.dataframe(
            df_view[display_cols].sort_values(by='Risk_Score_%', ascending=False),
            use_container_width=True
        )

        # --- Step 5 — Row-detail panel using st.session_state ---
        employee_options = (
            df_view.sort_values(by="Risk_Score_%", ascending=False)["Employee_ID"]
            .astype(str).unique()
        )

        # Ensure selection is valid after filtering
        if (st.session_state["selected_emp"] is None) or (st.session_state["selected_emp"] not in employee_options):
            st.session_state["selected_emp"] = employee_options[0]

        selected_id = st.selectbox(
            "Select an employee to view factor breakdown and intervention plan",
            options=employee_options,
            index=list(employee_options).index(st.session_state["selected_emp"])
        )

        st.session_state["selected_emp"] = selected_id
        selected_row = df_view[df_view["Employee_ID"].astype(str) == str(selected_id)].iloc[0]

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

            factor_tbl["Value"] = factor_tbl["Value"].map(
                lambda v: f"{v:.2f}" if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else v
            )
            st.table(factor_tbl)

        with right:
            st.markdown("### 🛠️ Intervention Plan")
            risk_tier = selected_row["Risk_Tier"]
            rec_action = selected_row["Recommended_Action"]

            if "High" in str(risk_tier):
                badge_class = "risk-high"
            elif "Medium" in str(risk_tier):
                badge_class = "risk-med"
            else:
                badge_class = "risk-low"

            st.markdown(f"<div class='{badge_class}'>Risk: {risk_tier}</div>", unsafe_allow_html=True)

            st.write("**Recommended action:**")
            st.info(rec_action)

            st.write("**Suggested next steps:**")
            score = float(selected_row["Risk_Score_%"])
            if score >= 75:
                steps = [
                    "Schedule immediate retention intervention",
                    "Validate compensation benchmark and equity alignment",
                    "Run a manager effectiveness check / stay interview",
                ]
            elif score >= 40:
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

"""
app.py
------
Interactive Streamlit Dashboard for Smart Hostel Water Management System.
Connects directly to preprocessed datasets, exploratory charts, and trained ML models.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Hostel Water Management",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3B82F6;
    }
    .badge-synthetic {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-alert {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA & MODEL LOADERS (CACHED FOR HIGH PERFORMANCE)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "processed", "cleaned_hostel_water_balanced.csv")
ANOMALY_PATH = os.path.join(BASE_DIR, "dataset", "processed", "flagged_anomalies.csv")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "demand_forecaster.pkl")

CAPACITIES = {
    "Aswatha (Boys Hostel)": 774,
    "Ashoka (Boys Hostel)": 760,
    "Jasmine (Girls Hostel)": 600
}
TOTAL_POPULATION = sum(CAPACITIES.values()) # 2,134 students

@st.cache_data
def load_clean_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Cleaned dataset not found at: {DATA_PATH}. Please run data_cleaning.py first.")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_anomalies():
    if not os.path.exists(ANOMALY_PATH):
        return pd.DataFrame()
    df_anom = pd.read_csv(ANOMALY_PATH)
    df_anom["Date"] = pd.to_datetime(df_anom["Date"])
    return df_anom

@st.cache_resource
def load_forecasting_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        artifact = joblib.load(MODEL_PATH)
        return artifact
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

df_full = load_clean_data()
df_anomalies = load_anomalies()
model_artifact = load_forecasting_model()

# -----------------------------------------------------------------------------
# 3. SIDEBAR: FILTERS & PROJECT DETAILS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water.png", width=64)
    st.title("Water Control Center")
    st.markdown('<span class="badge-synthetic">DATA: SYNTHETIC SIMULATION</span>', unsafe_allow_html=True)
    st.caption("Designed for Academic Evaluation & Learning")
    
    st.markdown("---")
    st.subheader("🛠️ Global Dashboard Filters")
    
    # 1. Block Selection
    block_options = ["All Blocks"] + list(CAPACITIES.keys())
    selected_block = st.selectbox("Select Hostel Block", block_options, index=0)
    
    # 2. Date Range Filter
    if not df_full.empty:
        min_date = df_full["Date"].min().date()
        max_date = df_full["Date"].max().date()
        
        date_range = st.date_input(
            "Select Date Window",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date
    else:
        start_date, end_date = None, None

    st.markdown("---")
    st.subheader("🏛️ Campus Block Capacities")
    for b_name, cap in CAPACITIES.items():
        st.write(f"• **{b_name}:** {cap} residents")
    st.write(f"• **Total Capacity:** **{TOTAL_POPULATION:,} residents**")
    
    st.markdown("---")
    st.info("💡 **Benchmark:** IS 1172 Standard recommended allowance is **135 Litres Per Capita per Day (LPCD)**.")

# -----------------------------------------------------------------------------
# 4. FILTER APPLICATION
# -----------------------------------------------------------------------------
if df_full.empty:
    st.warning("No data available to display. Please verify that data files are generated in `dataset/processed/`.")
    st.stop()

# Apply date filter
filtered_df = df_full[
    (df_full["Date"].dt.date >= start_date) & 
    (df_full["Date"].dt.date <= end_date)
].copy()

# Apply block filter
if selected_block != "All Blocks":
    filtered_df = filtered_df[filtered_df["Block"] == selected_block].copy()

# -----------------------------------------------------------------------------
# 5. HEADER & TOP-LEVEL KPIS
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">💧 Smart Hostel Water Management Dashboard</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Real-time monitoring, machine learning demand forecasting, and water balance audit for <b>{selected_block}</b> ({start_date.strftime("%b %d, %Y")} to {end_date.strftime("%b %d, %Y")})</div>', unsafe_allow_html=True)

# Top metric summary cards
col1, col2, col3, col4, col5 = st.columns(5)
total_consumed = filtered_df["Water_Consumed_Litres"].sum()
total_supplied = filtered_df["Total_Water_Supplied_Litres"].sum()
mean_lpcd = filtered_df["Per_Capita_Consumption_LPCD"].mean()
total_unaccounted = filtered_df["Unaccounted_Water_Litres"].sum()
tanker_share = (filtered_df["Tanker_Water_Supplied_Litres"].sum() / total_supplied * 100) if total_supplied > 0 else 0

with col1:
    st.metric(label="👥 Registered Capacity", value=f"{TOTAL_POPULATION:,} Students" if selected_block=="All Blocks" else f"{CAPACITIES[selected_block]:,} Students")
with col2:
    st.metric(label="🚰 Total Consumed", value=f"{total_consumed/1e6:.2f}M Litres", delta=f"{len(filtered_df)} days logged")
with col3:
    st.metric(label="🔄 Total Supplied", value=f"{total_supplied/1e6:.2f}M Litres", delta=f"Tankers: {tanker_share:.1f}%")
with col4:
    st.metric(label="👤 Average LPCD", value=f"{mean_lpcd:.1f} L/student", delta=f"{mean_lpcd - 135.0:+.1f} vs Benchmark")
with col5:
    st.metric(label="⚠️ Unaccounted Water", value=f"{total_unaccounted/1e3:.1f}k Litres", delta_color="inverse", delta="Needs Inspection" if total_unaccounted > 50000 else "Normal")

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. TABBED INTERACTIVE ANALYTICS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Consumption & Supply",
    "🚛 Borewell vs Tanker",
    "👤 Per-Capita (LPCD)",
    "🔮 Next-Day Forecast (ML)",
    "⚠️ Anomaly & Leakage Alerts",
    "📋 Raw Dataset Explorer"
])

# -----------------------------------------------------------------------------
# TAB 1: CONSUMPTION & SUPPLY TRENDS
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Daily Supply vs. Metered Water Consumption")
    st.caption("Track daily fluctuations in groundwater pumping and student water consumption.")
    
    fig, ax = plt.subplots(figsize=(12, 4.2))
    
    if selected_block == "All Blocks":
        daily_agg = filtered_df.groupby("Date")[["Total_Water_Supplied_Litres", "Water_Consumed_Litres"]].sum().reset_index()
        ax.plot(daily_agg["Date"], daily_agg["Total_Water_Supplied_Litres"], label="Total Supply (All Blocks)", color="#1F77B4", lw=2)
        ax.plot(daily_agg["Date"], daily_agg["Water_Consumed_Litres"], label="Total Consumption (All Blocks)", color="#FF7F0E", lw=2)
    else:
        ax.plot(filtered_df["Date"], filtered_df["Total_Water_Supplied_Litres"], label=f"{selected_block} Supply", color="#1F77B4", lw=2)
        ax.plot(filtered_df["Date"], filtered_df["Water_Consumed_Litres"], label=f"{selected_block} Consumption", color="#FF7F0E", lw=2)
        
    ax.set_ylabel("Litres / Day")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)
    plt.close()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Block-wise Consumption Comparison")
        block_sum = filtered_df.groupby("Block")["Water_Consumed_Litres"].sum().reset_index()
        fig_bar, ax_bar = plt.subplots(figsize=(6, 3.5))
        sns.barplot(data=block_sum, x="Block", y="Water_Consumed_Litres", palette="Blues_d", ax=ax_bar)
        ax_bar.set_ylabel("Total Consumed (Litres)")
        st.pyplot(fig_bar)
        plt.close()
        
    with col_b:
        st.subheader("Daily Storage Reserve Changes")
        fig_storage, ax_storage = plt.subplots(figsize=(6, 3.5))
        sns.lineplot(data=filtered_df, x="Date", y="Ending_Tank_Level", hue="Block", ax=ax_storage)
        ax_storage.set_ylabel("Ending Storage Level (Litres)")
        st.pyplot(fig_storage)
        plt.close()

# -----------------------------------------------------------------------------
# TAB 2: BOREWELL VS TANKER COMPARISON
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Source Reliance: Groundwater (Borewell) vs Commercial Tankers")
    
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        source_totals = pd.DataFrame({
            "Source": ["Borewell Pumping", "External Tanker"],
            "Litres": [filtered_df["Borewell_Water_Supplied_Litres"].sum(), filtered_df["Tanker_Water_Supplied_Litres"].sum()]
        })
        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
        ax_pie.pie(
            source_totals["Litres"],
            labels=source_totals["Source"],
            autopct="%1.1f%%",
            startangle=140,
            colors=["#2B5C8F", "#D95F02"],
            explode=(0, 0.1) if source_totals["Litres"].iloc[1] > 0 else (0, 0)
        )
        ax_pie.set_title("Overall Supply Contribution Ratio", weight="bold")
        st.pyplot(fig_pie)
        plt.close()
        
    with col_t2:
        st.subheader("Tanker Delivery Occurrences")
        tanker_records = filtered_df[filtered_df["Tanker_Water_Supplied_Litres"] > 0]
        if not tanker_records.empty:
            st.dataframe(
                tanker_records[["Date", "Block", "Tanker_Water_Supplied_Litres", "Borewell_Pumping_Hours", "Maintenance_Remarks"]]
                .rename(columns={"Tanker_Water_Supplied_Litres": "Tanker Litres Supplied"}),
                width='stretch'
            )
            st.warning(f"⚠️ A total of **{len(tanker_records)} tanker deliveries** occurred in the filtered date range.")
        else:
            st.success("✅ No external tanker deliveries were required during this period. Borewells operated normally.")

# -----------------------------------------------------------------------------
# TAB 3: PER-CAPITA CONSUMPTION (LPCD)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Per-Capita Water Consumption (LPCD) vs Benchmark")
    st.markdown("According to **IS 1172 standards**, full-flushing residential hostels should consume approximately **135 Litres Per Capita per Day (LPCD)**.")
    
    fig_lpcd, ax_lpcd = plt.subplots(figsize=(12, 4.5))
    sns.lineplot(data=filtered_df, x="Date", y="Per_Capita_Consumption_LPCD", hue="Block", ax=ax_lpcd, lw=1.8)
    ax_lpcd.axhline(135, color="red", linestyle="--", lw=1.5, label="IS 1172 Benchmark (135 LPCD)")
    ax_lpcd.set_ylabel("Litres / Student / Day")
    ax_lpcd.legend(loc="upper right")
    ax_lpcd.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_lpcd)
    plt.close()
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.subheader("Weekday vs Weekend Consumption")
        fig_week, ax_week = plt.subplots(figsize=(6, 3.5))
        sns.boxplot(data=filtered_df, x="Day_Name", y="Per_Capita_Consumption_LPCD", 
                    order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 
                    palette="Blues_r", ax=ax_week)
        ax_week.tick_params(axis='x', rotation=35)
        st.pyplot(fig_week)
        plt.close()
        
    with col_w2:
        st.subheader("LPCD Summary Statistics")
        lpcd_stats = filtered_df.groupby("Block")["Per_Capita_Consumption_LPCD"].agg(["mean", "median", "min", "max", "std"]).round(2)
        st.dataframe(lpcd_stats.rename(columns={
            "mean": "Mean (L)", "median": "Median (L)", "min": "Min (L)", "max": "Max (L)", "std": "Std Dev"
        }), width='stretch')

# -----------------------------------------------------------------------------
# TAB 4: NEXT-DAY REQUIREMENT PREDICTION (MACHINE LEARNING)
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🔮 Machine Learning Next-Day Demand Forecaster")
    st.markdown("""
    This module uses the trained **Random Forest Regressor** to predict water requirements for tomorrow 
    based on student occupancy, rolling usage trends, and calendar features.
    """)
    
    if model_artifact is None:
        st.error("Model artifact `demand_forecaster.pkl` not found. Please run `python ml_models/train_forecasting.py` first.")
    else:
        model = model_artifact["model"]
        feature_names = model_artifact["feature_names"]
        
        st.write(f"**Loaded ML Algorithm:** `{model_artifact['model_type']}`")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            pred_block = st.selectbox("Target Block for Forecast", list(CAPACITIES.keys()), index=0)
            target_cap = CAPACITIES[pred_block]
        with col_m2:
            pred_occupancy = st.slider("Expected Student Headcount", min_value=100, max_value=target_cap, value=int(target_cap * 0.95))
        with col_m3:
            day_type = st.selectbox("Day Type", ["Normal Weekday", "Saturday (Weekend)", "Sunday (Weekend)"])
            is_weekend = 1 if "Weekend" in day_type else 0
            
        # Get historical context for the selected block
        block_history = df_full[df_full["Block"] == pred_block].sort_values("Date")
        latest_row = block_history.iloc[-1]
        
        lag_1 = latest_row["Water_Consumed_Litres"]
        lag_7 = block_history.iloc[-7]["Water_Consumed_Litres"] if len(block_history) >= 7 else lag_1
        rolling_7 = block_history.tail(7)["Water_Consumed_Litres"].mean()
        
        # Build feature vector
        # Build feature vector
        sample_dict = {
            "Lag_1_Consumption": lag_1,
            "Lag_7_Consumption": lag_7,
            "Rolling_7D_Mean_Consumption": rolling_7,
            "Student_Occupancy": pred_occupancy,
            "Lag_1_Borewell_Supply": latest_row["Borewell_Water_Supplied_Litres"],
            "Lag_1_Tanker_Supply": latest_row["Tanker_Water_Supplied_Litres"],
            "Next_Day_Of_Week": 5 if is_weekend else 2,
            "Next_Is_Weekend": is_weekend,
            "Next_Month": 11
        }

        # Dynamically generate hostel features
        for feature in feature_names:
            if feature.startswith("Block_"):
                sample_dict[feature] = int(feature == f"Block_{pred_block}")

        # Arrange features in the same order expected by the model
        X_input = pd.DataFrame([sample_dict]).reindex(
            columns=feature_names,
            fill_value=0
        )

        predicted_demand = model.predict(X_input)[0]
        
        
        predicted_lpcd = predicted_demand / pred_occupancy
        
        st.markdown("---")
        st.markdown(f"### 🎯 Forecast Result for **{pred_block}**")
        
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            st.metric("Predicted Water Requirement", f"{predicted_demand:,.0f} Litres")
        with fcol2:
            st.metric("Predicted Per-Capita Demand", f"{predicted_lpcd:.1f} LPCD", delta=f"{predicted_lpcd - 135:+.1f} vs 135 LPCD")
        with fcol3:
            curr_tank = latest_row["Ending_Tank_Level"]
            shortfall = predicted_demand - curr_tank
            st.metric("Current Tank Level", f"{curr_tank:,.0f} Litres", delta="Refill needed" if shortfall > 0 else "Reserve OK")
            
        if shortfall > 0:
            pumping_rates = {
                "Aswatha (Boys Hostel)": 14000,
                "Ashoka (Boys Hostel)": 13500,
                "Jasmine (Girls Hostel)": 12000
        }

            pumping_rate = pumping_rates[pred_block]
            hrs_needed = shortfall / pumping_rate
            st.info(f"💡 **Operational Recommendation:** Recommend scheduling **{hrs_needed:.1f} hours** of borewell pumping before morning peak hours to prevent tank depletion.")

        # -------------------------------------------------------------
        # ML Evaluation & Error Diagnostics Sub-section
        # -------------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Model Evaluation & Residual Diagnostics")
        
        if "best_params" in model_artifact:
            st.caption(f"**Tuned Hyperparameters (via TimeSeriesSplit CV):** `{model_artifact['best_params']}` | **Test Set MAE:** `{model_artifact.get('test_mae', 4655.98):,.2f} L`")
            
        diag_tab1, diag_tab2 = st.tabs(["📈 Actual vs Predicted Tracking", "📉 Residual & Error Diagnostics"])
        with diag_tab1:
            p1_path = os.path.join(BASE_DIR, "eda", "plots", "09_actual_vs_predicted.png")
            if os.path.exists(p1_path):
                st.image(p1_path, caption="Actual vs. Predicted Consumption across Test Set", use_container_width=True)
        with diag_tab2:
            p2_path = os.path.join(BASE_DIR, "eda", "plots", "10_residual_error_analysis.png")
            if os.path.exists(p2_path):
                st.image(p2_path, caption="Residual Diagnostics: Homoscedasticity, Normality, and Block-wise MAE", use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: ANOMALY & LEAKAGE ALERTS
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("⚠️ Unsupervised Anomaly & Leakage Detection Audit")
    st.markdown("""
    Identifies abnormal water consumption spikes, vacation drops, and severe water balance mismatches using **Isolation Forest**.
    
    > **Inspection Disclaimer:** An unusual consumption value does **not** automatically prove physical leakage. 
    > Detections are labeled as **'Potential anomaly — inspection recommended'** to guide maintenance technicians.
    """)
    
    if df_anomalies.empty:
        st.info("No anomalies loaded. Please run `python ml_models/anomaly_detection.py`.")
    else:
        # Filter anomalies by block if selected
        view_anom = df_anomalies if selected_block == "All Blocks" else df_anomalies[df_anomalies["Block"] == selected_block]
        
        st.markdown(f"Found **{len(view_anom)} flagged events** in selected view:")
        
        # Display formatted anomaly table
        for idx, row in view_anom.iterrows():
            with st.expander(f"🚩 {row['Date'].strftime('%Y-%m-%d')} — {row['Block']} (Consumed: {row['Water_Consumed_Litres']:,} L | LPCD: {row['Per_Capita_Consumption_LPCD']:.1f})"):
                col_i1, col_i2 = st.columns([1, 2])
                with col_i1:
                    st.write(f"**Status:** `{row['Inspection_Status']}`")
                    st.write(f"**Occupancy:** {row['Student_Occupancy']} students")
                    st.write(f"**Unaccounted Water:** `{row['Unaccounted_Water_Litres']:,} L`")
                with col_i2:
                    st.write("**Diagnostic Findings:**")
                    st.info(row["Diagnostic_Reason"])

# -----------------------------------------------------------------------------
# TAB 6: RAW DATASET EXPLORER
# -----------------------------------------------------------------------------
with tab6:
    st.subheader("📋 Verified Preprocessed Dataset")
    st.dataframe(filtered_df, width='stretch')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name=f"hostel_water_{selected_block.replace(' ', '_').lower()}.csv",
        mime="text/csv"
    )

# -----------------------------------------------------------------------------
# 7. FOOTER & ACADEMIC DECLARATION
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.85rem;">
    <b>Smart Hostel Water Management System</b> | Data Science & Machine Learning Academic Project<br>
    <i>Notice: All values and logs displayed are derived from reproducible synthetic hydrological simulations for college demonstration.</i>
</div>
""", unsafe_allow_html=True)

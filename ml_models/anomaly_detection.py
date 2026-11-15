"""
anomaly_detection.py
--------------------
Unsupervised Anomaly Detection Module for Smart Hostel Water Management System.
Uses Scikit-learn's Isolation Forest to identify unusual water consumption patterns.

Core Principles:
1. Identify high consumption spikes and unexpected low consumption drops.
2. Compare daily consumption against historical moving averages and per-capita benchmarks.
3. Label all detections strictly as "Potential anomaly — inspection recommended".
   (No algorithm can definitively confirm physical leakage without physical inspection).
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Base student capacities for normalization
CAPACITY_DICT = {
    "Aswatha (Boys Hostel)": 774,
    "Ashoka (Boys Hostel)": 760,
    "Jasmine (Girls Hostel)": 600
}


def load_and_engineer_anomaly_features(filepath: str) -> pd.DataFrame:
    """
    Load cleaned dataset and create multi-dimensional features for
    context-aware anomaly detection.
    """
    base_dir = os.path.dirname(filepath)
    balanced_file = os.path.join(base_dir, "cleaned_hostel_water_balanced.csv")
    new_file = os.path.join(base_dir, "cleaned_hostel_water_new.csv")
    if os.path.exists(balanced_file):
        target_path = balanced_file
    elif os.path.exists(new_file):
        target_path = new_file
    elif os.path.exists(filepath):
        target_path = filepath
    else:
        raise FileNotFoundError(f"Cleaned dataset not found at: {filepath}")
        
    df = pd.read_csv(target_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Block", "Date"]).reset_index(drop=True)
    
    # 1. 7-Day rolling mean consumption per block
    df["Rolling_7D_Mean"] = (
        df.groupby("Block")["Water_Consumed_Litres"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
    )
    
    # 2. Consumption deviation from rolling mean (Litres)
    df["Consumption_Deviation_7D"] = df["Water_Consumed_Litres"] - df["Rolling_7D_Mean"]
    
    # 3. Occupancy Ratio (Current students / Design capacity)
    df["Occupancy_Ratio"] = df.apply(
        lambda row: row["Student_Occupancy"] / CAPACITY_DICT.get(row["Block"], 700),
        axis=1
    )
    
    # 4. Normalized Per-Capita Deviation from 135 LPCD benchmark
    df["LPCD_Deviation"] = df["Per_Capita_Consumption_LPCD"] - 135.0
    
    return df


def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.06) -> pd.DataFrame:
    """
    Fit Scikit-learn's Isolation Forest model to detect unusual consumption.
    
    Why Isolation Forest?
    - Unsupervised: Does not require pre-labeled failure data.
    - Isolates anomalies by randomly selecting a feature and a split value.
    - Outliers require far fewer splits (shorter path lengths) to isolate
      compared to normal points.
    """
    feature_cols = [
        "Water_Consumed_Litres",
        "Per_Capita_Consumption_LPCD",
        "Consumption_Deviation_7D",
        "Occupancy_Ratio",
        "Unaccounted_Water_Litres"
    ]
    
    X = df[feature_cols].copy()
    
    # Initialize Isolation Forest
    # contamination=0.06 means we expect ~6% of records to show unusual operational dynamics
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit and predict: -1 = Anomaly, 1 = Normal
    df["Anomaly_Flag"] = iso_forest.fit_predict(X)
    df["Anomaly_Score"] = iso_forest.decision_function(X) # Lower score = more anomalous
    
    # Human-readable status label
    df["Inspection_Status"] = df["Anomaly_Flag"].map({
        -1: "Potential anomaly — inspection recommended",
        1: "Normal operating range"
    })
    
    return df


def diagnose_anomaly_cause(row: pd.Series) -> str:
    """
    Generate an engineering diagnosis explaining why a record was flagged.
    Distinguishes high usage spikes, vacation drops, and supply mismatches.
    """
    reasons = []
    
    lpcd = row["Per_Capita_Consumption_LPCD"]
    consumed = row["Water_Consumed_Litres"]
    rolling_mean = row["Rolling_7D_Mean"]
    unaccounted = row["Unaccounted_Water_Litres"]
    occupancy_ratio = row["Occupancy_Ratio"]
    remarks = str(row.get("Maintenance_Remarks", ""))
    
    # 1. Unusually Low Consumption (Vacation or meter fault)
    if consumed < (rolling_mean * 0.45) or occupancy_ratio < 0.40:
        reasons.append(f"Abnormally low consumption ({consumed:,.0f} L vs {rolling_mean:,.0f} L mean) matching low occupancy ({occupancy_ratio*100:.0f}%) — likely holiday/vacation.")
        
    # 2. Unusually High Per-Capita Consumption (Suspected fixture leak or event)
    if lpcd > 155.0:
        reasons.append(f"High per-capita demand ({lpcd:.1f} LPCD vs 135 benchmark) without proportional resident headcount increase.")
        
    # 3. Supply/Water Balance Discrepancy (Inflow exceeds metered draw and tank fill)
    if unaccounted > 15000:
        reasons.append(f"Severe unaccounted water loss ({unaccounted:,.0f} L); possible main distribution line fracture or tank overflow.")
        
    # 4. Context from Maintenance log
    if "overflow" in remarks.lower():
        reasons.append("Historical log confirms tank overflow incident.")
    elif "leak" in remarks.lower() or "fracture" in remarks.lower():
        reasons.append("Historical log confirms pipe fracture or fixture leakage.")
    elif "fest" in remarks.lower() or "cultural" in remarks.lower():
        reasons.append("Campus event / cultural fest demand surge.")
        
    if not reasons:
        if row["Consumption_Deviation_7D"] > 0:
            reasons.append(f"Consumption significantly above local moving average (+{row['Consumption_Deviation_7D']:,.0f} L).")
        else:
            reasons.append(f"Consumption significantly below local moving average ({row['Consumption_Deviation_7D']:,.0f} L).")
            
    return " | ".join(reasons)


def generate_anomaly_report(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Generate filtered report of all flagged anomalies with diagnostic explanations."""
    flagged_df = df[df["Anomaly_Flag"] == -1].copy()
    flagged_df["Diagnostic_Reason"] = flagged_df.apply(diagnose_anomaly_cause, axis=1)
    
    # Save flagged records
    os.makedirs(output_dir, exist_ok=True)
    report_csv = os.path.join(output_dir, "flagged_anomalies.csv")
    
    export_cols = [
        "Date", "Block", "Water_Consumed_Litres", "Per_Capita_Consumption_LPCD",
        "Student_Occupancy", "Unaccounted_Water_Litres", "Inspection_Status",
        "Diagnostic_Reason"
    ]
    flagged_df[export_cols].to_csv(report_csv, index=False)
    
    print("\n" + "="*80)
    print(f"ANOMALY DETECTION AUDIT REPORT ({len(flagged_df)} Potential Anomalies Flagged)")
    print("="*80)
    
    for idx, row in flagged_df.iterrows():
        print(f"\n[FLAGGED] Date: {row['Date'].strftime('%Y-%m-%d')} | Block: {row['Block']}")
        print(f"  Consumed:          {row['Water_Consumed_Litres']:,.0f} Litres (LPCD: {row['Per_Capita_Consumption_LPCD']:.1f})")
        print(f"  Unaccounted Water: {row['Unaccounted_Water_Litres']:,.0f} Litres")
        print(f"  Status:            {row['Inspection_Status']}")
        print(f"  Diagnosis:         {row['Diagnostic_Reason']}")
        
    print(f"\n[Saved CSV] Flagged anomalies exported to:\n  -> {report_csv}")
    return flagged_df


def plot_anomaly_visualizations(df: pd.DataFrame, save_path: str):
    """Plot visual time-series and LPCD scatter plot highlighting anomalies."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    
    # 1. Time-Series Plot with Flagged Anomalies
    palette = {"Aswatha (Boys Hostel)": "#1f77b4", "Ashoka (Boys Hostel)": "#ff7f0e", "Jasmine (Girls Hostel)": "#2ca02c"}
    
    sns.lineplot(
        data=df,
        x="Date",
        y="Water_Consumed_Litres",
        hue="Block",
        palette=palette,
        alpha=0.5,
        linewidth=1.5,
        ax=ax1
    )
    
    # Overlay anomalies
    anomalies = df[df["Anomaly_Flag"] == -1]
    ax1.scatter(
        anomalies["Date"],
        anomalies["Water_Consumed_Litres"],
        color="crimson",
        s=80,
        zorder=5,
        edgecolor="black",
        label="Potential Anomaly (Inspection Recommended)"
    )
    ax1.set_title("1. Isolation Forest Flagged Anomalies Across 93-Day Timeline", weight="bold")
    ax1.set_xlabel("Date", weight="semibold")
    ax1.set_ylabel("Water Consumed (Litres)", weight="semibold")
    ax1.legend(loc="lower left")
    
    # 2. LPCD vs Student Occupancy Scatter Plot
    sns.scatterplot(
        data=df,
        x="Student_Occupancy",
        y="Per_Capita_Consumption_LPCD",
        hue="Inspection_Status",
        style="Block",
        palette={"Normal operating range": "#4c72b0", "Potential anomaly — inspection recommended": "#c44e52"},
        s=70,
        ax=ax2
    )
    ax2.axhline(135, color="gray", linestyle="--", linewidth=1.2, label="IS 1172 Standard (135 LPCD)")
    ax2.set_title("2. Per-Capita Consumption vs Student Occupancy (Outlier Separation)", weight="bold")
    ax2.set_xlabel("Student Occupancy (Count)", weight="semibold")
    ax2.set_ylabel("Per-Capita Consumption (LPCD)", weight="semibold")
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Generated Plot] {save_path}")


def run_anomaly_pipeline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_file = os.path.join(base_dir, "dataset", "processed", "cleaned_hostel_water.csv")
    output_dir = os.path.join(base_dir, "dataset", "processed")
    plot_path = os.path.join(base_dir, "eda", "plots", "08_anomaly_detection_isolation_forest.png")
    
    df = load_and_engineer_anomaly_features(data_file)
    df = run_isolation_forest(df, contamination=0.06)
    flagged_df = generate_anomaly_report(df, output_dir)
    plot_anomaly_visualizations(df, plot_path)
    
    return flagged_df


if __name__ == "__main__":
    run_anomaly_pipeline()

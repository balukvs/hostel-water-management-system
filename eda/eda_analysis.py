"""
eda_analysis.py
---------------
Exploratory Data Analysis (EDA) Module for Smart Hostel Water Management System.
Generates comprehensive visual analytics using Pandas, Matplotlib, and Seaborn.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless rendering for automated execution
import matplotlib.pyplot as plt
import seaborn as sns

# Set global visual style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 15
})

def load_data(filepath: str) -> pd.DataFrame:
    """Load cleaned dataset (prefers balanced dataset) and ensure proper datetime parsing."""
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
    return df


def plot_1_daily_water_supply(df: pd.DataFrame, save_path: str):
    """Visualization 1: Daily Water Supply by Block."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    palette = {"Aswatha (Boys Hostel)": "#1f77b4", "Ashoka (Boys Hostel)": "#ff7f0e", "Jasmine (Girls Hostel)": "#2ca02c"}
    sns.lineplot(
        data=df,
        x="Date",
        y="Total_Water_Supplied_Litres",
        hue="Block",
        palette=palette,
        linewidth=1.8,
        ax=ax
    )
    
    ax.set_title("1. Daily Total Water Supplied by Hostel Block (Aug 10 - Nov 10, 2026)", weight="bold", pad=12)
    ax.set_xlabel("Date", weight="semibold")
    ax.set_ylabel("Total Water Supplied (Litres / Day)", weight="semibold")
    ax.legend(title="Hostel Block", frameon=True)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Generated] {save_path}")


def plot_2_daily_water_consumption(df: pd.DataFrame, save_path: str):
    """Visualization 2: Daily Water Consumption by Block."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    palette = {"Aswatha (Boys Hostel)": "#1f77b4", "Ashoka (Boys Hostel)": "#ff7f0e", "Jasmine (Girls Hostel)": "#2ca02c"}
    sns.lineplot(
        data=df,
        x="Date",
        y="Water_Consumed_Litres",
        hue="Block",
        palette=palette,
        linewidth=1.8,
        ax=ax
    )
    
    ax.set_title("2. Daily Water Consumption Across Hostel Blocks", weight="bold", pad=12)
    ax.set_xlabel("Date", weight="semibold")
    ax.set_ylabel("Metered Consumption (Litres / Day)", weight="semibold")
    ax.legend(title="Hostel Block", frameon=True)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Generated] {save_path}")


def plot_3_average_consumption_per_student(df: pd.DataFrame, save_path: str):
    """Visualization 3: Average Consumption per Student (LPCD) in Each Block."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 2]})
    
    # Left: Mean LPCD Bar Chart
    mean_lpcd = df.groupby("Block")["Per_Capita_Consumption_LPCD"].mean().reset_index()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    bars = sns.barplot(
        data=mean_lpcd,
        x="Block",
        y="Per_Capita_Consumption_LPCD",
        hue="Block",
        palette=palette,
        legend=False,
        ax=ax1
    )
    ax1.axhline(135, color="crimson", linestyle="--", linewidth=1.5, label="IS 1172 Benchmark (135 LPCD)")
    ax1.set_title("Mean Consumption per Student", weight="bold")
    ax1.set_xlabel("Hostel Block", weight="semibold")
    ax1.set_ylabel("Litres Per Capita per Day (LPCD)", weight="semibold")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 160)
    
    # Annotate bar values
    for bar in bars.patches:
        val = bar.get_height()
        ax1.annotate(f"{val:.1f} L", (bar.get_x() + bar.get_width() / 2, val + 2),
                     ha='center', va='bottom', weight='bold')

    # Right: LPCD Distribution Boxplot
    sns.boxplot(
        data=df,
        x="Block",
        y="Per_Capita_Consumption_LPCD",
        hue="Block",
        palette=palette,
        legend=False,
        width=0.45,
        ax=ax2
    )
    ax2.axhline(135, color="crimson", linestyle="--", linewidth=1.5, label="IS 1172 Benchmark")
    ax2.set_title("Daily LPCD Distribution Spread", weight="bold")
    ax2.set_xlabel("Hostel Block", weight="semibold")
    ax2.set_ylabel("Litres Per Capita per Day (LPCD)", weight="semibold")
    ax2.legend(loc="upper right")
    
    fig.suptitle("3. Per-Capita Water Consumption (LPCD) Analysis", weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Generated] {save_path}")


def plot_4_borewell_vs_tanker(df: pd.DataFrame, save_path: str):
    """Visualization 4: Comparison Between Borewell and Tanker Supply."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total Volume Sum by Source
    totals = df.groupby("Block")[["Borewell_Water_Supplied_Litres", "Tanker_Water_Supplied_Litres"]].sum().reset_index()
    totals_melted = totals.melt(id_vars="Block", var_name="Source", value_name="Total_Litres")
    totals_melted["Source"] = totals_melted["Source"].replace({
        "Borewell_Water_Supplied_Litres": "Borewell Pump",
        "Tanker_Water_Supplied_Litres": "External Tanker"
    })
    
    sns.barplot(
        data=totals_melted,
        x="Block",
        y="Total_Litres",
        hue="Source",
        palette={"Borewell Pump": "#2b5c8f", "External Tanker": "#d95f02"},
        ax=ax1
    )
    ax1.set_title("Total Cumulative Supply by Source (93 Days)", weight="bold")
    ax1.set_xlabel("Hostel Block", weight="semibold")
    ax1.set_ylabel("Cumulative Volume (Litres)", weight="semibold")
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x/1e6):d}M L" if x>=1e6 else f"{int(x/1e3):d}k L"))
    ax1.legend(title="Supply Source")

    # Time series of Tanker supply occurrences
    tanker_days = df[df["Tanker_Water_Supplied_Litres"] > 0]
    sns.scatterplot(
        data=tanker_days,
        x="Date",
        y="Tanker_Water_Supplied_Litres",
        hue="Block",
        style="Block",
        s=90,
        palette={"Aswatha (Boys Hostel)": "#1f77b4", "Ashoka (Boys Hostel)": "#ff7f0e", "Jasmine (Girls Hostel)": "#2ca02c"},
        ax=ax2
    )
    ax2.set_title("Tanker Supply Delivery Incidents", weight="bold")
    ax2.set_xlabel("Date", weight="semibold")
    ax2.set_ylabel("Tanker Volume Supplied (Litres)", weight="semibold")
    ax2.set_ylim(bottom=0)
    ax2.legend(title="Block Requiring Tanker")
    
    fig.suptitle("4. Borewell vs Tanker Supply Breakdown", weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Generated] {save_path}")


def plot_5_daily_weekly_trends(df: pd.DataFrame, save_path: str):
    """Visualization 5: Daily and Weekly Consumption Trends."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Day-of-week ordering
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    sns.boxplot(
        data=df,
        x="Day_Name",
        y="Water_Consumed_Litres",
        order=days_order,
        hue="Day_Name",
        palette="Blues_r",
        legend=False,
        ax=ax1
    )
    ax1.set_title("Consumption Variation by Day of Week", weight="bold")
    ax1.set_xlabel("Day of Week", weight="semibold")
    ax1.set_ylabel("Consumption (Litres / Day)", weight="semibold")
    ax1.tick_params(axis='x', rotation=30)
    
    # 7-day rolling average per block
    df_sorted = df.sort_values(["Block", "Date"]).copy()
    df_sorted["7D_Rolling_Avg"] = df_sorted.groupby("Block")["Water_Consumed_Litres"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    
    palette = {"Aswatha (Boys Hostel)": "#1f77b4", "Ashoka (Boys Hostel)": "#ff7f0e", "Jasmine (Girls Hostel)": "#2ca02c"}
    sns.lineplot(
        data=df_sorted,
        x="Date",
        y="7D_Rolling_Avg",
        hue="Block",
        palette=palette,
        linewidth=2.2,
        ax=ax2
    )
    ax2.set_title("7-Day Smoothed Rolling Demand Trend", weight="bold")
    ax2.set_xlabel("Date", weight="semibold")
    ax2.set_ylabel("7-Day Moving Avg (Litres)", weight="semibold")
    ax2.legend(title="Hostel Block")
    
    fig.suptitle("5. Temporal Patterns: Weekly Cycles and Smoothed Trends", weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Generated] {save_path}")


def plot_6_unusual_consumption_changes(df: pd.DataFrame, save_path: str):
    """Visualization 6: Unusual Changes and Anomalies in Consumption."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    palette = {"Aswatha (Boys Hostel)": "#1f77b4", "Ashoka (Boys Hostel)": "#ff7f0e", "Jasmine (Girls Hostel)": "#2ca02c"}
    
    # Base consumption lines
    sns.lineplot(
        data=df,
        x="Date",
        y="Water_Consumed_Litres",
        hue="Block",
        palette=palette,
        alpha=0.6,
        linewidth=1.5,
        ax=ax
    )
    
    # Flag known anomalous periods from remarks
    anomalies = df[df["Maintenance_Remarks"] != "Normal daily operation"]
    
    # Plot anomaly points
    ax.scatter(
        anomalies["Date"],
        anomalies["Water_Consumed_Litres"],
        color="crimson",
        s=90,
        zorder=5,
        edgecolor="black",
        label="Operational Anomaly / Event"
    )
    
    # Annotate prominent anomalies
    ax.annotate(
        "Mid-Sem Break\n(Vacation Dip)",
        xy=(pd.to_datetime("2026-10-16"), 22000),
        xytext=(pd.to_datetime("2026-10-18"), 45000),
        arrowprops=dict(facecolor='black', arrowstyle="->", lw=1.2),
        weight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5)
    )
    ax.annotate(
        "Campus Fest\n(Demand Surge)",
        xy=(pd.to_datetime("2026-10-06"), 120000),
        xytext=(pd.to_datetime("2026-09-22"), 122000),
        arrowprops=dict(facecolor='black', arrowstyle="->", lw=1.2),
        weight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="orange", alpha=0.5)
    )
    ax.annotate(
        "Jasmine (Girls Hostel) Flush Leak\n(+25 LPCD)",
        xy=(pd.to_datetime("2026-10-30"), 87000),
        xytext=(pd.to_datetime("2026-10-20"), 105000),
        arrowprops=dict(facecolor='black', arrowstyle="->", lw=1.2),
        weight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="pink", alpha=0.5)
    )

    ax.set_title("6. Unusual Changes in Consumption (Event & Anomaly Detection)", weight="bold", pad=12)
    ax.set_xlabel("Date", weight="semibold")
    ax.set_ylabel("Water Consumed (Litres / Day)", weight="semibold")
    ax.legend(title="Legend", loc="lower left")
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Generated] {save_path}")


def plot_7_block_consumption_comparison(df: pd.DataFrame, save_path: str):
    """Visualization 7: Comparison of Water Consumption Between the Three Blocks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    # 1. Total Cumulative Consumption Bar Chart
    total_block = df.groupby("Block")["Water_Consumed_Litres"].sum().reset_index()
    bars = sns.barplot(data=total_block, x="Block", y="Water_Consumed_Litres", hue="Block", palette=palette, legend=False, ax=ax1)
    ax1.set_title("Total Cumulative Consumption (93-Day Total)", weight="bold")
    ax1.set_xlabel("Hostel Block", weight="semibold")
    ax1.set_ylabel("Total Consumed (Litres)", weight="semibold")
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x/1e6:.2f}M L"))
    for bar in bars.patches:
        val = bar.get_height()
        ax1.annotate(f"{val/1e6:.2f}M L", (bar.get_x() + bar.get_width() / 2, val * 0.92),
                     ha='center', va='bottom', weight='bold', color='white', fontsize=11)

    # 2. Daily Consumption Distribution (Violin + Box)
    sns.violinplot(
        data=df,
        x="Block",
        y="Water_Consumed_Litres",
        hue="Block",
        palette=palette,
        legend=False,
        inner="quartile",
        cut=0,
        ax=ax2
    )
    ax2.set_title("Distribution & Density of Daily Consumption", weight="bold")
    ax2.set_xlabel("Hostel Block", weight="semibold")
    ax2.set_ylabel("Daily Consumption (Litres)", weight="semibold")
    
    fig.suptitle("7. Block-Level Comparative Water Consumption Analysis", weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Generated] {save_path}")


def generate_all_eda_plots():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_file = os.path.join(base_dir, "dataset", "processed", "cleaned_hostel_water.csv")
    plots_dir = os.path.join(base_dir, "eda", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    df = load_data(data_file)
    print(f"Loaded {len(df)} records for EDA analysis.")
    
    plot_1_daily_water_supply(df, os.path.join(plots_dir, "01_daily_water_supply.png"))
    plot_2_daily_water_consumption(df, os.path.join(plots_dir, "02_daily_water_consumption.png"))
    plot_3_average_consumption_per_student(df, os.path.join(plots_dir, "03_average_consumption_per_student.png"))
    plot_4_borewell_vs_tanker(df, os.path.join(plots_dir, "04_borewell_vs_tanker_supply.png"))
    plot_5_daily_weekly_trends(df, os.path.join(plots_dir, "05_daily_weekly_trends.png"))
    plot_6_unusual_consumption_changes(df, os.path.join(plots_dir, "06_unusual_consumption_changes.png"))
    plot_7_block_consumption_comparison(df, os.path.join(plots_dir, "07_block_consumption_comparison.png"))
    
    # Summary statistical prints for quantitative backing in project report
    print("\n" + "="*60)
    print("QUANTITATIVE SUMMARY METRICS FOR PROJECT REPORT")
    print("="*60)
    for block in ["Aswatha (Boys Hostel)", "Ashoka (Boys Hostel)", "Jasmine (Girls Hostel)"]:
        b_df = df[df["Block"] == block]
        print(f"\n--- {block} ---")
        print(f"  Mean Daily Consumption: {b_df['Water_Consumed_Litres'].mean():,.0f} L/day (Median: {b_df['Water_Consumed_Litres'].median():,.0f} L)")
        print(f"  Mean Per-Capita (LPCD):  {b_df['Per_Capita_Consumption_LPCD'].mean():.2f} LPCD (Min: {b_df['Per_Capita_Consumption_LPCD'].min():.2f}, Max: {b_df['Per_Capita_Consumption_LPCD'].max():.2f})")
        print(f"  Total Borewell Supply:   {b_df['Borewell_Water_Supplied_Litres'].sum():,.0f} L")
        print(f"  Total Tanker Supply:     {b_df['Tanker_Water_Supplied_Litres'].sum():,.0f} L ({(b_df['Tanker_Water_Supplied_Litres'].sum()/b_df['Total_Water_Supplied_Litres'].sum())*100:.1f}% of total supply)")
        print(f"  Total Unaccounted Water: {b_df['Unaccounted_Water_Litres'].sum():,.0f} L")


if __name__ == "__main__":
    generate_all_eda_plots()

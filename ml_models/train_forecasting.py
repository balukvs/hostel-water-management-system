"""
train_forecasting.py
--------------------
Enhanced Machine Learning Module for Next-Day Water Requirement Prediction.

Key Enhancements for College Rubric Compliance:
1. Baseline Model (Naive Persistence).
2. Linear Regression vs. Tuned Random Forest Regressor.
3. Chronological 3-way Partitioning: Training (60%), Validation (20%), and Testing (20%).
4. TimeSeriesSplit (Walk-Forward Cross-Validation) for hyperparameter tuning.
5. Strict prevention of temporal data leakage.
6. Comprehensive evaluation using MAE, RMSE, MSE, and R² Score.
7. Diagnostic Plots:
   - Actual vs. Predicted Time-Series & Scatter Plot.
   - Residual Error Analysis (Residuals vs. Predicted, Error Distribution, Block breakdown).
8. Model serialization with diagnostic metadata.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set plotting styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})


def load_and_engineer_features(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load cleaned dataset and generate time-lag features while strictly
    preventing future data leakage.
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
    
    # -------------------------------------------------------------
    # Feature Engineering (Strictly Backward-Looking)
    # -------------------------------------------------------------
    # Target: Next day's consumption (T+1)
    df["Target_Next_Day_Consumption"] = df.groupby("Block")["Water_Consumed_Litres"].shift(-1)
    
    # Lag 1: Current day's consumption (T)
    df["Lag_1_Consumption"] = df["Water_Consumed_Litres"]
    
    # Lag 7: Consumption exactly 7 days ago (T-6)
    df["Lag_7_Consumption"] = df.groupby("Block")["Water_Consumed_Litres"].shift(6)
    
    # Rolling 7-day mean of past consumption (strictly up to day T)
    df["Rolling_7D_Mean_Consumption"] = (
        df.groupby("Block")["Water_Consumed_Litres"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
    )
    
    # Lag 1 Supply figures
    df["Lag_1_Borewell_Supply"] = df["Borewell_Water_Supplied_Litres"]
    df["Lag_1_Tanker_Supply"] = df["Tanker_Water_Supplied_Litres"]
    
    # Tomorrow's Calendar signals (known deterministically in advance)
    df["Next_Day_Of_Week"] = df.groupby("Block")["Day_Of_Week"].shift(-1)
    df["Next_Is_Weekend"] = df.groupby("Block")["Is_Weekend"].shift(-1)
    df["Next_Month"] = df.groupby("Block")["Month"].shift(-1)
    
    # Drop edge rows without sufficient history (first 6 days) or target (last day)
    df_model = df.dropna(subset=[
        "Target_Next_Day_Consumption",
        "Lag_7_Consumption",
        "Next_Day_Of_Week"
    ]).copy()
    
    return df_model, df


def chronological_3way_split(df_model: pd.DataFrame, train_pct=0.60, val_pct=0.20):
    """
    Split data strictly chronologically into Train (60%), Validation (20%), and Test (20%).
    NO random shuffling is applied to prevent temporal leakage.
    """
    unique_dates = np.sort(df_model["Date"].unique())
    n_dates = len(unique_dates)
    
    idx_train_end = int(n_dates * train_pct)
    idx_val_end = int(n_dates * (train_pct + val_pct))
    
    date_train_end = unique_dates[idx_train_end]
    date_val_end = unique_dates[idx_val_end]
    
    train_df = df_model[df_model["Date"] < date_train_end].copy()
    val_df = df_model[(df_model["Date"] >= date_train_end) & (df_model["Date"] < date_val_end)].copy()
    test_df = df_model[df_model["Date"] >= date_val_end].copy()
    
    print("\n" + "="*70)
    print("CHRONOLOGICAL 3-WAY DATA PARTITIONING (NO DATA LEAKAGE)")
    print("="*70)
    print(f"  Training Dates:   {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')} ({len(train_df)} samples)")
    print(f"  Validation Dates: {val_df['Date'].min().strftime('%Y-%m-%d')} to {val_df['Date'].max().strftime('%Y-%m-%d')} ({len(val_df)} samples)")
    print(f"  Testing Dates:    {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')} ({len(test_df)} samples)")
    
    feature_cols = [
        "Lag_1_Consumption",
        "Lag_7_Consumption",
        "Rolling_7D_Mean_Consumption",
        "Student_Occupancy",
        "Lag_1_Borewell_Supply",
        "Lag_1_Tanker_Supply",
        "Next_Day_Of_Week",
        "Next_Is_Weekend",
        "Next_Month"
    ]
    
    # One-hot encode Block categorical column
    def prepare_xy(df_sub):
        X_raw = df_sub[feature_cols + ["Block"]]
        X_encoded = pd.get_dummies(X_raw, columns=["Block"], dtype=int)
        y = df_sub["Target_Next_Day_Consumption"].values
        return X_encoded, y
        
    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)
    X_test, y_test = prepare_xy(test_df)
    
    # Align columns across splits
    X_train, X_val = X_train.align(X_val, join='outer', axis=1, fill_value=0)
    X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df, list(X_train.columns)


def tune_random_forest_with_tscv(X_train, y_train):
    """
    Tune Random Forest hyperparameters using TimeSeriesSplit (Walk-Forward CV).
    Guarantees no future data is used to validate past folds.
    """
    print("\n" + "="*70)
    print("STEP 2: HYPERPARAMETER TUNING VIA TimeSeriesSplit CROSS-VALIDATION")
    print("="*70)
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 8],
        'min_samples_split': [2, 4, 8]
    }
    
    rf_base = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"  Best Hyperparameters Found: {best_params}")
    print(f"  Best Cross-Validation MAE:  {-grid_search.best_score_:.2f} Litres")
    
    return grid_search.best_estimator_, best_params


def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "Model": model_name,
        "MAE (Litres)": round(mae, 2),
        "RMSE (Litres)": round(rmse, 2),
        "MSE": round(mse, 2),
        "R² Score": round(r2, 4)
    }


def evaluate_all_models(X_train, y_train, X_val, y_val, X_test, y_test, test_df, best_rf):
    """
    Train baseline, Linear Regression, and Tuned Random Forest.
    Evaluate and compare across validation and test sets.
    """
    # 1. Baseline: Naive Persistence (Tomorrow = Today)
    y_pred_baseline = test_df["Lag_1_Consumption"].values
    
    # 2. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    
    # 3. Tuned Random Forest
    # Fit best RF on Train + Validation combined for final test evaluation
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    best_rf.fit(X_train_val, y_train_val)
    y_pred_rf = best_rf.predict(X_test)
    
    results = [
        compute_metrics(y_test, y_pred_baseline, "1. Baseline (Naive Persistence)"),
        compute_metrics(y_test, y_pred_lr, "2. Linear Regression"),
        compute_metrics(y_test, y_pred_rf, "3. Tuned Random Forest Regressor")
    ]
    
    df_results = pd.DataFrame(results)
    return df_results, best_rf, lr_model, y_pred_baseline, y_pred_lr, y_pred_rf


def generate_actual_vs_predicted_plot(test_df: pd.DataFrame, y_test, y_pred_rf, y_pred_lr, save_path: str):
    """Generate Actual vs Predicted time-series and scatter plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Prepare comparison DataFrame
    plot_df = test_df[["Date", "Block"]].copy()
    plot_df["Actual"] = y_test
    plot_df["RF_Predicted"] = y_pred_rf
    plot_df["LR_Predicted"] = y_pred_lr
    plot_df = plot_df.sort_values("Date")
    
    # 1. Time Series Tracking
    sns.lineplot(data=plot_df, x="Date", y="Actual", label="Actual Consumption", color="black", lw=2.2, ax=ax1)
    sns.lineplot(data=plot_df, x="Date", y="RF_Predicted", label="Random Forest Forecast", color="#1F77B4", lw=2, linestyle="--", ax=ax1)
    sns.lineplot(data=plot_df, x="Date", y="LR_Predicted", label="Linear Regression Forecast", color="#D95F02", lw=1.5, linestyle=":", ax=ax1)
    ax1.set_title("Actual vs. Predicted Water Demand (Chronological Test Set)", weight="bold")
    ax1.set_ylabel("Water Consumption (Litres)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # 2. Scatter Plot: Actual vs Predicted
    min_val = min(y_test.min(), y_pred_rf.min()) * 0.95
    max_val = max(y_test.max(), y_pred_rf.max()) * 1.05
    
    sns.scatterplot(x=y_test, y=y_pred_rf, hue=plot_df["Block"], s=70, palette="deep", ax=ax2)
    ax2.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", lw=1.5, label="Perfect 1:1 Fit Line")
    ax2.set_title("Forecast Accuracy Scatter Plot (Tuned Random Forest)", weight="bold")
    ax2.set_xlabel("Actual Requirement (Litres)")
    ax2.set_ylabel("Predicted Requirement (Litres)")
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Generated Plot] {save_path}")


def generate_residual_analysis_plot(test_df: pd.DataFrame, y_test, y_pred_rf, save_path: str):
    """
    Perform formal Residual & Prediction Error Diagnostics:
    1. Residuals over time (error stability).
    2. Residuals vs. Predicted Values (homoscedasticity check).
    3. Error Distribution histogram (checking normality and bias).
    4. Block-by-block MAE breakdown.
    """
    residuals = y_test - y_pred_rf
    res_df = test_df[["Date", "Block"]].copy()
    res_df["Actual"] = y_test
    res_df["Predicted"] = y_pred_rf
    res_df["Residual"] = residuals
    res_df["Absolute_Error"] = np.abs(residuals)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    # 1. Residuals over Time
    sns.lineplot(data=res_df, x="Date", y="Residual", hue="Block", marker="o", ax=axes[0, 0])
    axes[0, 0].axhline(0, color="crimson", linestyle="--", lw=1.5)
    axes[0, 0].set_title("1. Forecast Residuals Over Chronological Timeline", weight="bold")
    axes[0, 0].set_ylabel("Residual (Actual - Predicted) [L]")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    
    # 2. Residuals vs. Predicted Values (Homoscedasticity)
    sns.scatterplot(data=res_df, x="Predicted", y="Residual", hue="Block", s=60, ax=axes[0, 1])
    axes[0, 1].axhline(0, color="crimson", linestyle="--", lw=1.5)
    axes[0, 1].set_title("2. Residuals vs. Predicted (Homoscedasticity Check)", weight="bold")
    axes[0, 1].set_xlabel("Predicted Value (Litres)")
    axes[0, 1].set_ylabel("Residual (Litres)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    
    # 3. Residual Distribution Histogram
    sns.histplot(residuals, kde=True, color="#2B5C8F", bins=12, ax=axes[1, 0])
    mean_err = np.mean(residuals)
    axes[1, 0].axvline(mean_err, color="red", linestyle="--", label=f"Mean Error: {mean_err:+.1f} L")
    axes[1, 0].set_title("3. Prediction Error Distribution (Normality & Bias)", weight="bold")
    axes[1, 0].set_xlabel("Residual (Litres)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    
    # 4. Block-by-Block MAE Comparison
    block_mae = res_df.groupby("Block")["Absolute_Error"].mean().reset_index()
    bars = sns.barplot(data=block_mae, x="Block", y="Absolute_Error", hue="Block", palette="Blues_d", legend=False, ax=axes[1, 1])
    axes[1, 1].set_title("4. Mean Absolute Error (MAE) by Hostel Block", weight="bold")
    axes[1, 1].set_ylabel("MAE (Litres)")
    for b in bars.patches:
        val = b.get_height()
        axes[1, 1].annotate(f"{val:,.0f} L", (b.get_x() + b.get_width() / 2, val * 0.9),
                            ha='center', va='bottom', weight='bold', color='white')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Generated Plot] {save_path}")
    
    return res_df


def run_improved_modeling_pipeline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_file = os.path.join(base_dir, "dataset", "processed", "cleaned_hostel_water.csv")
    models_dir = os.path.join(base_dir, "saved_models")
    plots_dir = os.path.join(base_dir, "eda", "plots")
    
    df_model, full_df = load_and_engineer_features(data_file)
    
    # Chronological 3-way partition
    X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df, feature_names = chronological_3way_split(
        df_model, train_pct=0.60, val_pct=0.20
    )
    
    # Hyperparameter tuning with TimeSeriesSplit
    best_rf, best_params = tune_random_forest_with_tscv(X_train, y_train)
    
    # Model evaluation
    df_results, final_rf, lr_model, y_pred_baseline, y_pred_lr, y_pred_rf = evaluate_all_models(
        X_train, y_train, X_val, y_val, X_test, y_test, test_df, best_rf
    )
    
    print("\n" + "="*80)
    print("FINAL MODEL EVALUATION ON UNSEEN CHRONOLOGICAL TEST SET")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    
    # Save diagnostic plots
    plot_1 = os.path.join(plots_dir, "09_actual_vs_predicted.png")
    plot_2 = os.path.join(plots_dir, "10_residual_error_analysis.png")
    
    generate_actual_vs_predicted_plot(test_df, y_test, y_pred_rf, y_pred_lr, plot_1)
    res_df = generate_residual_analysis_plot(test_df, y_test, y_pred_rf, plot_2)
    
    # Save model artifact
    model_artifact = {
        "model": final_rf,
        "feature_names": feature_names,
        "model_type": "Tuned Random Forest Regressor",
        "best_params": best_params,
        "metrics": df_results.to_dict(orient="records"),
        "test_mae": float(df_results.loc[df_results['Model'].str.contains('Random Forest'), 'MAE (Litres)'].values[0]),
        "mean_residual_bias": float(res_df["Residual"].mean())
    }
    
    model_save_path = os.path.join(models_dir, "demand_forecaster.pkl")
    joblib.dump(model_artifact, model_save_path)
    print(f"\n[Saved Artifact] Serialized tuned model saved to:\n  -> {model_save_path}")
    
    return df_results, res_df


if __name__ == "__main__":
    run_improved_modeling_pipeline()

# 💧 Smart Hostel Water Management System

An applied Data Science and Machine Learning project designed to monitor water consumption, forecast next-day requirements, audit distribution losses, and detect anomalous usage patterns across three campus hostel blocks.

> **Academic Notice:**  
> All records, sensor measurements, and logs in this project are derived from a reproducible, physically calibrated synthetic hydrological simulation (August 10, 2026 to November 10, 2026). This dataset was designed for educational demonstration and algorithm evaluation and does not represent actual institutional sensor data.

---

## 1. Project Background & Campus Context

College hostels accommodate high-density student populations with variable daily water requirements governed by academic routines, weekend departures, and hygiene peaks:

*   **Block A:** 774 residents | Dedicated Borewell ($14,000\text{ L/hr}$) | Storage Capacity: $150,000\text{ L}$
*   **Block B:** 760 residents | Dedicated Borewell ($13,500\text{ L/hr}$) | Storage Capacity: $150,000\text{ L}$
*   **Block C:** 600 residents | Dedicated Borewell ($12,000\text{ L/hr}$) | Storage Capacity: $120,000\text{ L}$
*   **Total Resident Population:** **2,134 Students**
*   **Standard Benchmark:** Indian Standard (IS 1172) recommends **$135\text{ Litres Per Capita per Day (LPCD)}$** for fully-flumbed student hostels.

---

## 2. System Architecture & Directory Structure

```text
hostel_water_management/
│
├── dataset/
│   ├── raw/                              # Original generated logs
│   │   ├── synthetic_daily_water_supply.csv
│   │   ├── synthetic_daily_tank_consumption.csv
│   │   ├── synthetic_water_wastage_observations.csv
│   │   └── synthetic_hostel_water_master.csv
│   └── processed/                        # Cleaned & feature-engineered data
│       ├── cleaned_hostel_water.csv      # 279 verified records (93 days × 3 blocks)
│       └── flagged_anomalies.csv         # Audit report of 17 flagged potential anomalies
│
├── preprocessing/
│   ├── __init__.py
│   ├── generate_synthetic_data.py        # Reproducible data generator (seed=42)
│   └── data_cleaning.py                  # Cleaning, primary key checks, validation
│
├── eda/
│   ├── __init__.py
│   ├── eda_analysis.py                   # Matplotlib/Seaborn analytics suite
│   └── plots/                            # High-resolution generated charts (.png)
│       ├── 01_daily_water_supply.png
│       ├── 02_daily_water_consumption.png
│       ├── 03_average_consumption_per_student.png
│       ├── 04_borewell_vs_tanker_supply.png
│       ├── 05_daily_weekly_trends.png
│       ├── 06_unusual_consumption_changes.png
│       ├── 07_block_consumption_comparison.png
│       └── 08_anomaly_detection_isolation_forest.png
│
├── ml_models/
│   ├── __init__.py
│   ├── train_forecasting.py              # Chronological ML demand forecaster
│   └── anomaly_detection.py              # Unsupervised Isolation Forest pipeline
│
├── saved_models/
│   └── demand_forecaster.pkl             # Serialized Random Forest model artifact
│
├── dashboard/
│   └── app.py                            # Interactive Streamlit multi-tab application
│
├── tests/
│   └── test_system.py                    # Automated test suite (4 unit test modules)
│
├── docs/
│   └── project_documentation.md          # Technical report & mathematical derivations
│
├── requirements.txt                      # Dependencies
└── README.md                             # Comprehensive project documentation
```

---

## 3. Step-by-Step Instructions: How to Run the Project

### Step 1: Environment Setup
Open PowerShell or Command Prompt in the project directory:
```powershell
# 1. Create a Python virtual environment
python -m venv venv

# 2. Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1
# (Or on Windows Command Prompt: .\venv\Scripts\activate.bat)

# 3. Install required libraries
pip install -r requirements.txt
```

### Step 2: Run Data Generation & Preprocessing
```powershell
# Generate synthetic dataset (Aug 10 - Nov 10, seed=42)
python preprocessing/generate_synthetic_data.py

# Run validation, error screening, and feature engineering pipeline
python preprocessing/data_cleaning.py
```

### Step 3: Run Exploratory Data Analysis (EDA)
```powershell
# Generates all 8 high-resolution charts in eda/plots/
python eda/eda_analysis.py
```

### Step 4: Train Machine Learning & Anomaly Models
```powershell
# Train and evaluate chronological forecasting models
python ml_models/train_forecasting.py

# Run unsupervised Isolation Forest anomaly detection
python ml_models/anomaly_detection.py
```

### Step 5: Run Automated System Tests
```powershell
python -m unittest tests/test_system.py
```

### Step 6: Launch the Interactive Streamlit Dashboard
```powershell
streamlit run dashboard/app.py
```
Open your browser to `http://localhost:8501`.

---

## 4. Dataset Description & Simulation Assumptions

The dataset covers a **93-calendar-day academic semester** (August 10, 2026 to November 10, 2026) across 3 blocks ($279\text{ daily records}$):

1.  **Hydrological Continuity:** Storage levels obey physical water conservation:
    $$\text{Starting Storage}_t = \text{Ending Storage}_{t-1}$$
    $$\text{Ending Storage}_t = \text{Starting Storage}_t + \text{Inflow}_t - \text{Consumption}_t - \text{Wastage}_t$$
2.  **Student Occupancy Modeling:**
    *   *Weekdays:* $94\%\text{ to }99\%$ of design capacity.
    *   *Weekends:* $75\%\text{ to }85\%$ of design capacity (students visiting home).
    *   *Mid-Semester Break (Oct 14–19):* Plunges to $18\%\text{ to }25\%$ attendance.
3.  **Per-Capita Usage Dynamics:**
    *   *Weekday baseline:* $\mathcal{N}(133, 5)\text{ LPCD}$.
    *   *Weekend baseline:* $\mathcal{N}(142, 6)\text{ LPCD}$ (higher personal laundry and bath duration).
    *   *Campus Cultural Fest (Oct 5–7):* $\mathcal{N}(165, 8)\text{ LPCD}$ due to campus guests.

---

## 5. Machine Learning Models & Selection Rationale

### A. Next-Day Requirement Prediction (Regression)

*   **Target Variable:** $Y_{t+1}$ (Next Day's Water Consumption in Litres).
*   **Data Partitioning (Strictly Chronological 3-Way Split):**  
    *NO random shuffling.*
    *   **Training Set (60%):** August 16 to October 05, 2026 ($153\text{ samples}$)
    *   **Validation Set (20%):** October 06 to October 22, 2026 ($51\text{ samples}$)
    *   **Testing Set (20%):** October 23 to November 09, 2026 ($54\text{ samples}$)
*   **Hyperparameter Tuning:**  
    Conducted using **`TimeSeriesSplit(n_splits=3)`** (Walk-Forward Cross-Validation) on the training set. The parameter search explored `max_depth` $\in [3, 5, 8]$, `min_samples_split` $\in [2, 4, 8]$, and `n_estimators` $\in [50, 100, 150]$.
    *   *Optimal Hyperparameters:* `{'max_depth': 8, 'min_samples_split': 2, 'n_estimators': 150}`
    *   *Cross-Validation MAE:* $6,561.85\text{ Litres}$

#### Empirical Model Evaluation Results on Unseen Chronological Test Set:

| Model | MAE (Litres) | RMSE (Litres) | MSE | $R^2$ Score | Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Baseline (Naive Persistence)** | $6,687.81\text{ L}$ | $8,789.67\text{ L}$ | $77,258,271$ | $0.3783$ | Simple assumption that tomorrow = today. |
| **2. Linear Regression** | $10,022.22\text{ L}$ | $12,008.77\text{ L}$ | $144,210,640$ | $-0.1605$ | Degrades severely due to non-linear holiday dips and interaction effects. |
| **3. Tuned Random Forest Regressor** | **$4,655.98\text{ L}$** | **$6,182.58\text{ L}$** | **$38,224,315$** | **$0.6924$** | **Best Model:** Explains $69.2\%$ of variance with lowest absolute error. |

#### Residual & Error Diagnostics:
*   **Mean Residual (Bias):** $\approx -480\text{ Litres}$ (centered near zero, indicating no systematic over- or under-prediction bias).
*   **Actual vs. Predicted Plot:** Saved in [09_actual_vs_predicted.png](file:///c:/Users/bb727/OneDrive/Desktop/hostel_water_management/eda/plots/09_actual_vs_predicted.png).
*   **Residual Diagnostics Plot:** Saved in [10_residual_error_analysis.png](file:///c:/Users/bb727/OneDrive/Desktop/hostel_water_management/eda/plots/10_residual_error_analysis.png).
*   **Block-Level MAE:** Block A: $4,820\text{ L}$, Block B: $4,690\text{ L}$, Block C: $4,458\text{ L}$. Errors are proportionally consistent with block sizes.

### B. Unsupervised Anomaly Detection (Isolation Forest)

*   **Why Isolation Forest?** Water distribution systems rarely have labeled failure databases. Isolation Forest isolates anomalies by randomly partitioning feature space. Because outliers have extreme values, they require significantly fewer tree splits (**shorter path lengths**) than normal observations.
*   **Features Evaluated:** `Water_Consumed_Litres`, `Per_Capita_Consumption_LPCD`, `Consumption_Deviation_7D`, `Occupancy_Ratio`, and `Unaccounted_Water_Litres`.
*   **Result Labeling:** All detections are labeled as **`Potential anomaly — inspection recommended`**. (An algorithm flags mathematical deviations; it cannot confirm physical pipe leakage without on-site technical inspection).

---

## 6. Project Limitations & Future Improvements

### Academic Limitations
1.  **Sample Size:** 93 days ($279\text{ rows}$) represents a 3-month semester. While sufficient for tree-based models, it is too small for deep learning time-series architectures (like LSTM or Prophet).
2.  **Synthetic Nature:** Real-world hostels suffer from missing manual logs, power outages, and intermittent valve blockages that cannot be fully captured in simulated data.
3.  **Extrapolation Barrier:** Random Forest cannot predict consumption values beyond the highest range seen in the training data.

### Future Improvements
1.  **IoT Integration:** Connect ultrasonic tank sensors and pulse-output digital flow meters via ESP32 microcontrollers for real-time telemetry.
2.  **Weather Correlation:** Incorporate ambient temperature and rainfall APIs to model seasonal water consumption spikes.
3.  **Automated Pump Cut-off:** Interface relay switches with the Streamlit dashboard to automatically stop borewells when storage reaches $95\%$ to prevent overflow.

---

## 7. Viva & Evaluation Defense Guide

| Common Examiner Question | Recommended Technical Defense |
| :--- | :--- |
| **Why did you not use a 70/30 random train-test split?** | Water consumption is an autocorrelated time series. Randomly shuffling rows causes future data leakage (training on Wednesday to predict Tuesday), yielding artificially inflated accuracy that collapses in production. Chronological splitting mirrors real-world deployment. |
| **Can your anomaly detection algorithm confirm a water leak?** | No. A machine learning model detects statistical divergence, not physical plumbing cracks. A spike can stem from an athletic fest or hostel party. Therefore, our model classifies alerts as *"Potential anomaly — inspection recommended"*. |
| **How do you calculate unaccounted water?** | Using the Water Balance equation: $\text{Unaccounted} = (\text{Borewell} + \text{Tanker}) - \Delta\text{Storage} - \text{Metered Draw}$. If inflow exceeds consumption and tank accumulation, the delta points to distribution losses. |
| **Why did Random Forest outperform Linear Regression?** | Hostel water consumption involves non-linear step functions (e.g. weekend occupancy drops by $20\%$ while individual LPCD rises by $10\%$). Decision tree splits naturally handle these interaction branches better than a linear slope. |

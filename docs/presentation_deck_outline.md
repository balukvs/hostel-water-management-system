# Smart Hostel Water Management System — 10-Slide Viva Presentation Outline

Use this slide-by-slide structure to defend your data science project during evaluation:

---

### Slide 1: Title & Overview
*   **Project Title:** Smart Hostel Water Management System
*   **Domain:** Time-Series Forecasting & Unsupervised Anomaly Detection in Campus Hydrology
*   **Target Population:** 2,134 students across 3 Hostel Blocks (Block A: 774, Block B: 760, Block C: 600).
*   **Tech Stack:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit.

---

### Slide 2: Problem Statement & Measurable Objectives
*   **Core Problem:** High-density hostel blocks suffer from unpredictable water draw, peak morning shortages, expensive emergency tankers, and hidden plumbing leakages.
*   **Measurable Objectives:**
    1. Daily consumption tracking and LPCD benchmarking (IS 1172 Standard = 135 LPCD).
    2. Next-day water requirement forecasting (evaluated via MAE, RMSE, R²).
    3. Unsupervised anomaly detection via Isolation Forest.
    4. Quantitative distribution loss auditing via Water Balance equations.
    5. Interactive decision-support dashboard for hostel wardens.

---

### Slide 3: Hydrological Infrastructure & Water Balance Physics
*   **Infrastructure:** Each block has a dedicated deep borewell ($12\text{k to }14\text{k L/hr}$) and storage tanks ($120\text{k to }150\text{k L}$).
*   **Water Conservation Balance Equation:**
    $$\text{Unaccounted Water} = (\text{Borewell Inflow} + \text{Tanker Supply}) - \Delta\text{Storage} - \text{Metered Draw}$$
*   **Continuity Principle:** Today's starting tank level strictly equals yesterday's ending tank level.

---

### Slide 4: Data Engineering & Preventing Data Leakage
*   **Dataset Scope:** 93-calendar-day autumn semester timeline (August 10 – November 10, 2026; 279 records).
*   **Data Cleaning:** Primary key uniqueness `(Date, Block)`, non-negative flow bounds, pump run limit ($\le 24\text{ hrs}$).
*   **Feature Engineering (Strictly Backward-Looking):**
    *   $T-1$ Lag, $T-7$ Seasonal Lag, 7-Day Rolling Mean.
    *   Calendar signals (Day of week, Month, Is_Weekend) to capture personal laundry peaks.
*   **No Future Leakage:** Target is strictly shifted forward ($T+1$). Training data never peeks into future dates.

---

### Slide 5: Exploratory Data Analysis Key Findings
*   **Consistency Across Blocks:** When normalized by student headcount, mean consumption across all three blocks aligns tightly with the national standard:
    *   Block A: $136.36\text{ LPCD}$ | Block B: $136.16\text{ LPCD}$ | Block C: $137.66\text{ LPCD}$.
*   **Temporal Cycles:** Weekends show higher individual hygiene usage ($142\text{ LPCD}$ vs $133\text{ LPCD}$), but overall campus volume drops by $15\%\text{–}20\%$ due to students traveling home.
*   **Source Vulnerability:** Block C required $540,000\text{ L}$ ($7.9\%$) in tanker water, identifying it as the most vulnerable block during dry spells.

---

### Slide 6: Machine Learning Methodology & Chronological 3-Way Partitioning
*   **Why Standard Random Train-Test Split Fails:** Water data is autocorrelated. Shuffling leads to future data leakage (training on Wednesday to predict Tuesday).
*   **Chronological Partition:**
    *   **Train (60%):** August 16 – October 05 (153 samples)
    *   **Validation (20%):** October 06 – October 22 (51 samples)
    *   **Test (20%):** October 23 – November 09 (54 samples)
*   **Hyperparameter Tuning:** Conducted using `TimeSeriesSplit(n_splits=3)` (Walk-Forward Cross-Validation) on the training set.
    *   *Optimal parameters:* `max_depth=8, min_samples_split=2, n_estimators=150`.

---

### Slide 7: Model Evaluation & Performance Comparison
*Results on the held-out Chronological Test Set:*

| Model | MAE (Litres) | RMSE (Litres) | $R^2$ Score | Key Finding |
| :--- | :---: | :---: | :---: | :--- |
| **Naive Persistence Baseline** | $6,687.81\text{ L}$ | $8,789.67\text{ L}$ | $0.3783$ | Struggles on weekend/holiday transitions |
| **Linear Regression** | $10,022.22\text{ L}$ | $12,008.77\text{ L}$ | $-0.1605$ | Collapses due to non-linear interaction terms |
| **Tuned Random Forest** | **$4,655.98\text{ L}$** | **$6,182.58\text{ L}$** | **$0.6924$** | **Best:** Captures multi-variable step functions |

---

### Slide 8: Residual & Prediction Error Diagnostics
*   **Residual Analysis:** Computed $\text{Residual} = y - \hat{y}$.
*   **Error Bias Check:** Mean residual is $-480\text{ Litres}$, centered close to zero, proving the model is not systematically over- or under-predicting.
*   **Homoscedasticity:** Residual scatter plots show constant variance across prediction ranges.
*   **Block Errors:** Block A MAE = $4,820\text{ L}$, Block B MAE = $4,690\text{ L}$, Block C MAE = $4,458\text{ L}$. Errors scale proportionally with block capacities.

---

### Slide 9: Unsupervised Anomaly Detection (Isolation Forest)
*   **Algorithm:** Scikit-learn's Isolation Forest isolates anomalies using random recursive feature splits. Outliers require significantly fewer splits (**shorter tree paths**).
*   **Audit Results:** 17 operational events flagged across 93 days ($6\%$ contamination rate).
    *   *Aug 29–30:* Flagged severe unaccounted water ($>31\text{k L/day}$) $\to$ Subsurface pipe crack.
    *   *Oct 5–6:* Flagged extreme per-capita surge ($>165\text{ LPCD}$) $\to$ Cultural fest guest influx.
    *   *Oct 14–19:* Flagged extreme low consumption ($<25\text{k L/day}$) $\to$ Mid-semester vacation.
*   **Safety Status:** Displayed as **`Potential anomaly — inspection recommended`**. (ML models detect statistical divergence, not confirmed physical cracks).

---

### Slide 10: Conclusion, Limitations & Future Work
*   **Key Contribution:** Demonstrated that combining non-linear machine learning with hydrological water balancing enables automated demand forecasting and leakage alerting.
*   **Academic Limitations:** 93-day single-semester dataset; cannot extrapolate to expanded hostel buildings without retraining.
*   **Future Scope:**
    1. Deploy ESP32 IoT ultrasonic depth transmitters for real-time tank telemetry.
    2. Incorporate weather and temperature APIs to model seasonal summer peaks.
    3. Relay-controlled automated pump cut-off when tanks reach $95\%$ to eliminate overflow.

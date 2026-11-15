# Smart Hostel Water Management System
## Final Project Report & Academic Defense Document

---

### 1. Abstract
Water scarcity and erratic distribution are critical operational challenges in residential college campuses. This project presents a data-driven **Smart Hostel Water Management System** designed to monitor, forecast, and audit water consumption across three hostel blocks housing **2,134 students** (Block A: 774, Block B: 760, Block C: 600). Using a 93-day physically consistent synthetic dataset (August 10 – November 10, 2026), the system analyzes consumption dynamics against the Indian Standard benchmark of **135 Litres Per Capita per Day (LPCD)**. A machine learning pipeline featuring a Naive Persistence Baseline, Linear Regression, and a Tuned Random Forest Regressor was developed using strictly chronological 3-way partitioning and `TimeSeriesSplit` cross-validation. The tuned Random Forest achieved a **Mean Absolute Error (MAE) of 4,655.98 Litres** and an **$R^2$ score of 0.6924**, substantially outperforming Linear Regression ($R^2 = -0.1605$). Furthermore, an unsupervised **Isolation Forest** module flagged 17 operational anomalies (including pipe fractures, tank overflows, and vacation dips), labeled under the safety guideline *"Potential anomaly — inspection recommended"*. The complete solution is deployed as an interactive Streamlit web dashboard supported by an automated test suite.

---

### 2. Introduction
In university hostels, water consumption exhibits strong temporal periodicity driven by academic timetables, weekend laundry routines, and seasonal breaks. Hostel administrations frequently suffer from two opposing issues: unexpected morning water depletion requiring expensive emergency commercial tankers, and undetected underground pipe fractures causing massive unmetered wastage. By combining hydrological water balance equations with machine learning demand forecasting, this project demonstrates an intelligent decision-support system for estate managers and hostel wardens.

---

### 3. Problem Statement
Hostels A, B, and C operate independent deep borewells, storage tanks, and distribution risers, supplemented by occasional external tankers. The estate office lacks:
1. Automated forecasting of next-day water demand to plan borewell pumping run-times.
2. Normalization of water use against variable daily student headcount (LPCD).
3. Early detection of underground distribution leaks and tank float-valve overflows before substantial financial and volumetric loss occurs.

---

### 4. Measurable Objectives
1. **Consumption Analytics:** Track daily consumption across all blocks and benchmark against IS 1172 ($135\text{ LPCD}$).
2. **Next-Day Demand Forecasting:** Predict next-day water demand with an MAE lower than naive persistence using machine learning.
3. **Distribution Audit:** Quantify daily unaccounted water using the physical Water Balance Conservation Equation.
4. **Anomaly Identification:** Detect unusual consumption patterns (high spikes and low drops) using unsupervised Isolation Forest.
5. **Interactive Interface:** Present all models, historical trends, and real-time operational advisories in a Streamlit dashboard.

---

### 5. Dataset Description & Synthetic Simulation Justification

#### 5.1 Infrastructure Baseline:
*   **Block A:** 774 design capacity | Borewell yield: $14,000\text{ L/hr}$ | Tank capacity: $150,000\text{ L}$
*   **Block B:** 760 design capacity | Borewell yield: $13,500\text{ L/hr}$ | Tank capacity: $150,000\text{ L}$
*   **Block C:** 600 design capacity | Borewell yield: $12,000\text{ L/hr}$ | Tank capacity: $120,000\text{ L}$
*   **Total Campus Hostel Population:** **2,134 Students**

#### 5.2 Simulation Justification:
Due to institutional restrictions preventing the installation of IoT meters across hostel pipelines, a physically grounded, reproducible synthetic dataset was generated (`seed=42`) spanning 93 calendar days (August 10 – November 10, 2026; 279 total master records). Hydrological continuity was strictly maintained:
$$\text{Starting Storage}_t = \text{Ending Storage}_{t-1}$$
$$\text{Ending Storage}_t = \text{Starting Storage}_t + \text{Inflow}_t - \text{Consumption}_t - \text{Wastage}_t$$
Every record is marked with `Data_Source: SYNTHETIC_SIMULATION` to maintain academic integrity.

---

### 6. Data Preprocessing
The preprocessing pipeline (`preprocessing/data_cleaning.py`) implements:
*   **Missing Value Audit:** Screens for unrecorded flow cells without silently inventing values.
*   **Primary Key Uniqueness:** Enforces distinct `(Date, Block)` pairs, dropping accidental duplicate logs.
*   **Physical Boundary Validation:** Asserts non-negative flow volumes, student headcounts $> 0$, and daily pump run hours $\le 24.0$.
*   **Type Casting:** Converts raw date strings to `datetime64[ns]` and strips extraneous whitespace from block names.

---

### 7. Exploratory Data Analysis (EDA) Key Findings
EDA was executed using Matplotlib and Seaborn, producing 8 analytical figures:
1.  **Per-Capita Equity:** When normalized by active student headcount, average water usage across blocks is remarkably uniform: Block A = $136.36\text{ LPCD}$, Block B = $136.16\text{ LPCD}$, and Block C = $137.66\text{ LPCD}$, tightly matching the national benchmark of $135\text{ LPCD}$.
2.  **Weekly Hygiene Cycle:** Saturdays and Sundays exhibit higher per-capita usage ($\approx 142\text{ LPCD}$ vs $\approx 133\text{ LPCD}$) due to laundry and extended bathing, but overall campus volumetric draw drops by $15\%\text{–}20\%$ because local students travel home.
3.  **Source Vulnerability:** Block C required $540,000\text{ Litres}$ ($7.9\%$) in commercial tanker deliveries across the term, identifying it as the most vulnerable infrastructure during groundwater drawdown.

---

### 8. Feature Engineering
To predict tomorrow's consumption ($Y_{t+1}$), all features were engineered strictly backward-looking to eliminate data leakage:
*   `Lag_1_Consumption`: Usage at day $T$ (yesterday's baseline).
*   `Lag_7_Consumption`: Usage at day $T-6$ (captures weekly cyclicality).
*   `Rolling_7D_Mean_Consumption`: Moving average filtering short-term noise.
*   `Student_Occupancy`: Expected headcount for the target day.
*   `Next_Day_Of_Week`, `Next_Is_Weekend`, `Next_Month`: Calendar indicators known in advance.
*   `Block_Block A, B, C`: One-hot encoded categorical indicators.

---

### 9. Machine Learning Algorithms

1.  **Baseline Model (Naive Persistence):**  
    Assumes tomorrow's consumption equals today's ($\hat{Y}_{t+1} = Y_t$). Minimal benchmark for time-series forecasting.
2.  **Linear Regression:**  
    Models consumption as a linear weighted sum of past lags and occupancy.
3.  **Random Forest Regressor:**  
    An ensemble of 150 decorrelated decision trees that recursively partition feature space into homogeneous sub-regions.
4.  **Isolation Forest (Anomaly Detection):**  
    An unsupervised ensemble algorithm that isolates anomalies using random axis-aligned splits. Outliers require fewer tree partitions (shorter path lengths) than normal data.

---

### 10. Model Training & Validation Methodology

#### Chronological 3-Way Partitioning:
Standard random shuffling was strictly prohibited to prevent **future data leakage**. The 93-day timeline was partitioned chronologically:
*   **Training Set (60%):** August 16 to October 05, 2026 ($153\text{ samples}$)
*   **Validation Set (20%):** October 06 to October 22, 2026 ($51\text{ samples}$)
*   **Testing Set (20%):** October 23 to November 09, 2026 ($54\text{ samples}$)

#### Walk-Forward Hyperparameter Tuning:
Using `TimeSeriesSplit(n_splits=3)` on the training partition, a systematic grid search identified the optimal Random Forest architecture:
*   `max_depth`: 8
*   `min_samples_split`: 2
*   `n_estimators`: 150
*   *Validation Cross-Validation MAE:* $6,561.85\text{ Litres}$

---

### 11. Model Evaluation & Comparison

Performance on the held-out **unseen chronological test set**:

| Model | MAE (Litres) | RMSE (Litres) | MSE | $R^2$ Score | Evaluation Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Baseline (Naive Persistence)** | $6,687.81\text{ L}$ | $8,789.67\text{ L}$ | $77,258,271$ | $0.3783$ | Poor adaptability during weekend shifts |
| **2. Linear Regression** | $10,022.22\text{ L}$ | $12,008.77\text{ L}$ | $144,210,640$ | $-0.1605$ | Fails completely due to non-linear holiday dips |
| **3. Tuned Random Forest Regressor** | **$4,655.98\text{ L}$** | **$6,182.58\text{ L}$** | **$38,224,315$** | **$0.6924$** | **Best Model:** Explains $69.2\%$ of variance |

---

### 12. Residual & Prediction Error Diagnostics

Residuals were computed as $\text{Residual} = Y_{\text{actual}} - \hat{Y}_{\text{pred}}$:
1.  **Systematic Bias Check:** Mean residual error is **$-480\text{ Litres}$**, demonstrating zero-centered symmetry without persistent over- or under-prediction bias.
2.  **Homoscedasticity:** Scatter plot of residuals against predicted values demonstrates uniform dispersion across the entire dynamic range ($60\text{k to }110\text{k L}$).
3.  **Block-by-Block Error Consistency:**
    *   Block A (774 cap): $\text{MAE} = 4,820\text{ L}$ ($4.8\%$ relative error)
    *   Block B (760 cap): $\text{MAE} = 4,690\text{ L}$ ($5.1\%$ relative error)
    *   Block C (600 cap): $\text{MAE} = 4,458\text{ L}$ ($5.8\%$ relative error)

---

### 13. Results & Discussion
The experimental findings demonstrate that student water consumption cannot be accurately modeled with linear regression ($R^2 < 0$) because occupancy drops and weekend laundry routines interact non-linearly. The Tuned Random Forest reduced forecasting error to **$\pm 4,656\text{ Litres}$**, allowing hostel staff to predict pumping hours within $\approx 20\text{ minutes}$ of precision. Concurrently, the Isolation Forest flagged 17 critical operational anomalies, including a simulated pipe fracture in Block A (August 29–30; $>31\text{k L}$ loss) and flush leaks in Block C (October 31; $175.7\text{ LPCD}$).

---

### 14. Academic Limitations
1.  **Temporal Scope:** 93 days ($279$ records) models a single semester; seasonal summer monsoon variations are omitted.
2.  **Extrapolation Limit:** Decision trees cannot predict target values outside the maximum training bound ($140,000\text{ L}$).
3.  **Synthetic Data Constraint:** Real pipelines experience gradual silt accumulation, electrical brownouts, and human logging errors not fully simulated.

---

### 15. Conclusion
The Smart Hostel Water Management System successfully fulfills all five measurable objectives. By combining physical water balance tracking with machine learning, the system eliminates reliance on guesswork, ensures fair per-capita resource auditing, and provides wardens with an intuitive, interactive control center.

---

### 16. Future Scope
1.  **IoT Hardware Integration:** Interfacing ESP32 microcontrollers with JSN-SR04T waterproof ultrasonic sensors for real-time tank level telemetry.
2.  **Automated Pump Automation:** Integrating relay-actuated contactors to automatically cut power to borewell motors when tank levels reach $95\%$.
3.  **Weather API Integration:** Incorporating ambient temperature and rainfall data to model summer dehydration peaks.

---

### 17. References
1. Bureau of Indian Standards (BIS). *IS 1172: Code of Basic Requirements for Water Supply, Drainage and Sanitation*. New Delhi, 1993.
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
3. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. IEEE Eighth International Conference on Data Mining (ICDM), pp. 413-422.
4. McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. Proceedings of the 9th Python in Science Conference, pp. 51-56.

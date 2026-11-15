# Smart Hostel Water Management System — Comprehensive Technical Report

## 1. Project Background & System Description
Hostel residential blocks accommodate dense student populations with dynamic water demands:
- **ashwatha:** 774 residents | Dedicated Borewell ($14,000\text{ L/hr}$) | Storage Capacity: $150,000\text{ L}$
- **ashoka:** 760 residents | Dedicated Borewell ($13,500\text{ L/hr}$) | Storage Capacity: $150,000\text{ L}$
- **jasmine:** 600 residents | Dedicated Borewell ($12,000\text{ L/hr}$) | Storage Capacity: $120,000\text{ L}$
- **Total Campus Hostel Population:** **2,134 Students**

---

## 2. Mathematical Framework & Hydrological Physics

### 2.1 Per-Capita Consumption (LPCD)
$$\text{LPCD} = \frac{\text{Daily Water Consumed (Litres)}}{\text{Active Student Occupancy}}$$
*Benchmark (IS 1172):* $135\text{ Litres/Capita/Day}$.

### 2.2 Storage Dynamics & Continuity
$$\Delta\text{Storage}_t = \text{Ending Tank Level}_t - \text{Starting Tank Level}_t$$
$$\text{Starting Tank Level}_t = \text{Ending Tank Level}_{t-1}$$

### 2.3 Water Balance & Unaccounted Water Audit
$$\text{Unaccounted Water} = \text{Total Inflow} - \Delta\text{Storage} - \text{Metered Consumption}$$
Where $\text{Total Inflow} = \text{Borewell Water Supplied} + \text{Tanker Water Supplied}$.

---

## 3. Machine Learning Formulation

### 3.1 Next-Day Demand Regression
* **Target:** $Y_{t+1} \in \mathbb{R}^+$ (Next Day's Water Consumption).
* **Partitioning:** Strictly Chronological ($80\%$ Train, $20\%$ Test) to eliminate temporal data leakage.
* **Results:**
  * Baseline (Naive): $\text{MAE} = 6,687.81\text{ L}$, $R^2 = 0.3783$
  * Linear Regression: $\text{MAE} = 7,433.95\text{ L}$, $R^2 = 0.3761$
  * **Random Forest Regressor:** $\text{MAE} = \mathbf{4,363.45\text{ L}}$, $R^2 = \mathbf{0.7437}$

### 3.2 Unsupervised Anomaly Detection (Isolation Forest)
* Ensemble of 150 random partition trees.
* Contamination parameter: $0.06$ ($6\%$ anomaly threshold).
* Safety Labeling: **`Potential anomaly — inspection recommended`**.

---

## 4. Test Suite Summary
Located at `tests/test_system.py`:
1. `test_01_processed_dataset_exists_and_valid`: Validates clean data existence and keys.
2. `test_02_water_supply_and_balance_math`: Proves mathematical compliance with water balance equations.
3. `test_03_prediction_pipeline`: Validates model loading, schema alignment, and physical plausibility.
4. `test_04_anomaly_detection_labeling`: Verifies mandatory diagnostic reasons and caution labeling.

# Smart Hostel Water Management System

A data science-based application for analyzing hostel water consumption, forecasting water demand and identifying unusual water usage patterns using machine learning.

**Academic Project | Data Science**

## 1. Project Submission

**IIITDM KANCHEEPURAM · IDSE Project**

| Team Member | Roll Number |
|---|---|
| Kalapati Balaji | EC24B1054 |
| Sudeep Nanda | EC24B1050 |

**Project Title:** Smart Hostel Water Management System

## 2. Project Overview

Water management is an important requirement in residential hostels. Understanding daily water consumption, estimating future demand and identifying unusual usage patterns can help improve water resource planning.

The Smart Hostel Water Management System uses Python, data analysis and machine learning to demonstrate how hostel water consumption can be monitored and analyzed.

The project includes data preprocessing, exploratory data analysis, machine learning-based demand forecasting, anomaly detection and an interactive Streamlit dashboard.

**Data disclosure:** This academic prototype uses simulated hostel water data. It does not use verified water meter readings or actual hostel supply records.

## 3. Project Objectives

- Analyze water consumption across different hostels.
- Study water supply and demand patterns.
- Estimate future water requirements using machine learning.
- Identify unusual water consumption patterns.
- Compare estimated demand with available water supply.
- Visualize important information through an interactive dashboard.
- Demonstrate the application of data science to water resource management.

## 4. Hostel Information

The project considers three hostels:

| Hostel Name | Category | Student Capacity |
|---|---|---:|
| Aswatha | Boys Hostel | 774 |
| Ashoka | Boys Hostel | 760 |
| Jasmine | Girls Hostel | 600 |
| **Total** | | **2,134** |

The capacities are project inputs and should be verified against official records before real-world use.

## 5. Key Features

### Data Preprocessing
- Loading and organizing hostel water datasets.
- Handling missing and inconsistent values.
- Preparing data for analysis and machine learning.
- Saving processed datasets.

### Exploratory Data Analysis
- Hostel-wise water consumption analysis.
- Water supply and demand comparisons.
- Consumption trend visualization.
- Graphical analysis of water usage patterns.

### Water Demand Forecasting
- Uses historical consumption information.
- Applies a trained machine learning model.
- Estimates future water demand.
- Displays predictions through the dashboard.

### Anomaly Detection
- Identifies unusual water consumption patterns.
- Highlights potentially abnormal observations.
- Helps investigate possible water wastage or leakage.

An anomaly indicates an unusual observation; it does not independently confirm a water leak.

### Interactive Dashboard
- Displays hostel water management information.
- Provides consumption visualizations.
- Displays demand predictions.
- Presents anomaly detection results.
- Helps understand estimated water requirements.

## 6. Technology Stack

| Technology | Application |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive dashboard |
| Pickle | Saving and loading the trained model |
| Git and GitHub | Version control and project hosting |

## 7. Project Structure

```text
hostel_water_management/
│
├── dashboard/
│   └── app.py
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── Smart_Hostel_Water_Management_Project_Report.pdf
│   ├── project_documentation.md
│   
│
├── eda/
│
├── ml_models/
│
├── preprocessing/
│
├── saved_models/
│   └── demand_forecaster.pkl
│
├── tests/
│   └── test_system.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 8. Dataset Information

The project uses simulated data for demonstrating hostel water management and machine learning workflows.

The dataset includes information used for analyzing:

- Hostel-wise water consumption.
- Borewell water supply.
- Tanker water supply.
- Student occupancy.
- Historical water consumption.
- Water demand patterns.

The project stores source datasets in `dataset/raw/` and processed datasets in `dataset/processed/`.

### Data Limitations

- The data is simulated and is not collected from actual hostel water meters.
- Supply and consumption values are demonstration inputs.
- Pumping rates are illustrative assumptions, not verified operational measurements.
- The model predictions are intended for academic demonstration and require validation using real data before practical deployment.

## 9. Machine Learning Workflow

The project follows these general stages:

1. Load the dataset.
2. Perform data cleaning and preprocessing.
3. Conduct exploratory data analysis.
4. Prepare the required machine learning features.
5. Train the forecasting model.
6. Save the trained model.
7. Load the model into the dashboard.
8. Generate water demand predictions.
9. Identify unusual consumption patterns.
10. Display results using dashboard visualizations.

The saved forecasting model is located in:

```text
saved_models/demand_forecaster.pkl
```

## 10. Installation and Execution

### Prerequisites

- Python 3
- Visual Studio Code or another Python editor
- Git (for cloning the repository)

### Step 1: Clone the repository

```bash
git clone https://github.com/balukvs/hostel-water-management-system.git
```

### Step 2: Open the project directory

```bash
cd hostel-water-management-system
```

### Step 3: Create a virtual environment

For Windows:

```powershell
python -m venv .venv
```

### Step 4: Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install the required libraries

```powershell
python -m pip install -r requirements.txt
```

### Step 6: Run the Streamlit application

```powershell
streamlit run dashboard/app.py
```

### Step 7: Open the dashboard

After starting the application, open the local URL displayed in the terminal.

Usually:

```text
http://localhost:8501
```

The dashboard will be available in your browser while the Streamlit application is running.

## 11. Limitations

- The current implementation uses simulated data.
- Predictions depend on the quality and assumptions of the available dataset.
- Anomaly detection identifies unusual patterns but cannot confirm the physical cause.
- Real-world accuracy has not been established using verified hostel measurements.
- Operational deployment would require real-time data collection and further testing.

## 12. Future Enhancements

- Integration with real-time water meters.
- IoT-based water level monitoring.
- Automated leakage alerts.
- Improved forecasting with additional historical data.
- User authentication and administrative access.
- Automated water supply scheduling.
- Deployment for remote dashboard access.

## 13. Conclusion

The Smart Hostel Water Management System demonstrates how data science and machine learning can support hostel water resource management.

By combining data preprocessing, exploratory data analysis, demand forecasting, anomaly detection and an interactive dashboard, the project provides an academic prototype for understanding water consumption and supporting better water planning.

Future development using verified hostel water measurements and IoT devices could help extend the system towards practical water management applications.

---

## Project Repository

GitHub: https://github.com/balukvs/hostel-water-management-system

**Project Title:** Smart Hostel Water Management System

**Institution:** IIITDM Kancheepuram

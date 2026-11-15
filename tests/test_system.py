"""
test_system.py
--------------
Automated Test Suite for Smart Hostel Water Management System.

Tests Included:
1. CSV Loading and Preprocessing Integrity.
2. Water Supply and Water Balance Calculations.
3. Machine Learning Forecasting Functionality.
4. Anomaly Detection and Safety Labeling.
"""

import os
import unittest
import pandas as pd
import numpy as np
import joblib

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
CLEANED_DATA_PATH = os.path.join(PROCESSED_DIR, "cleaned_hostel_water.csv")
ANOMALIES_PATH = os.path.join(PROCESSED_DIR, "flagged_anomalies.csv")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "demand_forecaster.pkl")


class TestHostelWaterSystem(unittest.TestCase):
    
    # -------------------------------------------------------------
    # TEST 1: CSV LOADING AND PREPROCESSING INTEGRITY
    # -------------------------------------------------------------
    def test_01_processed_dataset_exists_and_valid(self):
        """Verify cleaned dataset exists, has records, and contains required schema."""
        self.assertTrue(os.path.exists(CLEANED_DATA_PATH), "Cleaned dataset does not exist!")
        df = pd.read_csv(CLEANED_DATA_PATH)
        
        # Check non-empty
        self.assertGreater(len(df), 0, "Cleaned dataset is empty!")
        
        # Check primary key completeness
        self.assertIn("Date", df.columns)
        self.assertIn("Block", df.columns)
        
        # Verify block designations
        valid_blocks = {"Block A", "Block B", "Block C"}
        self.assertTrue(set(df["Block"].unique()).issubset(valid_blocks), "Unexpected block names found!")
        
        # Verify no duplicate primary keys
        duplicates = df.duplicated(subset=["Date", "Block"]).sum()
        self.assertEqual(duplicates, 0, f"Found {duplicates} duplicate (Date, Block) pairs!")
        
        # Verify date range
        df["Date"] = pd.to_datetime(df["Date"])
        self.assertEqual(df["Date"].min().strftime("%Y-%m-%d"), "2026-08-10")
        self.assertEqual(df["Date"].max().strftime("%Y-%m-%d"), "2026-11-10")

    # -------------------------------------------------------------
    # TEST 2: WATER SUPPLY & WATER BALANCE CALCULATIONS
    # -------------------------------------------------------------
    def test_02_water_supply_and_balance_math(self):
        """Verify mathematical consistency of water supply and water balance."""
        df = pd.read_csv(CLEANED_DATA_PATH)
        
        # 1. Supply Calculation Check: Total = Borewell + Tanker
        expected_supply = df["Borewell_Water_Supplied_Litres"] + df["Tanker_Water_Supplied_Litres"]
        diff_supply = np.abs(df["Total_Water_Supplied_Litres"] - expected_supply).max()
        self.assertAlmostEqual(diff_supply, 0, delta=1.0, msg="Total supply does not equal Borewell + Tanker!")
        
        # 2. Non-negative volumetric measurements
        self.assertTrue((df["Borewell_Water_Supplied_Litres"] >= 0).all(), "Negative borewell supply found!")
        self.assertTrue((df["Tanker_Water_Supplied_Litres"] >= 0).all(), "Negative tanker supply found!")
        self.assertTrue((df["Water_Consumed_Litres"] >= 0).all(), "Negative consumption found!")
        self.assertTrue((df["Student_Occupancy"] > 0).all(), "Zero or negative occupancy found!")
        
        # 3. Water Balance Check:
        # Unaccounted = Inflow - DeltaStorage - Consumed
        delta_storage = df["Ending_Tank_Level"] - df["Starting_Tank_Level"]
        expected_unaccounted = df["Total_Water_Supplied_Litres"] - delta_storage - df["Water_Consumed_Litres"]
        diff_balance = np.abs(df["Unaccounted_Water_Litres"] - expected_unaccounted).max()
        self.assertAlmostEqual(diff_balance, 0, delta=1.0, msg="Water balance equation violated!")

    # -------------------------------------------------------------
    # TEST 3: PREDICTION FUNCTIONALITY (ML MODEL ARTIFACT)
    # -------------------------------------------------------------
    def test_03_prediction_pipeline(self):
        """Verify the trained ML model loads, takes features, and outputs valid predictions."""
        self.assertTrue(os.path.exists(MODEL_PATH), "Trained model artifact not found!")
        
        artifact = joblib.load(MODEL_PATH)
        self.assertIn("model", artifact)
        self.assertIn("feature_names", artifact)
        
        model = artifact["model"]
        feature_names = artifact["feature_names"]
        
        # Test input record (Sample weekday for Block A with 735 students)
        sample_input = {
            "Lag_1_Consumption": 95000.0,
            "Lag_7_Consumption": 96000.0,
            "Rolling_7D_Mean_Consumption": 94500.0,
            "Student_Occupancy": 735,
            "Lag_1_Borewell_Supply": 100000.0,
            "Lag_1_Tanker_Supply": 0.0,
            "Next_Day_Of_Week": 2, # Wednesday
            "Next_Is_Weekend": 0,
            "Next_Month": 11,
            "Block_Block A": 1,
            "Block_Block B": 0,
            "Block_Block C": 0
        }
        
        df_test = pd.DataFrame([sample_input])[feature_names]
        prediction = model.predict(df_test)[0]
        
        # Prediction must be positive and physically realistic (~70k to 130k Litres)
        self.assertGreater(prediction, 50000.0, "Predicted consumption implausibly low!")
        self.assertLess(prediction, 160000.0, "Predicted consumption implausibly high!")
        
        # Per capita should be within reasonable bounds (100 to 180 LPCD)
        predicted_lpcd = prediction / 735
        self.assertGreaterEqual(predicted_lpcd, 100.0)
        self.assertLessEqual(predicted_lpcd, 180.0)

    # -------------------------------------------------------------
    # TEST 4: ANOMALY DETECTION & CAUTION LABELING
    # -------------------------------------------------------------
    def test_04_anomaly_detection_labeling(self):
        """Verify anomaly detection results and mandatory safety labeling."""
        self.assertTrue(os.path.exists(ANOMALIES_PATH), "Flagged anomalies CSV not found!")
        df_anom = pd.read_csv(ANOMALIES_PATH)
        
        self.assertGreater(len(df_anom), 0, "No anomalies flagged in report!")
        
        # Mandatory Data Science Requirement: Status must be "Potential anomaly — inspection recommended"
        self.assertIn("Inspection_Status", df_anom.columns)
        unique_statuses = df_anom["Inspection_Status"].unique()
        self.assertEqual(len(unique_statuses), 1)
        self.assertEqual(unique_statuses[0], "Potential anomaly — inspection recommended")
        
        # Verify diagnostic reasons exist for every flagged record
        self.assertIn("Diagnostic_Reason", df_anom.columns)
        self.assertTrue(df_anom["Diagnostic_Reason"].str.len().gt(5).all(), "Missing diagnostic reasons in anomaly report!")


if __name__ == "__main__":
    unittest.main()

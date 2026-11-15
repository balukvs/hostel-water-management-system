"""
data_cleaning.py
----------------
Data Preprocessing and Validation Pipeline for Smart Hostel Water Management System.

Core Responsibilities:
1. Load raw CSV logs (Supply, Tank/Consumption, Wastage).
2. Check and report missing values (without silently inventing data).
3. Identify and handle duplicate records.
4. Parse and validate dates into proper datetime format.
5. Validate hostel block names against authorized blocks (Block A, B, C).
6. Screen for invalid, negative, or physically impossible measurements.
7. Calculate total water supplied only when constituent sources are available.
8. Engineer calendar and date-based features (Day of week, Month, Is_Weekend).
9. Merge into a clean, verified dataset for analysis and ML.
"""

import os
import pandas as pd
import numpy as np

# Authorized hostel blocks
VALID_BLOCKS = ["Aswatha (Boys Hostel)", "Ashoka (Boys Hostel)", "Jasmine (Girls Hostel)"]


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Safely load a CSV file into a Pandas DataFrame.
    Includes basic error handling for missing or unreadable files.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[Error] Dataset file not found at: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"[Loaded] '{os.path.basename(file_path)}' with {len(df)} rows and {len(df.columns)} columns.")
        return df
    except Exception as e:
        raise IOError(f"[Error] Failed to read {file_path}: {e}")


def check_missing_values(df: pd.DataFrame, dataset_name: str = "Dataset") -> pd.Series:
    """
    Check and report missing (null/NaN) values in each column.
    Does NOT impute or invent fake values.
    """
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) == 0:
        print(f"[Missing Values] {dataset_name}: No missing values detected.")
    else:
        print(f"[Missing Values] {dataset_name}: Missing values found:")
        for col, count in missing_cols.items():
            print(f"  - Column '{col}': {count} missing records ({(count / len(df)) * 100:.2f}%)")
            
    return missing_counts


def check_duplicates(df: pd.DataFrame, subset_cols: list = None, dataset_name: str = "Dataset") -> pd.DataFrame:
    """
    Identify duplicate records based on primary keys (e.g., Date and Block).
    Drops exact duplicate rows to maintain data integrity.
    """
    if subset_cols is None:
        subset_cols = ["Date", "Block"]
        
    # Check if the subset columns exist in the DataFrame
    valid_subset = [col for col in subset_cols if col in df.columns]
    
    if valid_subset:
        duplicate_count = df.duplicated(subset=valid_subset).sum()
        if duplicate_count > 0:
            print(f"[Duplicates] {dataset_name}: Found {duplicate_count} duplicate rows on keys {valid_subset}. Removing duplicates...")
            df = df.drop_duplicates(subset=valid_subset, keep="first").reset_index(drop=True)
        else:
            print(f"[Duplicates] {dataset_name}: No duplicate records on keys {valid_subset}.")
    else:
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            print(f"[Duplicates] {dataset_name}: Found {duplicate_count} duplicate rows across all columns. Removing duplicates...")
            df = df.drop_duplicates(keep="first").reset_index(drop=True)
            
    return df


def convert_and_validate_dates(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Convert date column to datetime64 format.
    Flags invalid dates that cannot be parsed.
    """
    if date_col not in df.columns:
        raise KeyError(f"[Error] Required column '{date_col}' missing from DataFrame.")
    
    # errors='coerce' turns unparseable values into NaT (Not-a-Time) instead of crashing
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    nat_count = df[date_col].isna().sum()
    if nat_count > 0:
        print(f"[Warning] {nat_count} invalid date entries found and flagged as NaT.")
    else:
        print(f"[Dates] '{date_col}' successfully converted to datetime format (Range: {df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}).")
        
    return df


def validate_block_names(df: pd.DataFrame, block_col: str = "Block") -> pd.DataFrame:
    """
    Validate that all records correspond to the authorized hostel blocks:
    'Block A', 'Block B', or 'Block C'.
    """
    if block_col not in df.columns:
        return df
        
    # Clean whitespace and standardize casing
    df[block_col] = df[block_col].astype(str).str.strip()
    
    invalid_mask = ~df[block_col].isin(VALID_BLOCKS)
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        invalid_entries = df.loc[invalid_mask, block_col].unique().tolist()
        print(f"[Warning] Found {invalid_count} records with unauthorized block names: {invalid_entries}")
    else:
        print(f"[Blocks] All records have valid block designations: {VALID_BLOCKS}.")
        
    return df


def validate_numeric_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Screen for physically impossible or negative measurements:
    - Pumping hours must be between 0 and 24.
    - Water volumes (Borewell, Tanker, Consumed) must be >= 0.
    - Student occupancy must be >= 0.
    """
    # Columns that must strictly be non-negative (>= 0)
    non_negative_columns = [
        "Borewell_Pumping_Hours",
        "Borewell_Water_Supplied_Litres",
        "Tanker_Water_Supplied_Litres",
        "Total_Water_Supplied_Litres",
        "Student_Occupancy",
        "Starting_Tank_Level",
        "Ending_Tank_Level",
        "Water_Consumed_Litres",
        "Estimated_Loss_Litres"
    ]
    
    for col in non_negative_columns:
        if col in df.columns:
            # Check for negative values
            neg_mask = df[col] < 0
            neg_count = neg_mask.sum()
            if neg_count > 0:
                print(f"[Validation Alert] Column '{col}' contains {neg_count} negative entries! Flagging invalid records.")
                # Keep original data visible but flag or set invalid entries to NaN rather than guessing
                df.loc[neg_mask, col] = np.nan
                
    # Specific check: Pump hours cannot exceed 24 hours in a single day
    if "Borewell_Pumping_Hours" in df.columns:
        excess_pump = df["Borewell_Pumping_Hours"] > 24.0
        if excess_pump.sum() > 0:
            print(f"[Validation Alert] Found {excess_pump.sum()} records where Borewell_Pumping_Hours exceeds 24 hrs!")
            df.loc[excess_pump, "Borewell_Pumping_Hours"] = np.nan
            
    return df


def calculate_total_supplied_water(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Total_Water_Supplied_Litres when source values are available.
    
    Crucial Rule:
    We do NOT fill missing values with 0. If either source is NaN, the total
    is preserved as NaN to avoid inventing false supply figures.
    """
    if "Borewell_Water_Supplied_Litres" in df.columns and "Tanker_Water_Supplied_Litres" in df.columns:
        # Standard Pandas addition: NaN + value = NaN (safe, non-inventive)
        # However, tanker is frequently 0 (no tanker ordered). If tanker is logged as 0, sum is valid.
        df["Calculated_Total_Supplied"] = df["Borewell_Water_Supplied_Litres"] + df["Tanker_Water_Supplied_Litres"]
        print("[Supply Calculation] 'Calculated_Total_Supplied' computed (Borewell + Tanker).")
        
    return df


def add_date_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Extract calendar and day-of-week signals to support EDA and ML forecasting.
    """
    if date_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = convert_and_validate_dates(df, date_col)
        
    df["Day_Of_Week"] = df[date_col].dt.day_of_week       # 0=Monday, 6=Sunday
    df["Day_Name"] = df[date_col].dt.day_name()            # Monday, Tuesday, ...
    df["Is_Weekend"] = df["Day_Of_Week"].isin([5, 6]).astype(int) # 1 if Saturday/Sunday, else 0
    df["Month"] = df[date_col].dt.month                    # 8, 9, 10, 11
    df["Month_Name"] = df[date_col].dt.month_name()        # August, September, ...
    df["Day_Of_Month"] = df[date_col].dt.day               # 1 to 31
    
    print("[Feature Engineering] Extracted date features: Day_Of_Week, Day_Name, Is_Weekend, Month, Month_Name.")
    return df


def run_preprocessing_pipeline(data_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Executes the complete preprocessing pipeline on all three hostel datasets,
    merges them on Date and Block, and saves the cleaned dataset.
    """
    print("\n" + "="*60)
    print("STEP 1: LOADING RAW DATASETS")
    print("="*60)
    
    supply_file = os.path.join(data_dir, "synthetic_daily_water_supply.csv")
    consumption_file = os.path.join(data_dir, "synthetic_daily_tank_consumption.csv")
    wastage_file = os.path.join(data_dir, "synthetic_water_wastage_observations.csv")
    
    df_supply = load_dataset(supply_file)
    df_consumption = load_dataset(consumption_file)
    df_wastage = load_dataset(wastage_file)
    
    print("\n" + "="*60)
    print("STEP 2: CHECKING MISSING VALUES & DUPLICATES")
    print("="*60)
    
    check_missing_values(df_supply, "Water Supply")
    check_missing_values(df_consumption, "Tank & Consumption")
    check_missing_values(df_wastage, "Wastage Observations")
    
    df_supply = check_duplicates(df_supply, ["Date", "Block"], "Water Supply")
    df_consumption = check_duplicates(df_consumption, ["Date", "Block"], "Tank & Consumption")
    df_wastage = check_duplicates(df_wastage, ["Date", "Block"], "Wastage Observations")
    
    print("\n" + "="*60)
    print("STEP 3: VALIDATING DATES & BLOCK NAMES")
    print("="*60)
    
    for name, df_item in [("Supply", df_supply), ("Consumption", df_consumption), ("Wastage", df_wastage)]:
        df_item = convert_and_validate_dates(df_item)
        df_item = validate_block_names(df_item)
        df_item = validate_numeric_measurements(df_item)
        
    print("\n" + "="*60)
    print("STEP 4: SUPPLY CALCULATION & FEATURE ENGINEERING")
    print("="*60)
    
    df_supply = calculate_total_supplied_water(df_supply)
    
    # Merge datasets cleanly on Date and Block
    df_merged = pd.merge(df_supply, df_consumption, on=["Date", "Block"], how="inner", suffixes=("", "_cons"))
    df_merged = pd.merge(df_merged, df_wastage, on=["Date", "Block"], how="inner", suffixes=("", "_wastage"))
    
    # Clean up redundant duplicate columns from joins if present
    drop_cols = [c for c in df_merged.columns if "_cons" in c or "_wastage" in c]
    df_merged = df_merged.drop(columns=drop_cols, errors="ignore")
    
    # Add calendar features
    df_merged = add_date_features(df_merged, "Date")
    
    # Compute Per-Capita Consumption (LPCD)
    if "Water_Consumed_Litres" in df_merged.columns and "Student_Occupancy" in df_merged.columns:
        df_merged["Per_Capita_Consumption_LPCD"] = (
            df_merged["Water_Consumed_Litres"] / df_merged["Student_Occupancy"]
        ).round(2)
        
    # Compute Storage Change and Water Balance (Unaccounted Water)
    df_merged["Storage_Change_Litres"] = df_merged["Ending_Tank_Level"] - df_merged["Starting_Tank_Level"]
    df_merged["Unaccounted_Water_Litres"] = (
        df_merged["Total_Water_Supplied_Litres"] 
        - df_merged["Storage_Change_Litres"] 
        - df_merged["Water_Consumed_Litres"]
    )
    
    print("\n" + "="*60)
    print("STEP 5: SAVING CLEANED DATASET")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, "cleaned_hostel_water.csv")
    try:
        df_merged.to_csv(output_filepath, index=False)
        print(f"[SUCCESS] Cleaned dataset successfully saved to:\n  -> {output_filepath}")
    except PermissionError:
        fallback_path = os.path.join(output_dir, "cleaned_hostel_water_new.csv")
        df_merged.to_csv(fallback_path, index=False)
        print(f"[Notice] '{os.path.basename(output_filepath)}' is currently open in Excel.")
        print(f"Saved updated dataset to:\n  -> {fallback_path}")
        print("Please close Excel to allow overwriting cleaned_hostel_water.csv.")
    print(f"[Summary] Rows: {len(df_merged)}, Columns: {len(df_merged.columns)}")
    
    return df_merged


def run_unit_test_sample():
    """
    Self-contained verification test using small synthetic sample records
    explicitly designed to test validation, duplicate removal, and date parsing.
    """
    print("\n" + "#"*60)
    print("RUNNING UNIT TEST WITH LABELLED TEST RECORDS")
    print("#"*60)
    
    # Intentionally crafted test data containing edge cases:
    # 1. Row 0: Valid Aswatha record
    # 2. Row 1: Duplicate of Row 0 (should be detected and removed)
    # 3. Row 2: Negative water measurement (-500 L) (should be flagged)
    # 4. Row 3: Invalid block name ('Block X') (should be flagged)
    # 5. Row 4: Missing tanker value (None/NaN) (should NOT assume 0 blindly)
    test_data = {
        "Date": ["2026-08-10", "2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13"],
        "Block": ["Aswatha (Boys Hostel)", "Aswatha (Boys Hostel)", "Ashoka (Boys Hostel)", "Block X", "Jasmine (Girls Hostel)"],
        "Borewell_Pumping_Hours": [7.0, 7.0, 6.5, 7.2, 6.0],
        "Borewell_Water_Supplied_Litres": [98000, 98000, -500, 100800, 72000], # Notice -500
        "Tanker_Water_Supplied_Litres": [0, 0, 12000, 0, np.nan],                # Notice NaN
        "Data_Label": ["TEST_DATA"] * 5
    }
    
    test_df = pd.DataFrame(test_data)
    print("[Test Input DataFrame]")
    print(test_df)
    
    # 1. Test duplicate removal
    cleaned_test_df = check_duplicates(test_df, ["Date", "Block"], "Test Sample")
    assert len(cleaned_test_df) == 4, f"Expected 4 rows after duplicate removal, got {len(cleaned_test_df)}"
    
    # 2. Test block validation
    cleaned_test_df = validate_block_names(cleaned_test_df)
    
    # 3. Test negative measurement detection
    cleaned_test_df = validate_numeric_measurements(cleaned_test_df)
    assert np.isnan(cleaned_test_df.loc[cleaned_test_df["Date"] == "2026-08-11", "Borewell_Water_Supplied_Litres"].values[0]), \
        "Negative value was not converted to NaN!"
        
    # 4. Test safe supply calculation
    cleaned_test_df = calculate_total_supplied_water(cleaned_test_df)
    # Row for Jasmine had tanker as NaN, so calculated total should be NaN (not falsely 72000 + 0)
    jasmine_total = cleaned_test_df.loc[cleaned_test_df["Block"] == "Jasmine (Girls Hostel)", "Calculated_Total_Supplied"].values[0]
    assert np.isnan(jasmine_total), "Calculated total should be NaN when a constituent is missing!"
    
    # 5. Test date features
    cleaned_test_df = add_date_features(cleaned_test_df, "Date")
    assert "Is_Weekend" in cleaned_test_df.columns, "Feature 'Is_Weekend' was not created!"
    
    print("\n[TEST RESULT] All unit test assertions PASSED successfully!")
    print("#"*60 + "\n")


if __name__ == "__main__":
    # 1. Run unit test on dummy test records
    run_unit_test_sample()
    
    # 2. Run pipeline on the actual project datasets
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_path = os.path.join(base_dir, "dataset", "raw")
    processed_path = os.path.join(base_dir, "dataset", "processed")
    
    run_preprocessing_pipeline(data_dir=raw_path, output_dir=processed_path)

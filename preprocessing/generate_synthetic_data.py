"""
generate_synthetic_data.py
--------------------------
Generates realistic synthetic datasets for the Smart Hostel Water Management System.
Timeline: August 10, 2026 to November 10, 2026 (93 calendar days, ~3-month semester term).
All values are mathematically simulated for academic demonstration and reproducible
using Python's standard library with a fixed random seed (seed=42).
No external packages required.
"""

import os
import csv
import random
from datetime import datetime, timedelta

def generate_hostel_water_datasets(seed=42, start_str="2026-08-10", end_str="2026-11-10"):
    random.seed(seed)
    
    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_str, "%Y-%m-%d")
    num_days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    # Block specifications
    blocks_config = {
        'Aswatha (Boys Hostel)': {
            'capacity': 774,
            'tank_capacity': 150000,      # Litres
            'borewell_rate': 14000,       # Litres/hour
            'avg_pump_hours': 7.0,
            'tanker_capacity': 12000      # Litres per tanker trip
        },
        'Ashoka (Boys Hostel)': {
            'capacity': 760,
            'tank_capacity': 150000,
            'borewell_rate': 13500,
            'avg_pump_hours': 7.0,
            'tanker_capacity': 12000
        },
        'Jasmine (Girls Hostel)': {
            'capacity': 600,
            'tank_capacity': 120000,
            'borewell_rate': 12000,
            'avg_pump_hours': 6.5,
            'tanker_capacity': 12000
        }
    }

    supply_records = []
    consumption_records = []
    wastage_records = []
    master_records = []

    for block_name, config in blocks_config.items():
        base_capacity = config['capacity']
        tank_cap = config['tank_capacity']
        pump_rate = config['borewell_rate']
        tanker_unit = config['tanker_capacity']
        
        # Initial tank level at 65% of tank capacity
        current_storage = tank_cap * 0.65
        
        for day_idx, current_date in enumerate(dates):
            date_str = current_date.strftime('%Y-%m-%d')
            day_of_week = current_date.weekday() # 0=Monday, 6=Sunday
            is_weekend = day_of_week in [5, 6]
            
            # 1. Base Occupancy Simulation
            if is_weekend:
                # 15% to 25% of students leave on weekends
                occupancy_ratio = random.uniform(0.75, 0.85)
            else:
                # Normal weekday attendance
                occupancy_ratio = random.uniform(0.94, 0.99)
                
            # 2. Injected Event Contexts (Anomalies & Specific Days adapted for Aug 10 - Nov 10)
            is_leakage_event = (18 <= day_idx <= 20 and block_name == 'Aswatha (Boys Hostel)') # Aug 28-30: Pipe crack
            is_pump_breakdown = (day_idx == 35 and block_name == 'Aswatha (Boys Hostel)')      # Sep 14: Pump tripped
            is_overflow_event = (day_idx == 45 and block_name == 'Ashoka (Boys Hostel)')       # Sep 24: Float valve overflow
            is_fest = (56 <= day_idx <= 58)                                                    # Oct 5-7: Campus Fest
            is_break = (65 <= day_idx <= 70)                                                   # Oct 14-19: Mid-sem break
            is_flush_leak = (80 <= day_idx <= 82 and block_name == 'Jasmine (Girls Hostel)')  # Oct 29-31: Running flushes
            
            if is_break:
                occupancy_ratio = random.uniform(0.18, 0.25) # Mass vacation dip
            elif is_fest:
                occupancy_ratio = 1.05 # Visiting guests & alumni
                
            occupancy = int(round(base_capacity * occupancy_ratio))
            
            # 3. Base Per-Capita Consumption (LPCD)
            # Baseline mean ~135 LPCD with natural variation
            if is_weekend:
                lpcd = random.gauss(142, 6) # Higher laundry/hygiene usage
            elif is_fest:
                lpcd = random.gauss(165, 8)
            else:
                lpcd = random.gauss(133, 5)
                
            if is_flush_leak:
                lpcd += 25 # Continuous valve leak adds unmetered waste
                
            lpcd = max(80.0, lpcd) # Physical lower bound
            consumed_litres = int(round(occupancy * lpcd))
            
            # 4. Wastage & Leakage Simulation
            overflow_observed = "No"
            leakage_observed = "No"
            estimated_loss = 0
            maintenance_remarks = "Normal daily operation"
            tanker_trips = 0
            
            if is_pump_breakdown:
                pump_hours = 0.0
                tanker_trips = 4
                maintenance_remarks = "Borewell motor tripped; emergency tankers requisitioned"
            elif is_overflow_event:
                pump_hours = 9.8
                overflow_observed = "Yes"
                estimated_loss = 14500
                maintenance_remarks = "Overhead tank float valve stuck open; overflow to storm drain"
            elif is_leakage_event:
                pump_hours = round(random.uniform(7.8, 8.5), 1)
                leakage_observed = "Yes"
                estimated_loss = 18000
                tanker_trips = 1
                maintenance_remarks = "Subsurface pipe fracture suspected between riser and wing A2"
            elif is_flush_leak:
                pump_hours = round(random.uniform(7.5, 8.2), 1)
                leakage_observed = "Yes"
                estimated_loss = 7000
                tanker_trips = 0
                maintenance_remarks = "Defective dual-flush siphon valves reported across floor 2"
            else:
                # Normal background operation:
                # If current storage is below 35% capacity, order a tanker
                if current_storage < (tank_cap * 0.35):
                    tanker_trips = random.choice([1, 2])
                    maintenance_remarks = "Routine tanker augmentation due to storage drawdown"
                
                # Normal pumping hours to balance daily draw
                daily_deficit = consumed_litres - (tanker_trips * tanker_unit)
                expected_hours = max(3.5, daily_deficit / pump_rate)
                simulated_hours = random.gauss(expected_hours, 0.4)
                pump_hours = round(max(3.0, min(11.0, simulated_hours)), 1)
            
            # 5. Inflow Calculations
            borewell_supplied = int(round(pump_hours * pump_rate))
            tanker_supplied = tanker_trips * tanker_unit
            total_supplied = borewell_supplied + tanker_supplied
            
            # 6. Physical Water Balance & Storage Tracking
            starting_level = int(round(current_storage))
            net_change = total_supplied - consumed_litres - estimated_loss
            
            # Storage bounds between 5% and 100% capacity
            ending_level = int(round(max(tank_cap * 0.05, min(float(tank_cap), starting_level + net_change))))
            
            # Update storage for continuity to next day
            current_storage = ending_level
            
            # Append records
            supply_records.append({
                'Date': date_str,
                'Block': block_name,
                'Borewell_Pumping_Hours': pump_hours,
                'Borewell_Water_Supplied_Litres': borewell_supplied,
                'Tanker_Water_Supplied_Litres': tanker_supplied,
                'Total_Water_Supplied_Litres': total_supplied,
                'Data_Source': 'SYNTHETIC_SIMULATION'
            })
            
            consumption_records.append({
                'Date': date_str,
                'Block': block_name,
                'Student_Occupancy': occupancy,
                'Starting_Tank_Level': starting_level,
                'Ending_Tank_Level': ending_level,
                'Water_Consumed_Litres': consumed_litres,
                'Data_Source': 'SYNTHETIC_SIMULATION'
            })
            
            wastage_records.append({
                'Date': date_str,
                'Block': block_name,
                'Overflow_Observed': overflow_observed,
                'Leakage_Observed': leakage_observed,
                'Estimated_Loss_Litres': estimated_loss,
                'Maintenance_Remarks': maintenance_remarks,
                'Data_Source': 'SYNTHETIC_SIMULATION'
            })
            
            storage_change = ending_level - starting_level
            unaccounted = total_supplied - storage_change - consumed_litres
            
            master_records.append({
                'Date': date_str,
                'Block': block_name,
                'Student_Occupancy': occupancy,
                'Borewell_Pumping_Hours': pump_hours,
                'Borewell_Water_Supplied_Litres': borewell_supplied,
                'Tanker_Water_Supplied_Litres': tanker_supplied,
                'Total_Water_Supplied_Litres': total_supplied,
                'Starting_Tank_Level': starting_level,
                'Ending_Tank_Level': ending_level,
                'Water_Consumed_Litres': consumed_litres,
                'Per_Capita_Consumption_LPCD': round(consumed_litres / occupancy, 2),
                'Storage_Change_Litres': storage_change,
                'Unaccounted_Water_Litres': unaccounted,
                'Overflow_Observed': overflow_observed,
                'Leakage_Observed': leakage_observed,
                'Estimated_Loss_Litres': estimated_loss,
                'Maintenance_Remarks': maintenance_remarks,
                'Data_Source': 'SYNTHETIC_SIMULATION'
            })

    return supply_records, consumption_records, wastage_records, master_records


def write_csv(filepath, records):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_dir = os.path.join(base_dir, 'dataset', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    supply, consumption, wastage, master = generate_hostel_water_datasets(
        seed=42, 
        start_str="2026-08-10", 
        end_str="2026-11-10"
    )
    
    supply_path = os.path.join(raw_dir, 'synthetic_daily_water_supply.csv')
    consumption_path = os.path.join(raw_dir, 'synthetic_daily_tank_consumption.csv')
    wastage_path = os.path.join(raw_dir, 'synthetic_water_wastage_observations.csv')
    master_path = os.path.join(raw_dir, 'synthetic_hostel_water_master.csv')
    
    write_csv(supply_path, supply)
    write_csv(consumption_path, consumption)
    write_csv(wastage_path, wastage)
    write_csv(master_path, master)
    
    print(f"SUCCESS: Generated dataset for August 10, 2026 to November 10, 2026 ({len(master)//3} days per block, {len(master)} total master records).")
    print(f"Saved CSV files in: {raw_dir}")

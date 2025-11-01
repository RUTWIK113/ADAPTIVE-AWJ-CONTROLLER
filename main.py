# In main.py
import os
import pandas as pd
from control.ga_optimizer import run_genetic_algorithm

# --- 1. DEFINE YOUR REAL-WORLD INPUTS HERE ---

# This is the 4th input your 81-point dataset uses.
# Change this variable tomorrow to match your data file.
#
# Scenario A: "focusing_nozzle_diameter"
# Scenario B: "orifice_diameter"
#
FOURTH_INPUT_COLUMN = "orifice_diameter"  # <-- CHANGE THIS TOMORROW

# Define the full range of all your parameters
# This will be used to train the scaler correctly
PARAM_RANGES = {
    "water_pressure": [100.0, 4000.0],  # Use the widest possible range (Bar/MPa)
    "abrasive_flow": [0.1, 1.0],  # (kg/min)
    "traverse_speed": [100.0, 5000.0],  # (mm/min)
    "orifice_diameter": [0.1, 0.3],  # (mm)
    "focusing_nozzle_diameter": [0.76, 1.6]  # (mm)
}

# --- 2. DEFINE YOUR SIMULATED "LIVE" PARAMETERS ---
# Set your "live" measured diameter and desired depth
DESIRED_DEPTH = 10.0
LIVE_MEASURED_DIAMETER = 0.25  # <-- This is your "measured" orifice_diameter


# --------------------------------------------------


def main_control_loop():
    """
    Main loop for the adaptive control system.
    """
    print("--- Adaptive AWJ Control System Initialized ---")

    # 1. Prepare Inputs for the GA

    # These are the 3 parameters the GA will optimize
    optimizable_params = {
        "water_pressure": PARAM_RANGES["water_pressure"],
        "abrasive_flow": PARAM_RANGES["abrasive_flow"],
        "traverse_speed": PARAM_RANGES["traverse_speed"]
    }

    # This is the 1 static parameter we "measured"
    static_inputs = [LIVE_MEASURED_DIAMETER]

    print(f"Target Depth: {DESIRED_DEPTH} mm")
    print(f"Measured Diameter ({FOURTH_INPUT_COLUMN}): {LIVE_MEASURED_DIAMETER} mm")

    # 2. Control (Optimization) Phase
    print("\n--- Control Phase ---")

    optimal_params = run_genetic_algorithm(
        param_ranges=optimizable_params,
        static_inputs=static_inputs,
        desired_depth=DESIRED_DEPTH
    )

    # 3. Execution Phase
    print("\n--- Execution Phase ---")
    print("TEST SUCCESSFUL. System would now send parameters to the AWJ controller:")
    print(f"  SET Pressure = {optimal_params['pressure']:.2f} MPa")
    print(f"  SET Flow Rate = {optimal_params['flow_rate']:.3f} kg/min")
    print(f"  SET Traverse Rate = {optimal_params['traverse_rate']:.2f} mm/min")


# Run the main loop
if __name__ == "__main__":
    main_control_loop()
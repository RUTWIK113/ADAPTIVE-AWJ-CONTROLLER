# In create_dataset_physics.py

import numpy as np
import pandas as pd
import os

# --- Define Constants (from PDF and physics) ---
# Density of water (kg/m^3)
RHO_W = 1000.0
# Specific energy for steel (J/m^3), from 13.4 J/mm^3
U_STEEL = 13.4e9
# Discharge coefficient (assumed 1.0)
C_D = 1.0
# Jet width (w), assumed 1.0 mm (0.001 m)
W_JET = 0.001

# --- Define Output Path ---
DATA_DIR = 'data'
CSV_FILE = os.path.join(DATA_DIR, 'awj_training_data.csv')
NUM_SAMPLES = 5000  # Increased to 5000 for better training

# --- Input Parameter Ranges (from PDF Page 48) ---
# Water Pressure (bar)
P_MIN, P_MAX = 2500.0, 4000.0
# Abrasive Flow Rate (kg/min)
MA_MIN, MA_MAX = 0.1, 1.0
# Traverse Speed (mm/min), from 0.1-5 m/min
VC_MIN, VC_MAX = 100.0, 5000.0
# Orifice Diameter (mm)
D0_MIN, D0_MAX = 0.1, 0.3

# --- Generate Random Inputs ---
p_w_bar = np.random.uniform(P_MIN, P_MAX, NUM_SAMPLES)
m_ab_kgmin = np.random.uniform(MA_MIN, MA_MAX, NUM_SAMPLES)
v_c_mmmin = np.random.uniform(VC_MIN, VC_MAX, NUM_SAMPLES)
d_0_mm = np.random.uniform(D0_MIN, D0_MAX, NUM_SAMPLES)


def calculate_depth_of_cut(p_w_bar, m_ab_kgmin, v_c_mmmin, d_0_mm):
    """
    Calculates depth of cut (h) using formulas from the PDF.
    All calculations are done in base SI units (meters, kg, sec, Pa).
    """

    # --- 1. Convert all inputs to SI units ---
    p_w_pa = p_w_bar * 1e5  # Pascals (N/m^2)
    m_ab_kgs = m_ab_kgmin / 60.0  # kg/s
    v_c_ms = v_c_mmmin / 60000.0  # m/s
    d_0_m = d_0_mm / 1000.0  # meters

    # --- 2. Calculate Water Mass Flow Rate (m_water) ---
    # From Q_w = c_d * (pi/4) * d_0^2 * sqrt(2*p_w / rho_w)
    m_water_kgs = C_D * (np.pi / 4) * (d_0_m ** 2) * np.sqrt(2 * p_w_pa * RHO_W)

    # --- 3. Calculate Loading Factor (R) ---
    # R = m_ab / m_water
    R = m_ab_kgs / m_water_kgs

    # --- 4. Calculate Depth of Penetration (h) ---
    # h = (1/u) * (pi/4) * c_d * d_0^2 * R / (1+R)^2 * (p_w^1.5 / (w*v_c)) * sqrt(2/rho_w)

    term1 = 1.0 / U_STEEL
    term2 = (np.pi / 4) * C_D * (d_0_m ** 2)
    term3 = R / ((1 + R) ** 2)
    term4 = (p_w_pa ** 1.5) / (W_JET * v_c_ms)
    term5 = np.sqrt(2.0 / RHO_W)

    h_m = term1 * term2 * term3 * term4 * term5

    # --- 5. Convert back to mm and add noise ---
    h_mm = h_m * 1000.0

    # Add some random "experimental" noise (1%)
    noise = np.random.normal(0, h_mm * 0.01)

    return max(0, h_mm + noise)


# --- Create and Save the DataFrame ---
print("Generating synthetic dataset using physics-based formulas...")

h = [calculate_depth_of_cut(p_w_bar[i], m_ab_kgmin[i], v_c_mmmin[i], d_0_mm[i])
     for i in range(NUM_SAMPLES)]

# --- UPDATED SECTION: Column Mapping for Training Script ---
# We synthesize df (Focusing Tube) as roughly 3x the Orifice Diameter
d_f_mm = d_0_mm * 3.0

df = pd.DataFrame({
    'P (MPa)': p_w_bar / 10.0,      # Converted: 1 Bar = 0.1 MPa
    'mf (kg/min)': m_ab_kgmin,      # Matches: mf (kg/min)
    'v (mm/min)': v_c_mmmin,        # Matches: v (mm/min)
    'df (mm)': d_f_mm,              # Synthesized: df (mm)
    'do (mm)': d_0_mm,              # Matches: do (mm)
    'h (mm)': h                     # Matches: h (mm)
})

# Ensure the 'data' directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Save to CSV
df.to_csv(CSV_FILE, index=False)

print(f"Successfully generated {NUM_SAMPLES} data points.")
print(f"Dataset saved to: {CSV_FILE}")
print("\nFirst 5 rows of data (Formatted for Training):")
print(df.head())
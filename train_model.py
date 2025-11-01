# In train_model.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle

# Import the model builder and file path
from control.ann_model import build_ann_model, MODEL_FILE

# --- 1. DEFINE YOUR INPUTS ---
# This MUST match the column name in your 81-point CSV
#
# Scenario A: "focusing_nozzle_diameter"
# Scenario B: "orifice_diameter"
#
FOURTH_INPUT_COLUMN = "orifice_diameter" # <-- CHANGE THIS TOMORROW

# This file MUST exist
DATA_FILE = os.path.join('data', 'awj_training_data.csv') # <-- Use your 81-point file name

# ------------------------------

# --- 2. Load Data ---
if not os.path.exists(DATA_FILE):
    print(f"Error: Data file not found at {DATA_FILE}")
    print("Please add your 81-point data CSV to the 'data' folder.")
    exit()

print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

# --- 3. Prepare Data for ANN ---
features = [
    'water_pressure',
    'abrasive_flow',
    'traverse_speed',
    FOURTH_INPUT_COLUMN # This now matches your choice
]
target = 'depth_of_cut'

# Verify all columns exist
for col in features + [target]:
    if col not in df.columns:
        print(f"FATAL ERROR: Column '{col}' not found in {DATA_FILE}")
        print(f"Please check your CSV file column names.")
        exit()

X = df[features].values
y = df[target].values

# --- 4. Split and Scale Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# We use the scaler to learn the "shape" of our training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Build and Train the Model ---
print("Building new ANN model...")
model = build_ann_model()

print("Starting model training...")
history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=500, # Train for more epochs on the smaller 81-point dataset
    batch_size=8,
    verbose=1
)

# --- 6. Evaluate and Save ---
print("\nTraining complete.")
test_loss = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss (MSE): {test_loss}")

print(f"Saving trained model to {MODEL_FILE}...")
model.save(MODEL_FILE)

SCALER_FILE = os.path.join('data', 'scaler.pkl')
print(f"Saving scaler to {SCALER_FILE}...")
with open(SCALER_FILE, 'wb') as f:
    pickle.dump(scaler, f)

print("\n--- Model Training Successfully Completed ---")
# In create_dummy_model.py
from control.ann_model import build_ann_model, MODEL_FILE
import os

print(f"Attempting to create a new, untrained model at {MODEL_FILE}...")

# 1. Get the directory path
data_dir = os.path.dirname(MODEL_FILE)
print(f"Ensuring directory exists: {data_dir}")

# 2. Create the directory if it doesn't exist
# This was the line that failed
os.makedirs(data_dir, exist_ok=True)

# 3. Build the model
model = build_ann_model()

# 4. Save the model to the correct path
model.save(MODEL_FILE)

print("\nDummy model created successfully.")
import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("data/augmented_training_data.csv")

print("Any NaN:", df.isna().any().any())
print("Any Inf:", np.isinf(df).any().any())
print(df.describe(), "\n")

scaler = pickle.load(open("data/scaler.pkl", "rb"))
print("Scaler mean shape:", scaler.mean_.shape)


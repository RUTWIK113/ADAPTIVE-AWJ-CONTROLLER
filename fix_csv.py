import pandas as pd

df = pd.read_csv("data/augmented_training_data.csv")

print("Before cleaning:", df.isna().sum())

df_clean = df.dropna().reset_index(drop=True)

print("After cleaning:", df_clean.isna().sum())
print("Rows removed:", len(df) - len(df_clean))

df_clean.to_csv("data/augmented_training_data_clean.csv", index=False)

print("Saved cleaned dataset.")

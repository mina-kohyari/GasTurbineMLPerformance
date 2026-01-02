import pandas as pd
from pathlib import Path


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "turbine_data.csv"
CLEAN_PATH = BASE_DIR / "data" / "turbine_data_clean.csv"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)  # tab-separated file

print("Dataset loaded successfully")
print("Original shape:", df.shape)

print("\nColumn names:")
print(df.columns)

# -----------------------------
# Convert numeric columns
# -----------------------------
for col in df.columns:
    if col.lower() not in ["hour_time", "time", "date"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# Handle missing values
# -----------------------------
print("\nMissing values before cleaning:")
print(df.isnull().sum())

df.dropna(inplace=True)

print("\nCleaned shape:", df.shape)

# -----------------------------
# Save cleaned dataset
# -----------------------------
df.to_csv(CLEAN_PATH, index=False)

print(f"\nCleaned dataset saved to: {CLEAN_PATH}")
print("Data preprocessing completed successfully.")
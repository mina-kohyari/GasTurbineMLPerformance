import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load dataset

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "turbine_data_clean.csv"
df = pd.read_csv(DATA_PATH)  # tab-separated file

# 🔥 CRITICAL: clean column names
df.columns = df.columns.str.strip()

print("Loaded data shape:", df.shape)
print("Columns:")
print(df.columns.tolist())

# -----------------------------
# Target & features
# -----------------------------
TARGET = "Real power"
DROP_COLS = ["Hour_Time"]

X = df.drop(columns=[TARGET] + DROP_COLS)
y = df[TARGET]

# Ensure numeric only
X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

# Drop NaNs
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)



print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")
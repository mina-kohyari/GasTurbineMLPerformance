from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Import modules
from src.data_preprocessing import process_data
from src.model_training import train_model
from src.control_optimization import optimize

# Path to cleaned dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "turbine_data_clean.csv"

# Data processing
X, y = process_data(DATA_PATH)

# Train model
model, metrics = train_model(X, y)

# Optimization / Predictions
result = optimize(model, X)
y_pred = result["predictions"]

# -----------------------------
# Plot True vs Predicted
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("Actual Power (MW)")
plt.ylabel("Predicted Power (MW)")
plt.title("Actual vs Predicted Turbine Power")
plt.grid(True)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # diagonal line
plt.tight_layout()
plt.savefig(BASE_DIR / "plots" / "true_vs_predicted.png")
plt.show()

# -----------------------------
# Plot Feature Importance
# -----------------------------
importances = model.feature_importances_
feature_names = X.columns

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance characters", ascending=False)

plt.figure(figsize=(10,6))
plt.barh(fi_df["Feature"].head(15)[::-1], fi_df["Importance"].head(15)[::-1])
plt.xlabel("Importance")
plt.title("Top 15 Feature Importance characters")
plt.tight_layout()
plt.savefig(BASE_DIR / "plots" / "feature_importance.png")
plt.show()

# -----------------------------
# Summary
# -----------------------------
print("\nPipeline completed successfully")
print("Metrics:", metrics)
print("Sample predictions:", y_pred[:5])
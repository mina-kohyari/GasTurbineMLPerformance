import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/turbine_data.csv')

X = df[['FuelFlow', 'RPM', 'Pressure', 'Temperature']]
y = df['Efficiency']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Plot predicted vs actual
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Efficiency")
plt.ylabel("Predicted Efficiency")
plt.title("Gas Turbine Efficiency Prediction")
plt.plot([0,1],[0,1], color='red')
plt.show()

# Save model
import joblib
joblib.dump(model, 'src/efficiency_model.pkl')

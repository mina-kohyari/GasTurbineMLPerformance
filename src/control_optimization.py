import joblib
import numpy as np
from scipy.optimize import minimize

# Load model
model = joblib.load('src/efficiency_model.pkl')

# Control: Optimize FuelFlow and RPM for max efficiency
def objective(u):
    # u[0] = FuelFlow, u[1] = RPM
    x = np.array([[u[0], u[1], 20, 300]])  # Pressure=20 bar, Temperature=300°C
    pred = model.predict(x)[0]
    return -pred  # negative because we want to maximize


# Initial guess
u0 = [100, 4000]
bounds = [(50, 200), (3000, 6000)]


res = minimize(objective, u0, bounds=bounds)
optimal_fuel, optimal_rpm = res.x
max_efficiency = -res.fun

print(f"Optimal Fuel Flow: {optimal_fuel:.2f} kg/s")
print(f"Optimal RPM: {optimal_rpm:.2f} rev/min")
print(f"Predicted Max Efficiency: {max_efficiency:.4f}")
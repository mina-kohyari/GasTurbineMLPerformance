import numpy as np
import pandas as pd

# Number of data points
n_samples = 1000
np.random.seed(42)

# Inputs: Fuel flow (kg/s), RPM (rev/min), Pressure (bar), Temperature (°C)
fuel_flow = np.random.uniform(50, 200, n_samples)
rpm = np.random.uniform(3000, 6000, n_samples)
pressure = np.random.uniform(10, 25, n_samples)
temperature = np.random.uniform(200, 400, n_samples)

# Efficiency formula (simplified)
efficiency = 0.3 + 0.4 * (rpm/6000) + 0.2 * (pressure/25) - 0.1 * (temperature/400) + np.random.normal(0, 0.02, n_samples)

# Power output (MW)
power = fuel_flow * efficiency * 0.5  # simplified relation

# Create DataFrame
df = pd.DataFrame({
    'FuelFlow': fuel_flow,
    'RPM': rpm,
    'Pressure': pressure,
    'Temperature': temperature,
    'Efficiency': efficiency,
    'Power': power
})

# Save dataset
df.to_csv('data/turbine_data.csv', index=False)
print("Data saved to data/turbine_data.csv")
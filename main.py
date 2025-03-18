import os
import fastf1 as ff1
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from flask import Flask

# Ensure cache directory exists
if not os.path.exists('cache'):
    os.makedirs('cache')

# Enable FastF1 cache
ff1.Cache.enable_cache('cache')

# Load session data
session = ff1.get_session(2023, 'Monza', 'R')
session.load()

# Extract telemetry data
laps = session.laps.pick_quicklaps()

# Features and target
features = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
laps = laps.dropna(subset=features + ['DriverNumber'])

# Convert times to seconds
for feature in features:
    laps[feature] = laps[feature].dt.total_seconds()

# Prepare data
X = laps[features]
y = laps['DriverNumber'].astype(int)

# Train model
gbr = GradientBoostingRegressor()
gbr.fit(X, y)

# Evaluate model
y_pred = gbr.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse}")

# Create Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "🏁 F1 Winner Prediction API - Visit /predict to get predicted winner!"

@app.route("/predict")
def predict():
    # Predict on the average lap (as an example)
    avg_features = X.mean().values.reshape(1, -1)
    prediction = gbr.predict(avg_features)[0]
    
    # Round prediction to nearest driver number
    predicted_driver_number = int(round(prediction))

    # Map driver number to driver name (from session data)
    drivers = {int(laps.iloc[i]['DriverNumber']): laps.iloc[i]['Driver'] for i in range(len(laps))}
    driver_name = drivers.get(predicted_driver_number, "Unknown Driver")

    return f"Predicted Winner: {driver_name} (Driver No: {predicted_driver_number})"

if __name__ == "__main__":
    app.run(debug=True)

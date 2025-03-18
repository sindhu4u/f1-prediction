import fastf1 as ff1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify
import numpy as np
import threading
import time

# Enable cache for faster data retrieval
ff1.Cache.enable_cache('cache')

# Function to collect and prepare data
def collect_data():
    # Example: Load a specific race session
    race = ff1.get_session(2022, 'Monza', 'R')
    race.load()
    results = race.results

    # Extract features and target variable
    # This is a placeholder; you'll need to define how to extract meaningful features
    features = np.random.rand(len(results), 5)  # Replace with actual feature extraction
    targets = np.random.rand(len(results))      # Replace with actual target extraction

    return features, targets

# Function to train the model
def train_model():
    features, targets = collect_data()
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    return model

# Train the model (this could be run in a separate thread if desired)
model = train_model()

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

# Function to run the Flask app
def run_flask():
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    # Optionally, run the Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # If you have other tasks to run in parallel, you can do so here
    # For example, periodic data collection or model retraining
    while True:
        time.sleep(3600)  # Placeholder for periodic tasks

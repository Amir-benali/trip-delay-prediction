import pandas as pd
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from flask import Flask, request, jsonify
import joblib
import requests
from math import radians, sin, cos, sqrt, asin
import os 

# OpenWeatherMap API key (replace with your own key)
OPENWEATHER_API_KEY = "7c36f813f7d404cfc40bd3470734da52"

# Haversine function to calculate distance between two coordinates
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in km
    return c * r

# Step 1: Generate a Simulated Dataset
def generate_large_dataset(num_rows=50000):
    def generate_coordinates(lat_range, lon_range):
        lat = round(random.uniform(lat_range[0], lat_range[1]), 6)
        lon = round(random.uniform(lon_range[0], lon_range[1]), 6)
        return lat, lon

    def simulate_weather():
        return random.choice(["Clear", "Rain", "Snow", "Clouds", "Unknown"])

    def simulate_traffic():
        return random.choice([0, 1, 2])  # 0 = Low, 1 = Medium, 2 = High

    def calculate_delay(distance, start_weather, dest_weather, traffic):
        base_delay = distance * 0.1  # Base delay in minutes

        # Weather impact
        weather_delay = {
            "Clear": 0,
            "Rain": random.uniform(5, 15),
            "Snow": random.uniform(10, 20),
            "Clouds": random.uniform(2, 5),
            "Unknown": random.uniform(10, 25),
        }
        base_delay += weather_delay[start_weather] + weather_delay[dest_weather]

        # Traffic impact
        traffic_delay = {0: 0, 1: random.uniform(5, 10), 2: random.uniform(10, 20)}
        base_delay += traffic_delay[traffic]

        return round(base_delay, 2)

    data = []
    for _ in range(num_rows):
        start_lat, start_lon = generate_coordinates((50.0, 75.0), (10.0, 180.0))
        dest_lat, dest_lon = generate_coordinates((50.0, 75.0), (10.0, 180.0))
        start_weather = simulate_weather()
        dest_weather = simulate_weather()
        traffic = simulate_traffic()
        distance = haversine(start_lat, start_lon, dest_lat, dest_lon)
        delay = calculate_delay(distance, start_weather, dest_weather, traffic)

        data.append([start_lat, start_lon, dest_lat, dest_lon, traffic, distance, start_weather, dest_weather, delay])

    df = pd.DataFrame(data, columns=["start_lat", "start_lon", "dest_lat", "dest_lon", "traffic", "distance", "start_weather", "dest_weather", "delay_minutes"])

    df.to_csv("trip_data.csv", index=False)
    print(f"Dataset generated with {num_rows} rows.")

# Step 2: Train the Model
def train_model():
    df = pd.read_csv("trip_data.csv")

    # One-Hot Encoding for categorical variables
    df = pd.get_dummies(df, columns=["start_weather", "dest_weather"])

    # Features and labels
    X = df.drop(columns=["delay_minutes"])
    y = df["delay_minutes"]

    # Save feature names for later use
    feature_names = list(X.columns)
    joblib.dump(feature_names, "feature_names.pkl")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")

    # Save the model
    joblib.dump(model, "trip_delay_model.pkl")
    print("Model trained and saved!")

# Step 3: Flask API
app = Flask(__name__)

# Load the model and feature names if they exist, otherwise train the model
if os.path.exists("trip_delay_model.pkl") and os.path.exists("feature_names.pkl"):
    model = joblib.load("trip_delay_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
else:
    print("Model not found. Training the model...")
    generate_large_dataset(50000)  # Generate dataset
    train_model()  # Train model
    model = joblib.load("trip_delay_model.pkl")
    feature_names = joblib.load("feature_names.pkl")

def get_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['weather'][0]['main']
    return "Unknown"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        start_lat, start_lon = data['start_lat'], data['start_lon']
        dest_lat, dest_lon = data['dest_lat'], data['dest_lon']

        # Fetch real-time weather
        start_weather = get_weather(start_lat, start_lon)
        dest_weather = get_weather(dest_lat, dest_lon)

        # Haversine distance
        distance = haversine(start_lat, start_lon, dest_lat, dest_lon)

        # Simulate traffic
        traffic = random.choice([0, 1, 2])

        # Create DataFrame for prediction
        input_df = pd.DataFrame([[start_lat, start_lon, dest_lat, dest_lon, traffic, distance, start_weather, dest_weather]], 
                                columns=["start_lat", "start_lon", "dest_lat", "dest_lon", "traffic", "distance", "start_weather", "dest_weather"])

        # Apply One-Hot Encoding to match training features
        input_df = pd.get_dummies(input_df, columns=["start_weather", "dest_weather"])

        # Ensure all required columns exist
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0  # Add missing columns

        # Reorder columns to match training set
        input_df = input_df[feature_names]

        # Predict
        prediction = model.predict(input_df)

        return jsonify({
            "predicted_delay": round(float(prediction[0]), 2),
            "start_weather_condition": start_weather,
            "dest_weather_condition": dest_weather
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
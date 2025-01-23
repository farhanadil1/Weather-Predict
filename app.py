import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st
import numpy as np
import joblib
import os

# Train the model if not already trained
if not os.path.exists('weather_model.pkl'):
    # Load dataset
    data = pd.read_csv('./data/weatherHistory.csv')
    X = data[['Apparent Temperature (C)', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Visibility (km)']]
    y = data['Precip Type']
    y = y.map({'rain': 1, 'snow': 0})  # Convert target to numeric

    # Drop rows with missing values
    data = data[~y.isnull()]
    y = y.dropna()
    X = X.loc[y.index]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'weather_model.pkl')

# Streamlit App
st.title("Weather Prediction App")

# Input fields for prediction
apparent_temp = st.slider("Apparent Temperature (°C)", min_value=-10, max_value=50, value=20)
temperature = st.slider("Temperature (°C)", min_value=-10, max_value=50, value=25)
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=50)
wind_speed = st.slider("Wind Speed (km/h)", min_value=0, max_value=50, value=10)
visibility = st.slider("Visibility (km)", min_value=0, max_value=20, value=10)

# Ensure the model file exists
if not os.path.exists('weather_model.pkl'):
    st.error("Model file not found! Train the model and save it as 'weather_model.pkl'.")
else:
    # Load the trained model
    model = joblib.load('weather_model.pkl')

    # Predict button
    if st.button("Predict Precipitation Type"):
        input_data = np.array([[apparent_temp, temperature, humidity, wind_speed, visibility]])
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.write("Predicted Precipitation: Rain")
        else:
            st.write("Predicted Precipitation: Snow")

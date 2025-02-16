import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# API Details
API_KEY = 'a1b6f556b9085bf3baf738ea6566975c'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# Load trained models
with open("rainfall_prediction_model.pkl", "rb") as f:
    rain_model = pickle.load(f)

with open("temperature_prediction_model.pkl", "rb") as f:
    temperature_model = pickle.load(f)

with open("humidity_prediction_model.pkl", "rb") as f:
    humidity_model = pickle.load(f)

# Load historical data to match preprocessing
df = pd.read_csv("weather (1) (1).csv")  # Updated file name
le = LabelEncoder()
df["WindGustDir"] = le.fit_transform(df["WindGustDir"])

# Function to get real-time weather data
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        return {
            'city': data['name'],
            'MinTemp': round(data['main']['temp_min']),
            'MaxTemp': round(data['main']['temp_max']),
            'WindGustDir': data['wind']['deg'],
            'WindGustSpeed': data['wind']['speed'],
            'Humidity': round(data['main']['humidity']),
            'Pressure': data['main']['pressure'],
            'Temp': round(data['main']['temp']),
        }
    else:
        return None

# Function to predict future values
def predict_future(model, current_value, years=5, base_variation=5, extra_variation=3, is_humidity=False):
    predictions = []
    last_value = current_value

    for _ in range(years):
        next_value = model.predict([[last_value]])[0]
        min_value = next_value - np.random.uniform(base_variation, base_variation + extra_variation)
        max_value = next_value + np.random.uniform(base_variation, base_variation + extra_variation)

        if is_humidity:
            min_value = max(0, min_value)
            max_value = min(100, max_value + np.random.uniform(10, 30))

        predictions.append((min_value, max_value))
        last_value = next_value

    return predictions

# Streamlit UI
st.title("Weather Prediction App 🌤️")
st.write("Enter City Name and Year for Weather Prediction")

# User Inputs
city = st.text_input("Enter any city name:")
year = st.number_input("Enter the year for prediction:", min_value=2025, step=1)


if st.button("Predict"):
    if not city or not year:
        st.error("Please enter a valid city and year!")
    else:
        # Fetch real-time weather data
        weather_data = get_current_weather(city)

        if not weather_data:
            st.error("City not found. Please enter a valid city name.")
        else:
            # Encode WindGustDir
            wind_dir = weather_data['WindGustDir']
            encoded_wind_dir = le.transform([wind_dir])[0] if wind_dir in le.classes_ else 0

            # Prepare input features
            input_features = np.array([
                weather_data['MinTemp'],
                weather_data['MaxTemp'],
                encoded_wind_dir,
                weather_data['WindGustSpeed'],
                weather_data['Humidity'],
                weather_data['Pressure'],
                weather_data['Temp']
            ]).reshape(1, -1)

            try:
                # Rain Prediction
                predicted_rainfall = rain_model.predict(input_features)[0]

                # Temperature & Humidity Predictions
                years_ahead = year - datetime.now().year
                future_temp = predict_future(temperature_model, weather_data['Temp'], years=years_ahead)
                future_humidity = predict_future(humidity_model, weather_data['Humidity'], years=years_ahead, is_humidity=True)

                # Extract predictions
                years_list = list(range(datetime.now().year + 1, year + 1))
                min_temps, max_temps = zip(*future_temp)
                min_hums, max_hums = zip(*future_humidity)

                # Display Results
                st.write(f"🌧 **Rain Prediction:** {'Yes' if predicted_rainfall else 'No'}")
                st.write(f"### Predicted Weather for {city} in {year}:")
                for yr, min_temp, max_temp in zip(years_list, min_temps, max_temps):
                    st.write(f"📅 {yr}: **Min:** {round(min_temp, 1)}°C, **Max:** {round(max_temp, 1)}°C")

                st.write(f"### Future Humidity Prediction:")
                for yr, min_hum, max_hum in zip(years_list, min_hums, max_hums):
                    st.write(f"📅 {yr}: **Min:** {round(min_hum, 1)}%, **Max:** {round(max_hum, 1)}%")

                # Plot Predictions
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(years_list, min_temps, 'b-o', label="Min Temperature (°C)")
                ax.plot(years_list, max_temps, 'r-o', label="Max Temperature (°C)")
                ax.set_xlabel("Year")
                ax.set_ylabel("Temperature (°C)")
                ax.set_title(f"Predicted Temperature Trends for {city}")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(years_list, min_hums, 'g-o', label="Min Humidity (%)")
                ax.plot(years_list, max_hums, 'y-o', label="Max Humidity (%)")
                ax.set_xlabel("Year")
                ax.set_ylabel("Humidity (%)")
                ax.set_title(f"Predicted Humidity Trends for {city}")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {e}")

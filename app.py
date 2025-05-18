import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib

# Load model and scaler trained on 10 features
model = tf.keras.models.load_model('best_model.keras')
scaler = joblib.load('scaler.save')  # scaler trained on 10 features

st.title("🚀 Green Logistics Optimization System")
st.write("Predict the optimal route based on carbon emission estimates.")

# Input fields for all 10 features
route_distance = st.number_input("Route Distance (km)", min_value=0.0, value=300.0)
fuel_usage = st.number_input("Fuel Usage (liters)", min_value=0.0, value=20.0)
traffic_index = st.number_input("Traffic Index", min_value=0.0, value=1.2)
weather_severity = st.number_input("Weather Severity", min_value=0.0, value=0.5)
cargo_weight = st.number_input("Cargo Weight (kg)", min_value=0.0, value=15000.0)
feature6 = st.number_input("Feature 6", min_value=0.0, value=0.0)
feature7 = st.number_input("Feature 7", min_value=0.0, value=0.0)
feature8 = st.number_input("Feature 8", min_value=0.0, value=0.0)
feature9 = st.number_input("Feature 9", min_value=0.0, value=0.0)
feature10 = st.number_input("Feature 10", min_value=0.0, value=0.0)

if st.button("Suggest Optimal Route"):
    input_data = pd.DataFrame([[route_distance, fuel_usage, traffic_index, weather_severity, cargo_weight,
                                feature6, feature7, feature8, feature9, feature10]],
                              columns=['route_distance', 'fuel_usage', 'traffic_index', 'weather_severity', 'cargo_weight',
                                       'feature6', 'feature7', 'feature8', 'feature9', 'feature10'])

    # Scale using the 10-feature scaler
    scaled_input = scaler.transform(input_data.values)

    predicted_emission = model.predict(scaled_input)

    st.success(f"Estimated Carbon Emission: {predicted_emission[0][0]:.2f} kg")
    st.bar_chart(pd.DataFrame(predicted_emission, columns=["Estimated Emission (kg)"]))

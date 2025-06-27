import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the trained models and the scaler
# Make sure the paths match where your saved files are located
try:
    # Load the best performing model (based on your evaluation metrics)
    # Assuming Random Forest was the best based on typical performance on such tasks
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('feature_scaler.pkl') # Load the scaler fitted on training features
    st.success("Model and scaler loaded successfully!")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please make sure 'random_forest_model.pkl' and 'feature_scaler.pkl' are in the same directory as the Streamlit app.")
    st.stop() # Stop the app if files are not found

# Define the feature columns based on your training script
feature_cols = ['Dirección del viento (Grados)', 'Presión atmosférica (mm Hg)', 'Radiación Solar Global (W/m2)', 'Temperatura 10cm (°C)'] # Adjust if your features were different

# Set the title of the Streamlit app
st.title("Predicción de PM10 en La Candelaria")

# Add a brief description
st.write("Esta aplicación predice los niveles de PM10 (ug/m3) basándose en las condiciones meteorológicas.")

# Create input fields for the features
st.header("Ingrese los valores de las variables meteorológicas:")

input_data = {}
for col in feature_cols:
    input_data[col] = st.number_input(f"Ingrese el valor para '{col}':", value=0.0, step=0.1)

# Create a button to trigger the prediction
if st.button("Predecir PM10"):
    # Create a DataFrame from the user input
    input_df = pd.DataFrame([input_data])

    # Ensure the input DataFrame has the same column order as the training features
    input_df = input_df[feature_cols]

    # Scale the input data using the loaded scaler
    # It's crucial to use the scaler fitted on the training data
    try:
        scaled_input = scaler.transform(input_df)
        scaled_input_df = pd.DataFrame(scaled_input, columns=feature_cols)

        # Make the prediction
        prediction = model.predict(scaled_input_df)

        # Display the prediction
        st.header("Resultado de la Predicción:")
        st.write(f"El nivel de PM10 predicho es: **{prediction[0]:.4f} ug/m3**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.warning("Please ensure the input values are valid.")

# Optional: Add a section to explain the model or the process
st.sidebar.header("Acerca de la Aplicación")
st.sidebar.write("""
Esta aplicación utiliza un modelo de Machine Learning (Random Forest Regressor)
entrenado con datos históricos para predecir los niveles de PM10.
""")
st.sidebar.write("Las variables de entrada son:")
for col in feature_cols:
    st.sidebar.write(f"- {col}")

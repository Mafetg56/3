import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the pre-trained models and scaler
# Ensure these files ('random_forest_model.pkl', 'feature_scaler.pkl') exist
# and are in the same directory or provide the correct path.
try:
    rf_model = joblib.load('random_forest_model.pkl')
    feature_scaler = joblib.load('feature_scaler.pkl')
    st.success("Modelos y scaler cargados exitosamente.")
except FileNotFoundError:
    st.error("Error: Asegúrate de que los archivos 'random_forest_model.pkl' y 'feature_scaler.pkl' existan.")
    st.stop() # Stop the app if files are not found

st.title("Predicción de PM10 en La Candelaria")

st.write("""
Esta aplicación predice el nivel de PM10 basado en diferentes variables ambientales
utilizando un modelo de Random Forest previamente entrenado.
""")

st.header("Introduce los valores de las variables ambientales:")

# Define input fields for the features used in the model
# These should match the features the model was trained on
# Based on the provided code, the features were:
# 'Dirección del viento (Grados)', 'Presión atmosférica (mm Hg)',
# 'Radiación Solar Global (W/m2)', 'Temperatura 10cm (°C)',
# 'Velocidad del Viento (m/s)' (this was not explicitly dropped in the final features list)
# Let's assume 'Velocidad del Viento (m/s)' is also a feature based on the prompt.

# Inspect the feature_cols from your training script if possible
# feature_cols = ['Dirección del viento (Grados)', 'Presión atmosférica (mm Hg)', 'Radiación Solar Global (W/m2)', 'Temperatura 10cm (°C)', 'Velocidad del Viento (m/s)'] # Example based on likely features

# Dynamically get feature names from the loaded scaler/model if possible
# Or hardcode based on your training script
# Assuming the features are:
feature_names = ['Dirección del viento (Grados)', 'Presión atmosférica (mm Hg)',
                 'Radiación Solar Global (W/m2)', 'Temperatura 10cm (°C)',
                 'Velocidad del Viento (m/s)']

input_data = {}

# Create input widgets for each feature
for feature in feature_names:
    # Provide sensible default values or ranges
    if 'Dirección del viento' in feature:
        input_data[feature] = st.slider(feature, min_value=0.0, max_value=360.0, value=180.0, step=0.1)
    elif 'Presión atmosférica' in feature:
        input_data[feature] = st.number_input(feature, min_value=0.0, value=700.0, step=1.0)
    elif 'Radiación Solar Global' in feature:
        input_data[feature] = st.number_input(feature, min_value=0.0, value=200.0, step=1.0)
    elif 'Temperatura 10cm' in feature:
        input_data[feature] = st.number_input(feature, value=20.0, step=0.1)
    elif 'Velocidad del Viento' in feature:
        input_data[feature] = st.number_input(feature, min_value=0.0, value=5.0, step=0.1)
    else:
         input_data[feature] = st.number_input(feature, value=0.0, step=0.1)


# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Scale the relevant features using the loaded scaler
# Ensure the columns to be scaled match the 'columns_to_scale' from your training script
# which were ['Dirección del viento (Grados)', 'Presión atmosférica (mm Hg)', 'Radiación Solar Global (W/m2)', 'Temperatura 10cm (°C)']
# Note: 'Velocidad del Viento (m/s)' was not in the original scaling list in your code.
# You need to confirm if 'Velocidad del Viento (m/s)' should be scaled and update the scaler fitting accordingly.
# For now, let's assume the scaler was fitted on the exact 'feature_cols' used for training X_train.

# Assuming feature_scaler was fitted on the full feature_cols used for training:
input_scaled = feature_scaler.transform(input_df)

# Convert the scaled numpy array back to a DataFrame with correct column names
input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)


if st.button("Predecir PM10"):
    # Make prediction using the Random Forest model
    prediction = rf_model.predict(input_scaled_df)

    st.header("Resultado de la Predicción:")
    st.write(f"El nivel de PM10 predicho es: **{prediction[0]:.2f} ug/m3**")

st.sidebar.header("Información")
st.sidebar.write("""
Esta aplicación utiliza un modelo de Random Forest entrenado con datos
históricos de calidad del aire en La Candelaria para predecir los niveles de PM10.
""")
st.sidebar.write("Desarrollado por [Tu Nombre o Organización]") # Optional: Add your name/organization

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Load the models and scaler
# Ensure the paths to your saved files are correct
try:
    linear_regression_model = joblib.load('linear_regression_model.pkl')
    knn_model = joblib.load('knn_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    lasso_model = joblib.load('lasso_model.pkl')
    decision_tree_model = joblib.load('decision_tree_model.pkl')
    voting_regressor_model = joblib.load('voting_regressor_model.pkl')
    random_forest_model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('feature_scaler.pkl') # Load the scaler fitted on features
except FileNotFoundError:
    st.error("Model or scaler files not found. Please ensure 'linear_regression_model.pkl', 'knn_model.pkl', 'svm_model.pkl', 'lasso_model.pkl', 'decision_tree_model.pkl', 'voting_regressor_model.pkl', 'random_forest_model.pkl', and 'feature_scaler.pkl' are in the same directory.")
    st.stop()

# Define the feature columns used during training
# Make sure this list matches the feature columns used in your training script
feature_cols = ['Dirección del viento (Grados)', 'Presión atmosférica (mm Hg)', 'Radiación Solar Global (W/m2)', 'Temperatura 10cm (°C)']

# Streamlit App Layout
st.title("Predicción de Calidad del Aire (PM10)")

st.write("""
Esta aplicación utiliza varios modelos de regresión entrenados para predecir
los niveles de PM10 basados en las condiciones ambientales.
""")

# Input fields for features
st.sidebar.header("Introduce las variables ambientales:")

def user_input_features():
    wind_direction = st.sidebar.slider('Dirección del viento (Grados)', 0.0, 360.0, 180.0)
    pressure = st.sidebar.slider('Presión atmosférica (mm Hg)', 600.0, 800.0, 720.0)
    solar_radiation = st.sidebar.slider('Radiación Solar Global (W/m2)', 0.0, 1500.0, 300.0)
    temperature = st.sidebar.slider('Temperatura 10cm (°C)', -10.0, 40.0, 20.0)

    data = {'Dirección del viento (Grados)': wind_direction,
            'Presión atmosférica (mm Hg)': pressure,
            'Radiación Solar Global (W/m2)': solar_radiation,
            'Temperatura 10cm (°C)': temperature}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('Valores de entrada especificados:')
st.write(input_df)

# Scale the input features using the loaded scaler
try:
    scaled_input = scaler.transform(input_df)
    scaled_input_df = pd.DataFrame(scaled_input, columns=feature_cols) # Optional: show scaled input
    # st.subheader('Valores de entrada escalados:') # Uncomment to show scaled input
    # st.write(scaled_input_df) # Uncomment to show scaled input
except Exception as e:
    st.error(f"Error al escalar los datos de entrada: {e}")
    st.stop()


# Make predictions with each model
st.subheader("Predicciones de PM10:")

if st.button("Predecir"):
    try:
        prediction_lr = linear_regression_model.predict(scaled_input)
        prediction_knn = knn_model.predict(scaled_input)
        prediction_svm = svm_model.predict(scaled_input)
        prediction_lasso = lasso_model.predict(scaled_input)
        prediction_dt = decision_tree_model.predict(scaled_input)
        prediction_voting = voting_regressor_model.predict(scaled_input)
        prediction_rf = random_forest_model.predict(scaled_input)

        st.write(f"**Linear Regression:** {prediction_lr[0]:.4f} ug/m3")
        st.write(f"**KNN Regressor:** {prediction_knn[0]:.4f} ug/m3")
        st.write(f"**Support Vector Regressor:** {prediction_svm[0]:.4f} ug/m3")
        st.write(f"**Lasso Regressor:** {prediction_lasso[0]:.4f} ug/m3")
        st.write(f"**Decision Tree Regressor:** {prediction_dt[0]:.4f} ug/m3")
        st.write(f"**Voting Regressor:** {prediction_voting[0]:.4f} ug/m3")
        st.write(f"**Random Forest Regressor:** {prediction_rf[0]:.4f} ug/m3")

    except Exception as e:
        st.error(f"Error al realizar predicciones: {e}")

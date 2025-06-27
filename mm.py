import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os # Import os module to check for file existence

# Load the scaler and trained models
# Use st.cache_resource to cache the loading of models and scaler
@st.cache_resource
def load_resources():
  try:
    scaler = joblib.load('feature_scaler.pkl')
    linear_regression_model = joblib.load('linear_regression_model.pkl')
    knn_model = joblib.load('knn_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    lasso_model = joblib.load('lasso_model.pkl')
    decision_tree_model = joblib.load('decision_tree_model.pkl')
    voting_regressor_model = joblib.load('voting_regressor_model.pkl')
    random_forest_model = joblib.load('random_forest_model.pkl')
    # gradient_boosting_model = joblib.load('gradient_boosting_model.pkl') # Uncomment if GBR model exists and is saved

    # Create a dictionary of models
    models = {
        'Linear Regression': linear_regression_model,
        'KNN': knn_model,
        'SVM': svm_model,
        'Lasso': lasso_model,
        'Decision Tree': decision_tree_model,
        'Voting Regressor': voting_regressor_model,
        'Random Forest': random_forest_model,
        # 'Gradient Boosting': gradient_boosting_model # Uncomment if GBR model exists and is saved
    }
    return scaler, models
  except FileNotFoundError as e:
    st.error(f"Error loading model or scaler: {e}. Please ensure the necessary files ('feature_scaler.pkl', 'linear_regression_model.pkl', etc.) are in the same directory as your Streamlit app.")
    st.stop() # Stop the app if essential files are missing

scaler, models = load_resources()

# Define the feature columns used during training
# Ensure this matches the feature_cols used in the training script
feature_cols = ['Dirección del viento (Grados)', 'Presión atmosférica (mm Hg)', 'Radiación Solar Global (W/m2)', 'Temperatura 10cm (°C)'] # Adjust if your training script used different columns

# Streamlit App Title
st.title("Predicción de PM10 en La Candelaria")

st.write("Esta aplicación utiliza modelos de regresión para predecir los niveles de PM10 basándose en datos meteorológicos.")

# User Input
st.header("Ingrese los valores de las variables meteorológicas:")

# Create input fields for each feature
user_inputs = {}
for col in feature_cols:
    user_inputs[col] = st.number_input(f"Ingrese el valor para {col}", value=0.0)

# Create a DataFrame from user inputs
input_df = pd.DataFrame([user_inputs])

# Scale the user inputs using the loaded scaler
scaled_input = scaler.transform(input_df)

# Convert the scaled input back to a DataFrame with the original feature names
scaled_input_df = pd.DataFrame(scaled_input, columns=feature_cols)

# Model Selection
st.header("Seleccione el modelo para la predicción:")
selected_model_name = st.selectbox("Elija un modelo:", list(models.keys()))

selected_model = models[selected_model_name]

# Make Prediction
if st.button("Predecir PM10"):
    try:
        prediction = selected_model.predict(scaled_input_df)
        st.header("Resultado de la Predicción:")
        st.success(f"La predicción de PM10 ({selected_model_name}) es: {prediction[0]:.4f} ug/m3")

        # Optional: Show feature importances if the selected model has them (e.g., Random Forest)
        if hasattr(selected_model, 'feature_importances_'):
            st.subheader(f"Importancia de las Características ({selected_model_name})")
            feature_importances = pd.Series(selected_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            feature_importances.plot(kind='bar', ax=ax)
            ax.set_title(f'Importancia de las Características para {selected_model_name}')
            ax.set_ylabel('Importancia')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")

st.markdown("---")
st.write("Nota: Los valores de entrada se escalan automáticamente antes de la predicción utilizando el escalador entrenado.")

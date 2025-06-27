import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Define the file paths for the saved models and scaler
linear_regression_model_filename = 'linear_regression_model.pkl'
knn_model_filename = 'knn_model.pkl'
svm_model_filename = 'svm_model.pkl'
lasso_model_filename = 'lasso_model.pkl'
decision_tree_model_filename = 'decision_tree_model.pkl'
voting_regressor_model_filename = 'voting_regressor_model.pkl'
random_forest_model_filename = 'random_forest_model.pkl'
feature_scaler_filename = 'feature_scaler.pkl'
# Add gradient boosting model filename if it exists
# gradient_boosting_model_filename = 'gradient_boosting_model.pkl'


# Load the models and scaler
@st.cache_resource
def load_models():
    models = {}
    try:
        models['Linear Regression'] = joblib.load(linear_regression_model_filename)
        models['KNN'] = joblib.load(knn_model_filename)
        models['SVR'] = joblib.load(svm_model_filename)
        models['Lasso'] = joblib.load(lasso_model_filename)
        models['Decision Tree'] = joblib.load(decision_tree_model_filename)
        models['Voting Regressor'] = joblib.load(voting_regressor_model_filename)
        models['Random Forest'] = joblib.load(random_forest_model_filename)
        # Load gradient boosting model if it exists
        # models['Gradient Boosting'] = joblib.load(gradient_boosting_model_filename)

        scaler = joblib.load(feature_scaler_filename)
        return models, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading model or scaler file: {e}. Please ensure the model and scaler files are in the correct directory.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        return None, None

models, scaler = load_models()

# Define the expected feature columns in the order they were trained
# This should match the feature_cols list from your training script
expected_features = ['Velocidad del Viento (m/s)', 'Dirección del viento (Grados)',
                   'Temperatura 10cm (°C)', 'Presión atmosférica (mm Hg)', 
                   'Radiación Solar Global (W/m2)']


st.title("Predicción de PM10 (ug/m3) en La Candelaria")

if models and scaler:
    st.sidebar.header("Seleccione los Parámetros de Entrada")

    # Create input fields for each feature
    input_data = {}
    for feature in expected_features:
        input_data[feature] = st.sidebar.number_input(f"Ingrese {feature}", value=0.0) # You can set a default value

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Scale the input data using the loaded scaler
    try:
        scaled_input = scaler.transform(input_df)
        scaled_input_df = pd.DataFrame(scaled_input, columns=expected_features)
    except Exception as e:
        st.error(f"Error during scaling of input data: {e}")
        st.stop()

    st.subheader("Valores de Entrada (Escalados)")
    st.write(scaled_input_df)

    st.sidebar.header("Seleccione el Modelo para Predecir")
    selected_model_name = st.sidebar.selectbox("Modelo", list(models.keys()))

    selected_model = models[selected_model_name]

    # Make prediction
    if st.sidebar.button("Predecir"):
        try:
            prediction = selected_model.predict(scaled_input_df)
            st.subheader(f"Predicción de PM10 ({selected_model_name})")
            st.success(f"La predicción de PM10 es: {prediction[0]:.4f} ug/m3")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    st.sidebar.markdown("---")
    st.sidebar.write("Nota: Asegúrese de que los archivos del modelo (.pkl) y el scaler (.pkl) estén en el mismo directorio que la aplicación Streamlit.")

else:
    st.warning("No se pudieron cargar los modelos. Por favor, asegúrese de que los archivos existen.")



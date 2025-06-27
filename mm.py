import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Define the file paths for the saved models and scaler
# Assuming the Excel file is also in the same directory or a known path
EXCEL_FILE = 'Mediciones_Calidad_Aire_La_Candelaria_Organizado.xlsx'
SCALER_FILE = 'feature_scaler.pkl' # Use the scaler fitted on training data
LINEAR_REGRESSION_MODEL_FILE = 'linear_regression_model.pkl'
KNN_MODEL_FILE = 'knn_model.pkl'
SVM_MODEL_FILE = 'svm_model.pkl'
LASSO_MODEL_FILE = 'lasso_model.pkl'
DECISION_TREE_MODEL_FILE = 'decision_tree_model.pkl'
VOTING_REGRESSOR_MODEL_FILE = 'voting_regressor_model.pkl'
RANDOM_FOREST_MODEL_FILE = 'random_forest_model.pkl'

# Load the dataset and perform the same initial processing as in the notebook
@st.cache_data # Cache the data loading and processing
def load_and_process_data(excel_file):
  try:
    df = pd.read_excel(excel_file)
    # Drop columns as done in the notebook
    df = df.drop(['Fecha y Hora de Inicio (dd/MM/aaaa  HH:mm:ss)', 'Fecha y Hora de Finalización (dd/MM/aaaa  HH:mm:ss)', 'Precipitación (mm)', 'Humedad Relativa 10m (%)'], axis=1)

    # Define the target variable and feature columns
    target_variable = 'PM10 (ug/m3)\nCondición Estándar'
    numerical_cols = df.select_dtypes(include=np.number).columns
    feature_cols = [col for col in numerical_cols if col != target_variable]

    # Separate features and target before scaling
    X = df[feature_cols].copy()
    y = df[target_variable].copy()

    # Identify columns to scale (these should be the same as in the notebook before training)
    columns_to_scale = ['Dirección del viento (Grados)', 'Presión atmosférica (mm Hg)', 'Radiación Solar Global (W/m2)', 'Temperatura 10cm (°C)']

    return X, y, feature_cols, columns_to_scale
  except FileNotFoundError:
    st.error(f"Error: El archivo '{excel_file}' no se encontró.")
    return None, None, None, None
  except Exception as e:
    st.error(f"Error al cargar o procesar los datos: {e}")
    return None, None, None, None


# Load the trained models and scaler
@st.cache_resource # Cache the loading of models and scaler
def load_resources():
    try:
        scaler = joblib.load(SCALER_FILE)
        linear_regression_model = joblib.load(LINEAR_REGRESSION_MODEL_FILE)
        knn_model = joblib.load(KNN_MODEL_FILE)
        svm_model = joblib.load(SVM_MODEL_FILE)
        lasso_model = joblib.load(LASSO_MODEL_FILE)
        decision_tree_model = joblib.load(DECISION_TREE_MODEL_FILE)
        voting_regressor_model = joblib.load(VOTING_REGRESSOR_MODEL_FILE)
        random_forest_model = joblib.load(RANDOM_FOREST_MODEL_FILE)
        return scaler, linear_regression_model, knn_model, svm_model, lasso_model, decision_tree_model, voting_regressor_model, random_forest_model
    except FileNotFoundError as e:
        st.error(f"Error: Archivo de modelo o scaler no encontrado: {e}")
        return None, None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error al cargar los recursos: {e}")
        return None, None, None, None, None, None, None, None


# Load data and resources
X, y, feature_cols, columns_to_scale = load_and_process_data(EXCEL_FILE)
scaler, linear_regression_model, knn_model, svm_model, lasso_model, decision_tree_model, voting_regressor_model, random_forest_model = load_resources()

# Streamlit App Title
st.title('Predicción de Calidad del Aire (PM10)')

if X is not None and scaler is not None:
    st.header('Ingresa los valores para predecir PM10')

    # Create input fields for each feature
    input_data = {}
    for col in feature_cols:
        # Provide default values or ranges based on the dataset's characteristics
        if col in columns_to_scale:
             # For scaled features, it's better to get the original min/max from the unscaled data
             # Need to reload the original data or compute min/max before scaling
             # For simplicity here, we'll use a general number input
             input_data[col] = st.number_input(f'{col}', value=float(X[col].mean()))
        else:
             input_data[col] = st.number_input(f'{col}', value=float(X[col].mean()))


    # Create a button to trigger prediction
    if st.button('Predecir PM10'):
        try:
            # Create a DataFrame from the input data
            input_df = pd.DataFrame([input_data])

            # Apply the same scaling as used during training
            input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])

            st.subheader('Resultados de la Predicción:')

            # Make predictions using each loaded model
            if linear_regression_model:
                pred_lr = linear_regression_model.predict(input_df)[0]
                st.write(f'Predicción con Regresión Lineal: {pred_lr:.2f} ug/m³')

            if knn_model:
                 pred_knn = knn_model.predict(input_df)[0]
                 st.write(f'Predicción con KNN: {pred_knn:.2f} ug/m³')

            if svm_model:
                 pred_svm = svm_model.predict(input_df)[0]
                 st.write(f'Predicción con SVR: {pred_svm:.2f} ug/m³')

            if lasso_model:
                 pred_lasso = lasso_model.predict(input_df)[0]
                 st.write(f'Predicción con Lasso: {pred_lasso:.2f} ug/m³')

            if decision_tree_model:
                 pred_dt = decision_tree_model.predict(input_df)[0]
                 st.write(f'Predicción con Árbol de Decisión: {pred_dt:.2f} ug/m³')

            if voting_regressor_model:
                 pred_voting = voting_regressor_model.predict(input_df)[0]
                 st.write(f'Predicción con Voting Regressor: {pred_voting:.2f} ug/m³')

            if random_forest_model:
                 pred_rf = random_forest_model.predict(input_df)[0]
                 st.write(f'Predicción con Random Forest: {pred_rf:.2f} ug/m³')

        except Exception as e:
            st.error(f"Error durante la predicción: {e}")

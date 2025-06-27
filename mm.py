import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the trained models and the scaler
try:
    linear_regression_model = joblib.load('linear_regression_model.pkl')
    knn_model = joblib.load('knn_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    lasso_model = joblib.load('lasso_model.pkl')
    decision_tree_model = joblib.load('decision_tree_model.pkl')
    voting_regressor_model = joblib.load('voting_regressor_model.pkl')
    random_forest_model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('feature_scaler.pkl') # Load the scaler fitted on training features
except FileNotFoundError:
    st.error("Error: One or more model files or scaler file not found.")
    st.stop() # Stop execution if files are missing

st.title("Predicción de PM10")

st.write("""
Esta aplicación utiliza varios modelos de regresión para predecir los niveles de PM10
basados en datos meteorológicos. Por favor, suba un archivo Excel con los datos a predecir.
""")

uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read the uploaded Excel file
        df_predict = pd.read_excel(uploaded_file)

        st.subheader("Datos Cargados:")
        st.dataframe(df_predict.head())

        # --- Data Preprocessing (Must match the training preprocessing) ---

        # Define the columns that were used as features during training
        # This should match the feature_cols list from your training code
        # Check your training code's feature_cols list
        # Example based on your training code (adjust if necessary):
        feature_columns = ['Velocidad del Viento (m/s)', 'Dirección del viento (Grados)',
                   'Temperatura 10cm (°C)', 'Presión atmosférica (mm Hg)', 
                   'Radiación Solar Global (W/m2)']

        # Check if the uploaded DataFrame contains all required feature columns
        missing_cols = [col for col in feature_columns if col not in df_predict.columns]
        if missing_cols:
            st.error(f"Error: El archivo Excel cargado no contiene las siguientes columnas requeridas: {', '.join(missing_cols)}")
            st.stop()

        # Select only the feature columns
        X_predict = df_predict[feature_columns]

        # Ensure numerical columns are treated as such (handle potential non-numeric entries)
        for col in X_predict.columns:
             X_predict[col] = pd.to_numeric(X_predict[col], errors='coerce')

        # Handle potential NaN values that might arise from coercion (e.g., forward fill or mean imputation)
        # For simplicity, let's use forward fill here, but choose a strategy that suits your data
        X_predict.fillna(method='ffill', inplace=True)
        X_predict.fillna(method='bfill', inplace=True) # Handle potential NaNs at the beginning

        # Apply the same scaling used during training
        try:
            X_predict_scaled = scaler.transform(X_predict)
            X_predict_scaled_df = pd.DataFrame(X_predict_scaled, columns=feature_columns) # Keep as DataFrame
        except Exception as e:
            st.error(f"Error durante la escala de los datos: {e}")
            st.stop()


        st.subheader("Realizando Predicciones...")

        # --- Make Predictions using the loaded models ---
        try:
            df_predict['PM10 Predicho (Linear Regression)'] = linear_regression_model.predict(X_predict_scaled_df)
            df_predict['PM10 Predicho (KNN)'] = knn_model.predict(X_predict_scaled_df)
            df_predict['PM10 Predicho (SVR)'] = svm_model.predict(X_predict_scaled_df)
            df_predict['PM10 Predicho (Lasso)'] = lasso_model.predict(X_predict_scaled_df)
            df_predict['PM10 Predicho (Decision Tree)'] = decision_tree_model.predict(X_predict_scaled_df)
            df_predict['PM10 Predicho (Voting Regressor)'] = voting_regressor_model.predict(X_predict_scaled_df)
            df_predict['PM10 Predicho (Random Forest)'] = random_forest_model.predict(X_predict_scaled_df)


            st.subheader("Predicciones Generadas:")
            st.dataframe(df_predict[['PM10 Predicho (Linear Regression)',
                                     'PM10 Predicho (KNN)',
                                     'PM10 Predicho (SVR)',
                                     'PM10 Predicho (Lasso)',
                                     'PM10 Predicho (Decision Tree)',
                                     'PM10 Predicho (Voting Regressor)',
                                     'PM10 Predicho (Random Forest)']])

            # Option to download the results
            @st.cache_data
            def convert_df_to_excel(df):
                # IMPORTANT: Cache the conversion to prevent dataframe conversion on every rerun
                return df.to_excel("predicciones_PM10.xlsx", index=False)

            excel_data = convert_df_to_excel(df_predict)

            st.download_button(
                label="Descargar Predicciones (Excel)",
                data=open("predicciones_PM10.xlsx", "rb").read(), # Read the file directly
                file_name="predicciones_PM10.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


        except Exception as e:
            st.error(f"Error durante la predicción: {e}")

    except Exception as e:
        st.error(f"Error al leer el archivo Excel: {e}")


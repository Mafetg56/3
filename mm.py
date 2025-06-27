import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar modelos y scaler
@st.cache_resource
def load_models():
    try:
        models = {
            'Linear Regression': joblib.load('linear_regression_model.pkl'),
            'KNN': joblib.load('knn_model.pkl'),
            'SVR': joblib.load('svm_model.pkl'),
            'Lasso': joblib.load('lasso_model.pkl'),
            'Decision Tree': joblib.load('decision_tree_model.pkl'),
            'Voting Regressor': joblib.load('voting_regressor_model.pkl'),
            'Random Forest': joblib.load('random_forest_model.pkl')
        }
        scaler = joblib.load('feature_scaler.pkl')
        return models, scaler
    except FileNotFoundError as e:
        st.error(f"Archivo no encontrado: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None

models, scaler = load_models()

# Columnas esperadas
expected_features = ['Velocidad del Viento (m/s)', 'Dirección del viento (Grados)',
                     'Temperatura 10cm (°C)', 'Presión atmosférica (mm Hg)',
                     'Radiación Solar Global (W/m2)']

st.title("Predicción de PM10 (ug/m3) en La Candelaria")

# Cargar archivo Excel
uploaded_file = st.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None and models and scaler:
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("Datos cargados:")
        st.dataframe(df.head())

        # Validación de columnas
        missing_cols = [col for col in expected_features if col not in df.columns]
        if missing_cols:
            st.error(f"Faltan columnas requeridas: {', '.join(missing_cols)}")
            st.stop()

        # Procesamiento
        X = df[expected_features].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        X.fillna(method='ffill', inplace=True)
        X.fillna(method='bfill', inplace=True)

        st.subheader("Datos Procesados:")
        st.dataframe(X.head())

        try:
            X_scaled = scaler.transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=expected_features)
        except Exception as e:
            st.error(f"Error al escalar los datos: {e}")
            st.stop()

        # Realizar predicciones
        for name, model in models.items():
            df[f'PM10 Predicho ({name})'] = model.predict(X_scaled_df)

        # Mostrar las predicciones al final
        st.subheader("Predicciones por Modelo:")
        pred_cols = [col for col in df.columns if col.startswith("PM10 Predicho")]
        st.dataframe(df[pred_cols])

        # Mostrar resultados completos
        st.subheader("Resultados Completos:")
        st.dataframe(df)

        # Botón de descarga
        @st.cache_data
        def convert_df_to_excel(df):
            return df.to_excel("predicciones_PM10.xlsx", index=False)

        excel_data = convert_df_to_excel(df)

        st.download_button(
            label="Descargar Predicciones (Excel)",
            data=open("predicciones_PM10.xlsx", "rb").read(),
            file_name="predicciones_PM10.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
else:
    if uploaded_file is None:
        st.info("Por favor, sube un archivo Excel para continuar.")
    if models is None:
        st.warning("No se pudieron cargar los modelos. Asegúrate de que los archivos existan.")


import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# -------------------------
# Cargar modelos y scaler
# -------------------------
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
        st.error(f"Error cargando archivos: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error inesperado al cargar modelos: {e}")
        return None, None

# -------------------------
# Configuración inicial
# -------------------------
models, scaler = load_models()
expected_features = [
    'Velocidad del Viento (m/s)',
    'Dirección del viento (Grados)',
    'Temperatura 10cm (°C)',
    'Presión atmosférica (mm Hg)',
    'Radiación Solar Global (W/m2)'
]

st.title("Predicción de PM10 (ug/m3) en La Candelaria")

# -------------------------
# Subir archivo Excel
# -------------------------
st.subheader("1. Carga de Datos")
file = st.file_uploader("Sube un archivo Excel (.xlsx)", type=["xlsx"])

if file and models and scaler:
    try:
        df = pd.read_excel(file)
        st.write("Vista previa de los datos cargados:")
        st.dataframe(df.head())

        # Verificación de columnas requeridas
        missing_cols = [col for col in expected_features if col not in df.columns]
        if missing_cols:
            st.error(f"Faltan columnas requeridas: {', '.join(missing_cols)}")
            st.stop()

        # Preprocesamiento
        X = df[expected_features].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        X.fillna(method='ffill', inplace=True)
        X.fillna(method='bfill', inplace=True)

        try:
            X_scaled = scaler.transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=expected_features)
        except Exception as e:
            st.error(f"Error durante la escala de los datos: {e}")
            st.stop()

        st.subheader("2. Predicción con Diferentes Modelos")
        for name, model in models.items():
            try:
                df[f'Predicción {name}'] = model.predict(X_scaled_df)
                st.write(f"Predicciones con {name}:")
                st.dataframe(df[[f'Predicción {name}']].head())
            except Exception as e:
                st.warning(f"No se pudo predecir con {name}: {e}")

        st.subheader("3. Resultados Finales")
        st.dataframe(df)

        # Botón de descarga
        @st.cache_data
        def convert_df_to_excel(data):
            return data.to_excel("predicciones_PM10.xlsx", index=False)

        convert_df_to_excel(df)
        st.download_button(
            label="Descargar Resultados en Excel",
            data=open("predicciones_PM10.xlsx", "rb").read(),
            file_name="predicciones_PM10.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")

elif not models:
    st.warning("Los modelos no se pudieron cargar. Verifica los archivos .pkl.")



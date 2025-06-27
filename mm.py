import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import base64
import io

# Function to download a template Excel file
def get_excel_template(feature_cols):
    """Generates an Excel template with the required feature columns."""
    template_df = pd.DataFrame(columns=feature_cols)
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    template_df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Generates HTML for a file download link."""
    bin_str = base64.b64encode(bin_file).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}.xlsx">Descargar plantilla</a>'
    return href

st.title("Aplicación de Predicción de Calidad del Aire")

st.write("Esta aplicación utiliza un modelo de Machine Learning para predecir la calidad del aire (PM10) basada en datos meteorológicos.")

# Define the list of expected feature columns (based on your original notebook)
# Ensure this list matches the columns used for training the models, excluding the target variable.
# You might need to adjust this list based on the final features used in your models.
feature_cols = ['Dirección del viento (Grados)', 'Presión atmosférica (mm Hg)', 'Radiación Solar Global (W/m2)', 'Temperatura 10cm (°C)']
target_variable = 'PM10 (ug/m3)\nCondición Estándar' # Keep the target variable name for clarity

# Option to download the template
template_excel = get_excel_template(feature_cols)
st.markdown(get_binary_file_downloader_html(template_excel, 'plantilla_prediccion'), unsafe_allow_html=True)


# File uploader
uploaded_file = st.file_uploader("Carga un archivo Excel (.xlsx) con los datos a predecir", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read the uploaded Excel file
        prediction_df = pd.read_excel(uploaded_file)
        st.write("Datos cargados:")
        st.dataframe(prediction_df)

        # --- Data Preparation (Matching the original notebook) ---
        # Ensure required columns are present
        missing_cols = [col for col in feature_cols if col not in prediction_df.columns]
        if missing_cols:
            st.error(f"El archivo Excel cargado no contiene las siguientes columnas requeridas para la predicción: {', '.join(missing_cols)}")
        else:
            # Select only the feature columns for prediction
            prediction_df_features = prediction_df[feature_cols]

            # Load the trained scaler
            try:
                scaler = joblib.load('feature_scaler.pkl') # Load the scaler used for features during training
                st.write("Scaler cargado exitosamente.")
            except FileNotFoundError:
                st.error("Error: No se encontró el archivo del scaler ('feature_scaler.pkl'). Asegúrate de que está en el mismo directorio.")
                scaler = None # Set scaler to None if not found

            if scaler is not None:
                # Scale the features
                try:
                    prediction_df_scaled = scaler.transform(prediction_df_features)
                    st.write("Datos escalados exitosamente.")
                    prediction_df_scaled = pd.DataFrame(prediction_df_scaled, columns=feature_cols) # Convert back to DataFrame for consistency

                    # --- Model Loading and Prediction ---
                    st.subheader("Selecciona el modelo para la predicción")
                    model_option = st.selectbox(
                        "Elige un modelo:",
                        ('Linear Regression', 'KNN', 'SVR', 'Lasso', 'Decision Tree', 'Voting Regressor', 'Random Forest')
                    )

                    model_filename = None
                    if model_option == 'Linear Regression':
                        model_filename = 'linear_regression_model.pkl'
                    elif model_option == 'KNN':
                        model_filename = 'knn_model.pkl'
                    elif model_option == 'SVR':
                        model_filename = 'svm_model.pkl'
                    elif model_option == 'Lasso':
                        model_filename = 'lasso_model.pkl'
                    elif model_option == 'Decision Tree':
                        model_filename = 'decision_tree_model.pkl'
                    elif model_option == 'Voting Regressor':
                        model_filename = 'voting_regressor_model.pkl'
                    elif model_option == 'Random Forest':
                        model_filename = 'random_forest_model.pkl'

                    if model_filename:
                        try:
                            # Load the selected model
                            model = joblib.load(model_filename)
                            st.write(f"{model_option} cargado exitosamente.")

                            # Make predictions
                            predictions = model.predict(prediction_df_scaled)

                            # Add predictions to the original DataFrame
                            prediction_df['PM10_Predicted'] = predictions

                            st.subheader("Resultados de la Predicción:")
                            st.dataframe(prediction_df)

                            # Option to download the predictions
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                prediction_df.to_excel(writer, index=False, sheet_name='Predicciones')
                            processed_data = output.getvalue()

                            st.markdown(get_binary_file_downloader_html(processed_data, 'predicciones_PM10'), unsafe_allow_html=True)


                        except FileNotFoundError:
                            st.error(f"Error: No se encontró el archivo del modelo '{model_filename}'. Asegúrate de que todos los archivos de modelo entrenados están en el mismo directorio.")
                        except Exception as e:
                            st.error(f"Ocurrió un error al cargar el modelo o realizar la predicción: {e}")

                except Exception as e:
                    st.error(f"Ocurrió un error durante el escalado de los datos: {e}")

    except Exception as e:
        st.error(f"Ocurrió un error al leer el archivo Excel: {e}")



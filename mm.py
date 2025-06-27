import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os


# --- Data Preprocessing (as in your original script) ---
        st.header("2. Preprocesamiento de Datos")
        st.write("Limpiando y escalando los datos...")

        df = df.drop(['Fecha y Hora de Inicio (dd/MM/aaaa  HH:mm:ss)',
                      'Fecha y Hora de Finalización (dd/MM/aaaa  HH:mm:ss)',
                      'Precipitación (mm)',
                      'Humedad Relativa 10m (%)'], axis=1, errors='ignore') # Use errors='ignore' in case columns are already dropped

        # Select the columns to scale - Ensure these columns exist after dropping others
        columns_to_scale = ['Dirección del viento (Grados)', 'Presión atmosférica (mm Hg)', 'Radiación Solar Global (W/m2)', 'Temperatura 10cm (°C)']
        existing_columns_to_scale = [col for col in columns_to_scale if col in df.columns]

        if existing_columns_to_scale:
            # Initialize and fit the StandardScaler
            scaler = StandardScaler()
            df[existing_columns_to_scale] = scaler.fit_transform(df[existing_columns_to_scale])
            st.success("Datos escalados exitosamente.")
            st.dataframe(df.head())
            # Save the scaler
            joblib.dump(scaler, 'feature_scaler.pkl')
            st.write("Scaler guardado como 'feature_scaler.pkl'.")
        else:
            st.warning("No se encontraron columnas para escalar. Asegúrate de que las columnas 'Dirección del viento (Grados)', 'Presión atmosférica (mm Hg)', 'Radiación Solar Global (W/m2)' y 'Temperatura 10cm (°C)' estén en el archivo.")

        # Define features (X) and target (y)
        target_variable = 'PM10 (ug/m3)\nCondición Estándar'
        if target_variable in df.columns:
             numerical_cols = df.select_dtypes(include=np.number).columns
             feature_cols = [col for col in numerical_cols if col != target_variable]
             X = df[feature_cols]
             y = df[target_variable]

             st.write(f"Características (X): {feature_cols}")
             st.write(f"Variable objetivo (y): {target_variable}")

             # Splitting data (needed for training, but for inference we just need X structure)
             # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # Not needed for inference only

             # --- Model Loading (Assuming models are pre-trained and saved) ---
             st.header("3. Carga de Modelos Pre-entrenados")
             st.write("Cargando los modelos de regresión guardados...")

             models = {}
             model_filenames = {
                 'Linear Regression': 'linear_regression_model.pkl',
                 'KNN': 'knn_model.pkl',
                 'SVR': 'svm_model.pkl',
                 'Lasso': 'lasso_model.pkl',
                 'Decision Tree': 'decision_tree_model.pkl',
                 'Voting Regressor': 'voting_regressor_model.pkl',
                 'Random Forest': 'random_forest_model.pkl',
                 # 'Gradient Boosting': 'gradient_boosting_model.pkl' # Add if you have this model
             }

             loaded_models = {}
             for name, filename in model_filenames.items():
                 try:
                     # Check if the file exists before loading
                     if os.path.exists(filename):
                         loaded_models[name] = joblib.load(filename)
                         st.success(f"Modelo '{name}' cargado exitosamente.")
                     else:
                         st.warning(f"Archivo de modelo '{filename}' no encontrado. Salteando la carga de '{name}'.")
                 except Exception as e:
                     st.error(f"Error al cargar el modelo '{name}' desde '{filename}': {e}")

             if loaded_models:
                 # --- Prediction Section ---
                 st.header("4. Realizar Predicciones")
                 st.write("Selecciona un modelo para predecir los niveles de PM10.")

                 # Ensure the scaler for feature scaling is loaded or available
                 scaler_filename = 'feature_scaler.pkl'
                 try:
                     if os.path.exists(scaler_filename):
                          loaded_scaler = joblib.load(scaler_filename)
                          st.success("Scaler de características cargado exitosamente.")
                     else:
                         st.error(f"Scaler de características '{scaler_filename}' no encontrado. No se puede realizar la predicción sin el scaler.")
                         loaded_scaler = None
                 except Exception as e:
                      st.error(f"Error al cargar el scaler: {e}")
                      loaded_scaler = None


                 if loaded_scaler is not None:
                      selected_model_name = st.selectbox("Selecciona el Modelo de Predicción", list(loaded_models.keys()))

                      if st.button("Predecir PM10"):
                          if selected_model_name in loaded_models:
                              model_to_predict = loaded_models[selected_model_name]

                              # Ensure the input data has the same features as the training data
                              # It's crucial that the order and names of columns match the training data features (feature_cols)
                              # If you were processing a single new input, you'd scale that input using the loaded_scaler
                              # For predicting on the entire dataset, the df is already scaled based on the training process
                              # Here we assume we are predicting on the preprocessed df (which corresponds to X)

                              # Prepare data for prediction - make sure column order matches feature_cols
                              X_pred = df[feature_cols]

                              try:
                                  predictions = model_to_predict.predict(X_pred)
                                  st.subheader(f"Resultados de la Predicción con {selected_model_name}")
                                  prediction_df = pd.DataFrame({'Predicción PM10': predictions})
                                  st.dataframe(prediction_df)

                                  # Optional: Add a simple plot of actual vs predicted (if y is available)
                                  if 'y' in locals():
                                      st.subheader("Comparación de Valores Actuales vs. Predichos (Primeras 100 filas)")
                                      plt.figure(figsize=(10, 6))
                                      plt.plot(y.values[:100], label='Actual')
                                      plt.plot(predictions[:100], label='Predicción')
                                      plt.xlabel("Índice de la Muestra")
                                      plt.ylabel("PM10 (ug/m3)")
                                      plt.title(f"Actual vs. Predicho PM10 ({selected_model_name})")
                                      plt.legend()
                                      st.pyplot(plt)
                                      plt.clf() # Clear the figure

                              except Exception as e:
                                  st.error(f"Error durante la predicción: {e}")
                                  st.warning("Asegúrate de que las columnas de entrada coincidan con las columnas con las que se entrenó el modelo.")
                                  st.write("Columnas esperadas para la predicción:", feature_cols)
                                  st.write("Columnas disponibles en los datos cargados:", X_pred.columns.tolist())

                          else:
                              st.warning("Por favor, selecciona un modelo de la lista.")

                 else:
                      st.warning("El scaler no se cargó correctamente. No se pueden realizar predicciones.")


             else:
                 st.error("No se pudo cargar ningún modelo pre-entrenado. Asegúrate de que los archivos .pkl estén en el mismo directorio que este script.")


        else:
            st.error(f"La columna objetivo '{target_variable}' no se encontró en el archivo Excel.")


    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")

else:
    st.info("Por favor, sube un archivo Excel para comenzar.")


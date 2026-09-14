"""
AeroShield: AI-Based Real-Time Air Quality Forecasting System
Fixed vs. the original Colab notebook version:
  - Removed the dead fallback cell that saved a completely different model
    architecture to 'aqi_deep_learning_model.h5' — a filename this app never
    loaded in the first place, so it silently did nothing useful.
  - Artifact filenames below now exactly match what the fixed training
    notebook saves: xgboost_aqi_model.pkl, scaler_X.pkl, scaler_y.pkl,
    cnn_bilstm_model.keras.
  - No behavior change to the three prediction modes themselves — that logic
    was already sound; only the model/artifact plumbing around it was broken.

Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="AeroShield AI - Air Quality Forecasting System",
    page_icon="🌬️",
    layout="wide",
)

st.title("🌬️ AeroShield: AI-Based Real-Time Air Quality Forecasting System")
st.write(
    "This application uses **XGBoost Regressor** and **CNN-BiLSTM Deep Learning** "
    "models to predict CO(GT) concentrations."
)

FEATURE_COLS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S5(O3)", "T", "RH", "AH"]
TARGET_COL = "CO(GT)"
LOOKBACK = 24  # 24-hour sequence length for the CNN-BiLSTM model

MODEL_DIR = "models"
XGB_MODEL_PATH = f"{MODEL_DIR}/xgboost_aqi_model.pkl"
SCALER_X_PATH = f"{MODEL_DIR}/scaler_X.pkl"
SCALER_Y_PATH = f"{MODEL_DIR}/scaler_y.pkl"
DL_MODEL_PATH = f"{MODEL_DIR}/cnn_bilstm_model.keras"


@st.cache_resource
def load_artifacts():
    """Load all model artifacts once per session. Missing files degrade
    gracefully (the affected tab just tells the user what's missing) instead
    of crashing the whole app."""
    xgb_model = None
    scaler_X = None
    scaler_y = None
    dl_model = None

    try:
        xgb_model = joblib.load(XGB_MODEL_PATH)
    except Exception as e:
        st.warning(f"⚠️ XGBoost model ('{XGB_MODEL_PATH}') could not be loaded: {e}")

    try:
        scaler_X = joblib.load(SCALER_X_PATH)
    except Exception as e:
        st.warning(f"⚠️ Feature scaler ('{SCALER_X_PATH}') could not be loaded: {e}")

    try:
        scaler_y = joblib.load(SCALER_Y_PATH)
    except Exception:
        pass  # optional — only needed to un-scale DL predictions

    # Import TensorFlow lazily so the app can still start (and the XGBoost
    # tab still work) even in environments where TF is slow to import or unavailable.
    try:
        import tensorflow as tf
        try:
            dl_model = tf.keras.models.load_model(DL_MODEL_PATH)
        except Exception as e:
            st.info(f"ℹ️ Deep learning model ('{DL_MODEL_PATH}') could not be loaded: {e}")
    except Exception:
        st.info("ℹ️ TensorFlow is unavailable — the Deep Learning tab will fall back to the XGBoost model.")

    return xgb_model, scaler_X, scaler_y, dl_model


xgb_model, scaler_X, scaler_y, dl_model = load_artifacts()

st.sidebar.header("Navigation & Settings")
model_option = st.sidebar.selectbox(
    "Choose Prediction Mode",
    [
        "1. XGBoost Regressor (Single Instance)",
        "2. CNN-BiLSTM (24-Hour Sequence Forecast)",
        "3. Batch CSV Data Processing",
    ],
)

# --- OPTION 1: Single Instance XGBoost Prediction ---
if model_option == "1. XGBoost Regressor (Single Instance)":
    st.subheader("📊 Single-Timestamp Prediction (XGBoost Regressor)")
    st.write("Enter the ambient sensor values below to estimate current **CO(GT)** levels:")

    col1, col2, col3 = st.columns(3)
    with col1:
        pt08_s1 = st.number_input("PT08.S1(CO) [Tin Oxide]", value=1360.0)
        pt08_s2 = st.number_input("PT08.S2(NMHC) [Tungsten Oxide]", value=1046.0)
        pt08_s3 = st.number_input("PT08.S3(NOx) [Tungsten Oxide]", value=1056.0)
    with col2:
        pt08_s5 = st.number_input("PT08.S5(O3) [Indium Oxide]", value=1268.0)
        temp = st.number_input("Temperature (°C)", value=13.6)
    with col3:
        rh = st.number_input("Relative Humidity (%)", value=48.9)
        ah = st.number_input("Absolute Humidity (AH)", value=0.7578)

    if st.button("Predict CO(GT) Concentration"):
        if xgb_model is not None:
            input_data = pd.DataFrame(
                [[pt08_s1, pt08_s2, pt08_s3, pt08_s5, temp, rh, ah]], columns=FEATURE_COLS
            )
            prediction = xgb_model.predict(input_data)[0]
            st.success(f"### Predicted CO(GT): `{prediction:.3f}` mg/m³")
        else:
            st.error(f"XGBoost model file (`{XGB_MODEL_PATH}`) is missing from the app directory!")

# --- OPTION 2: CNN-BiLSTM 24-Hour Sequence Prediction ---
elif model_option == "2. CNN-BiLSTM (24-Hour Sequence Forecast)":
    st.subheader("🧠 Deep Learning Time-Series Forecasting (CNN-BiLSTM)")
    st.write(
        "Upload a CSV file containing at least 24 recent **hourly, chronologically ordered** "
        "sensor readings to generate a sequence forecast."
    )

    uploaded_file = st.file_uploader("Upload Recent Sensor CSV", type=["csv"], key="dl_uploader")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.replace(-200, np.nan, inplace=True)

        if all(col in df.columns for col in FEATURE_COLS):
            df_features = df[FEATURE_COLS].copy()
            df_features.fillna(df_features.mean(), inplace=True)

            st.write("📋 **Uploaded Recent Readings Preview:**", df_features.tail(LOOKBACK))

            if len(df_features) >= LOOKBACK:
                if st.button("Run Sequence Forecast"):
                    sequence_data = df_features.tail(LOOKBACK).values

                    if scaler_X is not None:
                        scaled_seq = scaler_X.transform(sequence_data)
                        scaled_seq = np.expand_dims(scaled_seq, axis=0)  # shape: (1, 24, 7)

                        if dl_model is not None:
                            pred = dl_model.predict(scaled_seq)[0][0]
                            if scaler_y is not None:
                                pred = scaler_y.inverse_transform([[pred]])[0][0]
                            st.success(f"### Forecasted CO(GT) for Next Hour: `{pred:.3f}` mg/m³")
                        elif xgb_model is not None:
                            st.info(f"ℹ️ DL model ('{DL_MODEL_PATH}') not found. Falling back to XGBoost on the most recent reading:")
                            avg_pred = xgb_model.predict(df_features.tail(1))[0]
                            st.metric(label="XGBoost Next-Hour Estimate", value=f"{avg_pred:.3f} mg/m³")
                        else:
                            st.error("Neither the deep learning model nor the XGBoost fallback is available.")
                    else:
                        st.error(f"Scaler file (`{SCALER_X_PATH}`) missing from the app directory!")
            else:
                st.error(f"The uploaded dataset must have at least {LOOKBACK} rows to construct the input sequence.")
        else:
            st.error(f"Dataset must include these feature columns: {FEATURE_COLS}")

# --- OPTION 3: Batch Data Clean & Evaluation ---
elif model_option == "3. Batch CSV Data Processing":
    st.subheader("📁 Batch CSV Upload & Inference")

    file = st.file_uploader("Upload AirQualityUCI.csv or a raw dataset:", type=["csv"], key="batch_uploader")
    if file is not None:
        data = pd.read_csv(file)
        st.write("📄 **Raw Input Data Preview:**", data.head())

        data.replace(-200, np.nan, inplace=True)
        data.dropna(how="all", axis=0, inplace=True)

        if all(col in data.columns for col in FEATURE_COLS):
            X_batch = data[FEATURE_COLS].copy()
            X_batch.fillna(X_batch.mean(), inplace=True)

            if xgb_model is not None:
                preds = xgb_model.predict(X_batch)
                data["Predicted_CO(GT)"] = preds
                st.write("✅ **Processed Data with Predictions:**", data.head())

                csv_download = data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predicted Results CSV",
                    data=csv_download,
                    file_name="AQI_Predictions_Output.csv",
                    mime="text/csv",
                )
            else:
                st.error(f"XGBoost model file (`{XGB_MODEL_PATH}`) is missing from the app directory!")
        else:
            st.error(f"Dataset must include these feature columns: {FEATURE_COLS}")

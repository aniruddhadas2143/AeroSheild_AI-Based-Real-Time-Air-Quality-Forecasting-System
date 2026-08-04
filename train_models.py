"""
train_models.py

Run this ONCE, locally or in Colab, to produce the four artifact files
that app.py needs:
    - xgboost_aqi_model.pkl
    - scaler_X.pkl
    - scaler_y.pkl
    - cnn_bilstm_model.keras

Put AirQualityUCI.csv in the same folder before running:
    python train_models.py

This is the same pipeline as your original notebook, just consolidated
into one script and saving with filenames that match what app.py expects
(your notebook's deep-learning save used 'cnn_bilstm_model.keras', but the
fallback generator at the bottom of your old aqf_app.py saved a
differently-named 'aqi_deep_learning_model.h5' -- that mismatch is why the
DL tab could never find its model. Fixed here.)
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout

FEATURE_COLS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S5(O3)", "T", "RH", "AH"]
TARGET_COL = "CO(GT)"
LOOKBACK = 24


def load_and_clean(csv_path="AirQualityUCI.csv"):
    data = pd.read_csv(csv_path)
    data.replace(-200, np.nan, inplace=True)
    data.dropna(how="all", axis=0, inplace=True)
    data.dropna(how="all", axis=1, inplace=True)
    return data


def train_xgboost(x_train, y_train, x_test, y_test):
    model = XGBRegressor(n_estimators=100, learning_rate=0.5, random_state=42)
    model.fit(x_train, y_train.values.ravel())
    pred = model.predict(x_test)

    print("XGBoost — RMSE:", mean_squared_error(y_test, pred))
    print("XGBoost — MAE:", mean_absolute_error(y_test, pred))
    print("XGBoost — R2:", r2_score(y_test, pred))

    joblib.dump(model, "xgboost_aqi_model.pkl")
    print("Saved xgboost_aqi_model.pkl")
    return model


def build_sequences(X_scaled, Y_scaled, lookback=LOOKBACK):
    x_sequences, y_targets = [], []
    for i in range(lookback, len(X_scaled)):
        x_sequences.append(X_scaled[i - lookback:i])
    x_sequences = np.array(x_sequences)
    y_targets = np.array(Y_scaled[lookback:])
    return x_sequences, y_targets


def train_cnn_bilstm(x_train, y_train, n_features):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(x_train)
    Y_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")
    print("Saved scaler_X.pkl and scaler_y.pkl")

    x_sequences, y_targets = build_sequences(X_scaled, Y_scaled)
    split_idx = int(0.8 * len(x_sequences))
    X_train, X_test = x_sequences[:split_idx], x_sequences[split_idx:]
    Y_train, Y_test = y_targets[:split_idx], y_targets[split_idx:]

    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(LOOKBACK, n_features)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Bidirectional(LSTM(50, activation="tanh", return_sequences=False)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

    predictions = model.predict(X_test)
    predictions_actual = scaler_y.inverse_transform(predictions)
    Y_test_actual = scaler_y.inverse_transform(Y_test)
    print("CNN-BiLSTM — RMSE:", mean_squared_error(Y_test_actual, predictions_actual))
    print("CNN-BiLSTM — MAE:", mean_absolute_error(Y_test_actual, predictions_actual))

    model.save("cnn_bilstm_model.keras")
    print("Saved cnn_bilstm_model.keras")
    return model


def main():
    data = load_and_clean()

    x = data[FEATURE_COLS]
    y = data[[TARGET_COL]]
    x = x.fillna(x.mean())
    y = y.fillna(y.mean())

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    train_xgboost(x_train, y_train, x_test, y_test)
    train_cnn_bilstm(x_train, y_train, n_features=len(FEATURE_COLS))

    print("\nAll artifacts saved. Copy these 4 files into your app folder:")
    print(" - xgboost_aqi_model.pkl")
    print(" - scaler_X.pkl")
    print(" - scaler_y.pkl")
    print(" - cnn_bilstm_model.keras")


if __name__ == "__main__":
    main()

"""
Robust training script for a hybrid quantum-LSTM stock predictor.

Features:
- Structured configuration via dataclass
- CLI arguments for flexibility
- Deterministic seeding option
- Detailed logging and input validation
- Safe saving of model, scaler, and quantum weights
- No personal names or identifiers included anywhere
"""

import os
import argparse
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import pennylane as qml
from pennylane import numpy as pnp
import joblib

# ----------------------------
# Configuration & Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("train_hybrid_q_lstm")

@dataclass
class TrainConfig:
    ticker: str = "SBUX"
    period: str = "5y"
    interval: str = "1d"
    model_dir: str = "saved_models"
    model_name: str = "hybrid_quantum_lstm_stock_model.keras"
    weights_name: str = "quantum_weights.npy"
    scaler_name: str = "scaler.pkl"
    window_size: int = 365
    horizon: int = 7
    test_split: float = 0.3
    n_qubits: int = 4
    n_layers: int = 2
    batch_size: int = 16
    epochs: int = 60
    learning_rate: float = 1e-3
    random_seed: int = 42
    pennylane_device: str = "default.qubit"


# ----------------------------
# Utilities
# ----------------------------
def set_global_seed(seed: int):
    """Set seeds for numpy, tensorflow, and pennylane for reproducibility."""
    try:
        import tensorflow as tf  # local import
        np.random.seed(seed)
        tf.random.set_seed(seed)
        pnp.random.seed(seed)
        logger.info("Global random seed set to %s", seed)
    except Exception as e:
        logger.warning("Failed to set global seed: %s", e)


# ----------------------------
# Data utilities
# ----------------------------
def download_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Download daily Close prices using yfinance, reindex to continuous daily calendar,
    forward-fill missing days, and return DataFrame with 'Close' column and Date index.
    """
    logger.info("Downloading %s data for period=%s interval=%s", ticker, period, interval)
    df = yf.download(ticker, period=period, interval=interval)[['Close']]

    if df.empty:
        raise ValueError(f"No data downloaded for ticker={ticker} period={period} interval={interval}")

    all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(all_dates)
    df.ffill(inplace=True)
    df.index.name = 'Date'
    logger.info("Downloaded data range: %s to %s (rows=%d)", df.index.min(), df.index.max(), len(df))
    return df


def create_dataset(series: np.ndarray, window_size: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 1D series to supervised dataset:
      - X: windows of length window_size
      - y: next horizon values
    """
    if series.ndim == 2 and series.shape[1] == 1:
        series = series.flatten()
    if series.ndim != 1:
        raise ValueError("Series must be 1D array or (n,1) array.")

    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i:(i + window_size)])
        y.append(series[(i + window_size):(i + window_size + horizon)])
    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    logger.info("Created dataset: X=%s, y=%s", X_arr.shape, y_arr.shape)
    return X_arr, y_arr


# ----------------------------
# Quantum utilities
# ----------------------------
def build_quantum_feature_map(n_qubits: int, n_layers: int, device_name: str = "default.qubit"):
    """Return a pennylane QNode and weight_shapes dict."""
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[l, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return circuit, weight_shapes


def reduce_window_to_qubits(window: np.ndarray, n_qubits: int) -> np.ndarray:
    """Downsample or pad a window to length n_qubits."""
    w = window.flatten()
    if len(w) == 0:
        raise ValueError("Window must contain at least one element.")
    if len(w) < n_qubits:
        pad_size = n_qubits - len(w)
        w = np.concatenate([w, np.repeat(w[-1], pad_size)])
    idx = np.linspace(0, len(w) - 1, n_qubits).astype(int)
    return w[idx]


def compute_quantum_features_for_dataset(X_windows: np.ndarray, circuit, weights, n_qubits: int) -> np.ndarray:
    """
    For each training window, compute quantum features using circuit and weights.
    Returns array shape (n_samples, n_qubits).
    """
    features = []
    for i in range(X_windows.shape[0]):
        window = X_windows[i]
        reduced = reduce_window_to_qubits(window, n_qubits)
        angles = reduced * np.pi
        q_out = circuit(pnp.array(angles), weights)
        features.append(np.array(q_out, dtype=np.float64))
    features_arr = np.array(features, dtype=np.float64)
    logger.info("Computed quantum features: %s", features_arr.shape)
    return features_arr


# ----------------------------
# LSTM model
# ----------------------------
def build_lstm_model(input_shape: Tuple[int, int], learning_rate: float = 1e-3, horizon: int = 7) -> Sequential:
    """Build and compile an LSTM model."""
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(horizon)
    ])
    model.compile(optimizer=Adam(learning_rate), loss='mean_squared_error')
    logger.info("Compiled LSTM model with input_shape=%s horizon=%d", input_shape, horizon)
    return model


# ----------------------------
# Training pipeline
# ----------------------------
def train_pipeline(cfg: TrainConfig):
    """Main pipeline to download data, build features, train and save the model/scaler/weights."""
    os.makedirs(cfg.model_dir, exist_ok=True)
    set_global_seed(cfg.random_seed)

    df = download_stock_data(cfg.ticker, cfg.period, cfg.interval)

    # Save raw CSV
    csv_path = os.path.join(cfg.model_dir, f"{cfg.ticker}.csv")
    df.to_csv(csv_path, index=True)
    logger.info("Saved raw data to: %s", csv_path)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)  # shape (n_rows, 1)

    # Split into train/test by chronological order
    n_total = len(scaled_data)
    test_start = int(np.floor((1 - cfg.test_split) * n_total))
    train_data = scaled_data[:test_start].flatten()
    test_data = scaled_data[test_start:].flatten()
    logger.info("Train rows: %d, Test rows: %d", len(train_data), len(test_data))

    min_required = cfg.window_size + cfg.horizon
    if len(train_data) < min_required or len(test_data) < min_required:
        raise ValueError(f"Not enough data. Each split must have at least {min_required} rows.")

    X_train_raw, y_train = create_dataset(train_data, cfg.window_size, cfg.horizon)
    X_test_raw, y_test = create_dataset(test_data, cfg.window_size, cfg.horizon)

    circuit, weight_shapes = build_quantum_feature_map(cfg.n_qubits, cfg.n_layers, cfg.pennylane_device)
    q_weights = pnp.random.uniform(0, np.pi, size=weight_shapes['weights'], requires_grad=False)
    logger.info("Initialized quantum weights with shape %s", weight_shapes['weights'])

    logger.info("Computing quantum features for training and test sets...")
    X_train_q = compute_quantum_features_for_dataset(X_train_raw, circuit, q_weights, cfg.n_qubits)
    X_test_q = compute_quantum_features_for_dataset(X_test_raw, circuit, q_weights, cfg.n_qubits)

    X_train = X_train_q.reshape((X_train_q.shape[0], 1, cfg.n_qubits))
    X_test = X_test_q.reshape((X_test_q.shape[0], 1, cfg.n_qubits))

    model = build_lstm_model((1, cfg.n_qubits), learning_rate=cfg.learning_rate, horizon=cfg.horizon)
    tb_callback = TensorBoard(log_dir=os.path.join(cfg.model_dir, "logs"), histogram_freq=1, write_graph=True)

    logger.info("Starting training for %d epochs (batch_size=%d)...", cfg.epochs, cfg.batch_size)
    model.fit(X_train, y_train,
              epochs=cfg.epochs,
              batch_size=cfg.batch_size,
              validation_data=(X_test, y_test),
              callbacks=[tb_callback])

    model_path = os.path.join(cfg.model_dir, cfg.model_name)
    scaler_path = os.path.join(cfg.model_dir, cfg.scaler_name)
    weights_path = os.path.join(cfg.model_dir, cfg.weights_name)

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    np.save(weights_path, np.array(q_weights))

    logger.info("Saved model to %s", model_path)
    logger.info("Saved scaler to %s", scaler_path)
    logger.info("Saved quantum weights to %s", weights_path)
    logger.info("Training complete and artifacts saved successfully.")


# ----------------------------
# CLI entrypoint
# ----------------------------
def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train hybrid Quantum-LSTM stock model")
    parser.add_argument("--ticker", type=str, default=TrainConfig.ticker, help="Ticker symbol (default: SBUX)")
    parser.add_argument("--period", type=str, default=TrainConfig.period, help="yfinance period (default: 5y)")
    parser.add_argument("--window_size", type=int, default=TrainConfig.window_size, help="Window size for history")
    parser.add_argument("--horizon", type=int, default=TrainConfig.horizon, help="Prediction horizon (days)")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size, help="Batch size")
    parser.add_argument("--n_qubits", type=int, default=TrainConfig.n_qubits, help="Number of qubits for quantum feature map")
    parser.add_argument("--n_layers", type=int, default=TrainConfig.n_layers, help="Number of quantum layers")
    parser.add_argument("--seed", type=int, default=TrainConfig.random_seed, help="Random seed")
    parser.add_argument("--model_dir", type=str, default=TrainConfig.model_dir, help="Directory to save model artifacts")
    args = parser.parse_args()

    cfg = TrainConfig(
        ticker=args.ticker,
        period=args.period,
        window_size=args.window_size,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        random_seed=args.seed,
        model_dir=args.model_dir
    )
    return cfg


if __name__ == "__main__":
    try:
        cfg = parse_args()
        logger.info("Starting training with config: %s", cfg)
        train_pipeline(cfg)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        raise

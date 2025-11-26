import os
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional

import joblib
import numpy as np
import tensorflow as tf
import pennylane as qml
from pennylane import numpy as pnp
from numpy.typing import NDArray

# ----------------------------
# Configuration & Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("hybrid_q_lstm")

@dataclass
class ModelConfig:
    model_dir: str = "saved_models"
    model_name: str = "hybrid_quantum_lstm_stock_model.keras"
    scaler_name: str = "scaler.pkl"
    weights_name: str = "quantum_weights.npy"
    n_qubits: int = 4
    n_layers: int = 2
    random_seed: Optional[int] = 42
    pennylane_device: str = "default.qubit"  # change to another device if available


# Default config instance
CFG = ModelConfig()


# ----------------------------
# Utilities
# ----------------------------
def set_global_seed(seed: Optional[int]) -> None:
    """
    Set seeds for numpy, tensorflow and pennylane. If seed is None, randomness is unchanged.
    """
    if seed is None:
        logger.debug("No seed provided; skipping deterministic seeding.")
        return
    try:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        pnp.random.seed(seed)
        logger.info("Random seed set to %s for numpy, tensorflow, and pennylane.", seed)
    except Exception as e:
        logger.warning("Failed to set seeds: %s", e)


# ----------------------------
# Quantum Circuit & Helpers
# ----------------------------
def build_quantum_feature_map(n_qubits: int, n_layers: int, device_name: str = "default.qubit"):
    """
    Build and return a Pennylane QNode (callable) that maps classical inputs to quantum expectations.
    The returned function has signature: circuit(inputs_angles, weights) -> array of expectations (length n_qubits)
    """
    if n_qubits <= 0 or n_layers <= 0:
        raise ValueError("n_qubits and n_layers must be positive integers.")

    try:
        dev = qml.device(device_name, wires=n_qubits)
    except Exception as e:
        logger.warning("Could not initialize device '%s'. Falling back to 'default.qubit'. Error: %s",
                       device_name, e)
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs: NDArray[np.floating_like], weights: NDArray[np.floating_like]) -> NDArray[np.floating_like]:
        """
        inputs: length n_qubits -- rotation angles
        weights: shape (n_layers, n_qubits)
        returns: expectation values (Z) for each qubit
        """
        if inputs.shape[0] != n_qubits:
            raise ValueError(f"Inputs length {inputs.shape[0]} does not match n_qubits {n_qubits}.")
        if weights.shape != (n_layers, n_qubits):
            raise ValueError(f"Weights shape {weights.shape} must be ({n_layers},{n_qubits}).")

        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)

        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[l, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        return pnp.array([qml.expval(qml.PauliZ(i)) for i in range(n_qubits)])

    return circuit


def reduce_window_to_qubits(window: NDArray[np.floating_like], n_qubits: int) -> NDArray[np.floating_like]:
    """
    Reduce a numeric window (1D array) to a length equal to n_qubits by sampling indices evenly.
    """
    if window.ndim != 1:
        window = window.flatten()
    if len(window) == 0:
        raise ValueError("Window must contain at least one element.")
    if n_qubits <= 0:
        raise ValueError("n_qubits must be > 0.")

    if len(window) < n_qubits:
        pad_size = n_qubits - len(window)
        padded = np.concatenate([window, np.repeat(window[-1], pad_size)])
        indices = np.linspace(0, len(padded) - 1, n_qubits).astype(int)
        return padded[indices]
    else:
        idx = np.linspace(0, len(window) - 1, n_qubits).astype(int)
        return window[idx]


def quantum_preprocess(window: NDArray[np.floating_like],
                        circuit,
                        weights: NDArray[np.floating_like],
                        scaler,
                        n_qubits: int) -> NDArray[np.floating_like]:
    """
    Scale the window using scaler, reduce length to n_qubits, convert to angles, feed to QNode and return feature vector.
    Output dtype is float64 for compatibility with downstream TensorFlow model.
    """
    arr = np.asarray(window).astype(np.float64).flatten()
    if arr.size == 0:
        raise ValueError("Input window for quantum_preprocess is empty.")

    scaled = scaler.transform(arr.reshape(-1, 1)).flatten()
    reduced = reduce_window_to_qubits(scaled, n_qubits)
    angles = reduced * np.pi
    angles_p = pnp.array(angles)
    weights_p = pnp.array(weights)
    q_out = circuit(angles_p, weights_p)
    return np.asarray(q_out, dtype=np.float64)


# ----------------------------
# Load / Save Helpers
# ----------------------------
def default_paths(cfg: ModelConfig) -> Tuple[str, str, str]:
    model_path = os.path.join(cfg.model_dir, cfg.model_name)
    scaler_path = os.path.join(cfg.model_dir, cfg.scaler_name)
    weights_path = os.path.join(cfg.model_dir, cfg.weights_name)
    return model_path, scaler_path, weights_path


def load_model_and_scaler(cfg: ModelConfig = CFG) -> Tuple[tf.keras.Model, object, NDArray[np.floating_like]]:
    """
    Load the trained TensorFlow model, scaler (joblib), and quantum weights (npy).
    If quantum weights are missing, create a random initialization and return it (and log a warning).
    """
    set_global_seed(cfg.random_seed)

    model_path, scaler_path, weights_path = default_paths(cfg)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")

    logger.info("Loading TensorFlow model from: %s", model_path)
    model = tf.keras.models.load_model(model_path)
    logger.info("Loading scaler from: %s", scaler_path)
    scaler = joblib.load(scaler_path)

    if os.path.exists(weights_path):
        logger.info("Loading quantum weights from: %s", weights_path)
        q_weights = np.load(weights_path, allow_pickle=True)
        q_weights = pnp.array(np.asarray(q_weights, dtype=np.float64))
    else:
        logger.warning("Quantum weights not found at %s. Initializing random quantum weights.", weights_path)
        q_weights = pnp.random.uniform(0, np.pi, size=(cfg.n_layers, cfg.n_qubits))

    if q_weights.shape != (cfg.n_layers, cfg.n_qubits):
        logger.info("Quantum weights shape %s doesn't match config (%s,%s). Attempting to reshape or reinitialize.",
                    q_weights.shape, cfg.n_layers, cfg.n_qubits)
        try:
            q_weights = pnp.array(q_weights).reshape((cfg.n_layers, cfg.n_qubits))
        except Exception:
            q_weights = pnp.random.uniform(0, np.pi, size=(cfg.n_layers, cfg.n_qubits))

    return model, scaler, q_weights


def save_quantum_weights(weights: NDArray[np.floating_like], cfg: ModelConfig = CFG) -> str:
    """
    Save provided quantum weights to disk (numpy .npy). Returns saved path.
    """
    _, _, weights_path = default_paths(cfg)
    os.makedirs(cfg.model_dir, exist_ok=True)
    np.save(weights_path, np.asarray(weights))
    logger.info("Quantum weights saved to %s", weights_path)
    return weights_path


# ----------------------------
# Prediction Logic
# ----------------------------
def predict_n_days(model: tf.keras.Model,
                   scaler,
                   q_weights: NDArray[np.floating_like],
                   recent_data: List[float],
                   window_size: int,
                   n_days: int,
                   cfg: ModelConfig = CFG) -> List[float]:
    """
    Predict next n_days using a hybrid approach:
    - Preprocess the last `window_size` values into quantum features
    - Feed those features into the loaded TF model to get scaled predictions
    - Inverse-transform predictions to original scale and append to recent_data for multi-step forecasting

    Returns a list of floats (length n_days) representing predicted Close prices.
    """
    if n_days <= 0:
        logger.info("n_days <= 0; returning empty list.")
        return []
    if window_size <= 0:
        raise ValueError("window_size must be > 0.")
    if len(recent_data) < 1:
        raise ValueError("recent_data must contain at least one historical value.")

    set_global_seed(cfg.random_seed)

    circuit = build_quantum_feature_map(cfg.n_qubits, cfg.n_layers, cfg.pennylane_device)

    recent_list = list(map(float, recent_data))
    preds_scaled = []

    for step in range(n_days):
        window = np.array(recent_list[-window_size:], dtype=np.float64)
        q_features = quantum_preprocess(window, circuit, q_weights, scaler, cfg.n_qubits)
        X_input = q_features.reshape((1, 1, cfg.n_qubits))

        try:
            next_scaled = model.predict(X_input, verbose=0).flatten()[0]
        except Exception as e:
            logger.error("Error during model.predict at step %s: %s", step, e)
            raise

        preds_scaled.append(next_scaled)
        next_unscaled = scaler.inverse_transform(np.array([[next_scaled]]))[0, 0]
        recent_list.append(float(next_unscaled))

        logger.debug("Step %d: scaled=%s, unscaled=%s", step + 1, next_scaled, next_unscaled)

    preds_scaled_arr = np.array(preds_scaled).reshape(-1, 1)
    preds_unscaled = scaler.inverse_transform(preds_scaled_arr).flatten().tolist()
    return [float(x) for x in preds_unscaled]


# ----------------------------
# Example helper for quick local test (not a full unit test suite)
# ----------------------------
def quick_local_test(cfg: ModelConfig = CFG) -> None:
    """
    Smoke-test: attempt to load model/scaler and run predict_n_days with dummy recent data.
    """
    try:
        model, scaler, q_weights = load_model_and_scaler(cfg)
    except Exception as e:
        logger.error("Quick local test failed during load: %s", e)
        return

    try:
        if hasattr(scaler, "mean_"):
            base = float(np.mean(scaler.mean_))
            recent = [base] * max(4, cfg.n_qubits)
        else:
            recent = [1.0] * max(4, cfg.n_qubits)
    except Exception:
        recent = [1.0] * max(4, cfg.n_qubits)

    try:
        preds = predict_n_days(model=model, scaler=scaler, q_weights=q_weights,
                               recent_data=recent, window_size=cfg.n_qubits, n_days=3, cfg=cfg)
        logger.info("Quick local test predictions: %s", preds)
    except Exception as e:
        logger.error("Quick local test failed during prediction: %s", e)


# ----------------------------
# If executed as script, run quick test
# ----------------------------
if __name__ == "__main__":
    logger.info("Running module as script for a quick smoke test.")
    quick_local_test(CFG)

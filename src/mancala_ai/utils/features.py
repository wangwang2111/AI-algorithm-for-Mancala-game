# src/mancala_ai/utils/features.py
import numpy as np

def encode_state(state: dict):
    """
    Returns a Torch tensor if torch is available, else a NumPy array (1, F).
    This avoids importing torch (and thus cuDNN) at module import time.
    """
    pits   = np.array(state["pits"]).reshape(-1)
    stores = np.array(state["stores"])
    cur    = np.array([state["current_player"]])
    arr = np.concatenate([pits, stores, cur]).astype(np.float32).reshape(1, -1)

    try:
        import torch  # lazy import to avoid cuDNN requirement if torch isn't available
        return torch.from_numpy(arr)  # shape (1, F)
    except Exception:
        # torch not installed/available -> return numpy, callers must handle fallback
        return arr

# src/mancala_ai/agents/dqn.py
import os
from typing import Dict, List
import numpy as np

from mancala_ai.engine.core import legal_actions as legal_actions_state

REGISTRY_DIR = os.getenv("MODEL_REGISTRY", "model_registry/latest")
WEIGHTS_PATH = os.path.join(REGISTRY_DIR, "policy.pt")

# --- lazy torch import so inference works even without CUDA/cuDNN ---
_TORCH = None
def _get_torch():
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    try:
        import torch  # noqa
        _TORCH = torch
    except Exception:
        _TORCH = None
    return _TORCH

def _state_to_board(state: Dict) -> Dict[str, List[int]]:
    """Convert engine 'state' to board dict sometimes expected by training utils."""
    return {
        "player_1": state["pits"][0][:] + [state["stores"][0]],
        "player_2": state["pits"][1][:] + [state["stores"][1]],
    }

class DQNWrapper:
    """
    Thin wrapper around mancala_ai.training.dqn.DQNAgent with robust fallbacks.
    Caches the agent and weights on first use.
    """
    def __init__(self, model_path: str = WEIGHTS_PATH):
        self.model_path = model_path
        self.agent = None
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        if self._loaded and self.agent is not None:
            return True
        torch = _get_torch()
        if torch is None:
            return False
        try:
            from mancala_ai.training.dqn import DQNAgent  # lazy import to avoid hard dep at import time
        except Exception:
            return False

        # Construct agent; your suggested default used state_shape=(29,)
        try:
            self.agent = DQNAgent(state_shape=(29,))
        except TypeError:
            # if signature differs, try default constructor
            self.agent = DQNAgent()

        # Load weights if present (prefer CPU when serving)
        try:
            if os.path.exists(self.model_path):
                self.agent.load_model(self.model_path)
        except Exception:
            # proceed with randomly-initialized agent if load fails
            pass

        self._loaded = True
        return True

    def choose(self, state: Dict) -> int:
        acts = legal_actions_state(state)
        if not acts:
            return 0
        if not self._ensure_loaded() or self.agent is None:
            # No torch/agent -> safe fallback
            return int(acts[0])

        board = _state_to_board(state)
        player = int(state.get("current_player", 0))
        try:
            # Preferred codepaths
            preprocess = getattr(self.agent, "preprocess_state", None)
            get_action = getattr(self.agent, "get_action", None)
            if callable(preprocess) and callable(get_action):
                s_feat = preprocess(board, player)
                action = int(get_action(s_feat, acts, greedy=True))
            else:
                # Fallback to q-value head if exposed
                forward = getattr(self.agent, "forward", None) or getattr(self.agent, "predict", None)
                if callable(forward):
                    q = np.array(forward(board, player)).reshape(-1)  # type: ignore
                    masked = np.full(6, -1e9, dtype=float)
                    masked[acts] = q[acts]
                    action = int(np.argmax(masked))
                else:
                    action = int(acts[0])
        except Exception:
            action = int(acts[0])

        if action not in acts:
            action = int(acts[0])
        return action

# Module-level singleton for serving
_DQN = None

def dqn_action(state: Dict) -> int:
    global _DQN
    if _DQN is None:
        _DQN = DQNWrapper(WEIGHTS_PATH)
    return _DQN.choose(state)

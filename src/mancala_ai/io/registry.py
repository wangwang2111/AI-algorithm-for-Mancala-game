# src/mancala_ai/io/registry.py
import os, json
from typing import Dict
from mancala_ai.engine.core import legal_actions as legal_actions_state

# ---------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------
REGISTRY_DIR = os.getenv("MODEL_REGISTRY", "model_registry/latest")
WEIGHTS_PATH = os.path.join(REGISTRY_DIR, "policy.pt")
META_PATH    = os.path.join(REGISTRY_DIR, "meta.json")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def current_meta() -> Dict:
    if os.path.exists(META_PATH):
        try:
            return json.load(open(META_PATH))
        except Exception:
            print(f"Failed to read {META_PATH}")
            pass
    return {"version": f"unknown, {META_PATH} not found"}

# ---------------------------------------------------------------------
# Agent adapters
# ---------------------------------------------------------------------
def _dqn_action(state: Dict) -> int:
    acts = legal_actions_state(state)
    if not acts:
        return 0
    try:
        # New home for the DQN; wrapper caches internally
        from mancala_ai.agents.dqn import dqn_action as _fn
        return int(_fn(state))
    except Exception:
        return int(acts[0])

def _minimax_action(state: Dict) -> int:
    acts = legal_actions_state(state)
    if not acts:
        return 0
    try:
        from mancala_ai.agents.minimax import simple_minimax as _fn  # type: ignore
    except Exception:
        try:
            from mancala_ai.agents.minimax import choose_move as _fn  # type: ignore
        except Exception:
            return int(acts[0])
    try:
        return int(_fn(state))
    except Exception:
        return int(acts[0])

def _alphabeta_action(state: Dict) -> int:
    acts = legal_actions_state(state)
    if not acts:
        return 0
    try:
        from mancala_ai.agents.alpha_beta import minimax_alpha_beta as _fn  # type: ignore
    except Exception:
        try:
            from mancala_ai.agents.alpha_beta import choose_move as _fn  # type: ignore
        except Exception:
            return int(acts[0])
    try:
        return int(_fn(state))
    except Exception:
        return int(acts[0])

def _mcts_action(state: Dict) -> int:
    acts = legal_actions_state(state)
    if not acts:
        return 0
    try:
        from mancala_ai.agents.MCTS import mcts_decide as _fn  # type: ignore
    except Exception:
        try:
            from mancala_ai.agents.mcts import choose_move as _fn  # type: ignore
        except Exception:
            return int(acts[0])
    try:
        return int(_fn(state))
    except Exception:
        return int(acts[0])

def _advanced_heuristic_action(state: Dict) -> int:
    """Advanced heuristic minimax from your module."""
    acts = legal_actions_state(state)
    if not acts:
        return 0
    try:
        from mancala_ai.agents.advanced_heuristic import advanced_heuristic_minimax as _fn
    except Exception:
        return int(acts[0])
    try:
        return int(_fn(state))
    except Exception:
        return int(acts[0])

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def pick_action(state: Dict, agent: str) -> int:
    agent = (agent or "dqn").lower()
    if agent == "dqn":
        return _dqn_action(state)
    elif agent == "minimax":
        return _minimax_action(state)
    elif agent in ("alpha_beta", "alphabeta", "alpha-beta"):
        return _alphabeta_action(state)
    elif agent == "mcts":
        return _mcts_action(state)
    elif agent in ("advanced", "advanced_heuristic", "adv", "ah"):
        return _advanced_heuristic_action(state)
    elif agent == "random":
        acts = legal_actions_state(state)
        return int(acts[0]) if acts else 0
    else:
        acts = legal_actions_state(state)
        return int(acts[0]) if acts else 0

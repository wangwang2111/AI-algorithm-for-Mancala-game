# Simple Minimax (STATE-BASED, no alpha-beta)
from __future__ import annotations
import copy, math
from typing import Tuple, Optional, List, Dict
from mancala_ai.engine.core import legal_actions, step, heuristic_evaluate

def _score_for(state: Dict, maximizing_for_idx: int) -> float:
    """
    heuristic_evaluate from a FIXED player's perspective by setting state['current_player']
    before calling engine.heuristic_evaluate(state).
    """
    s = copy.deepcopy(state)
    s["current_player"] = maximizing_for_idx
    return float(heuristic_evaluate(s))

def _minimax(
    state: Dict,
    depth: int,
    maximizing_for_idx: int,
) -> Tuple[float, Optional[int]]:
    # Terminal or leaf
    if depth == 0 or sum(state["pits"][0]) == 0 or sum(state["pits"][1]) == 0:
        return _score_for(state, maximizing_for_idx), None

    moves = legal_actions(state)
    if not moves:
        return _score_for(state, maximizing_for_idx), None

    is_max = (state["current_player"] == maximizing_for_idx)
    best_move = moves[0]

    if is_max:
        best = -math.inf
        for mv in moves:
            ns, _, _ = step(state, mv)
            # If extra turn occurred, ns['current_player'] == state['current_player']
            # → don't reduce depth (same ply); otherwise reduce by 1.
            reduce = 0 if ns["current_player"] == state["current_player"] else 1
            score, _ = _minimax(ns, depth - reduce, maximizing_for_idx)
            if score > best:
                best, best_move = score, mv
        return best, best_move
    else:
        best = math.inf
        for mv in moves:
            ns, _, _ = step(state, mv)
            reduce = 0 if ns["current_player"] == state["current_player"] else 1
            score, _ = _minimax(ns, depth - reduce, maximizing_for_idx)
            if score < best:
                best, best_move = score, mv
        return best, best_move

# Public helpers -------------------------------------------------------

def choose_move(state: Dict, depth: int = 5) -> int:
    """Return best action index for the current state using plain minimax (no alpha–beta)."""
    _, move = _minimax(state, depth=depth, maximizing_for_idx=state["current_player"])
    if move is None:
        acts = legal_actions(state)
        return int(acts[0]) if acts else 0
    return int(move)

# Backwards-compatible export name (used by registry)
def simple_minimax(state: Dict, depth: int = 5) -> int:
    return choose_move(state, depth=depth)

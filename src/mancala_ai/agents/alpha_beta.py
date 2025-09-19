from __future__ import annotations
import copy, math
from typing import Tuple, Optional, List, Dict
from mancala_ai.engine.core import legal_actions, step, heuristic_evaluate

# ------------ scoring & ordering helpers (fixed perspective) ---------------

def _score_for(state: Dict, maximizing_for_idx: int) -> float:
    """
    heuristic_evaluate a position from a FIXED player's perspective by setting
    state['current_player'] prior to calling engine.heuristic_evaluate(state).
    """
    s = copy.deepcopy(state)
    s["current_player"] = maximizing_for_idx
    return float(heuristic_evaluate(s))

def _order_moves(state: Dict, maximizing_for_idx: int) -> List[int]:
    """
    Simple one-ply lookahead ordering (improves pruning).
    """
    acts = legal_actions(state)
    scored = []
    for a in acts:
        ns, _, _ = step(state, a)
        scored.append((_score_for(ns, maximizing_for_idx), a))
    scored.sort(reverse=True)  # higher first
    return [a for _, a in scored]

# ----------------------------- alpha-beta core ------------------------------

def _search(
    state: Dict,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_for_idx: int,
) -> Tuple[float, Optional[int]]:
    # Leaf or terminal (either side has no stones)
    if depth == 0 or sum(state["pits"][0]) == 0 or sum(state["pits"][1]) == 0:
        return _score_for(state, maximizing_for_idx), None

    moves = legal_actions(state)
    if not moves:
        return _score_for(state, maximizing_for_idx), None

    is_max = (state["current_player"] == maximizing_for_idx)
    ordered = _order_moves(state, maximizing_for_idx)
    best_move = ordered[0]

    if is_max:
        best = -math.inf
        a = alpha
        for mv in ordered:
            ns, _, _ = step(state, mv)
            # If extra-turn occurs, current_player stays same → don't reduce depth
            reduce = 0 if ns["current_player"] == state["current_player"] else 1
            score, _ = _search(ns, depth - reduce, a, beta, maximizing_for_idx)
            if score > best:
                best, best_move = score, mv
            a = max(a, score)
            if beta <= a:
                break
        return best, best_move
    else:
        best = math.inf
        b = beta
        for mv in ordered:
            ns, _, _ = step(state, mv)
            reduce = 0 if ns["current_player"] == state["current_player"] else 1
            score, _ = _search(ns, depth - reduce, alpha, b, maximizing_for_idx)
            if score < best:
                best, best_move = score, mv
            b = min(b, score)
            if b <= alpha:
                break
        return best, best_move

# ------------------------------- public API ---------------------------------

def choose_move(state: Dict, depth: int = 5) -> int:
    """
    Return best action index for the current state using alpha-beta minimax.
    Perspective is the state's current player.
    """
    _, move = _search(state, depth=depth, alpha=-1e9, beta=1e9,
                      maximizing_for_idx=state["current_player"])
    if move is None:
        acts = legal_actions(state)
        return int(acts[0]) if acts else 0
    return int(move)

# Registry-compatible name
def minimax_alpha_beta(state: Dict, depth: int = 5) -> int:
    """
    Back-compat export expected by the registry. Returns just the MOVE (int).
    """
    return choose_move(state, depth=depth)

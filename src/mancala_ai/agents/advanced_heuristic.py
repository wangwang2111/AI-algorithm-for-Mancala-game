# Advanced heuristic minimax (state-based, uses custom heuristic instead of engine.evaluate)
from __future__ import annotations
import copy, math
from typing import Tuple, Optional, List
from mancala_ai.engine.core import legal_actions, step

# ----------------------- utils: state <-> board / keys ------------------------

def _player_key(idx: int) -> str:
    return "player_1" if idx == 0 else "player_2"

def _board_from_state(state: dict) -> dict:
    return {
        "player_1": state["pits"][0][:] + [state["stores"][0]],
        "player_2": state["pits"][1][:] + [state["stores"][1]],
    }

# --------------------------- your custom heuristic ----------------------------
# IMPORTANT: higher return value = better for 'maximizing_for'
def _heuristic_move_value_board(board: dict, move: int, maximizing_for: str) -> float:
    player_pits = board[maximizing_for]
    opponent = "player_1" if maximizing_for == "player_2" else "player_2"
    opponent_pits = board[opponent]
    stones = player_pits[move]
    value = 0.0

    # 1) Basic Movement Analysis (simplified ring; DOES NOT skip opp. store by design)
    landing_pit = (move + stones) % 14

    # 2) Immediate Rewards
    # - Extra turn if land in own store
    if landing_pit == 6:
        value += 22.0

    # - Capture potential
    if landing_pit < 6 and player_pits[landing_pit] == 0:
        captured = opponent_pits[5 - landing_pit]
        value += captured * 2.0

    # 3) Positional Advantage
    position_weights = [0.5, 0.8, 1.2, 1.5, 1.2, 0.8]
    value += position_weights[move] * 2.0

    # 4) Future Game State (danger zones)
    danger_zones = [0, 1, 5]
    for i in range(stones):
        pit = (move + i + 1) % 14
        if pit in danger_zones and pit < 6:
            value += 0.3

    # 5) Defensive Considerations (block opp captures)
    for opp_move in range(6):
        if opponent_pits[opp_move] == (13 - opp_move):
            value += 1.5

    # 6) Progressive Game Phase
    total_seeds = sum(player_pits) + sum(opponent_pits)
    game_phase = 1 - (total_seeds / 96.0)
    value += (player_pits[6] - opponent_pits[6]) * game_phase * 2.0
    value += (sum(player_pits[:6]) - sum(opponent_pits[:6])) * (1 - game_phase) * 1.5

    # 7) Mobility
    future_moves = sum(1 for i in range(6) if player_pits[i] > (6 - i))
    value += future_moves * 0.8

    # 8) Denial Strategy (opp extra turn threats)
    for i in range(6):
        if opponent_pits[i] == (13 - i):
            value -= 2.0

    # 9) Seed Conservation (avoid emptying non-scoring pits)
    if player_pits[move] == stones and landing_pit != 6:
        value -= 2.0

    # 10) Tempo Control
    if stones > 10:
        value += 1.2

    # Return "higher is better"
    return float(value)

def _position_heuristic_for(state: dict, maximizing_for_idx: int) -> float:
    """
    Position score for a fixed player using the custom move heuristic.
    We approximate position value by the best immediate move value available to that player.
    If no move exists (empty side), fall back to store difference.
    """
    board = _board_from_state(state)
    me = _player_key(maximizing_for_idx)
    opp = _player_key(1 - maximizing_for_idx)

    # moves available to 'me' (regardless of whose turn it is)
    my_moves = [i for i in range(6) if board[me][i] > 0]
    if not my_moves:
        return float(board[me][6] - board[opp][6])

    best = -math.inf
    for mv in my_moves:
        sc = _heuristic_move_value_board(board, mv, me)
        if sc > best:
            best = sc
    return best

# ------------------------------- move ordering --------------------------------

def _order_moves_for_side_to_play(state: dict) -> List[int]:
    """
    Order moves for the CURRENT side to move, scored using the custom heuristic
    from the mover's own perspective (not the global maximizing player).
    """
    acts = legal_actions(state)
    if not acts:
        return []

    board = _board_from_state(state)
    mover_key = _player_key(state["current_player"])

    scored = [(_heuristic_move_value_board(board, a, mover_key), a) for a in acts]
    scored.sort(reverse=True)
    return [a for _, a in scored]

# --------------------------------- minimax ------------------------------------

def _minimax(
    state: dict,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_for_idx: int,
) -> Tuple[float, Optional[int]]:
    # Terminal/leaf
    if depth == 0 or sum(state["pits"][0]) == 0 or sum(state["pits"][1]) == 0:
        return _position_heuristic_for(state, maximizing_for_idx), None

    moves = legal_actions(state)
    if not moves:
        return _position_heuristic_for(state, maximizing_for_idx), None

    is_max = (state["current_player"] == maximizing_for_idx)
    ordered = _order_moves_for_side_to_play(state)
    best_move = ordered[0]

    if is_max:
        best = -math.inf
        a = alpha
        for mv in ordered:
            ns, _, _ = step(state, mv)
            # don't reduce depth on extra turns (same side continues)
            reduce = 0 if ns["current_player"] == state["current_player"] else 1
            score, _ = _minimax(ns, depth - reduce, a, beta, maximizing_for_idx)
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
            score, _ = _minimax(ns, depth - reduce, alpha, b, maximizing_for_idx)
            if score < best:
                best, best_move = score, mv
            b = min(b, score)
            if b <= alpha:
                break
        return best, best_move

# ------------------------------- public API -----------------------------------

def choose_move_advanced(state: dict, depth: int = 5) -> int:
    """
    Best move for the current state using alpha-beta + your custom heuristic
    (no call to engine.core.evaluate).
    """
    _, move = _minimax(state, depth=depth, alpha=-1e9, beta=1e9,
                       maximizing_for_idx=state["current_player"])
    if move is None:
        acts = legal_actions(state)
        return int(acts[0]) if acts else 0
    return int(move)

# Backwards-compatible name (used by registry)
def advanced_heuristic_minimax(state: dict, depth: int = 5) -> int:
    return choose_move_advanced(state, depth=depth)

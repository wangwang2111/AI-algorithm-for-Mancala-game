# src/mancala_ai/engine/core.py
# Mancala core engine: state, rules, and evaluation
# State format (engine-wide standard):
# {
#   "pits":   [[int]*6, [int]*6],  # player_1 row=0, player_2 row=1
#   "stores": [int, int],          # stores[0] for player_1, stores[1] for player_2
#   "current_player": 0 | 1        # 0 = player_1, 1 = player_2
# }

from __future__ import annotations
import copy
from typing import Dict, List, Tuple

NUM_PITS = 6
INITIAL_STONES = 4

# --------- Helpers to bridge "board" (old dict) and "state" (new dict) ---------

def _board_from_state(state: Dict) -> Dict[str, List[int]]:
    """Convert state -> board with player keys (for internal move/eval helpers)."""
    return {
        "player_1": state["pits"][0][:] + [state["stores"][0]],
        "player_2": state["pits"][1][:] + [state["stores"][1]],
    }

def _state_from_board(board: Dict[str, List[int]], current_player: int) -> Dict:
    """Convert board -> state, preserving whose turn it is."""
    return {
        "pits":   [board["player_1"][:NUM_PITS], board["player_2"][:NUM_PITS]],
        "stores": [board["player_1"][6],         board["player_2"][6]],
        "current_player": current_player,
    }

def _player_key(player_idx: int) -> str:
    return "player_1" if player_idx == 0 else "player_2"

# --------------------------------- Public API ---------------------------------

def new_game() -> Dict:
    """Create a fresh Mancala state."""
    return {
        "pits":   [[INITIAL_STONES] * NUM_PITS, [INITIAL_STONES] * NUM_PITS],
        "stores": [0, 0],
        "current_player": 0,
    }

def legal_actions(state: Dict) -> List[int]:
    """All legal pit indices (0..5) for the current player."""
    row = state["current_player"]
    return [i for i in range(NUM_PITS) if state["pits"][row][i] > 0]

def step(state: Dict, action: int) -> Tuple[Dict, float, bool]:
    """
    Apply one move for state['current_player'] from pit index `action`.
    Returns: (next_state, reward, done)
      - reward: 0 for non-terminal; at terminal = store_diff from mover's perspective.
    """
    mover_idx = state["current_player"]
    mover_key = _player_key(mover_idx)
    opp_key   = _player_key(1 - mover_idx)

    board = _board_from_state(state)
    new_board, extra_turn = _make_move_board(board, mover_key, action)
    done = _is_terminal_board(new_board)

    # Reward only at terminal (store difference from mover's perspective).
    reward = float(new_board[mover_key][6] - new_board[opp_key][6]) if done else 0.0

    next_player = mover_idx if (extra_turn and not done) else 1 - mover_idx
    next_state = _state_from_board(new_board, next_player)
    return next_state, reward, done

# ------------------------------ Rule primitives -------------------------------

def _is_terminal_board(board: Dict[str, List[int]]) -> bool:
    """Terminal when either side has no stones in their 6 pits."""
    return sum(board["player_1"][:6]) == 0 or sum(board["player_2"][:6]) == 0

def _make_move_board(board: Dict[str, List[int]], player: str, pit: int) -> Tuple[Dict, bool]:
    """
    Internal move executor on 'board' format.
    Returns: (new_board, extra_turn)
    Implements:
      - sowing (skipping opponent's store),
      - capture,
      - extra turn,
      - end-of-game sweeping.
    """
    new_board = copy.deepcopy(board)
    seeds = new_board[player][pit]
    new_board[player][pit] = 0

    pits = ["player_1", "player_2"]
    opponent = "player_2" if player == "player_1" else "player_1"

    # Flat 14-pit mapping: p1 pits(0–5), p1 store(6), p2 pits(7–12), p2 store(13)
    pos_map = [("player_1", i) for i in range(6)] + [("player_1", 6)] + \
              [("player_2", i) for i in range(6)] + [("player_2", 6)]

    flat_index = pits.index(player) * 7 + pit  # starting index (pit chosen)

    # Sowing
    while seeds > 0:
        flat_index = (flat_index + 1) % 14
        side, idx = pos_map[flat_index]

        # Skip opponent's store
        if side != player and idx == 6:
            continue

        new_board[side][idx] += 1
        seeds -= 1

    last_side, last_idx = side, idx

    # Capture: last stone in empty own pit -> capture opposite pit + the placed stone
    if last_side == player and last_idx < 6 and new_board[player][last_idx] == 1:
        opposite_idx = 5 - last_idx
        captured = new_board[opponent][opposite_idx]
        if captured > 0:
            new_board[player][6] += captured + 1
            new_board[player][last_idx] = 0
            new_board[opponent][opposite_idx] = 0

    # Extra turn if last stone in own store
    extra_turn = (last_side == player and last_idx == 6)

    # End-of-game sweeping (when either row of pits becomes empty)
    if _is_terminal_board(new_board):
        for i in range(6):
            new_board["player_1"][6] += new_board["player_1"][i]
            new_board["player_1"][i] = 0
            new_board["player_2"][6] += new_board["player_2"][i]
            new_board["player_2"][i] = 0

    return new_board, extra_turn

# ------------------------------ Heuristic eval --------------------------------

def evaluate(state: Dict, last_move: int | None = None) -> float:
    """
    Advanced board evaluation for the CURRENT player (state['current_player']).
    Returns a scalar score: higher is better for the current player.
    - If terminal: exact store difference in [-48, 48].
    - Otherwise: mixes store diff, capture threats, extra-turn potential, phase, etc.
    """
    board = _board_from_state(state)
    current_player = _player_key(state["current_player"])
    opponent = _player_key(1 - state["current_player"])

    my_store = board[current_player][6]
    opp_store = board[opponent][6]
    my_pits = board[current_player][:6]
    opp_pits = board[opponent][:6]

    # Terminal returns true score diff
    if _is_terminal_board(board):
        return float(my_store - opp_store)

    # 1) Game phase (0 early -> 1 late) by total captured
    total_in_stores = my_store + opp_store
    game_phase = total_in_stores / 48.0

    # 2) Core heuristics
    score_diff = my_store - opp_store
    position_weights = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]

    # 3) Capture analysis
    def _potential_captures(pits: List[int], opp: List[int]) -> float:
        return sum(
            opp[5 - i] * position_weights[i]
            for i in range(6)
            if pits[i] == 0 and opp[5 - i] > 0
        )

    my_captures  = _potential_captures(my_pits, opp_pits)
    opp_captures = _potential_captures(opp_pits, my_pits)

    # 4) Extra turn potential (how many moves exactly land in store)
    extra_turns = sum(1 for i in range(6) if my_pits[i] == (6 - i))

    # 5) Move-specific analysis (optional, if last_move provided)
    move_bonus = 0.0
    if last_move is not None:
        stones = my_pits[last_move]
        if stones == 0:
            return -100.0  # penalize obviously invalid
        sim_board, extra = _make_move_board(copy.deepcopy(board), current_player, last_move)
        if extra:
            move_bonus += 3.0
        landing_pit = (last_move + stones) % 14
        if landing_pit < 6 and sim_board[current_player][landing_pit] == 0:
            opp_pit = 5 - landing_pit
            move_bonus += sim_board[opponent][opp_pit] * 0.4
        move_bonus += position_weights[last_move] * 0.5

    # 6) Opponent extra-turn threats
    opp_extra_threats = sum(position_weights[i] for i in range(6) if opp_pits[i] == (6 - i))

    # 7) Progressive strategy (phase-adjusted)
    late_game_value  = score_diff * 3.0 * game_phase
    early_game_value = (sum(my_pits) - sum(opp_pits)) * 2.5 * (1 - game_phase)

    # 8) Mobility / tempo
    future_moves = sum(1 for i in range(6) if my_pits[i] >= (6 - i))

    # 9) Combined score
    return float(
        score_diff * 2.0 +
        my_captures * 1.5 -
        opp_captures * 2.0 +
        extra_turns * 2.0 -
        opp_extra_threats * 1.8 +
        late_game_value +
        early_game_value +
        move_bonus +
        (sum(my_pits) * 0.1 - sum(opp_pits) * 0.1) +
        (future_moves * 0.3)
    )

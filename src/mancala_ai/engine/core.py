# Mancala core engine (state-based public API)
# State shape:
# {
#   "pits": [[int]*6, [int]*6],   # row 0 = player_1, row 1 = player_2
#   "stores": [int, int],         # stores[0] = player_1 store, stores[1] = player_2 store
#   "current_player": 0 | 1       # 0 = player_1 turn, 1 = player_2 turn
# }

from __future__ import annotations
import copy
from typing import Dict, List, Tuple

NUM_PITS = 6
INITIAL_STONES = 4

# ---------------------------------------------------------------------
# Conversions between "state" (public) and "board" (internal)
# ---------------------------------------------------------------------

def _player_key(player_idx: int) -> str:
    return "player_1" if player_idx == 0 else "player_2"

def _opp_key(player_key: str) -> str:
    return "player_2" if player_key == "player_1" else "player_1"

def _board_from_state(state: Dict) -> Dict[str, List[int]]:
    return {
        "player_1": state["pits"][0][:] + [state["stores"][0]],
        "player_2": state["pits"][1][:] + [state["stores"][1]],
    }

def _state_from_board(board: Dict[str, List[int]], current_player: int) -> Dict:
    return {
        "pits":   [board["player_1"][:NUM_PITS], board["player_2"][:NUM_PITS]],
        "stores": [board["player_1"][6],         board["player_2"][6]],
        "current_player": current_player,
    }

# ---------------------------------------------------------------------
# Public state-based API
# ---------------------------------------------------------------------

def new_game() -> Dict:
    return {
        "pits":   [[INITIAL_STONES] * NUM_PITS, [INITIAL_STONES] * NUM_PITS],
        "stores": [0, 0],
        "current_player": 0,
    }

def is_terminal_state(state: Dict) -> bool:
    board = _board_from_state(state)
    return _is_terminal_board(board)

def legal_actions(state: Dict) -> List[int]:
    row = state["current_player"]
    return [i for i in range(NUM_PITS) if state["pits"][row][i] > 0]

def step(state: Dict, action: int) -> Tuple[Dict, float, bool]:
    """
    Apply one move for state['current_player'] from pit index `action`.
    Returns: (next_state, reward, done)
      - reward: 0.0 for non-terminal; at terminal, store difference from mover's perspective.
    """
    mover_idx = state["current_player"]
    mover_key = _player_key(mover_idx)
    opp_key   = _player_key(1 - mover_idx)

    board = _board_from_state(state)
    new_board, extra_turn = _make_move_board(board, mover_key, action)
    done = _is_terminal_board(new_board)

    reward = float(new_board[mover_key][6] - new_board[opp_key][6]) if done else 0.0
    next_player = mover_idx if (extra_turn and not done) else 1 - mover_idx
    next_state = _state_from_board(new_board, next_player)
    return next_state, reward, done

# ---------------------------------------------------------------------
# Internal rule primitives (board-based)
# ---------------------------------------------------------------------

# Flat 14-slot map: P1 pits(0..5), P1 store(6), P2 pits(7..12), P2 store(13)
_POS_MAP = [("player_1", i) for i in range(6)] + [("player_1", 6)] + \
           [("player_2", i) for i in range(6)] + [("player_2", 6)]

def _start_flat_index(player_key: str, pit_idx: int) -> int:
    return (0 if player_key == "player_1" else 7) + pit_idx

def _is_terminal_board(board: Dict[str, List[int]]) -> bool:
    return sum(board["player_1"][:6]) == 0 or sum(board["player_2"][:6]) == 0

def _make_move_board(board: Dict[str, List[int]], player: str, pit: int) -> Tuple[Dict, bool]:
    new_board = copy.deepcopy(board)
    seeds = new_board[player][pit]
    new_board[player][pit] = 0

    opponent = _opp_key(player)
    flat = _start_flat_index(player, pit)

    # sowing: skip opponent's store
    last_side, last_idx = player, pit
    while seeds > 0:
        flat = (flat + 1) % 14
        side, idx = _POS_MAP[flat]
        if side != player and idx == 6:
            continue
        new_board[side][idx] += 1
        seeds -= 1
        last_side, last_idx = side, idx

    # capture
    if last_side == player and last_idx < 6 and new_board[player][last_idx] == 1:
        opp_idx = 5 - last_idx
        captured = new_board[opponent][opp_idx]
        if captured > 0:
            new_board[player][6] += captured + 1
            new_board[player][last_idx] = 0
            new_board[opponent][opp_idx] = 0

    # extra turn if land in own store
    extra_turn = (last_side == player and last_idx == 6)

    # terminal sweep
    if _is_terminal_board(new_board):
        for i in range(6):
            new_board["player_1"][6] += new_board["player_1"][i]
            new_board["player_1"][i] = 0
            new_board["player_2"][6] += new_board["player_2"][i]
            new_board["player_2"][i] = 0

    return new_board, extra_turn

def _landing_after_sow(player_key: str, pit_idx: int, stones: int) -> Tuple[str, int]:
    """Accurate landing (side_key, idx), skipping opponent's store."""
    flat = _start_flat_index(player_key, pit_idx)
    remaining = stones
    side, idx = player_key, pit_idx
    while remaining > 0:
        flat = (flat + 1) % 14
        side, idx = _POS_MAP[flat]
        if side != player_key and idx == 6:
            continue
        remaining -= 1
    return side, idx

# ---------------------------------------------------------------------
# Heuristic Evaluation (state -> score for current player)
# ---------------------------------------------------------------------

def evaluate(state: Dict, last_move: int | None = None) -> float:
    """
    Heuristic for the CURRENT player (state['current_player']).
    - If terminal: exact store diff (my_store - opp_store).
    - Otherwise: mix of store diff, capture threats, extra-turn potential,
      mobility, and phase-adjusted material.
    """
    board = _board_from_state(state)
    me = _player_key(state["current_player"])
    opp = _opp_key(me)

    my_store, opp_store = board[me][6], board[opp][6]
    my_pits,  opp_pits  = board[me][:6], board[opp][:6]

    # terminal: exact score
    if _is_terminal_board(board):
        return float(my_store - opp_store)

    # phase: 0 (early) -> 1 (late) by total captured
    total_in_stores = my_store + opp_store
    phase = total_in_stores / 48.0

    score_diff = my_store - opp_store
    pos_w = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]

    # capture potential
    def _pot_caps(pits: List[int], opp: List[int]) -> float:
        return sum(opp[5 - i] * pos_w[i] for i in range(6) if pits[i] == 0 and opp[5 - i] > 0)

    my_caps  = _pot_caps(my_pits,  opp_pits)
    opp_caps = _pot_caps(opp_pits, my_pits)

    # extra turn potential (exactly lands in my store)
    extra_turns = sum(1 for i in range(6) if my_pits[i] == (6 - i))

    # move-specific bonus if last_move provided
    move_bonus = 0.0
    if last_move is not None:
        stones = my_pits[last_move]
        if stones == 0:
            return -100.0  # discourage illegal
        sim_board, extra = _make_move_board(copy.deepcopy(board), me, last_move)
        if extra:
            move_bonus += 3.0
        land_side, land_idx = _landing_after_sow(me, last_move, stones)
        if land_side == me and land_idx < 6 and sim_board[me][land_idx] == 0:
            opp_idx = 5 - land_idx
            move_bonus += sim_board[opp][opp_idx] * 0.4
        move_bonus += pos_w[last_move] * 0.5

    # opponent extra-turn threats
    opp_extra_threats = sum(pos_w[i] for i in range(6) if opp_pits[i] == (6 - i))

    # phase-adjusted material
    late_val  = score_diff * 3.0 * phase
    early_val = (sum(my_pits) - sum(opp_pits)) * 2.5 * (1 - phase)

    # mobility (how many moves can reach store)
    mobility = sum(1 for i in range(6) if my_pits[i] >= (6 - i))

    score = (
        score_diff * 2.0 +
        my_caps * 1.5 - opp_caps * 2.0 +
        extra_turns * 2.0 - opp_extra_threats * 1.8 +
        late_val + early_val + move_bonus +
        (sum(my_pits) - sum(opp_pits)) * 0.1 +
        mobility * 0.3
    )
    return float(score)

# ---------------------------------------------------------------------
# Optional: lightweight compatibility shims (old board-based API)
# ---------------------------------------------------------------------

def is_terminal(board: Dict[str, List[int]]) -> bool:
    return _is_terminal_board(board)

def get_valid_moves(board: Dict[str, List[int]], player: str) -> List[int]:
    return [i for i in range(NUM_PITS) if board[player][i] > 0]

def make_move(board: Dict[str, List[int]], player: str, pit: int) -> Tuple[Dict, bool]:
    return _make_move_board(board, player, pit)

def evaluate_board(board: Dict[str, List[int]], player: str, last_move: int | None = None) -> float:
    state = _state_from_board(board, 0 if player == "player_1" else 1)
    return evaluate(state, last_move=last_move)

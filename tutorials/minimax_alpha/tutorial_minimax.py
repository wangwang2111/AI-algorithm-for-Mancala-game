# tutorials/minimax_alpha/tutorial_minimax.py
# -----------------------------------------------------------
# Plain Minimax for Mancala (STATE-BASED, no alpha-beta)
# Uses your engine: legal_actions(state), step(state, action), evaluate(state)
# Handles Mancala's "extra turn" rule by NOT reducing depth when the same
# player moves again after step().
# -----------------------------------------------------------

from __future__ import annotations
import copy, math, random, sys, pathlib
from typing import Dict, Optional, Tuple, List

# -- repo-root/src on sys.path so we can import the engine when run from repo root
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # .../tutorials/minimax_alpha/ -> repo root
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mancala_ai.engine.core import new_game, legal_actions, step, evaluate


# ---------- helpers ----------

def score_for(state: Dict, root_idx: int) -> float:
    """
    Evaluate from a FIXED player's perspective by temporarily setting
    state['current_player'] before calling engine.evaluate(state).
    """
    s = copy.deepcopy(state)
    s["current_player"] = root_idx
    return float(evaluate(s))

def print_state(s: Dict) -> None:
    p0, p1 = s["pits"][0], s["pits"][1]
    st0, st1 = s["stores"]
    turn = s["current_player"]
    print("+----- Mancala -----+")
    print("P1 store:", st1)
    print("P1 pits: ", list(reversed(p1)))
    print("P0 pits: ", p0)
    print("P0 store:", st0)
    print("Turn: P", turn, sep="")
    print("+-------------------+")

class Stats:
    def __init__(self):
        self.visits = 0  # node expansions


# ---------- minimax (no alpha-beta) ----------

def _minimax(state: Dict, depth: int, root_idx: int, stats: Stats) -> Tuple[float, Optional[int]]:
    stats.visits += 1

    # Leaf or terminal (either row empty)
    if depth == 0 or sum(state["pits"][0]) == 0 or sum(state["pits"][1]) == 0:
        return score_for(state, root_idx), None

    actions = legal_actions(state)
    if not actions:
        return score_for(state, root_idx), None

    is_max = (state["current_player"] == root_idx)
    best_move = actions[0]

    if is_max:
        best = -math.inf
        for a in actions:
            ns, _, _ = step(state, a)
            # Extra turn: current_player unchanged -> do NOT reduce depth
            reduce = 0 if ns["current_player"] == state["current_player"] else 1
            v, _ = _minimax(ns, depth - reduce, root_idx, stats)
            if v > best:
                best, best_move = v, a
        return best, best_move
    else:
        best = math.inf
        for a in actions:
            ns, _, _ = step(state, a)
            reduce = 0 if ns["current_player"] == state["current_player"] else 1
            v, _ = _minimax(ns, depth - reduce, root_idx, stats)
            if v < best:
                best, best_move = v, a
        return best, best_move


def choose_move_minimax(state: Dict, depth: int = 5) -> Tuple[int, Stats]:
    stats = Stats()
    _, mv = _minimax(state, depth, state["current_player"], stats)
    if mv is None:
        acts = legal_actions(state)
        mv = int(acts[0]) if acts else 0
    return int(mv), stats


# ---------- simple demo & exercises ----------

def random_move(state: Dict) -> int:
    acts = legal_actions(state)
    import random as _r
    return _r.choice(acts) if acts else 0

def play_game_minimax_vs_random(depth_minimax=5, verbose=False) -> Dict:
    s = new_game()
    while sum(s["pits"][0]) > 0 and sum(s["pits"][1]) > 0:
        if s["current_player"] == 0:
            mv, _ = choose_move_minimax(s, depth=depth_minimax)
        else:
            mv = random_move(s)
        s, _, _ = step(s, mv)
        if verbose:
            print(f"Move={mv}, next turn=P{s['current_player']}")
    return s


if __name__ == "__main__":
    # Demo: pick a move from the initial state
    s = new_game()
    print_state(s)
    mv, st = choose_move_minimax(s, depth=5)
    print("Chosen move:", mv, "| nodes visited:", st.visits)
    ns, _, _ = step(s, mv)
    print_state(ns)

    # Exercise (uncomment to try):
    # 1) Compare node counts for different depths:
    # for d in [3, 5, 7]:
    #     _, st = choose_move_minimax(new_game(), depth=d)
    #     print(f"depth={d}: nodes={st.visits}")

    # 2) Add simple move ordering:
    #    actions sorted by whether they land in store: pits[i] == 6 - i (for current row)
    #    Implement inside _minimax() before looping.

    # 3) Play a full game Minimax vs Random:
    # final_state = play_game_minimax_vs_random(depth_minimax=5, verbose=False)
    # print_state(final_state)

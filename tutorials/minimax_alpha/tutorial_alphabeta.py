# tutorials/minimax_alpha/tutorial_alphabeta.py
# -----------------------------------------------------------
# Alpha–Beta Minimax for Mancala (STATE-BASED)
# Includes optional one-ply move ordering to improve pruning.
# Correctly handles "extra turns" by NOT reducing depth when the same
# player moves again after step().
# -----------------------------------------------------------

from __future__ import annotations
import copy, math, sys, pathlib
from typing import Dict, Optional, Tuple, List

# -- repo-root/src on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
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


# ---------- optional one-ply ordering ----------

def order_moves_simple(state: Dict, root_idx: int) -> List[int]:
    """
    Order actions by the one-ply heuristic value of the child state
    from the ROOT player's perspective.
    """
    acts = legal_actions(state)
    scored = []
    for a in acts:
        ns, _, _ = step(state, a)
        scored.append((score_for(ns, root_idx), a))
    scored.sort(reverse=True)
    return [a for _, a in scored]


# ---------- alpha-beta core ----------

def _alphabeta(
    state: Dict,
    depth: int,
    alpha: float,
    beta: float,
    root_idx: int,
    stats: Stats,
    use_ordering: bool = True,
) -> Tuple[float, Optional[int]]:
    stats.visits += 1

    # Leaf or terminal
    if depth == 0 or sum(state["pits"][0]) == 0 or sum(state["pits"][1]) == 0:
        return score_for(state, root_idx), None

    acts = legal_actions(state)
    if not acts:
        return score_for(state, root_idx), None

    is_max = (state["current_player"] == root_idx)
    ordered = order_moves_simple(state, root_idx) if use_ordering else acts
    best_move = ordered[0]

    if is_max:
        val, a = -math.inf, alpha
        for mv in ordered:
            ns, _, _ = step(state, mv)
            # Extra turn: same player -> depth unchanged
            reduce = 0 if ns["current_player"] == state["current_player"] else 1
            child_v, _ = _alphabeta(ns, depth - reduce, a, beta, root_idx, stats, use_ordering)
            if child_v > val:
                val, best_move = child_v, mv
            a = max(a, val)
            if beta <= a:  # beta cut
                break
        return val, best_move
    else:
        val, b = math.inf, beta
        for mv in ordered:
            ns, _, _ = step(state, mv)
            reduce = 0 if ns["current_player"] == state["current_player"] else 1
            child_v, _ = _alphabeta(ns, depth - reduce, alpha, b, root_idx, stats, use_ordering)
            if child_v < val:
                val, best_move = child_v, mv
            b = min(b, val)
            if b <= alpha:  # alpha cut
                break
        return val, best_move


def choose_move_alphabeta(state: Dict, depth: int = 7, use_ordering: bool = True) -> Tuple[int, Stats]:
    stats = Stats()
    _, mv = _alphabeta(state, depth, -1e9, 1e9, state["current_player"], stats, use_ordering=use_ordering)
    if mv is None:
        acts = legal_actions(state)
        mv = int(acts[0]) if acts else 0
    return int(mv), stats


# ---------- quick experiments ----------

def _minimax_for_compare(state: Dict, depth: int) -> Tuple[int, int]:
    """
    Tiny inline minimax (no ordering, no pruning) just to compare node counts
    without importing another file.
    """
    class MStats:
        def __init__(self): self.visits = 0

    def mm(st: Dict, d: int, root: int, stats: MStats):
        stats.visits += 1
        if d == 0 or sum(st["pits"][0]) == 0 or sum(st["pits"][1]) == 0:
            return score_for(st, root), None
        acts = legal_actions(st)
        if not acts:
            return score_for(st, root), None
        is_max = (st["current_player"] == root)
        best_mv = acts[0]
        if is_max:
            val = -math.inf
            for a in acts:
                ns, _, _ = step(st, a)
                reduce = 0 if ns["current_player"] == st["current_player"] else 1
                v, _ = mm(ns, d - reduce, root, stats)
                if v > val: val, best_mv = v, a
            return val, best_mv
        else:
            val = math.inf
            for a in acts:
                ns, _, _ = step(st, a)
                reduce = 0 if ns["current_player"] == st["current_player"] else 1
                v, _ = mm(ns, d - reduce, root, stats)
                if v < val: val, best_mv = v, a
            return val, best_mv

    st = MStats()
    _, mv = mm(state, depth, state["current_player"], st)
    return int(mv if mv is not None else 0), st.visits


if __name__ == "__main__":
    s = new_game()
    print_state(s)

    # Compare node counts
    mv_min, nodes_min = _minimax_for_compare(s, depth=5)
    print(f"Minimax (d=5)           -> nodes={nodes_min}, move={mv_min}")

    mv_ab_no, st_ab_no = choose_move_alphabeta(s, depth=5, use_ordering=False)
    print(f"AlphaBeta no ordering   -> nodes={st_ab_no.visits}, move={mv_ab_no}")

    mv_ab, st_ab = choose_move_alphabeta(s, depth=5, use_ordering=True)
    print(f"AlphaBeta with ordering -> nodes={st_ab.visits}, move={mv_ab}")

    # Exercises (uncomment to try):
    # 1) Implement a simple transposition table (TT) keyed by:
    #    key = (tuple(state['pits'][0]), tuple(state['pits'][1]),
    #           state['stores'][0], state['stores'][1], state['current_player'])
    #    Cache (depth, value), consult before searching children.
    #
    # 2) Iterative deepening with time budget:
    #    - Loop depth 1..D, stop when time exceeds budget; return best move from
    #      the deepest completed search.

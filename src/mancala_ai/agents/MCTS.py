# Monte Carlo Tree Search (STATE-BASED)
from __future__ import annotations
import copy, math, random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mancala_ai.engine.core import legal_actions, step, evaluate

# ------------------------- utilities -------------------------

def _score_for(state: Dict, root_player_idx: int) -> float:
    """Heuristic score from the ROOT player's perspective (bounded to [-1, 1])."""
    s = copy.deepcopy(state)
    s["current_player"] = root_player_idx
    v = float(evaluate(s))
    # Bound scores so UCB doesn't blow up; tanh keeps shape but stabilizes.
    return math.tanh(v / 10.0)

def _is_terminal(state: Dict) -> bool:
    return sum(state["pits"][0]) == 0 or sum(state["pits"][1]) == 0

# ------------------------- node -------------------------

@dataclass
class _Node:
    state: Dict
    parent: Optional["_Node"] = None
    incoming_action: Optional[int] = None

    children: Dict[int, "_Node"] = field(default_factory=dict)
    untried_actions: List[int] = field(default_factory=list)

    N: int = 0           # visit count
    W: float = 0.0       # total value (from ROOT perspective)
    Q: float = 0.0       # mean value
    P: Dict[int, float] = field(default_factory=dict)  # priors per action (optional)

    def is_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def expand_one(self) -> "_Node":
        """Expand exactly one child (lazy expansion)."""
        a = self.untried_actions.pop()
        ns, _, _ = step(self.state, a)
        child = _Node(state=ns, parent=self, incoming_action=a)
        # Initialize child's untried actions
        child.untried_actions = list(legal_actions(ns))
        self.children[a] = child
        return child

    def ucb_score(self, action: int, c_puct: float) -> float:
        child = self.children[action]
        prior = self.P.get(action, 1.0 / max(1, len(self.children)))
        # Q + c * P * sqrt(N_parent) / (1 + N_child)
        return child.Q + c_puct * prior * math.sqrt(self.N + 1e-9) / (1 + child.N)

    def best_child_ucb(self, c_puct: float) -> Tuple[int, "_Node"]:
        best_a, best_node, best_score = None, None, -1e9
        for a, ch in self.children.items():
            s = self.ucb_score(a, c_puct)
            if s > best_score:
                best_score, best_a, best_node = s, a, ch
        return best_a, best_node

    def backup(self, value: float) -> None:
        """Backpropagate a ROOT-perspective value up the path."""
        node: Optional[_Node] = self
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            node = node.parent

# ------------------------- rollout policy -------------------------

def _rollout_random(state: Dict, max_depth: int) -> Dict:
    s = copy.deepcopy(state)
    d = 0
    while not _is_terminal(s) and d < max_depth:
        acts = legal_actions(s)
        if not acts:
            break
        a = random.choice(acts)
        s, _, _ = step(s, a)
        d += 1
    return s

def _rollout_greedy(state: Dict, root_idx: int, max_depth: int) -> Dict:
    """Greedy by one-ply heuristic to speed convergence."""
    s = copy.deepcopy(state)
    d = 0
    while not _is_terminal(s) and d < max_depth:
        acts = legal_actions(s)
        if not acts:
            break
        # choose action maximizing root-perspective value after 1 step
        best_a, best_v = acts[0], -1e9
        for a in acts:
            ns, _, _ = step(s, a)
            v = _score_for(ns, root_idx)
            if v > best_v:
                best_v, best_a = v, a
        s, _, _ = step(s, best_a)
        d += 1
    return s

# ------------------------- MCTS core -------------------------

def _mcts_search(
    root_state: Dict,
    n_simulations: int = 400,
    c_puct: float = 1.4,
    rollout: str = "random",          # "random" | "greedy"
    max_rollout_depth: int = 40,
) -> int:
    """
    Run MCTS from root_state and return the action with highest visit count.
    Everything is evaluated from the root player's perspective.
    """
    root_player_idx = int(root_state["current_player"])
    root = _Node(state=copy.deepcopy(root_state))
    root.untried_actions = list(legal_actions(root.state))

    # Uniform priors initially
    if root.untried_actions:
        p = 1.0 / len(root.untried_actions)
        root.P = {a: p for a in root.untried_actions}

    for _ in range(n_simulations):
        # 1) Selection
        node = root
        while node.is_expanded() and not _is_terminal(node.state) and node.children:
            _, node = node.best_child_ucb(c_puct)

        # 2) Expansion (if not terminal and still have moves)
        if not _is_terminal(node.state) and node.untried_actions:
            node = node.expand_one()
            # Optional: set uniform priors for the new node
            if node.untried_actions:
                p = 1.0 / len(node.untried_actions)
                node.P = {a: p for a in node.untried_actions}

        # 3) Rollout
        if rollout == "greedy":
            leaf = _rollout_greedy(node.state, root_player_idx, max_rollout_depth)
        else:
            leaf = _rollout_random(node.state, max_rollout_depth)

        value = _score_for(leaf, root_player_idx)  # ROOT perspective

        # 4) Backpropagate
        node.backup(value)

    # Pick the action with highest visit count from the root
    acts = list(root.children.keys())
    if not acts:
        # no legal moves; return a safe default
        safe = legal_actions(root_state)
        return int(safe[0]) if safe else 0

    best_a = max(acts, key=lambda a: root.children[a].N)
    return int(best_a)

# ------------------------- public API -------------------------

def choose_move_mcts(
    state: Dict,
    n_simulations: int = 400,
    c_puct: float = 1.4,
    rollout: str = "random",
    max_rollout_depth: int = 40,
) -> int:
    """
    Convenience wrapper for external callers.
    """
    return _mcts_search(
        state,
        n_simulations=n_simulations,
        c_puct=c_puct,
        rollout=rollout,
        max_rollout_depth=max_rollout_depth,
    )

# Registry-compatible name used in your registry.py
def mcts_decide(state: Dict, n_simulations: int = 400) -> int:
    return choose_move_mcts(state, n_simulations=n_simulations)

# tests/test_agents_smoke.py
from mancala_ai.engine.core import new_game, legal_actions
from mancala_ai.agents.advanced_heuristic import choose_move_advanced

def test_advanced_agent_returns_legal_move():
    s = new_game()
    mv = choose_move_advanced(s, depth=3)
    assert mv in legal_actions(s)

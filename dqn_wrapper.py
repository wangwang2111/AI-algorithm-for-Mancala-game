import torch
import numpy as np
from ai.dqn import DQNAgent  # assumes you placed DQNAgent in ai/dqn.py
from ai.rules import get_valid_moves

class DQNWrapper:
    def __init__(self, model_path="ai/save/policy7.pt"):
        self.agent = DQNAgent(state_shape=(29,))  # Use simplified 16-feature version
        self.agent.load_model(model_path)

    def __call__(self, board, player):
        state = self.agent.preprocess_state(board, player)
        valid_moves = get_valid_moves(board, player)
        return self.agent.get_action(state, valid_moves, greedy=True)


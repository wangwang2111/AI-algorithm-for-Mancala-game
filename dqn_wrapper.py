import torch
import numpy as np
from dqn import DQNAgent
from ai.rules import get_valid_moves

class DQNWrapper:
    def __init__(self, model_path="ai/save/policy_final.pt"):
        self.agent = DQNAgent(state_shape=(29,))  # Use simplified 16-feature version
        self.agent.load_model(model_path)

    def __call__(self, board, player):
        state = self.agent.preprocess_state(board, player)
        valid_moves = get_valid_moves(board, player)
        action = self.agent.get_action(state, valid_moves, greedy=True)
        if action not in valid_moves:
            print(f"WARNING: DQN selected illegal move: {action}")
        return action

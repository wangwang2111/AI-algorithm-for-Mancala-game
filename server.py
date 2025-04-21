from flask import Flask, request, jsonify
from flask_cors import CORS
import copy
import math
import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model

from ai.alpha_beta import minimax_alpha_beta
from ai.minimax import simple_minimax
from ai.MCTS import mcts_decide
from ai.advanced_heuristic import advanced_heuristic_minimax
from ai.rules import get_valid_moves, initialize_board, make_move, is_terminal
from dqn_wrapper import DQNWrapper

import random

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models (once at startup)
# a3c_model = load_model('models/mancala_a3c_final.h5', compile=False)  # Load A3C model
try:
    dqn_model = DQNWrapper('ai/save/policy_final.pt')
except Exception as e:
    print("Failed to load DQN model:", e)
    dqn_model = None

def js_to_python_board(js_board):
    return {
        "player_1": js_board[:7],
        "player_2": [js_board[7], js_board[8], js_board[9], js_board[10], js_board[11], js_board[12], js_board[13]]
    }

def python_to_js_board(python_board):
    return python_board["player_1"] + python_board["player_2"]

@app.route('/ai-move', methods=['POST'])
def get_ai_move():
    data = request.json
    js_board = data['board']
    
    if not js_board or len(js_board) != 14:
        return jsonify({'error': 'Invalid board format'}), 400
    
    ai_type = data.get('ai', 'advanced')  # Options: minimax, alpha_beta, advanced, dqn, a3c
    depth = data.get('depth', 7)
    currentPlayer = data.get('currentPlayer', 'player_1')
    
    print("currentPlayer is", currentPlayer)
    python_board = js_to_python_board(js_board)
    
    best_move = None
    
    if ai_type == 'minimax':
        _, best_move = simple_minimax(
            board=python_board,
            depth=depth,
            current_player=currentPlayer,
            maximizing_for=currentPlayer,
        )
    elif ai_type == 'alpha_beta':
        _, best_move = minimax_alpha_beta(
            board=python_board,
            depth=depth,
            alpha=-math.inf,
            beta=math.inf,
            current_player=currentPlayer,
            maximizing_for=currentPlayer,
        )
    elif ai_type == 'advanced':
        _, best_move = advanced_heuristic_minimax(
            board=python_board,
            depth=depth,
            alpha=-math.inf,
            beta=math.inf,
            current_player=currentPlayer,
            maximizing_for=currentPlayer,
        )
    elif ai_type == 'dqn':
        if dqn_model is None:
            return jsonify({'error': 'DQN model not available'}), 500
        best_move = dqn_model(python_board, currentPlayer)
    # elif ai_type == 'a3c':
    #     best_move = a3c_get_move(python_board, a3c_model, currentPlayer)
    elif ai_type == 'MCTS':
        best_move = mcts_decide(python_board, player=currentPlayer, time_limit=15)
    else:
        return jsonify({'error': 'Invalid AI type selected.'}), 400
    
    print("the best move found is:", best_move)
    if best_move is not None:
        if currentPlayer == 'player_1':
            js_move = best_move  # Player 1 moves are 0-5
        else:
            js_move = best_move + 7  # Player 2 moves are 7-12
    else:
        valid_moves = get_valid_moves(python_board, currentPlayer)
        js_move = random.choice(valid_moves) if valid_moves else None
    
    return jsonify({"move": int(js_move) if js_move is not None else None})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
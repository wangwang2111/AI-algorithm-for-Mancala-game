from flask import Flask, request, jsonify
from flask_cors import CORS
import copy
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from ai import minimax, alpha_beta, advanced_heuristic
from ai.rules import get_valid_moves
import random

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load DQN model (once at startup)
dqn_model = load_model('models/mancala_dqn.h5', compile=False)

# Your existing AI code
def js_to_python_board(js_board):
    return {
        "player_1": js_board[:7],
        "player_2": [js_board[7], js_board[8], js_board[9], js_board[10], js_board[11], js_board[12], js_board[13]]
    }

def python_to_js_board(python_board):
    return python_board["player_1"] + python_board["player_2"]

def dqn_get_move(board, model, player="player_2"):
    """Get best move using trained DQN model with validation"""
    state = np.array(
        board["player_1"][:6] + 
        [board["player_1"][6]] + 
        board["player_2"][:6] + 
        [board["player_2"][6]]
    )
    
    processed_state = (np.array(state, dtype=np.float32).reshape(1, -1) / 4.0)
    assert processed_state.shape == (1, 14), "Invalid state shape"
    
    valid_moves = get_valid_moves(board, player)
    if not valid_moves:
        return None

    q_values = model.predict(processed_state, verbose=0)[0]
    masked_q = [-np.inf if i not in valid_moves else q_values[i] for i in range(6)]
    
    best_move = np.argmax(masked_q)
    
    # Fallback to random valid move if needed
    return best_move if best_move in valid_moves else (
        random.choice(valid_moves) if valid_moves else None
    )

@app.route('/ai-move', methods=['POST'])
def get_ai_move():
    data = request.json
    js_board = data['board']
    ai_type = data.get('ai', 'alpha_beta')  # Options: minimax, alpha_beta, advanced, dqn
    depth = data.get('depth', 7)
    
    python_board = js_to_python_board(js_board)
    
    best_move = None
    
    if ai_type == 'minimax':
        _, best_move = minimax(
            board=python_board,
            depth=depth,
            maximizing_player=True,
            original_player=True
        )
    elif ai_type == 'alpha_beta':
        _, best_move = alpha_beta(
            board=python_board,
            depth=depth,
            alpha=-math.inf,
            beta=math.inf,
            maximizing_player=True,
            original_player=True
        )
    elif ai_type == 'advanced':
        _, best_move = advanced_heuristic(
            board=python_board,
            depth=depth,
            alpha=-math.inf,
            beta=math.inf,
            maximizing_player=True,
            original_player=True
        )
    elif ai_type == 'dqn':
        best_move = dqn_get_move(python_board, dqn_model)
    else:
        return jsonify({'error': 'Invalid AI type selected.'}), 400
    
    # Convert Python move (0-5) to JS format (7-12) if necessary
    js_move = best_move + 7 if best_move is not None else None
    
    return jsonify({"move": js_move})


if __name__ == '__main__':
    app.run(port=5000)
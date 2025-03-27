from flask import Flask, request, jsonify
from flask_cors import CORS
import copy
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from ai.alpha_beta import minimax_alpha_beta
from ai.minimax import simple_minimax
from ai.MCTS import mcts_decide
from ai.advanced_heuristic import advanced_heuristic_minimax
from ai.rules import get_valid_moves
import random

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load DQN model (once at startup)
dqn_model = load_model('mancala_dqn_final.h5', compile=False)

# Your existing AI code
def js_to_python_board(js_board):
    return {
        "player_1": js_board[:7],
        "player_2": [js_board[7], js_board[8], js_board[9], js_board[10], js_board[11], js_board[12], js_board[13]]
    }

def python_to_js_board(python_board):
    return python_board["player_1"] + python_board["player_2"]

def dqn_get_move(board, model, player="player_2"):
    """Get best move using trained DQN model with proper state preprocessing.
    
    Args:
        board: Current game board state as a dictionary with 'player_1' and 'player_2' keys
        model: Trained DQN model that expects input shape (2, 7, 2)
        player: The player making the move ('player_1' or 'player_2')
    
    Returns:
        int: Best move (0-5) according to the model, or random valid move if error occurs
    """
    try:
        # Preprocess the state exactly like in the MancalaDQN class
        state = np.zeros((2, 7, 2))
        state[:, :, 0] = np.array([
            board['player_1'][:6] + [board['player_1'][6]],  # Player 1 pits + store
            board['player_2'][:6] + [board['player_2'][6]]   # Player 2 pits + store
        ])
        state[:, :, 1] = 1 if player == 'player_1' else -1  # Player indicator
        
        # Add batch dimension
        state = state[np.newaxis, ...]
        
        # Get valid moves (0-5 indices)
        valid_moves = get_valid_moves(board, player)
        if not valid_moves:
            return None

        # Get Q-values from model
        q_values = model.predict(state, verbose=0)[0]
        
        # Mask invalid moves by setting their Q-values to -infinity
        masked_q = np.full(6, -np.inf)  # Initialize all moves as invalid
        for move in valid_moves:
            masked_q[move] = q_values[move]  # Only keep valid moves
            
        # Select move with highest Q-value
        best_move = np.argmax(masked_q)
        
        # Verify the move is valid (should always be true due to masking)
        if best_move in valid_moves:
            return int(best_move)  # Convert numpy.int64 to Python int
        else:
            # Fallback to random valid move if something went wrong
            return random.choice(valid_moves) if valid_moves else None
            
    except Exception as e:
        print(f"Error in dqn_get_move: {str(e)}")
        # Fallback to random valid move if error occurs
        valid_moves = get_valid_moves(board, player)
        return random.choice(valid_moves) if valid_moves else None
        
# def dqn_get_move(board, model, player="player_2"):
#     """Get best move using trained DQN model with proper serialization"""
#     try:
#         # Convert board to numpy array and normalize
#         state = np.array(
#             board["player_1"][:6] + 
#             [board["player_1"][6]] + 
#             board["player_2"][:6] + 
#             [board["player_2"][6]],
#             dtype=np.float32
#         ) / 4.0
        
#         processed_state = state.reshape(1, -1)
        
#         if processed_state.shape != (1, 14):
#             raise ValueError(f"Invalid state shape: {processed_state.shape}")

#         # Get valid moves
#         valid_moves = get_valid_moves(board, player)
#         if not valid_moves:
#             return None

#         # Get Q-values
#         q_values = model.predict(processed_state, verbose=0)[0]
        
#         # Mask invalid moves
#         masked_q = [-np.inf if i not in valid_moves else q_values[i] for i in range(6)]
        
#         # Select best move and convert to native Python int
#         best_move = int(np.argmax(masked_q))  # Convert numpy.int64 to Python int
        
#         return best_move if best_move in valid_moves else (
#             random.choice(valid_moves) if valid_moves else None
#         )
        
#     except Exception as e:
#         print(f"Error in dqn_get_move: {e}")
#         return None
        
    
# def dqn_get_move(board, model, player="player_2"):
#     """Get best move using trained DQN model with validation"""
#     state = np.array(
#         board["player_1"][:6] + 
#         [board["player_1"][6]] + 
#         board["player_2"][:6] + 
#         [board["player_2"][6]]
#     )
    
#     processed_state = (np.array(state, dtype=np.float32).reshape(1, -1) / 4.0)
#     assert processed_state.shape == (1, 14), "Invalid state shape"
    
#     valid_moves = get_valid_moves(board, player)
#     if not valid_moves:
#         return None

#     q_values = model.predict(processed_state, verbose=0)[0]
#     masked_q = [-np.inf if i not in valid_moves else q_values[i] for i in range(6)]
    
#     best_move = np.argmax(masked_q)
    
#     # Fallback to random valid move if needed
#     return best_move if best_move in valid_moves else (
#         random.choice(valid_moves) if valid_moves else None
#     )

@app.route('/ai-move', methods=['POST'])
def get_ai_move():
    data = request.json
    js_board = data['board']
    ai_type = data.get('ai', 'advanced')  # Options: minimax, alpha_beta, advanced, dqn
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
        best_move = dqn_get_move(python_board, dqn_model)
    elif ai_type == 'MCTS':
        best_move = mcts_decide(python_board, player=currentPlayer, simulations=1000, time_limit=5)
    else:
        return jsonify({'error': 'Invalid AI type selected.'}), 400
    
    print("the best move found is:", best_move)
    if best_move is not None:
        if currentPlayer == 'player_1':
            js_move = best_move  # Player 1 moves are 0-5
        else:
            js_move = best_move + 7  # Player 2 moves are 7-12
    else:
        js_move = random.choice()
    
    return jsonify({"move": js_move})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
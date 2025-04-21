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
from ai.rules import get_valid_moves, initialize_board, make_move, is_terminal
import random

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models (once at startup)
# dqn_model = load_model('models/mancala_dqn_best.h5', compile=False)
a3c_model = load_model('models/mancala_a3c_final.h5', compile=False)  # Load A3C model

def js_to_python_board(js_board):
    return {
        "player_1": js_board[:7],
        "player_2": [js_board[7], js_board[8], js_board[9], js_board[10], js_board[11], js_board[12], js_board[13]]
    }

def python_to_js_board(python_board):
    return python_board["player_1"] + python_board["player_2"]

def preprocess_state_for_a3c(board, current_player, agent_player="player_1"):
    """Preprocess board state for A3C model with 29 features"""
    p1_pits = np.array(board['player_1'][:6])
    p1_store = board['player_1'][6]
    p2_pits = np.array(board['player_2'][:6])
    p2_store = board['player_2'][6]
    
    if agent_player == 'player_1':
        features = [
            # Player 1 perspective (7)
            *(p1_pits / 24.0),  # 6 features
            p1_store / 48.0,     # 1 feature
            
            # Player 2 perspective (7)
            *(p2_pits / 24.0),   # 6
            p2_store / 48.0,     # 1
            
            # Relative differences (7)
            *(p1_pits - p2_pits) / 24.0,  # 6
            (p1_store - p2_store) / 48.0, # 1
            
            # Turn information (2)
            1.0 if current_player == 'player_1' else 0.0,
            1.0 if current_player == 'player_2' else 0.0,
            
            # Game phase (1)
            min(1.0, (p1_store + p2_store) / 30.0),
            
            # Strategic features (5)
            sum(p1_pits) / 24.0,
            sum(p2_pits) / 24.0,
            float(any(p == 0 for p in p1_pits)),
            float(any(p == 0 for p in p2_pits)),
            1.0  # Constant bias term to reach 29 features
        ]
    else:
        # When agent is player_2, we swap perspectives
        features = [
            # Player 2 perspective (7)
            *(p2_pits / 24.0),  # 6 features
            p2_store / 48.0,     # 1 feature
            
            # Player 1 perspective (7)
            *(p1_pits / 24.0),   # 6
            p1_store / 48.0,     # 1
            
            # Relative differences (7)
            *(p2_pits - p1_pits) / 24.0,  # 6
            (p2_store - p1_store) / 48.0, # 1
            
            # Turn information (2)
            1.0 if current_player == 'player_2' else 0.0,
            1.0 if current_player == 'player_1' else 0.0,
            
            # Game phase (1)
            min(1.0, (p1_store + p2_store) / 30.0),
            
            # Strategic features (5)
            sum(p2_pits) / 24.0,
            sum(p1_pits) / 24.0,
            float(any(p == 0 for p in p2_pits)),
            float(any(p == 0 for p in p1_pits)),
            1.0  # Constant bias term to reach 29 features
        ]
    
    return np.array(features, dtype=np.float32)

def a3c_get_move(board, model, player="player_1"):
    """Get best move using trained A3C model"""
    try:
        # Preprocess state
        state = preprocess_state_for_a3c(board, player, player)  # Pass player as agent_player
        
        # Get valid moves
        valid_moves = get_valid_moves(board, player)
        if not valid_moves:
            return None
            
        # Get policy and value from model
        policy, _ = model.predict(state[np.newaxis, ...], verbose=0)
        policy = policy[0]  # Remove batch dimension
        
        # Create mask for valid moves (0-5 for pits)
        valid_mask = np.zeros(6, dtype=bool)  # Assuming 6 actions (pits 0-5)
        for move in valid_moves:
            pit_index = move if move < 6 else move - 7  # Convert to 0-5 index
            valid_mask[pit_index] = True
        
        # Apply mask and renormalize
        masked_policy = policy.copy()
        masked_policy[~valid_mask] = 0  # Zero out invalid moves
        
        # Check if we have any valid probabilities left
        if np.sum(masked_policy) <= 0:
            # If all zeros (shouldn't happen with exploration), fall back to uniform
            masked_policy[valid_mask] = 1.0 / np.sum(valid_mask)
        else:
            # Normalize valid probabilities
            masked_policy = masked_policy / np.sum(masked_policy)
        
        # Sample action
        action = np.random.choice(6, p=masked_policy)
        
        # Convert back to move format if needed (0-5 for pits, 7-12 for stores)
        return action if action in valid_moves else None
        
    except Exception as e:
        print(f"Error in a3c_get_move: {str(e)}")
        valid_moves = get_valid_moves(board, player)
        return random.choice(valid_moves) if valid_moves else None

def dqn_get_move(board, model, player="player_2"):
    """Get best move using trained DQN model with proper state preprocessing."""
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

@app.route('/ai-move', methods=['POST'])
def get_ai_move():
    data = request.json
    js_board = data['board']
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
    # elif ai_type == 'dqn':
    #     best_move = dqn_get_move(python_board, dqn_model, currentPlayer)
    elif ai_type == 'a3c':
        best_move = a3c_get_move(python_board, a3c_model, currentPlayer)
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
    
    return jsonify({"move": js_move})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
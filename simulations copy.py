import math
import random
import time
import signal
from itertools import product
import pandas as pd
import numpy as np
import psutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Import your game components
from ai.rules import initialize_board, is_terminal, get_valid_moves, make_move
from ai.dqn import DQNAgent, MancalaEnv
from ai.A3C import A3CAgent
from ai.alpha_beta import minimax_alpha_beta
from ai.minimax import simple_minimax
from ai.advanced_heuristic import advanced_heuristic_minimax
from ai.MCTS import mcts_decide
import os
import tensorflow as tf
from server import preprocess_state_for_a3c  # At top of simulations.py
import multiprocessing as mp
mp.set_start_method('spawn', force=True)  # Add this right after imports

# multiprocessing.set_start_method('spawn')
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function timed out")

def run_with_timeout(func, timeout, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
    finally:
        signal.alarm(0)
    return result

# Initialize agents with proper error handling


# A3C Agent

# def init_worker():
#     """Initialize worker with proper cleanup handlers"""
#     import signal
#     signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore interrupts
    
#     # Initialize TF and GPU
#     tf.keras.backend.clear_session()
#     tf.config.run_functions_eagerly(True)
    
#     # Configure GPU
#     gpus = tf.config.list_physical_devices('GPU')
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#         except RuntimeError as e:
#             print(f"GPU config error: {e}")
    
#     # Set random seeds for reproducibility
#     np.random.seed(int(time.time()) % 1000)
#     tf.random.set_seed(int(time.time()) % 1000)
    
# Strategy implementations with memory checks
def memory_safe_strategy(func):
    def wrapper(board, player):
        try:
            if psutil.virtual_memory().percent > 99:
                raise MemoryError("High memory usage")
            return func(board, player)
        except Exception as e:
            print(f"Strategy {func.__name__} failed: {e}")
            valid_moves = get_valid_moves(board, player)
            return random.choice(valid_moves) if valid_moves else 0
    return wrapper

def random_strategy(board, player):
    return random.choice(get_valid_moves(board, player))

def dqn_strategy(board, player):
    try:
        # DQN Agent
        dqn_agent = DQNAgent()
        dqn_agent.model.load_weights("mancala_dqn_best_rules.h5")
        if psutil.virtual_memory().percent > 99:
            raise MemoryError("High memory usage")
        state = np.array(board["player_1"][:6] + [board["player_1"][6]] + 
                board["player_2"][:6] + [board["player_2"][6]])
        valid_moves = get_valid_moves(board, player)
        return dqn_agent.get_action(state, valid_moves)
    except Exception as e:
        print(f"DQN strategy failed: {e}")

def a3c_strategy(board, player):
    try:
        # Create isolated TF context
        tf.keras.backend.clear_session()
        tf.config.run_functions_eagerly(True)
        
        # Load model fresh for each call (necessary for multiprocessing)
        model = tf.keras.models.load_model('models/mancala_a3c_final.h5', compile=False)
        
        state = preprocess_state_for_a3c(board, player, player)
        valid_moves = get_valid_moves(board, player)
        
        if len(state.shape) == 1:
            state = state[np.newaxis, ...]
        
        policy, _ = model(state)
        policy = policy.numpy()[0]
        
        valid_mask = np.zeros(6, dtype=bool)
        for move in valid_moves:
            pit_index = move if move < 6 else move - 7
            valid_mask[pit_index] = True
        
        masked_policy = np.zeros_like(policy)
        masked_policy[valid_mask] = policy[valid_mask]
        
        if np.sum(masked_policy) <= 0:
            masked_policy[valid_mask] = 1.0 / np.sum(valid_mask)
        else:
            masked_policy = masked_policy / np.sum(masked_policy)
        
        action = np.random.choice(6, p=masked_policy)
        return action if action in valid_moves else random.choice(valid_moves)
        
    except Exception as e:
        print(f"A3C strategy failed: {str(e)}")
        valid_moves = get_valid_moves(board, player)
        return random.choice(valid_moves)
        
board = initialize_board()
print(a3c_strategy(board, "player_1"))

def simple_minimax_strategy(board, player):
    return simple_minimax(
        board=board,
        depth=5,
        current_player=player,
        maximizing_for=player,
    )[1]

def minimax_alpha_beta_strategy(board, player):
    return minimax_alpha_beta(
        board=board,
        depth=5,
        alpha=-math.inf,
        beta=math.inf,
        current_player=player,
        maximizing_for=player,
    )[1]

def advanced_heuristic_strategy(board, player):
    return advanced_heuristic_minimax(
        board=board,
        depth=5,
        alpha=-math.inf,
        beta=math.inf,
        current_player=player,
        maximizing_for=player,
    )[1]

def mcts_strategy(board, player):
    return mcts_decide(board, player)

strategies = [
    {"name": "A3C", "function": a3c_strategy},
    # {"name": "Random", "function": random_strategy},
    # {"name": "Simple Minimax", "function": simple_minimax_strategy},
    # {"name": "Minimax Alpha-Beta", "function": minimax_alpha_beta_strategy},
    # {"name": "Advanced Heuristic", "function": advanced_heuristic_strategy},
    {"name": "DQN", "function": dqn_strategy},
    
    # {"name": "MCTS", "function": mcts_strategy},
]

def simulate_game(player1_strategy, player2_strategy, max_moves=200):
    """Robust game simulation with time and memory limits"""
    try:
        start_time = time.time()
        board = initialize_board()
        current_player = "player_1"
        moves_count = 0
        
        while not is_terminal(board) and moves_count < max_moves:
            # if psutil.virtual_memory().percent > 99:
            #     raise MemoryError("Memory limit exceeded during game")
                
            if current_player == "player_1":
                move = run_with_timeout(
                    player1_strategy["function"], 
                    timeout=10,  # 5 second timeout per move
                    board=board, 
                    player=current_player
                )
            else:
                move = run_with_timeout(
                    player2_strategy["function"], 
                    timeout=10,
                    board=board, 
                    player=current_player
                )
                
            board, extra_turn = make_move(board, current_player, move)
            moves_count += 1
            
            if not extra_turn:
                current_player = "player_2" if current_player == "player_1" else "player_1"

        # Collect remaining seeds
        for player in ["player_1", "player_2"]:
            total_seeds = sum(board[player][:6])
            board[player][6] += total_seeds
            for i in range(6):
                board[player][i] = 0

        return {
            "player1": player1_strategy["name"],
            "player2": player2_strategy["name"],
            "p1_score": board["player_1"][6],
            "p2_score": board["player_2"][6],
            "winner": "Draw" if board["player_1"][6] == board["player_2"][6] else 
                     ("Player1" if board["player_1"][6] > board["player_2"][6] else "Player2"),
            "moves": moves_count,
            "time": time.time() - start_time
        }
        
    except Exception as e:
        print(f"Game error {player1_strategy['name']} vs {player2_strategy['name']}: {e}")
        return {
            "player1": player1_strategy["name"],
            "player2": player2_strategy["name"],
            "p1_score": 0,
            "p2_score": 0,
            "winner": "Error",
            "moves": 0,
            "time": 0
        }

def run_simulations_for_pair(pair, num_games):
    """Safe execution for strategy pairs"""
    player1_strat, player2_strat = pair
    results = []
    
    for _ in tqdm(range(num_games), desc=f"{player1_strat['name']} vs {player2_strat['name']}", leave=False):
        try:
            result = simulate_game(player1_strat, player2_strat)
            results.append({
                "Player1_Strategy": result["player1"],
                "Player2_Strategy": result["player2"],
                "Player1_Score": result["p1_score"],
                "Player2_Score": result["p2_score"],
                "Winner": result["winner"],
                "Moves": result["moves"],
                "Time_Seconds": round(result["time"], 3)
            })
        except Exception as e:
            print(f"Error in simulation: {e}")
    return results

def run_comprehensive_simulations(num_games=100):
    """Main simulation runner with robust process management"""
    # Initialize A3C model in main process first
    # with tf.device('/CPU:0'):  # Force CPU initialization
    #     _ = a3c_agent.model.predict(np.zeros((1,29)))  # Warmup
    results = []
    # combinations = list(product(strategies, strategies))
    
    selected_strategy = next(s for s in strategies if s["name"] == "A3C")
    other_strategies = [s for s in strategies if s["name"] != "A3C"]
    
    # Create all combinations where one player is  and the other is any other strategy
    combinations = []
    for strategy in other_strategies:
        combinations.append((selected_strategy, strategy))  # MCTS as player1
        combinations.append((strategy, selected_strategy))  # MCTS as player2
    print("Combinations: ", combinations)
    total_combinations = len(combinations)
    # Conservative parallel processing
    max_workers = max(1, min(4, cpu_count()))
    # max_workers = 2
    chunk_size = max(1, num_games // 10)  # Process in chunks
    
    print("total combinations: ", total_combinations)
    with tqdm(total=total_combinations, desc="Overall Progress") as pbar:
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = {}
            
            for pair in combinations:
                # Submit in chunks to reduce memory pressure
                for chunk_start in range(0, num_games, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_games)
                    future = executor.submit(
                        run_simulations_for_pair,
                        pair,
                        chunk_end - chunk_start
                    )
                    futures[future] = pair
                    pbar.set_postfix_str(f"{pair[0]['name']} vs {pair[1]['name']}")
            
            # Process completed futures
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    
                    # # Periodic saving
                    # if len(results) % 50 == 0:
                    #     pd.DataFrame(results).to_csv("progress_checkpoint.csv", index=False)
                        
                except Exception as e:
                    print(f"Process failed: {e}")
                finally:
                    pbar.update(1)
    
    return results

from pathlib import Path

if __name__ == "__main__":
    try:
        # Force clean shutdown of multiprocessing
        # import atexit
        # import multiprocessing as mp
        # atexit.register(mp.set_executable, '')  # Clean up resources
          
        # Initial memory check
        if psutil.virtual_memory().percent > 99:
            raise MemoryError("Insufficient memory to start simulations")
            
        print("Starting simulations...")
        results = run_comprehensive_simulations(num_games=80)
        
        df = pd.DataFrame(results)
        path = "mancala_simulation_results_a3c_dqn_4"
        if (Path.cwd() / path).exists():
            path = f'{path}-1.csv'
        df.to_csv(path, index=False)
        
        print("\nSummary Statistics:")
        print(df.groupby(['Player1_Strategy', 'Player2_Strategy']).agg({
            'Time_Seconds': 'mean',
            'Moves': 'median',
            'Winner': lambda x: x.value_counts(normalize=True).get('Player1', 0)
        }))
        
    except Exception as e:
        print(f"Fatal error in main execution: {e}")
    # finally:
    #     # Explicit cleanup
    #     import gc
    #     gc.collect()
    #     tf.keras.backend.clear_session()
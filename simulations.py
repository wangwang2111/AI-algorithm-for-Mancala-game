import math
import random
import time
from itertools import product, permutations
import pandas as pd

from ai.rules import initialize_board, is_terminal, evaluate, get_valid_moves, make_move
from ai.alpha_beta import minimax_alpha_beta
from ai.minimax import simple_minimax
from ai.advanced_heuristic import advanced_heuristic_minimax
from ai.MCTS import mcts_decide
from concurrent.futures import ProcessPoolExecutor
from dqn_wrapper import DQNWrapper
from ai.MCTS import mcts_decide

import multiprocessing as mp
mp.set_start_method('spawn', force=True)  # This to enable multiprocessing with pytorch model

def random_strategy(board, player):
    return random.choice(get_valid_moves(board, player))

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

# Updated strategies list
strategies = [
    {"name": "Random", "function": random_strategy},
    {"name": "Simple Minimax", "function": simple_minimax_strategy},
    {"name": "Minimax Alpha-Beta", "function": minimax_alpha_beta_strategy},
    {"name": "Advanced Heuristic", "function": advanced_heuristic_strategy},
    {"name": "MCTS", "function": mcts_strategy},
    {"name": "DQN", "function": None},  # Placeholder, will be replaced per process
]

# Global cache per process
# The model is only loaded once per process
# avoid race conditions or bad sharing between processes
_dqn_cache = {}

def get_cached_dqn():
    if "agent" not in _dqn_cache:
        _dqn_cache["agent"] = DQNWrapper('ai/save/policy_final.pt')
    return _dqn_cache["agent"]

def simulate_game(player1_strategy, player2_strategy):
    """Simulate a game between two strategies."""
    if player1_strategy["name"] == "DQN":
        player1_strategy["function"] = get_cached_dqn()
    if player2_strategy["name"] == "DQN":
        player2_strategy["function"] = get_cached_dqn()
        
    start_time = time.time()
    board = initialize_board()
    # Randomly select which player starts the game
    current_player = "player_1"
    moves_count = 0

    while not is_terminal(board):
        if current_player == "player_1":
            move = player1_strategy["function"](board, current_player)
        else:
            move = player2_strategy["function"](board, current_player)
        
        board, extra_turn = make_move(board, current_player, move)
        moves_count += 1
        
        if not extra_turn:
            current_player = "player_2" if current_player == "player_1" else "player_1"

    # --- ADD THIS SECTION TO COLLECT REMAINING SEEDS ---
    # When the game ends, move all remaining seeds to the respective stores
    for player in ["player_1", "player_2"]:
        total_seeds = sum(board[player][:6])
        board[player][6] += total_seeds
        for i in range(6):
            board[player][i] = 0  # Empty the pits

    # Collect results
    p1_score = board["player_1"][6]
    p2_score = board["player_2"][6]
    
    return {
        "player1": player1_strategy["name"],
        "player2": player2_strategy["name"],
        "p1_score": p1_score,
        "p2_score": p2_score,
        "winner": "Draw" if p1_score == p2_score else ("Player1" if p1_score > p2_score else "Player2"),
        "moves": moves_count,
        "time": time.time() - start_time
    }

# def run_comprehensive_simulations(num_games=100):
#     """Run simulations for all strategy combinations."""
#     results = []
#     combinations = list(product(strategies, repeat=2))  # All possible pairs
    
#     for pair in combinations:
#         player1_strat, player2_strat = pair
#         print(f"Running {num_games} games: {player1_strat['name']} vs {player2_strat['name']}")
        
#         for _ in range(num_games):
#             result = simulate_game(player1_strat, player2_strat)
#             results.append({
#                 "Player1_Strategy": result["player1"],
#                 "Player2_Strategy": result["player2"],
#                 "Player1_Score": result["p1_score"],
#                 "Player2_Score": result["p2_score"],
#                 "Winner": result["winner"],
#                 "Moves": result["moves"],
#                 "Time_Seconds": round(result["time"], 3)
#             })
    
#     return results
from tqdm import tqdm

def run_simulations_for_pair(pair, num_games):
    player1_strat, player2_strat = pair
    results = []
    # Progress bar for games within a single strategy pair
    for _ in tqdm(range(num_games), desc=f"{player1_strat['name']} vs {player2_strat['name']}", leave=False):
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
    return results

def run_comprehensive_simulations(num_games=1000):
    """Run simulations where one player is always MCTS and the other is any other strategy."""
    results = []
    
    # selected_strategy = next(s for s in strategies if s["name"] == "DQN")
    # other_strategies = [s for s in strategies if s["name"] != "DQN"]
    
    # # Create all combinations where one player is  and the other is any other strategy
    # combinations = []
    # for strategy in other_strategies:
    #     combinations.append((selected_strategy, strategy))
        # combinations.append((strategy, selected_strategy))
        # combinations.append((selected_strategy, selected_strategy))
    
    combinations = []
    combinations = list(product(strategies, strategies))
    
    print("Combinations: ", combinations)
    total_combinations = len(combinations)
    
    # Memory check at game start
    import psutil
    if psutil.virtual_memory().percent > 90:
        raise MemoryError("High memory at game start")
    
    # Master progress bar for all strategy pairs
    with tqdm(total=total_combinations, desc="Overall Progress") as overall_progress:
        with ProcessPoolExecutor() as executor:
            futures = []
            for pair in combinations:
                future = executor.submit(
                    run_simulations_for_pair, 
                    pair, 
                    num_games
                )
                future.add_done_callback(lambda _: overall_progress.update(1))
                futures.append(future)
            
            for future in futures:
                try:
                    results.extend(future.result())
                except Exception as e:
                    print(f"Error in worker: {e}")
                    continue
    
    return results

if __name__ == "__main__":
    simulation_results = run_comprehensive_simulations(num_games=100)
    df = pd.DataFrame(simulation_results)
    df.to_csv("evaluation/_result_final.csv", index=False)
    
    print("\nSummary Statistics:")
    print(df.groupby(['Player1_Strategy', 'Player2_Strategy'])['Winner'] \
                         .value_counts(normalize=True) \
                         .unstack() \
                         .fillna(0))
import math
import random
import time
from itertools import product
import pandas as pd

import sys
sys.path.append("/Project/src/")

from ai.rules import initialize_board, is_terminal, evaluate, get_valid_moves, make_move
from ai.alpha_beta import minimax_alpha_beta
from ai.minimax import simple_minimax
from ai.advanced_heuristic import advanced_heuristic_minimax

# Strategy definitions
strategies = [
    {
        "name": "Random",
        "function": lambda board, player: random.choice(get_valid_moves(board, player))
    },
    {
        "name": "Simple Minimax",
        "function": lambda board, player: simple_minimax(
            board=board,
            depth=5,
            current_player=player,
            maximizing_for=player,     # maximizing for self
        )[1]
    },
    {
        "name": "Minimax Alpha-Beta",
        "function": lambda board, player: minimax_alpha_beta(
            board=board,
            depth=5,
            alpha=-math.inf,
            beta=math.inf,
            current_player=player,
            maximizing_for=player,     # maximizing for self
        )[1]
    },
    {
        "name": "Advanced Heuristic",
        "function": lambda board, player: advanced_heuristic_minimax(
            board=board,
            depth=5,
            alpha=-math.inf,
            beta=math.inf,
            current_player=player,
            maximizing_for=player,     # maximizing for self
        )[1]
    }
]

def simulate_game(player1_strategy, player2_strategy):
    """Simulate a game between two strategies."""
    start_time = time.time()
    board = initialize_board()
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

def run_comprehensive_simulations(num_games=100):
    """Run simulations for all strategy combinations."""
    results = []
    combinations = list(product(strategies, repeat=2))  # All possible pairs
    
    for pair in combinations:
        player1_strat, player2_strat = pair
        print(f"Running {num_games} games: {player1_strat['name']} vs {player2_strat['name']}")
        
        for _ in range(num_games):
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

if __name__ == "__main__":
    # Run 100 simulations for each of the 16 possible strategy pairs (4x4)
    simulation_results = run_comprehensive_simulations(num_games=100)
    df = pd.DataFrame(simulation_results)
    df.to_csv("mancala_results.csv", index=False)
    
    # Optional: Quick analysis
    print("\nSummary Statistics:")
    print(df.groupby(['Player1_Strategy', 'Player2_Strategy']).agg({
        'Time_Seconds': 'mean',
        'Moves': 'median',
        'Winner': lambda x: x.value_counts(normalize=True).get('Player2', 0)
    }))

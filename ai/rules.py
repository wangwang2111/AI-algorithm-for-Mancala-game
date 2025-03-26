import copy

def initialize_board():
    return {
        "player_1": [4, 4, 4, 4, 4, 4, 0],
        "player_2": [4, 4, 4, 4, 4, 4, 0],
    }

def is_terminal(board):
    return sum(board["player_1"][:6]) == 0 or sum(board["player_2"][:6]) == 0

def evaluate(board, current_player):
    """Evaluate board state from the perspective of `current_player` using all 10 heuristics."""
    opponent = "player_2" if current_player == "player_1" else "player_1"
    
    my_store = board[current_player][6]
    opponent_store = board[opponent][6]
    my_pits = board[current_player][:6]
    opponent_pits = board[opponent][:6]

    # Heuristic 1: Score difference (store difference)
    score_diff = my_store - opponent_store

    # Heuristic 2: Stones captured (recent captures)
    # Note: In evaluation function, we need to calculate potential captures instead
    captured_stones = sum(
        opponent_pits[5 - i] 
        for i in range(6) 
        if my_pits[i] == 0 and opponent_pits[5 - i] > 0
    )

    # Heuristic 3: Extra turns potential
    extra_turn_potential = sum(
        1 for i in range(6)
        if my_pits[i] == (6 - i)  # Stones needed to land in store
    )

    # Heuristic 4: Empty pits (penalize)
    empty_pits = sum(1 for pit in my_pits if pit == 0)

    # Heuristic 5: Opponent's empty pits (reward)
    opponent_empty_pits = sum(1 for pit in opponent_pits if pit == 0)

    # Heuristic 6: Stones in store
    store_value = my_store

    # Heuristic 7: Stones in opponent's store (penalize)
    opponent_store_value = opponent_store

    # Heuristic 8: Potential captures
    potential_captures = captured_stones  # Same as heuristic 2 in this context

    # Heuristic 9: Balance of stones
    if len(my_pits) > 0:
        mean_pits = sum(my_pits) / len(my_pits)
        stone_imbalance = (sum((x - mean_pits) ** 2 for x in my_pits) / len(my_pits)) ** 0.5
    else:
        stone_imbalance = 0

    # Heuristic 10: Endgame advantage
    endgame_score = 0
    if is_terminal(board):
        if my_store > 24:
            endgame_score = 1000  # Win
        elif opponent_store > 24:
            endgame_score = -1000  # Lose
        else:
            endgame_score = 500 if score_diff > 0 else -500  # Draw with advantage

    # If terminal state, return immediately with endgame score
    if is_terminal(board):
        return endgame_score

    # Combine all heuristics with adjusted weights for evaluation
    evaluation = (
        score_diff * 5.0 +                   # Primary score difference
        captured_stones * 2.5 +              # Capture potential
        extra_turn_potential * 3.0 +         # Extra turn potential
        empty_pits * -0.5 +                  # Penalize empty pits
        opponent_empty_pits * 0.5 +          # Reward opponent's empty pits
        store_value * 0.3 +                 # Direct store value
        opponent_store_value * -0.3 +        # Penalize opponent's store
        potential_captures * 1.5 +           # Potential captures
        stone_imbalance * -0.2 +             # Penalize imbalance
        sum(my_pits) * 0.1 -                 # Reward stones in pits
        sum(opponent_pits) * 0.1             # Penalize opponent's stones
    )

    return evaluation

def get_valid_moves(board, player):
    return [i for i in range(6) if board[player][i] > 0]

def make_move(board, player, pit):
    new_board = copy.deepcopy(board)
    seeds = new_board[player][pit]
    new_board[player][pit] = 0

    original_player = player
    current_player = player
    index = pit  # Start at the selected pit

    # Track last pit for capturing
    last_pit = None

    while seeds > 0:
        index += 1

        # Handle wrap-around and store logic
        if current_player == original_player:
            # On original player's side: include their store (index 6)
            if index > 6:  # After store (index 6), switch to opponent's side
                current_player = "player_2" if current_player == "player_1" else "player_1"
                index = 0
        else:
            # On opponent's side: skip their store (only indices 0-5)
            if index >= 6:  # After pit 5, switch back to original player
                current_player = original_player
                index = 0

        # Add seed to the current pit/store
        new_board[current_player][index] += 1
        seeds -= 1
        last_pit = (current_player, index)

    # Check for capture (only on original player's side)
    if (
        last_pit[0] == original_player  # Last seed on original player's side
        and last_pit[1] < 6  # Landed in a pit (not the store)
        and new_board[original_player][last_pit[1]] == 1  # Pit was empty
    ):
        opposite_player = "player_2" if original_player == "player_1" else "player_1"
        opposite_pit = 5 - last_pit[1]
        captured = new_board[opposite_player][opposite_pit]
        new_board[original_player][6] += captured + 1  # Capture + last seed
        new_board[opposite_player][opposite_pit] = 0
        new_board[original_player][last_pit[1]] = 0

    # Check for extra turn (last seed in store)
    extra_turn = last_pit[0] == original_player and last_pit[1] == 6

    return new_board, extra_turn
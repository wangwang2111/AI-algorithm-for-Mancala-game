import copy

def initialize_board():
    return {
        "player_1": [4, 4, 4, 4, 4, 4, 0],
        "player_2": [4, 4, 4, 4, 4, 4, 0],
    }

def is_terminal(board):
    return sum(board["player_1"][:6]) == 0 or sum(board["player_2"][:6]) == 0

def evaluate(board, current_player):
    """Evaluate board state from the perspective of `current_player`."""
    opponent = "player_2" if current_player == "player_1" else "player_1"
    
    my_store = board[current_player][6]
    opponent_store = board[opponent][6]

    # Terminal state checks (win/loss conditions)
    if my_store > 24:
        return 1000  # Guaranteed win
    if opponent_store > 24:
        return -1000  # Guaranteed loss

    store_diff = my_store - opponent_store
    pit_diff = sum(board[current_player][:6]) - sum(board[opponent][:6])
    
    # Capture potential: Sum of stones in opponent's pits opposite to our empty pits
    capture_potential = sum(
        board[opponent][5 - i]  # Stones in opponent's opposite pit
        for i in range(6)
        if board[current_player][i] == 0  # Only if our pit is empty
    )
    
    # Extra turn potential: Pits that can land in the store
    extra_turn_potential = sum(
        1 for i in range(6)
        if board[current_player][i] == (6 - i)  # Stones needed to reach the store
    )

    # Adjusted weights for balanced evaluation
    return (
        store_diff * 5 +               # Prioritize store difference
        pit_diff * 0.7 +                # Favor controlling more pits
        capture_potential * 1 +       # Reward capture opportunities
        extra_turn_potential * 2 -      # Value extra turns moderately
        sum(board[opponent][:6]) * 0.3  # Penalize opponent's pit control
    )

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
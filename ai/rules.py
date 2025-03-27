import copy

def initialize_board():
    return {
        "player_1": [4, 4, 4, 4, 4, 4, 0],
        "player_2": [4, 4, 4, 4, 4, 4, 0],
    }

def is_terminal(board):
    return sum(board["player_1"][:6]) == 0 or sum(board["player_2"][:6]) == 0

def evaluate(board, current_player, last_move=None):
    """Advanced board evaluation combining all strategic heuristics"""
    opponent = "player_2" if current_player == "player_1" else "player_1"
    
    my_store = board[current_player][6]
    opp_store = board[opponent][6]
    my_pits = board[current_player][:6]
    opp_pits = board[opponent][:6]
    total_seeds = sum(my_pits) + sum(opp_pits) + my_store + opp_store
    
    # Terminal state evaluation
    if is_terminal(board):
        if my_store > 24: return 1000  # Win
        if opp_store > 24: return -1000  # Loss
        return 500 if my_store > opp_store else -500  # Draw advantage
    
    # 1. Game Phase Calculation
    game_phase = 1 - (total_seeds / 96.0)  # 0=early, 1=late
    
    # 2. Core Heuristics ------------------------------------------------------
    score_diff = my_store - opp_store
    position_weights = [0.5, 0.8, 1.2, 1.5, 1.2, 0.8]  # Central pits more valuable
    
    # 3. Capture Analysis
    def calculate_captures(pits, opp_pits):
        return sum(
            opp_pits[5-i] * position_weights[i] 
            for i in range(6) 
            if pits[i] == 0 and opp_pits[5-i] > 0
        )
    
    my_captures = calculate_captures(my_pits, opp_pits)
    opp_captures = calculate_captures(opp_pits, my_pits)
    
    # 4. Extra Turn Potential
    extra_turns = sum(
        1 for i in range(6) 
        if my_pits[i] == (6 - i)  # Exact stones to land in store
    )
    
    # 5. Move-Specific Analysis (if last_move provided)
    move_bonus = 0
    if last_move is not None:
        stones = my_pits[last_move]
        landing_pit = (last_move + stones) % 14
        
        # Immediate extra turn
        if landing_pit == 6:
            move_bonus += 3.0
            
        # Immediate capture
        if landing_pit < 6 and my_pits[landing_pit] == 0:
            move_bonus += opp_pits[5-landing_pit] * 0.4
            
        # Positional advantage
        move_bonus += position_weights[last_move] * 0.5
        
        # Danger zone seeding
        danger_zones = [0, 1, 5]
        for i in range(stones):
            pit = (last_move + i + 1) % 14
            if pit in danger_zones and pit < 6:
                move_bonus += 0.2
                
        # Seed conservation penalty
        if my_pits[last_move] == stones and landing_pit != 6:
            move_bonus -= 1.0 if game_phase < 0.5 else 0.5
    
    # 6. Defensive Considerations ---------------------------------------------
    # Opponent extra turn threats
    opp_extra_threats = sum(
        position_weights[i] 
        for i in range(6) 
        if opp_pits[i] == (13 - i)
    )
    
    # 7. Progressive Strategy -------------------------------------------------
    late_game_value = score_diff * 3.0 * game_phase
    early_game_value = (sum(my_pits) - sum(opp_pits)) * 2.5 * (1 - game_phase)
    
    # 8. Mobility and Tempo ---------------------------------------------------
    future_moves = sum(1 for i in range(6) if my_pits[i] >= (6 - i))
    large_groups = sum(1 for s in my_pits if s > 10)
    
    # 9. Combined Evaluation --------------------------------------------------
    evaluation = (
        # Core components
        score_diff * 2.0 +
        my_captures * 1.5 -
        opp_captures * 2.0 +
        extra_turns * 2.0 -
        opp_extra_threats * 1.8 +
        
        # Progressive strategy
        late_game_value +
        early_game_value +
        
        # Move-specific bonuses
        move_bonus +
        
        # Additional heuristics
        (sum(my_pits) * 0.1 - sum(opp_pits) * 0.1) +  # Stone count
        (future_moves * 0.3) +  # Mobility
        (large_groups * 0.4)  # Tempo control
    )
    
    # 10. Normalization and return
    return evaluation / 10.0  # Scale to reasonable range

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
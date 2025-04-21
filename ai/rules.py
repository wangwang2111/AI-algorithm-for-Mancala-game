import copy

def initialize_board():
    return {
        "player_1": [4, 4, 4, 4, 4, 4, 0],
        "player_2": [4, 4, 4, 4, 4, 4, 0],
    }

def is_terminal(board):
    return sum(board["player_1"][:6]) == 0 or sum(board["player_2"][:6]) == 0

def evaluate(board, current_player, last_move=None):
    """Advanced board evaluation with corrected heuristics"""
    opponent = "player_2" if current_player == "player_1" else "player_1"
    
    my_store = board[current_player][6]
    opp_store = board[opponent][6]
    my_pits = board[current_player][:6]
    opp_pits = board[opponent][:6]
    
    # Terminal state returns actual score difference
    if is_terminal(board):
        return my_store - opp_store  # Range: [-48, 48]
    
    # 1. Game Phase Calculation (based on captured seeds)
    total_in_stores = my_store + opp_store
    game_phase = total_in_stores / 48.0  # 0=early, 1=late
    
    # 2. Core Heuristics ------------------------------------------------------
    score_diff = my_store - opp_store
    position_weights = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]  # Pit weights
    
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
        if my_pits[i] == (6 - i)  # Correct stone count for store landing
    )
    
    # 5. Move-Specific Analysis (activated via MCTSNode)
    move_bonus = 0
    if last_move is not None:
        stones = my_pits[last_move]
        if stones == 0:  # Invalid move, skip
            return -100  # Penalize invalid
        
        # Simulate move accurately using game rules
        simulated_board, extra_turn = make_move(copy.deepcopy(board), current_player, last_move)
        
        # Check for immediate extra turn
        if extra_turn:
            move_bonus += 3.0
        
        # Check for captures in simulated board
        landing_pit = (last_move + stones) % 14
        if landing_pit < 6 and simulated_board[current_player][landing_pit] == 0:
            opp_pit = 5 - landing_pit
            move_bonus += simulated_board[opponent][opp_pit] * 0.4
        
        # Positional bonus
        move_bonus += position_weights[last_move] * 0.5
    
    # 6. Defensive Considerations
    opp_extra_threats = sum(
        position_weights[i] 
        for i in range(6) 
        if opp_pits[i] == (6 - i)  # Correct extra turn check
    )
    
    # 7. Progressive Strategy
    late_game_value = score_diff * 3.0 * game_phase
    early_game_value = (sum(my_pits) - sum(opp_pits)) * 2.5 * (1 - game_phase)
    
    # 8. Mobility and Tempo
    future_moves = sum(1 for i in range(6) if my_pits[i] >= (6 - i))
    
    # 9. Combined Evaluation
    evaluation = (
        score_diff * 2.0 +
        my_captures * 1.5 -
        opp_captures * 2.0 +
        extra_turns * 2.0 -
        opp_extra_threats * 1.8 +
        late_game_value +
        early_game_value +
        move_bonus +
        (sum(my_pits) * 0.1 - sum(opp_pits) * 0.1) +
        (future_moves * 0.3)
    )
    
    return evaluation  # No division; rely on rollout normalization

def get_valid_moves(board, player):
    return [i for i in range(6) if board[player][i] > 0]

def make_move(board, player, pit):
    new_board = copy.deepcopy(board)
    seeds = new_board[player][pit]
    new_board[player][pit] = 0

    pits = ["player_1", "player_2"]
    opponent = "player_2" if player == "player_1" else "player_1"
    idx = pit
    side = player

    # Flat 14-pit view: p1 pits(0–5), p1 store(6), p2 pits(7–12), p2 store(13)
    pos_map = [("player_1", i) for i in range(6)] + [("player_1", 6)] + \
              [("player_2", i) for i in range(6)] + [("player_2", 6)]

    # Compute starting index in pos_map
    flat_index = pits.index(player) * 7 + pit

    while seeds > 0:
        flat_index = (flat_index + 1) % 14
        side, idx = pos_map[flat_index]

        # Skip opponent's store
        if side != player and idx == 6:
            continue

        new_board[side][idx] += 1
        seeds -= 1

    last_pit = (side, idx)

    # Handle capture if last stone lands in empty pit on your side
    if (
        last_pit[0] == player and
        last_pit[1] < 6 and
        new_board[player][last_pit[1]] == 1
    ):
        opposite_pit = 5 - last_pit[1]
        captured = new_board[opponent][opposite_pit]
        if captured > 0:
            # 🧹 Remove the stone just placed (it's being captured)
            new_board[player][6] += captured + 1  # Only add what's captured + 1 for this stone
            new_board[player][last_pit[1]] = 0
            new_board[opponent][opposite_pit] = 0

    # Extra turn if last seed ends in own store
    extra_turn = last_pit[0] == player and last_pit[1] == 6

    # Terminal state seed sweeping
    if is_terminal(new_board):
        for i in range(6):
            new_board["player_1"][6] += new_board["player_1"][i]
            new_board["player_1"][i] = 0
            new_board["player_2"][6] += new_board["player_2"][i]
            new_board["player_2"][i] = 0

    return new_board, extra_turn

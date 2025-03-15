import copy

def initialize_board():
    return {
        "player_1": [4, 4, 4, 4, 4, 4, 0],
        "player_2": [4, 4, 4, 4, 4, 4, 0],
    }

def is_terminal(board):
    return sum(board["player_1"][:6]) == 0 or sum(board["player_2"][:6]) == 0

def evaluate(board, current_player):
    """Evaluate board state from perspective of current_player"""
    opponent = "player_2" if current_player == "player_1" else "player_1"
    
    my_store = board[current_player][6]
    opponent_store = board[opponent][6]

    if my_store > 24: return 1000
    if opponent_store > 24: return -1000

    store_diff = my_store - opponent_store
    pit_diff = sum(board[current_player][:6]) - sum(board[opponent][:6])
    
    # Capture potential calculation
    potential_captures = sum(1 for i in range(6) if 
        board[current_player][i] == 0 and board[opponent][5-i] > 0)
    
    # Extra turn potential calculation
    extra_turn_potential = sum(1 for i in range(6) if 
        (board[current_player][i] == (6 - i)))

    return (
        store_diff * 4 +
        pit_diff * 0.5 +
        potential_captures * 1.5 +
        extra_turn_potential * 4 -
        sum(board[opponent][:6]) * 0.2
    )

def get_valid_moves(board, player):
    return [i for i in range(6) if board[player][i] > 0]

def make_move(board, player, pit):
    new_board = copy.deepcopy(board)
    seeds = new_board[player][pit]
    new_board[player][pit] = 0
    index = pit
    player_turn = player

    while seeds > 0:
        index += 1
        if index == 7:
            if player_turn == "player_1":
                player_turn = "player_2"
                index = 0
            else:
                index = 0
        new_board[player_turn][index] += 1
        seeds -= 1

    if index < 6 and new_board[player_turn][index] == 1:
        opposite_index = 5 - index
        if player_turn == "player_1":
            captured = new_board["player_2"][opposite_index]
            new_board["player_2"][opposite_index] = 0
            new_board["player_1"][6] += captured + 1
            new_board["player_1"][index] = 0
        else:
            captured = new_board["player_1"][opposite_index]
            new_board["player_1"][opposite_index] = 0
            new_board["player_2"][6] += captured + 1
            new_board["player_2"][index] = 0

    return new_board, index == 6

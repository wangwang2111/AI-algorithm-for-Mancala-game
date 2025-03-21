from ai.rules import is_terminal, evaluate, get_valid_moves, make_move
import math

def simple_minimax(board, depth, current_player, maximizing_for):
    if depth == 0 or is_terminal(board):
        # Return evaluation from the perspective of the maximizing player
        return evaluate(board, maximizing_for), None
    
    valid_moves = get_valid_moves(board, current_player)
    if not valid_moves:
        return evaluate(board, maximizing_for), None

    best_move = None
    is_maximizing = current_player == maximizing_for

    if is_maximizing:
        max_eval = -math.inf
        for move in valid_moves:
            new_board, extra_turn = make_move(board, current_player, move)
            
            # Always decrement depth by 1
            new_depth = depth - 1
            
            # Determine next player
            next_player = current_player if extra_turn else (
                "player_2" if current_player == "player_1" else "player_1"
            )
            
            eval_score, _ = simple_minimax(
                new_board,
                new_depth,
                next_player,
                maximizing_for
            )
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

        return max_eval, best_move

    else:
        min_eval = math.inf
        for move in valid_moves:
            new_board, extra_turn = make_move(board, current_player, move)
            
            new_depth = depth - 1
            next_player = current_player if extra_turn else (
                "player_2" if current_player == "player_1" else "player_1"
            )
            
            eval_score, _ = simple_minimax(
                new_board,
                new_depth,
                next_player,
                maximizing_for
            )
            
            # Invert the evaluation for the minimizing player
            eval_score = -eval_score
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

        return min_eval, best_move
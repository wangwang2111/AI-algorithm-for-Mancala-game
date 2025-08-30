from mancala_ai.engine.core import _is_terminal_board, evaluate, legal_actions, _make_move_board
import math

def minimax_alpha_beta(board, depth, alpha, beta, current_player, maximizing_for):
    if depth == 0 or _is_terminal_board(board):
        return evaluate(board, maximizing_for), None
    
    valid_moves = legal_actions(board, current_player)
    if not valid_moves:
        return evaluate(board, maximizing_for), None

    best_move = None
    is_maximizing = current_player == maximizing_for

    if is_maximizing:
        max_eval = -math.inf
        for move in valid_moves:
            new_board, extra_turn = _make_move_board(board, current_player, move)
            next_player = current_player if extra_turn else ("player_2" if current_player == "player_1" else "player_1")
            new_depth = depth - 1  # Always decrement depth
            eval_score, _ = minimax_alpha_beta(new_board, new_depth, alpha, beta, next_player, maximizing_for)
            if eval_score > max_eval:
                max_eval, best_move = eval_score, move
            alpha = max(alpha, eval_score)
            if beta <= alpha: break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in valid_moves:
            new_board, extra_turn = _make_move_board(board, current_player, move)
            next_player = current_player if extra_turn else ("player_2" if current_player == "player_1" else "player_1")
            new_depth = depth - 1  # Always decrement depth
            eval_score, _ = minimax_alpha_beta(new_board, new_depth, alpha, beta, next_player, maximizing_for)
            if eval_score < min_eval:
                min_eval, best_move = eval_score, move
            beta = min(beta, eval_score)
            if beta <= alpha: break
        return min_eval, best_move
from ai.rules import is_terminal, evaluate, get_valid_moves, make_move
import math

def minimax_alpha_beta(board, depth, alpha, beta, current_player, maximizing_for):
    # Base case: terminal node or max depth reached
    if depth == 0 or is_terminal(board):
        return evaluate(board, maximizing_for), None  # Evaluate from the maximizing player's perspective
    
    valid_moves = get_valid_moves(board, current_player)
    if not valid_moves:
        return evaluate(board, maximizing_for), None

    best_move = valid_moves[0]
    is_maximizing = current_player == maximizing_for

    if is_maximizing:
        max_eval = -math.inf
        for move in valid_moves:
            new_board, extra_turn = make_move(board, current_player, move)

            # Determine the next player after move
            next_player = current_player if extra_turn else (
                "player_2" if current_player == "player_1" else "player_1"
            )

            new_depth = depth - (0 if extra_turn else 1)

            eval_score, _ = minimax_alpha_beta(
                new_board,
                new_depth,
                alpha,
                beta,
                next_player,
                maximizing_for
            )

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval, best_move

    else:
        min_eval = math.inf
        for move in valid_moves:
            new_board, extra_turn = make_move(board, current_player, move)

            next_player = current_player if extra_turn else (
                "player_2" if current_player == "player_1" else "player_1"
            )

            new_depth = depth - (0 if extra_turn else 1)

            eval_score, _ = minimax_alpha_beta(
                new_board,
                new_depth,
                alpha,
                beta,
                next_player,
                maximizing_for
            )

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval, best_move

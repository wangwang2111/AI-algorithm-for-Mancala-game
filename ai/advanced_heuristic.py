from ai.rules import is_terminal, evaluate, get_valid_moves, make_move
import math

def heuristic_move_value(board, move, player):
    """
    Assigns a heuristic score to a move for sorting and prioritization in the minimax search.
    Factors:
    - Extra turn potential (landing in store)
    - Capture potential
    - Pit difference advantage
    """
    player_pits = board[player]
    opponent = "player_1" if player == "player_2" else "player_2"
    opponent_pits = board[opponent]

    stones = player_pits[move]
    landing_pit = (move + stones) % 7
    value = 0

    # Prioritize extra turns
    if landing_pit == 6:
        value += 8

    # Prioritize captures
    if landing_pit < 6 and player_pits[landing_pit] == 0:
        opposite_pit = 5 - landing_pit
        captured_stones = board[opponent][opposite_pit]
        if captured_stones > 0:
            value += captured_stones * 0.5

    # Pit difference: Favor moves that leave us with more stones on our side
    pit_diff_after_move = (
        sum(player_pits[:6]) - player_pits[move]  # subtract stones being moved
        + (1 if landing_pit != 6 else 0)  # +1 if we land somewhere that isn't store
        + stones  # redistribute stones
    ) - sum(opponent_pits[:6])

    # Weigh the pit difference
    value += pit_diff_after_move * 0.2  

    return -value  # Negative so higher values come first when sorting (maximize benefit)


def advanced_heuristic_minimax(board, depth, alpha, beta, current_player, maximizing_for):
    if depth == 0 or is_terminal(board):
        return evaluate(board, maximizing_for), None  # Evaluate from maximizing_for's perspective

    valid_moves = get_valid_moves(board, current_player)
    if not valid_moves:
        return evaluate(board, maximizing_for), None

    # Sorting moves based on heuristic for current_player
    sorted_moves = sorted(
        valid_moves,
        key=lambda m: heuristic_move_value(board, m, current_player),
        reverse=True  # We want higher heuristic values first
    )

    best_move = sorted_moves[0]
    is_maximizing = current_player == maximizing_for

    if is_maximizing:
        max_eval = -math.inf
        for move in sorted_moves:
            new_board, extra_turn = make_move(board, current_player, move)

            next_player = current_player if extra_turn else (
                "player_1" if current_player == "player_2" else "player_2"
            )
            new_depth = depth - (0 if extra_turn else 1)

            eval_score, _ = advanced_heuristic_minimax(
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
        for move in sorted_moves:
            new_board, extra_turn = make_move(board, current_player, move)

            next_player = current_player if extra_turn else (
                "player_1" if current_player == "player_2" else "player_2"
            )
            new_depth = depth - (0 if extra_turn else 1)

            eval_score, _ = advanced_heuristic_minimax(
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

from ai.rules import is_terminal, evaluate, get_valid_moves, make_move
import math

def heuristic_move_value(board, move, maximizing_for):
    player_pits = board[maximizing_for]
    opponent = "player_1" if maximizing_for == "player_2" else "player_2"
    opponent_pits = board[opponent]
    stones = player_pits[move]
    value = 0

    # 1. Basic Movement Analysis
    landing_pit = (move + stones) % 14  # Simplified circular board assumption
    
    # 2. Immediate Rewards
    # - Extra turns
    if landing_pit == 6:  # Landing in own store
        value += 15  # Highest priority
        
    # - Capture potential
    if landing_pit < 6 and player_pits[landing_pit] == 0:
        captured = opponent_pits[5 - landing_pit]
        value += captured * 2  # Value actual captured stones

    # 3. Positional Advantage
    # - Central pit control (pits 2,3,4 are more valuable)
    position_weights = [0.5, 0.8, 1.2, 1.5, 1.2, 0.8]
    value += position_weights[move] * 2

    # 4. Future Game State
    # - Seeds that will end up in dangerous positions for opponent
    danger_zones = [0, 1, 5]  # Pits vulnerable to capture
    for i in range(stones):
        pit = (move + i + 1) % 14
        if pit in danger_zones and pit < 6:
            value += 0.3

    # 5. Defensive Considerations
    # - Block opponent's potential captures
    for opp_move in range(6):
        if opponent_pits[opp_move] == (13 - opp_move):  # Stones needed to reach our pit
            value += 1.5

    # 6. Progressive Game Phase
    total_seeds = sum(player_pits) + sum(opponent_pits)
    game_phase = 1 - (total_seeds / 96)  # 0=start, 1=endgame
    
    # - Endgame: Maximize store difference
    value += (player_pits[6] - opponent_pits[6]) * game_phase * 2
    
    # - Early game: Control pit advantage
    value += (sum(player_pits[:6]) - sum(opponent_pits[:6])) * (1 - game_phase) * 1.5

    # 7. Mobility
    # - Number of valid moves next turn
    future_moves = 0
    for i in range(6):
        if player_pits[i] > (6 - i):  # Can reach store
            future_moves += 1
    value += future_moves * 0.8

    # 8. Denial Strategy
    # - Prevent opponent from getting extra turns
    for i in range(6):
        if opponent_pits[i] == (13 - i):  # Stones needed for extra turn
            value -= 2  # Penalize moves that allow this

    # 9. Seed Conservation
    # - Avoid emptying pits unless beneficial
    if player_pits[move] == stones:  # This move empties the pit
        if landing_pit != 6:  # Only penalize if not scoring
            value -= 2

    # 10. Tempo Control
    # - Moves that force opponent into bad positions
    if stones > 10:  # Large seed groups create complex distributions
        value += 1.2

    return -value  # Negative for descending sort
def advanced_heuristic_minimax(board, depth, alpha, beta, current_player, maximizing_for):
    if depth == 0 or is_terminal(board):
        return evaluate(board, maximizing_for), None  # Evaluate from maximizing_for's perspective

    valid_moves = get_valid_moves(board, current_player)
    if not valid_moves:
        return evaluate(board, maximizing_for), None

    # Sort moves based on maximizing_for's perspective
    sorted_moves = sorted(
        valid_moves,
        key=lambda m: heuristic_move_value(board, m, maximizing_for),  # Use maximizing_for
        reverse=True
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

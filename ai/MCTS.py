import math
import random
from collections import defaultdict
import time
import copy
from functools import lru_cache
from ai.rules import initialize_board, get_valid_moves, make_move, is_terminal, evaluate

class MCTSNode:
    def __init__(self, board, player, parent=None, move=None):
        self.board = board  # Now storing original board (modified below)
        self.player = player
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.virtual_loss = 0
        self.untried_moves = get_valid_moves(self.board, player)
        
        # Progressive bias and RAVE
        self.heuristic_value = self.evaluate_board() if move is not None else 0
        self.rave_visits = defaultdict(int)
        self.rave_wins = defaultdict(float)
        
        # Depth tracking for adaptive strategies
        self.depth = parent.depth + 1 if parent else 0

    def board_to_tuple(self, board):
        """Convert board to hashable tuple"""
        return (
            tuple(board["player_1"]),
            tuple(board["player_2"])
        )

    def evaluate_board(self):
        """Quick evaluation from node's perspective"""
        return evaluate(self.board, self.player) / 24.0  # Normalized

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.4, heuristic_weight=0.3):
        """Enhanced child selection with RAVE and progressive bias"""
        if not self.children:
            return None
            
        log_visits = math.log(self.visits + 1e-10)
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            # UCT components
            exploit = (child.wins - child.virtual_loss) / (child.visits + 1e-5)
            explore = math.sqrt(log_visits / (child.visits + 1e-5))
            
            # Progressive bias (decays with visits)
            heuristic = child.heuristic_value * heuristic_weight / (child.visits**0.5 + 1)
            
            # RAVE component
            beta = math.sqrt(2000 / (3 * child.visits + 2000)) if child.visits > 0 else 1
            rave = beta * (self.rave_wins[child.move] / (self.rave_visits[child.move] + 1e-5))
            
            # Dynamic exploration decay
            dynamic_explore = exploration_weight * (0.99 ** self.depth)
            
            score = exploit + dynamic_explore * explore + heuristic + rave
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

    def expand(self):
        """Expansion with move ordering"""
        if not self.untried_moves:
            return None
            
        # Sort moves by heuristic (captures > extra turns > random)
        self.untried_moves.sort(key=lambda m: self.move_heuristic(m), reverse=True)
        
        move = self.untried_moves.pop()
        new_board, extra_turn = make_move(copy.deepcopy(self.board), self.player, move)
        next_player = self.player if extra_turn else self.opponent()
        
        child = MCTSNode(new_board, next_player, self, move)
        self.children.append(child)
        return child

    def move_heuristic(self, move):
        """Fast move evaluation without full simulation"""
        stones = self.board[self.player][move]
        landing_pit = move + stones
        if landing_pit % 13 == 6:  # Lands in store
            return 3
        if landing_pit < 6 and self.board[self.player][landing_pit] == 0:
            opposite_pit = 5 - landing_pit
            if self.board[self.opponent()][opposite_pit] > 0:
                return 2 + self.board[self.opponent()][opposite_pit]  # Capture bonus
        return 1 + stones/6  # Small bonus for spreading stones

    def rollout(self):
        """Enhanced rollout with adaptive policy"""
        current_board = copy.deepcopy(self.board)
        current_player = self.player
        rollout_moves = []
        max_depth = 10 + (48 - sum(current_board["player_1"][:6])) // 4  # Dynamic depth
        
        for _ in range(max_depth):
            if is_terminal(current_board):
                break
                
            moves = get_valid_moves(current_board, current_player)
            if not moves:
                break
                
            move = self.select_rollout_move(current_board, current_player, moves)
            rollout_moves.append(move)
            current_board, extra_turn = make_move(current_board, current_player, move)
            
            # Early termination if significant advantage
            if abs(evaluate(current_board, self.player)) > 12:
                break
                
            if not extra_turn:
                current_player = self.opponent()
                
        result = (evaluate(current_board, self.player) + 1) / 2  # Normalized to [0,1]
        return result, rollout_moves

    def select_rollout_move(self, board, player, moves):
        """Adaptive rollout policy (70% heuristic early, 30% late)"""
        game_phase = sum(board[player][:6])  # 0-24 stones remaining
        if random.random() < 0.7 - 0.4 * (game_phase / 24):
            # Heuristic selection
            best_score = -1
            best_move = moves[0]
            for move in moves:
                stones = board[player][move]
                landing_pit = move + stones
                if landing_pit % 13 == 6:  # Always take store moves
                    return move
                if landing_pit < 6 and board[player][landing_pit] == 0:
                    opposite_pit = 5 - landing_pit
                    if board[self.opponent()][opposite_pit] > 0:  # Capture
                        return move
                # Secondary heuristic
                score = stones + (6 - abs(3 - move))  # Prefer center pits
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_move
        return random.choice(moves)
    
    def backpropagate(self, result, rollout_moves):
        """Backpropagation with RAVE updates"""
        self.visits += 1
        self.wins += result
        self.virtual_loss = 0  # Reset if using parallel MCTS
        
        # Update RAVE statistics
        for move in rollout_moves:
            self.rave_visits[move] += 1
            self.rave_wins[move] += result
            
        if self.parent:
            self.parent.backpropagate(1 - result, rollout_moves)  # Alternate perspective

    def opponent(self):
        return "player_2" if self.player == "player_1" else "player_1"

    def update_pv(self):
        """Get principal variation (for debugging)"""
        if not self.children:
            return [self.move] if self.move else []
        best_child = max(self.children, key=lambda c: c.visits)
        return [self.move] + best_child.update_pv() if self.move else best_child.update_pv()


def mcts_decide(board, player, time_limit=3, simulations=2000):
    """Optimized MCTS decision function"""
    root = MCTSNode(copy.deepcopy(board), player)
    start_time = time.time()
    simulations_done = 0
    
    while time.time() - start_time < time_limit and simulations_done < simulations:
        node = root
        
        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            node.virtual_loss += 1
        
        # Expansion
        if not is_terminal(node.board) and not node.is_fully_expanded():
            node = node.expand()
        
        # Simulation
        result, rollout_moves = node.rollout()
        
        # Backpropagation
        node.backpropagate(result, rollout_moves)
        simulations_done += 1
    
    # Select best move
    if not root.children:
        valid_moves = get_valid_moves(board, player)
        return random.choice(valid_moves) if valid_moves else None
    
    best_child = max(root.children, key=lambda c: (
        c.visits,
        c.wins / (c.visits + 1e-5)
    ))
    
    # print(f"Best move: {best_child.move} (visits: {best_child.visits}, win rate: {best_child.wins/best_child.visits:.2%})")
    return best_child.move
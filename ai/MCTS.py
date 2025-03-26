import math
import random
import copy
from ai.rules import initialize_board, get_valid_moves, make_move, is_terminal, evaluate
import time

class MCTSNode:
    def __init__(self, board, player, parent=None, move=None):
        self.board = copy.deepcopy(board)
        self.player = player
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = get_valid_moves(self.board, player)
        
        # Store heuristic value for progressive bias
        self.heuristic_value = self.evaluate_board() if move is not None else 0

    def evaluate_board(self):
        """Evaluate the current board state from the node's player perspective"""
        return evaluate(self.board, self.player)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.4, heuristic_weight=0.3):
        """Select best child with combined UCT and heuristic value"""
        if not self.children:
            return None
            
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            # Normalize exploitation term
            exploitation = child.wins / child.visits if child.visits > 0 else 0
            
            # Exploration term
            exploration = math.sqrt(math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')
            
            # Heuristic term (progressive bias)
            heuristic = child.heuristic_value * heuristic_weight / (child.visits + 1)
            
            # Combined score
            score = exploitation + exploration_weight * exploration + heuristic
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

    def expand(self):
        """Expand a node with heuristic-based move ordering"""
        if not self.untried_moves:
            return None
            
        # Sort untried moves by heuristic value
        self.untried_moves.sort(
            key=lambda move: self.move_heuristic(move),
            reverse=True
        )
        
        move = self.untried_moves.pop()
        new_board, extra_turn = make_move(copy.deepcopy(self.board), self.player, move)
        next_player = self.player if extra_turn else ("player_2" if self.player == "player_1" else "player_1")
        child_node = MCTSNode(new_board, next_player, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def move_heuristic(self, move):
        """Simple heuristic to evaluate a move without making it"""
        stones = self.board[self.player][move]
        landing_pit = move + stones
        if landing_pit % 13 == 6:  # Landing in store
            return 3
        if landing_pit < 6 and self.board[self.player][landing_pit] == 0:
            opposite_pit = 5 - landing_pit
            if self.board[self.opponent()][opposite_pit] > 0:
                return 2 + self.board[self.opponent()][opposite_pit]
        return 1

    def opponent(self):
        return "player_2" if self.player == "player_1" else "player_1"

    def rollout(self):
        """Enhanced rollout with heuristic-based moves"""
        current_board = copy.deepcopy(self.board)
        current_player = self.player
        rollout_depth = 0
        max_depth = 20
        
        while not is_terminal(current_board) and rollout_depth < max_depth:
            moves = get_valid_moves(current_board, current_player)
            if not moves:
                break
                
            # Use heuristic to select moves during rollout
            move = self.select_rollout_move(current_board, current_player, moves)
            current_board, extra_turn = make_move(current_board, current_player, move)
            
            if not extra_turn:
                current_player = "player_2" if current_player == "player_1" else "player_1"
            rollout_depth += 1

        # Return evaluation score normalized to [0, 1]
        return (self.evaluate_rollout_result(current_board) + 1) / 2  # Normalize to 0-1 range

    def select_rollout_move(self, board, player, moves):
        """Select moves during rollout with heuristic guidance"""
        if random.random() < 0.7:  # 70% heuristic, 30% random
            best_score = -1
            best_move = moves[0]
            for move in moves:
                stones = board[player][move]
                landing_pit = move + stones
                if landing_pit % 13 == 6:
                    return move  # Always prefer moves that land in store
                if landing_pit < 6 and board[player][landing_pit] == 0:
                    opposite_pit = 5 - landing_pit
                    if board[self.opponent()][opposite_pit] > 0:
                        return move  # Always prefer capturing moves
            return random.choice(moves)
        else:
            return random.choice(moves)

    def evaluate_rollout_result(self, board):
        """Evaluate the final board state from node's player perspective"""
        return evaluate(board, self.player) / 24.0  # Normalize by max possible score difference

    def backpropagate(self, result):
        """Backpropagate the simulation result"""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(1 - result)  # Alternate perspective for opponent

    def _get_winner(self, board):
        """Determine the winner from the board state"""
        if not is_terminal(board):
            return None
        p1_score = board["player_1"][6]
        p2_score = board["player_2"][6]
        if p1_score > p2_score:
            return "player_1"
        elif p2_score > p1_score:
            return "player_2"
        else:
            return "draw"


def mcts_decide(board, player, simulations=1000, time_limit=5):
    """MCTS decision function with time limit and simulation count"""
    root = MCTSNode(board, player)
    start_time = time.time()
    simulations_done = 0
    
    while (time.time() - start_time < time_limit) and (simulations_done < simulations):
        node = root
        
        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        
        # Expansion
        if not is_terminal(node.board) and not node.is_fully_expanded():
            node = node.expand()
        
        # Simulation
        result = node.rollout()
        
        # Backpropagation
        node.backpropagate(result)
        simulations_done += 1
    
    # Select move with highest visit count
    if not root.children:
        valid_moves = get_valid_moves(board, player)
        return random.choice(valid_moves) if valid_moves else None
    
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.move
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, BatchNormalization, Add, Dropout
from tensorflow.keras.regularizers import l2
import math
import random
from collections import defaultdict, deque
from ai.rules import initialize_board, get_valid_moves, make_move, is_terminal

class NeuralMCTSHybrid:
    def __init__(self, model_path=None, simulations=800, c_puct=1.5):
        self.simulations = simulations
        self.c_puct = c_puct  # exploration constant
        self.Q = defaultdict(list)  # action values (now properly initialized as lists)
        self.N = defaultdict(list)  # visit counts (now properly initialized as lists)
        self.P = {}                 # policy predictions
        self.virtual_loss = 0.5     # virtual loss for parallel exploration
        
        # Load or create neural network
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = self.build_improved_model()
        
        # Initialize optimizer with learning rate schedule
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    def build_improved_model(self):
        """Build an enhanced dual-headed neural network with residual connections"""
        board_input = Input(shape=(2, 6, 1))  # 2 players x 6 pits
        
        # Initial convolutional layer
        x = Conv2D(128, (2, 2), activation='relu', padding='same')(board_input)
        x = BatchNormalization()(x)
        
        # Residual block 1
        residual = x
        x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, residual])
        
        # Residual block 2
        residual = x
        x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, residual])
        
        x = Flatten()(x)
        
        # Intermediate dense layers
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.0001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Policy head
        policy = Dense(128, activation='relu')(x)
        policy = Dense(6, activation='softmax', name='policy')(policy)
        
        # Value head
        value = Dense(128, activation='relu')(x)
        value = Dense(1, activation='tanh', name='value')(value)
        
        model = Model(inputs=board_input, outputs=[policy, value])
        
        # Compile with separate losses
        model.compile(
            optimizer=self.optimizer,
            loss={'policy': 'categorical_crossentropy', 'value': 'mse'},
            loss_weights={'policy': 1.0, 'value': 1.0}
        )
        
        return model
    
    def board_to_input(self, board):
        """Convert game board to neural network input format"""
        p1_pits = np.array(board['player_1'][:6]).reshape(1, 6, 1)
        p2_pits = np.array(board['player_2'][:6]).reshape(1, 6, 1)
        return np.concatenate([p1_pits, p2_pits], axis=0)
    
    def predict(self, board):
        """Neural network prediction with caching"""
        nn_input = self.board_to_input(board)
        state_key = tuple(nn_input.flatten())
        
        if state_key in self.P:
            return self.P[state_key], np.mean(self.Q[state_key]) if self.Q[state_key] else 0
        
        policy, value = self.model.predict(nn_input[np.newaxis, ...], verbose=0)
        return policy[0], value[0][0]
    
    def search(self, board, player, temperature=1.0):
        """Perform MCTS search guided by neural network with temperature"""
        root_state_key = tuple(self.board_to_input(board).flatten())
        
        # Add Dirichlet noise to root node for exploration
        if root_state_key in self.P:
            noise = np.random.dirichlet([0.03] * 6)
            self.P[root_state_key] = 0.75 * self.P[root_state_key] + 0.25 * noise
        
        for _ in range(self.simulations):
            self._simulate(board.copy(), player)
        
        moves = get_valid_moves(board, player)
        visit_counts = np.array([self.N[root_state_key][m] if m < len(self.N[root_state_key]) else 0 for m in moves])
        
        # Apply temperature
        if temperature > 0:
            visit_counts = visit_counts ** (1 / temperature)
            probs = visit_counts / np.sum(visit_counts)
            return np.random.choice(moves, p=probs)
        return moves[np.argmax(visit_counts)]
    
    def _simulate(self, board, player):
        """Enhanced MCTS simulation with virtual loss"""
        path = []
        current_player = player
        
        while not is_terminal(board):
            state_key = tuple(self.board_to_input(board).flatten())
            moves = get_valid_moves(board, current_player)
            
            if state_key not in self.P:
                # Neural network expansion
                policy, value = self.predict(board)
                self.P[state_key] = policy
                self.Q[state_key] = [0.0] * 6
                self.N[state_key] = [0] * 6
                
                # Apply virtual loss
                for move in moves:
                    if move < len(self.Q[state_key]):  # Safety check
                        self.Q[state_key][move] -= self.virtual_loss
                        self.N[state_key][move] += 1
                break
            
            # Select move with highest UCB score
            best_score = -float('inf')
            best_move = moves[0]  # Default to first move
            
            total_visits = sum(self.N[state_key])
            
            for move in moves:
                if move >= len(self.P[state_key]):  # Safety check
                    continue
                    
                u = (self.Q[state_key][move] if move < len(self.Q[state_key]) else 0) + \
                    self.c_puct * self.P[state_key][move] * \
                    math.sqrt(total_visits) / (1 + (self.N[state_key][move] if move < len(self.N[state_key]) else 0))
                
                if u > best_score:
                    best_score = u
                    best_move = move
            
            path.append((state_key, best_move, current_player))
            board, extra_turn = make_move(board, current_player, best_move)
            if not extra_turn:
                current_player = 'player_2' if current_player == 'player_1' else 'player_1'
        
        # Evaluation
        if is_terminal(board):
            winner = self._get_winner(board)
            value = 1.0 if winner == player else (-1.0 if winner else 0.0)
        else:
            _, value = self.predict(board)
            # Scale value based on remaining stones
            stones_remaining = sum(board['player_1'][:6]) + sum(board['player_2'][:6])
            value *= (1 - stones_remaining / 72)  # 72 is total stones
        
        # Backpropagation
        for state_key, move, node_player in reversed(path):
            if move >= len(self.Q[state_key]):  # Safety check
                continue
                
            perspective = 1 if node_player == player else -1
            self.N[state_key][move] += 1
            self.Q[state_key][move] += (perspective * value - self.Q[state_key][move]) / self.N[state_key][move]
            value = -value  # Alternate players
    
    def _get_winner(self, board):
        """Determine game winner"""
        p1_score = board['player_1'][6]
        p2_score = board['player_2'][6]
        if p1_score > p2_score:
            return 'player_1'
        elif p2_score > p1_score:
            return 'player_2'
        return None  # Draw
    
    def train(self, examples, epochs=10, batch_size=128):
        """Enhanced training process with validation"""
        if not examples:
            return
        
        states = []
        policy_targets = []
        value_targets = []
        
        for board, policy, value in examples:
            states.append(self.board_to_input(board))
            policy_targets.append(policy)
            value_targets.append(value)
        
        # Convert to numpy arrays
        states = np.array(states)
        policy_targets = np.array(policy_targets)
        value_targets = np.array(value_targets)
        
        # Train with validation split
        self.model.fit(
            states,
            {'policy': policy_targets, 'value': value_targets},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        
# Initialize
mcts = NeuralMCTSHybrid()

# During self-play
board = initialize_board()
move = mcts.search(board, 'player_1', temperature=1.0)  # Higher temp for exploration

# During competitive play
move = mcts.search(board, 'player_1', temperature=0.1)  # Lower temp for exploitation

# Training
mcts.train(examples, epochs=10, batch_size=128)
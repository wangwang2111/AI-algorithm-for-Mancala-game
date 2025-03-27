import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from ai.rules import initialize_board, get_valid_moves, make_move, is_terminal

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Add these to your training loop
win_rates = []
avg_rewards = []
loss_history = []

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class MancalaDQN:
    def __init__(self, state_shape=(2, 7, 2), action_size=6):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(capacity=50000)
        self.gamma = 0.99   # discount factor
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_decay = 0.995  # Slower decay
        self.epsilon_min = 0.05
        self.batch_size = 64  # Smaller batch size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.tensorboard = TensorBoard(log_dir='./logs/dqn')
    
        # Target network update parameters
        self.target_update_freq = 100  # Update every 100 training steps
        self.train_step = 0  # Counter for training steps
    
    def _build_model(self):
        """Simpler network with dense layers only"""
        inputs = Input(shape=self.state_shape)
        x = Flatten()(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.action_size, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='huber', optimizer=Adam(learning_rate=0.001))
        return model
        
    def update_target_model(self, tau=0.01):
        """Soft target network update with Polyak averaging"""
        q_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        self.target_model.set_weights(
            [tau*w + (1-tau)*tw for w, tw in zip(q_weights, target_weights)]
        )
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
    
    
    def get_action(self, state, valid_moves):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
        
        q_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
        return valid_moves[np.argmax(q_values[valid_moves])]
    
    def replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return None
        
        # Increment training step counter
        self.train_step += 1
        
        samples, indices, weights = self.memory.sample(self.batch_size)
        
        states = np.array([sample[0] for sample in samples])
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples])
        next_states = np.array([sample[3] for sample in samples])
        dones = np.array([sample[4] for sample in samples])
        
        # Double DQN update
        current_q = self.model.predict(next_states, verbose=0)
        target_q = self.target_model.predict(next_states, verbose=0)
        
        targets = self.model.predict(states, verbose=0)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        targets[batch_index, actions] = rewards + self.gamma * target_q[
            batch_index, np.argmax(current_q, axis=1)] * (1 - dones)
        
        # Train with importance sampling weights
        history = self.model.fit(
            states, targets,
            batch_size=self.batch_size,
            sample_weight=weights,
            verbose=0
        )
        
        # Update priorities
        preds = self.model.predict(states, verbose=0)
        errors = np.abs(targets - preds).mean(axis=1)
        self.memory.update_priorities(indices, errors)
        
        # Update target network periodically
        if self.train_step % self.target_update_freq == 0:
            self.update_target_model()
            print(f"Updated target network at step {self.train_step}")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
        
    def preprocess_state(self, board, current_player):
        state = np.zeros((2, 7, 2))
        state[:, :, 0] = np.array([
            board['player_1'][:6] + [board['player_1'][6]],
            board['player_2'][:6] + [board['player_2'][6]]
        ])
        state[:, :, 1] = 1 if current_player == 'player_1' else -1
        return state
    
    def evaluate(self, test_env, test_games=20, opponent='rules'):
        """Evaluate agent against specified opponent type
        Args:
            test_env: Environment instance
            test_games: Number of games to play
            opponent: Type of opponent ('random' or 'rules')
        Returns:
            avg_reward: Average reward per game
            win_rate: Percentage of games won
            avg_moves: Average moves per game
        """
        total_reward = 0
        wins = 0
        total_moves = 0
        
        for _ in range(test_games):
            test_env.reset()
            state = self.preprocess_state(test_env.board, test_env.current_player)
            done = False
            game_moves = 0
            
            while not done:
                game_moves += 1
                valid_moves = test_env.get_valid_moves()
                
                # Agent's turn
                if test_env.current_player == test_env.agent_player:
                    action = self.get_action(state, valid_moves)
                # Opponent's turn
                else:
                    if opponent == 'random':
                        action = random.choice(valid_moves)
                    elif opponent == 'rules':
                        action = self._rules_based_opponent(test_env.board, test_env.current_player)
                
                _, reward, done = test_env.step(action)
                state = self.preprocess_state(test_env.board, test_env.current_player)
                
                if done:
                    total_reward += reward
                    wins += 1 if reward > 0 else 0
                    total_moves += game_moves
        
        avg_reward = total_reward / test_games
        win_rate = wins / test_games
        avg_moves = total_moves / test_games
        
        return avg_reward, win_rate, avg_moves

    def _rules_based_opponent(self, board, current_player):
        """Simple rules-based opponent for evaluation"""
        valid_moves = get_valid_moves(board, current_player)
        
        # 1. Prioritize extra turns
        for move in valid_moves:
            stones = board[current_player][move]
            if (move + stones) % 14 == 6:  # Lands in store
                return move
        
        # 2. Prioritize captures
        for move in valid_moves:
            stones = board[current_player][move]
            landing_pit = (move + stones) % 14
            if landing_pit < 6 and board[current_player][landing_pit] == 0:
                if board[opponent_player(current_player)][5 - landing_pit] > 0:
                    return move
        
        # 3. Choose move with most stones
        return max(valid_moves, key=lambda x: board[current_player][x])

    def train(self, env, episodes=1000, early_stop=True):
        """Training loop with early stopping capabilities"""
        win_rates = []
        avg_rewards = []
        avg_moves = []
        best_win_rate = -np.inf
        no_improvement_count = 0
        patience = 20  # Episodes to wait before stopping
        plateau_window = 10  # Window for reward plateau detection
        min_improvement = 0.01  # Minimum win rate improvement
        
        test_env = MancalaEnv(agent_player='player_1')
        
        for e in range(episodes):
            state = env.reset()
            state = self.preprocess_state(env.board, env.current_player)
            done = False
            total_reward = 0
            
            while not done:
                valid_moves = env.get_valid_moves()
                action = self.get_action(state, valid_moves)
                
                _, reward, done = env.step(action)
                next_state = self.preprocess_state(env.board, env.current_player)
                
                self.remember(state, action, reward, next_state, done)
                loss = self.replay()
                
                state = next_state
                total_reward += reward
                
                if done:
                    # Save checkpoint periodically
                    if e % 100 == 0:
                        self.save_model(f"checkpoint_ep{e}.h5")
                    
                    # Evaluation and logging
                    if e % 10 == 0:
                        q_values = self.model.predict(state[np.newaxis,...], verbose=0)[0]
                        print(f"Episode {e+1} - Epsilon: {self.epsilon:.3f}")
                        print(f"Max Q-value: {np.max(q_values):.2f}, Min Q-value: {np.min(q_values):.2f}")
                        print(f"Gradients: {[np.mean(layer.weights[0].numpy()) for layer in self.model.layers if layer.weights]}")
                    
                    # Full evaluation every 50 episodes
                    if e % 50 == 0 or e == episodes-1:
                        test_avg_reward, test_win_rate, test_move = self.evaluate(test_env, test_games=20)
                        avg_rewards.append(test_avg_reward)
                        win_rates.append(test_win_rate)
                        avg_moves.append(test_move)
                        
                        print(f"Episode {e+1}/{episodes} - "
                            f"Win Rate: {test_win_rate:.2%} - "
                            f"Avg Reward: {test_avg_reward:.2f} - "
                            f"Avg Move: {test_move:.2f} - "
                            f"Loss: {loss if loss is not None else 0:.4f}")
                        
                        # Early stopping checks
                        if early_stop:
                            # Condition 1: Win rate threshold (95%)
                            if test_win_rate >= 0.95:
                                print(f"Early stopping: Reached 95% win rate")
                                self.save_model("mancala_dqn_final.h5")
                                return win_rates, avg_rewards
                            
                            # Condition 2: Improvement check
                            if test_win_rate > best_win_rate + min_improvement:
                                best_win_rate = test_win_rate
                                no_improvement_count = 0
                                self.save_model("mancala_dqn_best.h5")
                            else:
                                no_improvement_count += 1
                            
                            # Condition 3: Patience exceeded
                            if no_improvement_count >= patience:
                                print(f"Early stopping: No improvement for {patience} evaluations")
                                self.save_model("mancala_dqn_final.h5")
                                return win_rates, avg_rewards
                            
                            # Condition 4: Reward plateau (last N evaluations within 1% range)
                            if len(avg_rewards) >= plateau_window:
                                recent_rewards = avg_rewards[-plateau_window:]
                                if max(recent_rewards) - min(recent_rewards) < 0.5:
                                    print(f"Early stopping: Reward plateau detected")
                                    self.save_model("mancala_dqn_final.h5")
                                    return win_rates, avg_rewards
                    break
        
        self.save_model("mancala_dqn_final.h5")
        return win_rates, avg_rewards

    def save_model(self, filepath):
        self.model.save(filepath)

def opponent_player(current_player):
    return 'player_2' if current_player == 'player_1' else 'player_1'

class MancalaEnv:
    def __init__(self, agent_player='player_1'):
        self.board = initialize_board()
        self.current_player = 'player_1'
        self.agent_player = agent_player
        self.opponent_player = 'player_2' if agent_player == 'player_1' else 'player_1'
    
    def reset(self):
        self.board = initialize_board()
        self.current_player = 'player_1'
        return self.board
    
    def step(self, action):
        # Execute move
        self.board, extra_turn = make_move(self.board, self.current_player, action)
        
        # Calculate reward from agent's perspective
        reward = self._calculate_reward()
        
        # Switch players if no extra turn
        if not extra_turn:
            self.current_player = self.opponent_player if self.current_player == self.agent_player else self.agent_player
        
        done = is_terminal(self.board)
        return self.board, reward, done
    
    def get_valid_moves(self):
        return get_valid_moves(self.board, self.current_player)
    
    def _calculate_reward(self, last_move=None):
        """Advanced reward function combining:
        - Immediate rewards (captures, extra turns)
        - Positional advantages
        - Defensive considerations
        - Game phase awareness
        - Mobility and tempo control
        """
        # Base values
        my_store = self.board[self.agent_player][6]
        opp_store = self.board[self.opponent_player][6]
        my_pits = self.board[self.agent_player][:6]
        opp_pits = self.board[self.opponent_player][:6]
        total_seeds = sum(my_pits) + sum(opp_pits) + my_store + opp_store
        
        # Terminal state reward
        if is_terminal(self.board):
            return 5.0 if my_store > 24 else (-5.0 if my_store < 24 else 0)
        
        # 1. Game phase calculation (0=early, 1=late)
        game_phase = 1 - (total_seeds / 96.0)
        
        # 2. Immediate Rewards -----------------------------------------------------
        reward = 0
        
        # Extra turn detection (needs move tracking)
        if last_move is not None:
            stones = my_pits[last_move]
            landing_pit = (last_move + stones) % 14
            if landing_pit == 6:  # Extra turn
                reward += 3.0
        
        # 3. Capture Analysis ------------------------------------------------------
        def calculate_captures(pits, opp_pits):
            captures = 0
            for i in range(6):
                if pits[i] == 0 and opp_pits[5-i] > 0:
                    # Weight by position (central pits more valuable)
                    position_weights = [0.5, 0.8, 1.2, 1.5, 1.2, 0.8]
                    captures += opp_pits[5-i] * position_weights[i]
            return captures
        
        # Positive captures
        my_captures = calculate_captures(my_pits, opp_pits)
        reward += my_captures * 0.4
        
        # Negative captures (opponent's opportunities)
        opp_captures = calculate_captures(opp_pits, my_pits)
        reward -= opp_captures * 0.6  # Higher penalty
        
        # 4. Positional Advantage --------------------------------------------------
        position_weights = [0.5, 0.8, 1.2, 1.5, 1.2, 0.8]
        positional_value = sum(w*p for w,p in zip(position_weights, my_pits)) / 10.0
        reward += positional_value
        
        # 5. Defensive Considerations ----------------------------------------------
        # Block opponent's extra turns
        for i in range(6):
            if opp_pits[i] == (13 - i):  # Would give extra turn
                reward -= 2.0 * position_weights[i]
        
        # 6. Progressive Rewards ---------------------------------------------------
        # Late game: store difference matters more
        store_diff = (my_store - opp_store) / 24.0
        reward += store_diff * (3.0 * game_phase)
        
        # Early game: pit control matters more
        pit_diff = (sum(my_pits) - sum(opp_pits)) / 24.0
        reward += pit_diff * (2.5 * (1 - game_phase))
        
        # 7. Mobility and Tempo ----------------------------------------------------
        # Valid moves that can reach the store
        future_moves = sum(1 for i in range(6) if my_pits[i] >= (6 - i))
        reward += future_moves * 0.2
        
        # Large seed groups (tempo control)
        large_groups = sum(1 for s in my_pits if s > 10)
        reward += large_groups * 0.3
        
        # 8. Stone Conservation ----------------------------------------------------
        if last_move is not None:
            if my_pits[last_move] == 0:  # Emptied a pit
                reward -= 1.0 if game_phase < 0.5 else 0.5
        
        # Normalize and return
        return np.clip(reward, -5.0, 5.0)

# Training
if __name__ == "__main__":
    env = MancalaEnv(agent_player='player_1')
    agent = MancalaDQN(state_shape=(2,7,2))  # 2 players x 7 positions x 2 channels (stones + turn)
    
    print("Starting training...")
    agent.train(env, episodes=1000)
    agent.save_model("mancala_dqn_improved.h5")
    print("Training completed. Model saved.")
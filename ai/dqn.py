import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import copy
from ai.rules import initialize_board, get_valid_moves, make_move, is_terminal

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def opponent_player(current_player):
    """Utility function to determine the opponent of the given player."""
    return 'player_2' if current_player == 'player_1' else 'player_1'

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

class DQNAgent:
    def __init__(self, state_shape=(15,), action_size=6):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(capacity=1000000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.9985
        self.epsilon_min = 0.09
        self.batch_size = 256
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()  # Initialize target network
        self.tensorboard = TensorBoard(log_dir='./logs/dqn')
        self.summary_writer = tf.summary.create_file_writer('./logs/dqn')
        self.target_update_freq = 200
        self.train_step = 0
        self.tau = 0.005  # For soft updates (add this)
        
    def _build_model(self):
        inputs = Input(shape=self.state_shape)
        x = Dense(512, activation='relu')(inputs)  # Increased from 256
        x = tf.keras.layers.Dropout(0.3)(x)  # Increased from 0.2
        x = Dense(256, activation='relu')(x)  # Additional layer
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.action_size, 
                    activation='linear',
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.5))(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=tf.keras.losses.Huber(delta=0.2),  # Increased from 0.1
                    optimizer=Adam(learning_rate=0.0003))  # Reduced from 0.0005
        return model
    
    def update_target_model(self, tau=0.01):
        """Soft update target network weights."""
        q_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        self.target_model.set_weights([tau * w + (1 - tau) * tw for w, tw in zip(q_weights, target_weights)])

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def get_action(self, state, valid_moves):
        if not valid_moves:  # No valid moves available
            return None
            
        if np.random.rand() <= self.epsilon:  # Exploration
            return random.choice(valid_moves)  # Only choose valid moves
        
        # Exploitation with masking
        q_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
        masked_q = np.full_like(q_values, -np.inf)
        masked_q[valid_moves] = q_values[valid_moves]  # Only consider valid moves
        
        return np.argmax(masked_q)
        
    def replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return None

        self.train_step += 1
        # Sample with annealed beta value
        samples, indices, weights = self.memory.sample(
            self.batch_size,
            beta=min(0.4 + self.train_step / 100000, 1.0)
        )
        
        states = np.array([sample[0] for sample in samples])
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples])
        next_states = np.array([sample[3] for sample in samples])
        dones = np.array([sample[4] for sample in samples])

        # Single forward pass for all predictions
        current_predictions = self.model.predict(states, verbose=0)
        current_q = self.model.predict(next_states, verbose=0)
        target_q = self.target_model.predict(next_states, verbose=0)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        best_next_actions = np.argmax(current_q, axis=1)
        
        # Compute targets and advantages together
        targets = current_predictions.copy()
        td_targets = rewards + self.gamma * target_q[batch_index, best_next_actions] * (1 - dones)
        targets[batch_index, actions] = td_targets
        
        # Advantage calculation
        selected_actions_values = current_predictions[batch_index, actions]
        advantages = td_targets - selected_actions_values

        # Train model
        history = self.model.fit(
            states, 
            targets, 
            batch_size=self.batch_size, 
            sample_weight=weights, 
            verbose=0, 
            callbacks=[self.tensorboard]
        )
        
        # Update priorities
        errors = np.abs(advantages)  # Using advantages as TD errors
        self.memory.update_priorities(indices, errors)
        
        # Logging
        with self.summary_writer.as_default():
            tf.summary.histogram("advantages", advantages, step=self.train_step)
            tf.summary.scalar("avg_advantage", np.mean(advantages), step=self.train_step)
            tf.summary.scalar("max_q_value", np.max(current_predictions), step=self.train_step)

        if self.train_step % self.target_update_freq == 0:
            self.update_target_model()
            print(f"Updated target network at step {self.train_step}")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0]

    def preprocess_state(self, board, current_player, agent_player):
        """
        Preprocess the board state into a consistent 14-length vector (normalized 0-1).
        The state always represents the board from the agent's perspective.
        """
        # Normalize all values by dividing by max possible stones
        p1_pits = np.array(board['player_1'][:6]) / 24.0  # Max 4*6=24 stones in pits
        p1_store = board['player_1'][6] / 48.0  # Max 48 stones in store
        p2_pits = np.array(board['player_2'][:6]) / 24.0
        p2_store = board['player_2'][6] / 48.0
        
        if agent_player == 'player_1':
            # Agent is player_1 - state is [p1_pits, p1_store, p2_pits, p2_store, turn_indicator]
            turn_indicator = 1.0 if current_player == 'player_1' else -1.0
            return np.concatenate([p1_pits, [p1_store], p2_pits, [p2_store], [turn_indicator]])
        else:
            # Agent is player_2 - rotate perspective so agent's pits come first
            turn_indicator = 1.0 if current_player == 'player_2' else -1.0
            return np.concatenate([p2_pits, [p2_store], p1_pits, [p1_store], [turn_indicator]])

    def _rules_based_opponent(self, board, current_player):
        """
        A simple rules-based opponent. It first looks for a move that grants an extra turn,
        then looks for a capturing move, and finally selects the move with the most seeds.
        """
        valid_moves = get_valid_moves(board, current_player)
        for move in valid_moves:
            stones = board[current_player][move]
            if (move + stones) % 14 == 6:
                return move
        for move in valid_moves:
            stones = board[current_player][move]
            landing_pit = (move + stones) % 14
            if landing_pit < 6 and board[current_player][landing_pit] == 1:
                opp = opponent_player(current_player)
                if board[opp][5 - landing_pit] > 0:
                    return move
        return max(valid_moves, key=lambda x: board[current_player][x])
    
    def evaluate(self, test_env, test_games=20, opponent='rules'):
        """
        Evaluate the agent against a specified opponent type.
        Tests both player_1 and player_2 perspectives for balanced evaluation.
        
        Returns:
            avg_reward: Average reward per game
            win_rate: Percentage of games won
            avg_moves: Average moves per game
        """
        original_epsilon = self.epsilon
        self.epsilon = 0  # Use greedy policy for evaluation
        
        metrics = {
            'wins': 0,
            'losses': 0,
            'ties': 0,
            'total_reward': 0,
            'total_moves': 0,
            'store_diffs': []
        }
        
        
        # Test both player perspectives
        for agent_player in ['player_1', 'player_2']:
            test_env.agent_player = agent_player
            test_env.opponent_player = 'player_2' if agent_player == 'player_1' else 'player_1'
            
            for _ in range(test_games // 2):  # Split games between perspectives
                test_env.reset()
                state = self.preprocess_state(test_env.board, test_env.current_player, test_env.agent_player)
                done = False
                game_moves = 0
                game_reward = 0
                
                while not done:
                    game_moves += 1
                    valid_moves = test_env.get_valid_moves(test_env.current_player)
                    
                    if test_env.current_player == test_env.agent_player:
                        action = self.get_action(state, valid_moves)
                        _, reward, done = test_env.step(action)
                        game_reward += reward
                    else:
                        if opponent == 'random':
                            action = random.choice(valid_moves)
                        elif opponent == 'rules':
                            action = self._rules_based_opponent(test_env.board, test_env.current_player)
                        _, _, done = test_env.step(action)
                    
                    state = self.preprocess_state(test_env.board, test_env.current_player, test_env.agent_player)
                    
                    # In evaluate(), add tie handling:
                    if done:
                        store_diff = test_env.board[test_env.agent_player][6] - test_env.board[test_env.opponent_player][6]
                        metrics['store_diffs'].append(store_diff)
                        metrics['total_reward'] += game_reward / game_moves
                        metrics['total_moves'] += game_moves
                        
                        if store_diff > 0:
                            metrics['wins'] += 1
                        elif store_diff < 0:
                            metrics['losses'] += 1
                        else:
                            metrics['ties'] += 1
        
        self.epsilon = original_epsilon
        # In your training loop:
        # avg_rewards = metrics['total_reward'] / test_games
        # rewards = (avg_rewards - np.mean(avg_rewards)) / (np.std(avg_rewards) + 1e-8)
        return {
            'win_rate': metrics['wins'] / test_games,
            'avg_reward': metrics['total_reward'] / test_games,
            'avg_moves': metrics['total_moves'] / test_games,
            'avg_store_diff': np.mean(metrics['store_diffs']),
            'std_store_diff': np.std(metrics['store_diffs'])
        }

    def train(self, env, episodes=1000, early_stop=True):
        """
        Training loop for the DQNAgent with early stopping capabilities.
        """
        win_rates = []
        avg_rewards = []
        avg_moves = []
        best_win_rate = -np.inf
        no_improvement_count = 0
        patience = 200         # Episodes to wait before stopping
        plateau_window = 25    # Window for reward plateau detection
        min_improvement = 0.01 # Minimum win rate improvement
        
        test_env = MancalaEnv(agent_player='player_1')
        
        for e in range(episodes):
            # Alternate between random and rules-based opponents every 10 episodes
            # opponent_type = 'random' if (e // 25) % 2 == 0 else 'rules'  # Slower alternation
            # Dynamic opponent scheduling
            if e < episodes//3:
                opponent_type = 'random'  # Start with random opponent
            elif e < 2*episodes//3:
                opponent_type = 'random' if e % 20 < 10 else 'rules'  # Mixed phase
            else:
                opponent_type = 'rules'  # Final phase against rules
    
        
            state = env.reset()
            state = self.preprocess_state(env.board, env.current_player, env.agent_player)
            done = False
            total_reward = 0
            
            while not done:
                if env.current_player == env.agent_player:
                    valid_moves = env.get_valid_moves(env.current_player)
                    action = self.get_action(state, valid_moves)
                    _, reward, done = env.step(action)
                    next_state = self.preprocess_state(env.board, env.current_player, env.agent_player)
                    self.remember(state, action, reward, next_state, done)
                    loss = self.replay()
                    total_reward += reward
                    state = next_state
                else:
                    valid_moves = env.get_valid_moves(env.current_player)
                    if opponent_type == 'random':
                        action = random.choice(valid_moves)
                    else:
                        action = self._rules_based_opponent(env.board, env.current_player)
                    
                    _, _, done = env.step(action)
                    state = self.preprocess_state(env.board, env.current_player, env.agent_player)
                
                if done:
                    print(f"Episode {e+1} - Epsilon: {self.epsilon:.3f} - Reward: {reward:.2f}, Store diff: {env.board[env.agent_player][6] - env.board[env.opponent_player][6]}")
                    if e % 100 == 0:
                        env.agent_player = 'player_2' if env.agent_player == 'player_1' else 'player_1'
                        env.opponent_player = 'player_2' if env.agent_player == 'player_1' else 'player_1'
                    
                    if e % 10 == 0:
                        q_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
                        sample_state = np.zeros((1, self.state_shape[0]))  # Zero state
                        print("Initial Q-values:", self.model.predict(sample_state, verbose=0))
                        
                        random_state = np.random.rand(1, self.state_shape[0])
                        print("Random state Q-values:", self.model.predict(random_state, verbose=0))
                    
                    if e % 15 == 0 or e == episodes - 1:
                                                # Evaluate against both opponents
                        rules_metrics = self.evaluate(test_env, test_games=25, opponent='rules')
                        random_metrics = self.evaluate(test_env, test_games=40, opponent='random')

                        # Store metrics for tracking
                        avg_rewards.append(random_metrics['avg_reward'])
                        win_rates.append(random_metrics['win_rate'])
                        avg_moves.append(random_metrics['avg_moves'])

                        print(f"Evaluation Episode {e+1}:")
                        print(f"  Vs Random: WR {random_metrics['win_rate']:.2%} | Avg Reward: {random_metrics['avg_reward']:.2f} | Avg Moves: {random_metrics['avg_moves']:.1f}")
                        print(f"  Vs Rules: WR {rules_metrics['win_rate']:.2%} | Avg Store Diff: {rules_metrics['avg_store_diff']:.1f} ± {rules_metrics['std_store_diff']:.1f}")

                        # Calculate policy entropy
                        probs = tf.nn.softmax(q_values)
                        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10))  # Added epsilon for numerical stability

                        # Log to TensorBoard
                        with self.summary_writer.as_default():
                            tf.summary.scalar('epsilon', self.epsilon, step=e)
                            tf.summary.scalar('win_rate/random', random_metrics['win_rate'], step=e)
                            tf.summary.scalar('win_rate/rules', rules_metrics['win_rate'], step=e)
                            tf.summary.scalar('avg_reward', random_metrics['avg_reward'], step=e)
                            tf.summary.scalar('avg_store_diff', rules_metrics['avg_store_diff'], step=e)
                            tf.summary.scalar('max_q_value', np.max(q_values), step=e)
                            tf.summary.scalar('min_q_value', np.min(q_values), step=e)
                            tf.summary.scalar("policy_entropy", entropy, step=e)
                            tf.summary.scalar("avg_game_length", random_metrics['avg_moves'], step=e)

                        print(f"Training Loss: {loss if loss is not None else 0:.4f} | Policy Entropy: {entropy:.3f}")

                        if early_stop:
                            # Use rules-based win rate for early stopping (more meaningful metric)
                            current_win_rate = rules_metrics['win_rate']
                            
                            if current_win_rate >= 0.95:  # Slightly more reasonable target than 0.98
                                print(f"Early stopping: Reached {current_win_rate:.0%} win rate against rules-based opponent")
                                self.save_model("mancala_dqn_final.h5")
                                return win_rates, avg_rewards
                            
                            if current_win_rate > best_win_rate + min_improvement:
                                best_win_rate = current_win_rate
                                no_improvement_count = 0
                                self.save_model("mancala_dqn_best.h5")
                                print(f"New best model saved with {current_win_rate:.2%} win rate")
                            else:
                                no_improvement_count += 1
                                if no_improvement_count >= patience:
                                    print(f"Early stopping: No improvement for {patience} evaluations")
                                    self.save_model("mancala_dqn_final.h5")
                                    return win_rates, avg_rewards
                            
                            # if no_improvement_count >= patience:
                            #     print(f"Early stopping: No improvement for {patience} evaluations")
                            #     self.save_model("mancala_dqn_final.h5")
                            #     return win_rates, avg_rewards
                            
                            # if len(avg_rewards) >= plateau_window:
                            #     recent_rewards = avg_rewards[-plateau_window:]
                            #     if max(recent_rewards) - min(recent_rewards) < 0.5:
                            #         print("Early stopping: Reward plateau detected")
                            #         self.save_model("mancala_dqn_final.h5")
                            #         return win_rates, avg_rewards
                        # Add learning rate scheduling
                    if e % 500 == 0:
                        try:
                            lr = self.model.optimizer.learning_rate.numpy()  # Correct way to get LR
                            new_lr = lr * 0.9
                            self.model.optimizer.learning_rate.assign(new_lr)
                            print(f"Reduced learning rate from {lr:.6f} to {new_lr:.6f}")
                        except:
                            print("fail to reduce learning rate")
                    break
        
        self.save_model("mancala_dqn_final.h5")
        return win_rates, avg_rewards

    def save_model(self, filepath):
        self.model.save(filepath)

class MancalaEnv:
    def __init__(self, agent_player='player_1'):
        self.agent_player = agent_player
        self.opponent_player = 'player_2' if agent_player == 'player_1' else 'player_1'
        self.reset()

    def reset(self):
        self.board = initialize_board()
        self.current_player = 'player_1'
        return self.board

    def step(self, action):
        current_player_before = self.current_player
        reward = self._calculate_reward(last_action=action) if current_player_before == self.agent_player else 0.0
        self.board, extra_turn = make_move(self.board, self.current_player, action)
        if not extra_turn:
            self.current_player = self.opponent_player if self.current_player == self.agent_player else self.agent_player
        done = is_terminal(self.board)
        return self.board, reward, done

    def get_valid_moves(self, player=None):
        player = player or self.current_player
        return get_valid_moves(self.board, player)

    def _calculate_reward(self, last_action=None, done=False):
        """Precision reward function with proper defensive/capture logic"""
        if last_action is not None and self.board[self.agent_player][last_action] == 0:
            return -25  # Invalid move penalty
        
        # 1. Simulate move first
        if last_action is not None:
            simulated_board, extra_turn = make_move(copy.deepcopy(self.board),
                                            self.agent_player,
                                            last_action)
        else:
            simulated_board = copy.deepcopy(self.board)
            extra_turn = False
            
        my_store = simulated_board[self.agent_player][6]
        opp_store = simulated_board[self.opponent_player][6]
        store_diff = my_store - opp_store
        
        reward = 0.0
        defensive_bonus = 0.0
        capture_threat = 0.0

        # 1. Base store difference (normalized)
        reward += 10.0 * (store_diff / 48.0) #(0-10)

        if last_action is not None:
            stones = self.board[self.agent_player][last_action]
            landing_pit = (last_action + stones) % 14

            # CORRECTED CAPTURE LOGIC
            if 0 <= landing_pit < 6:  # Landed in our side
                # Valid capture requires:
                # 1. Pit was empty before sowing
                # 2. Opponent's opposite pit has stones
                if ((self.board[self.agent_player][landing_pit] == 0 or
                    self.board[self.agent_player][landing_pit] == self.board[self.agent_player][last_action]) and # in case landing pit = start pit
                    self.board[self.opponent_player][5 - landing_pit] > 0):
                    
                    captured = self.board[self.opponent_player][5 - landing_pit]
                    reward += 1.5 * captured

            # DEFENSIVE MOVE DETECTION (NEW)
            # Check opponent's potential captures we blocked
            for opp_pit in range(6):
                if self.board[self.opponent_player][opp_pit] == 0:
                    continue  # Opponent can't move from empty pit
                    
                # Simulate opponent sowing from this pit
                opp_stones = self.board[self.opponent_player][opp_pit]
                opp_landing = (opp_pit + opp_stones) % 14
                
                if 0 <= (opp_pit + opp_stones) % 14 < 6:  # Lands in their own side
                    # Would this create a capture threat?
                    if ((self.board[self.opponent_player][opp_landing] == 0 or 
                        self.board[self.opponent_player][opp_landing] == self.board[self.opponent_player][opp_pit]) and # in case landing pit = opp_pit
                        (opp_pit + opp_stones) // 13 == 0 and # Check if the landing pit passed the opp_pit
                        self.board[self.agent_player][5 - opp_landing] > 0):
                        
                        # Calculate how our move affected this threat
                        if last_action == (5 - opp_landing):
                            # We modified the threatened pit
                            defensive_bonus += (0.75 * stones)
                        if landing_pit > 6 and landing_pit - 7 >= opp_landing:
                            defensive_bonus += (0.75 * self.board[self.agent_player][5-opp_landing])
                            

            # Extra turn bonus
            if (last_action + stones) % 14 == 6:
                reward += 4.0

        # 2. Apply defensive bonuses
        reward += defensive_bonus

        # 3. Terminal state rewards
        if is_terminal(simulated_board):
            if my_store > opp_store:
                reward += 20.0 + store_diff * 0.5
            else:
                reward += -15.0 + (store_diff * 0.4)

        # 4. Progressive game phase bonus
        game_phase = (my_store + opp_store) / 48.0
        if game_phase > 0.7 and store_diff > 0:
            reward += 2.0 * game_phase * (store_diff / 24.0)

        reward = np.tanh(reward) *5
        return reward

if __name__ == "__main__":
    env = MancalaEnv(agent_player='player_1')
    agent = DQNAgent(state_shape=(15,))
    print("Starting training...")
    agent.train(env, episodes=5000)
    agent.save_model("mancala_dqn_improved.h5")
    print("Training completed. Model saved.")

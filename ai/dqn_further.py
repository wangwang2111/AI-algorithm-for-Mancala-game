import numpy as np
import random
from collections import deque
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
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
    def __init__(self, state_shape=(29,), action_size=6):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(capacity=100000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9988
        self.epsilon_min = 0.1
        self.batch_size = 128
        self.model = self._build_model()
        self.target_model = self._build_model()  # Create target model
        self.target_model.set_weights(self.model.get_weights())  # Initialize with same weights
        self._init_optimizer()
        self.tensorboard = TensorBoard(log_dir='./logs/dqn')
        self.summary_writer = tf.summary.create_file_writer('./logs/dqn')
        self.target_update_freq = 200
        self.train_step = 0

    def _init_optimizer(self):
        """Initialize a fresh optimizer with current model variables"""
        self.model.optimizer = Adam(learning_rate=0.0003, clipnorm=1.0)
        
    def _build_model(self):
        inputs = Input(shape=self.state_shape)
        x = Dense(512, activation='relu')(inputs)  # Increased from 256
        x = tf.keras.layers.Dropout(0.3)(x)  # Increased from 0.2
        x = Dense(256, activation='relu')(x)  # Additional layer
        # x = tf.keras.layers.Dropout(0.3)(x)  # Increased from 0.2
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
        """Select an action using an epsilon-greedy policy over only valid moves."""
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
        
        q_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
        
        # For both players, valid moves are 0-5 (their own pits)
        # The environment will translate these to actual pit numbers
        masked_q = np.full(6, -np.inf)  # Only 6 possible actions (pits)
        for move in valid_moves:
            # Convert pit number to 0-5 index (for player's own pits)
            pit_index = move if move < 6 else move - 7
            masked_q[pit_index] = q_values[pit_index]
        
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

    def preprocess_state(self, board, current_player):
        """Corrected state representation with 29 features"""
        p1_pits = np.array(board['player_1'][:6])
        p1_store = board['player_1'][6]
        p2_pits = np.array(board['player_2'][:6])
        p2_store = board['player_2'][6]
            
        features = [
            # Player 1 perspective (7)
            *(p1_pits / 24.0),  # 6 features
            p1_store / 48.0,     # 1 feature
            
            # Player 2 perspective (7)
            *(p2_pits / 24.0),   # 6
            p2_store / 48.0,     # 1
            
            # Relative differences (7)
            *(p1_pits - p2_pits) / 24.0,  # 6
            (p1_store - p2_store) / 48.0, # 1
            
            # Turn information (2)
            1.0 if current_player == 'player_1' else 0.0,
            1.0 if current_player == 'player_2' else 0.0,
            
            # Game phase (1)
            min(1.0, (p1_store + p2_store) / 30.0),
            
            # Strategic features (5)
            sum(p1_pits) / 24.0,
            sum(p2_pits) / 24.0,
            float(any(p == 0 for p in p1_pits)),
            float(any(p == 0 for p in p2_pits)),
            1.0  # Constant bias term to reach 29 features
        ]
        
        return np.array(features, dtype=np.float32)

        # def preprocess_state(self, board, current_player, agent_player):
        # """
        # Preprocess the board state into a consistent 14-length vector (normalized 0-1).
        # The state always represents the board from the agent's perspective.
        # """
        # # Normalize all values by dividing by max possible stones
        # p1_pits = np.array(board['player_1'][:6]) / 24.0  # Max 4*6=24 stones in pits
        # p1_store = board['player_1'][6] / 48.0  # Max 48 stones in store
        # p2_pits = np.array(board['player_2'][:6]) / 24.0
        # p2_store = board['player_2'][6] / 48.0
        
        # if agent_player == 'player_1':
        #     # Agent is player_1 - state is [p1_pits, p1_store, p2_pits, p2_store, turn_indicator]
        #     turn_indicator = 1.0 if current_player == 'player_1' else -1.0
        #     return np.concatenate([p1_pits, [p1_store], p2_pits, [p2_store], [turn_indicator]])
        # else:
        #     # Agent is player_2 - rotate perspective so agent's pits come first
        #     turn_indicator = 1.0 if current_player == 'player_2' else -1.0
        #     return np.concatenate([p2_pits, [p2_store], p1_pits, [p1_store], [turn_indicator]])

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
        
        total_reward = 0
        wins = 0
        total_moves = 0
        
        # Test both player perspectives
        for agent_player in ['player_1', 'player_2']:
            test_env.agent_player = agent_player
            test_env.opponent_player = 'player_2' if agent_player == 'player_1' else 'player_1'
            
            for _ in range(test_games // 2):  # Split games between perspectives
                test_env.reset()
                state = self.preprocess_state(test_env.board, test_env.current_player)
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
                    
                    state = self.preprocess_state(test_env.board, test_env.current_player)
                    
                    # In evaluate(), add tie handling:
                    if done:
                        total_reward += game_reward
                        if test_env.board[test_env.agent_player][6] > test_env.board[test_env.opponent_player][6]:
                            wins += 1
                        elif test_env.board[test_env.agent_player][6] == test_env.board[test_env.opponent_player][6]:
                            wins += 0.1  # Count ties as 0.1-wins
                        total_moves += game_moves
        
        # Calculate averages
        avg_reward = total_reward / test_games
        win_rate = wins / test_games
        avg_moves = total_moves / test_games
        
        self.epsilon = original_epsilon
        return avg_reward, win_rate, avg_moves

    def train(self, env, episodes=10000, early_stop=True):
        """
        Training loop for the DQNAgent with early stopping capabilities.
        """
        # Training phases configuration
        phases = [
            {'eps': 1000, 'opponent': 'random', 'epsilon': 1.0, 'batch_size': 64},
            {'eps': 3000, 'opponent': 'mixed', 'epsilon': 0.3, 'batch_size': 128},
            {'eps': 6000, 'opponent': 'rules', 'epsilon': 0.1, 'batch_size': 256},
            {'eps': 10000, 'opponent': 'rules', 'epsilon': 0.01, 'batch_size': 512}
        ]
        win_rates = []
        avg_rewards = []
        avg_moves = []
        
        current_phase = 0
        phase_start_episode = 0
        best_win_rate = -np.inf
        best_win_rate_rule = -np.inf
        no_improvement_count = 0
        
        patience = 200         # Episodes to wait before stopping
        plateau_window = 25    # Window for reward plateau detection
        min_improvement = 0.01 # Minimum win rate improvement
        
        test_env = MancalaEnv(agent_player='player_1')
        
        for e in range(episodes):
            if current_phase < len(phases)-1 and e >= phase_start_episode + phases[current_phase]['eps']:
                # Evaluate before phase transition
                rules_reward, rules_win, _ = self.evaluate(env, opponent='rules', test_games=20)
                
                if rules_win >= 0.7 or current_phase == 0:  # Always advance from phase 1
                    current_phase += 1
                    phase_start_episode = e
                    env.set_phase(current_phase + 1)  # Phases are 1-indexed
                    
                    # Update hyperparameters
                    self.epsilon = phases[current_phase]['epsilon']
                    self.batch_size = phases[current_phase]['batch_size']
                    
                    print(f"\n=== ADVANCING TO PHASE {current_phase+1} ===")
                    print(f"New params: ε={self.epsilon}, batch={self.batch_size}")
                    
            # Alternate between random and rules-based opponents every 10 episodes
            opponent_type = 'random'
            # opponent_type = 'random' if (e // 10) % 2 == 0 else 'rules'
        
            state = env.reset()
            state = self.preprocess_state(env.board, env.current_player)
            done = False
            total_reward = 0
            
            while not done:
                if env.current_player == env.agent_player:
                    valid_moves = env.get_valid_moves(env.current_player)
                    action = self.get_action(state, valid_moves)
                    _, reward, done = env.step(action)
                    next_state = self.preprocess_state(env.board, env.current_player)
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
                    state = self.preprocess_state(env.board, env.current_player)
                
                if done:
                    print(f"Episode {e+1} - Epsilon: {self.epsilon:.3f} - Reward: {reward:.2f}, Store diff: {env.board[env.agent_player][6] - env.board[env.opponent_player][6]}")
                    if e % 100 == 0:
                        env.agent_player = 'player_2' if env.agent_player == 'player_1' else 'player_1'
                        env.opponent_player = 'player_2' if env.agent_player == 'player_1' else 'player_1'
                    
                    if e % 10 == 0:
                        q_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
                    
                    if e % 15 == 0 or e == episodes - 1:
                        # Evaluate against random opponent
                        rules_avg_reward, rules_win, rules_move = self.evaluate(test_env, test_games=25, opponent='rules')
                        # Evaluate against rules-based opponent
                        test_avg_reward, test_win_rate, test_move = self.evaluate(test_env, opponent='random', test_games=40)
                        avg_rewards.append(test_avg_reward)
                        win_rates.append(test_win_rate)
                        avg_moves.append(test_move)
                        
                        print(f"Vs Random: WR {test_win_rate:.2%} | Avg Reward: {test_avg_reward} | Avg Moves: {test_move:.1f}")
                        print(f"Vs Rules: WR {rules_win:.2%} | Avg Store Diff: {rules_avg_reward:.1f} | {rules_move:.1f}")

                        # Calculate policy entropy
                        probs = tf.nn.softmax(q_values)
                        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10))  # Added epsilon for numerical stability

                        with self.summary_writer.as_default():
                            tf.summary.scalar('epsilon', self.epsilon, step=e)
                            tf.summary.scalar('win_rate', test_win_rate, step=e)
                            tf.summary.scalar('win_rate/random', test_win_rate, step=e)
                            tf.summary.scalar('win_rate/rules', rules_win, step=e)
                            tf.summary.scalar('avg_reward', test_avg_reward, step=e)
                            tf.summary.scalar('max_q_value', np.max(q_values), step=e)
                            tf.summary.scalar('min_q_value', np.min(q_values), step=e)
                            tf.summary.scalar("policy_entropy", entropy, step=e)
                            tf.summary.scalar("avg_game_length", test_move, step=e)
                        print(f"Training Loss: {loss if loss is not None else 0:.4f} | Policy Entropy: {entropy:.3f}")
                        print(f"Ep {e}/{episodes} | Phase {current_phase+1} | "
                            f"ε={self.epsilon:.3f} | Random WR: {test_win_rate:.1%} | "
                            f"Rules WR: {rules_win:.1%}")
                        if early_stop:
                            if test_win_rate >= 0.98:
                                print("Early stopping: Reached 95% win rate")
                                self.save_model("mancala_dqn_final.h5")
                                return win_rates, avg_rewards
                            if rules_win >= 0.90 and rules_win > best_win_rate_rule + min_improvement:
                                best_win_rate_rule = rules_win
                                print("Early stopping: Reached 95% win rate")
                                self.save_model("mancala_dqn_best_rules.h5")
                            
                            if test_win_rate > best_win_rate + min_improvement:
                                best_win_rate = test_win_rate
                                no_improvement_count = 0
                                self.save_model("mancala_dqn_best.h5")
                            else:
                                no_improvement_count += 1
                            
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
                    if e % 500 == 0 or (e == 0 and hasattr(self, 'loaded_weights')):
                        # Reinitialize optimizer when:
                        # 1. Periodic refresh
                        # 2. After loading weights
                        self._init_optimizer()
                        print("Optimizer reinitialized")
                        lr = self.model.optimizer.learning_rate.numpy()  # Correct way to get LR
                        new_lr = lr * 0.9
                        self.model.optimizer.learning_rate.assign(new_lr)
                        print(f"Reduced learning rate from {lr:.6f} to {new_lr:.6f}")
                        agent.epsilon = 0.35  # Start with some exploration
                        agent.epsilon_min = 0.05  # Start with some exploration
                      
                    break
        
        self.save_model("mancala_dqn_final.h5")
        return win_rates, avg_rewards

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        self.target_model = tf.keras.models.load_model(filepath)  # Load separate target model
        self._init_optimizer()
        self.loaded_weights = True
        return self  # For method chaining if needed
        
class MancalaEnv:
    def __init__(self, agent_player='player_1', training_phase=1):
        self.agent_player = agent_player
        self.opponent_player = 'player_2' if agent_player == 'player_1' else 'player_1'
        self.training_phase = training_phase  # Track training phase
        self.reset()
        
    def set_phase(self, phase):
        """Dynamically adjust reward complexity"""
        self.training_phase = phase
        
    def reset(self):
        self.board = initialize_board()
        self.current_player = 'player_1'
        return self.board

    def step(self, action):
        # Store the pre-move state for reward calculation
        pre_move_board = {
            'player_1': self.board['player_1'].copy(),
            'player_2': self.board['player_2'].copy()
        }
        current_player_before = self.current_player
        
        # Execute the move
        self.board, extra_turn = make_move(self.board, self.current_player, action)
        
        # Calculate reward based on pre-move state
        reward = 0.0
        if current_player_before == self.agent_player:
            reward = self._calculate_reward(pre_move_board, action)
        
        # Switch player if no extra turn
        if not extra_turn:
            self.current_player = self.opponent_player if self.current_player == self.agent_player else self.agent_player
        
        done = is_terminal(self.board)
        return self.board, reward, done

    def get_valid_moves(self, player=None):
        player = player or self.current_player
        return get_valid_moves(self.board, player)

    def _calculate_reward(self, pre_move_board, last_action=None):
        my_store = self.board[self.agent_player][6]
        opp_store = self.board[self.opponent_player][6]
        store_diff = (my_store - opp_store) * 0.7  # Increased weight
        
        reward = store_diff
        if is_terminal(self.board):
            if my_store > 24:
                reward += 18.0
            elif my_store > opp_store:
                reward += 10.0
            elif my_store == opp_store:
                reward -= 5.0
            else:
                reward -= 15.0
        
        if self.training_phase == 1:
            return np.tanh(reward)*20
        
        elif self.training_phase == 2:
            if last_action is not None:
                stones = pre_move_board[self.agent_player][last_action]
                landing_pit = (last_action + stones) % 14
                
                # Capture bonus
                if 0 <= landing_pit < 6 and pre_move_board[self.agent_player][landing_pit] == 0:
                    captured = self.board[self.opponent_player][5 - landing_pit] - \
                              pre_move_board[self.opponent_player][5 - landing_pit]
                    if captured > 0:
                        reward += 0.5 * captured
                
                # Extra turn bonus
                if landing_pit == 6:
                    reward += 0.3
                    
            return np.tanh(reward)*20
        
        elif self.training_phase == 3:
            # Phase 4: Expert rewards
            strategic_bonus = 0
            my_side = sum(self.board[self.agent_player][:6])
            opp_side = sum(self.board[self.opponent_player][:6])
            control_ratio = my_side / (opp_side + 1e-5)
            
            for pit in range(6):
                if self.board[self.agent_player][pit] == (6 - pit):
                    strategic_bonus += 0.2
            
            total_stones = my_side + opp_side
            
            endgame_weight = min(1.0, (48 - total_stones) / 24)
            reward = reward * (endgame_weight + 1)
        
            return np.tanh(reward)*20
            
        # if last_action is not None:
        #     stones = pre_move_board[self.agent_player][last_action]
        #     landing_pit = (last_action + stones) % 14
            
        #     # Strategic bonuses
        #     if 0 <= landing_pit < 6:
        #         if pre_move_board[self.agent_player][landing_pit] == 0:
        #             # Capture bonus
        #             if pre_move_board[self.opponent_player][5 - landing_pit] > 0:
        #                 captured = pre_move_board[self.opponent_player][5 - landing_pit]
        #                 reward += 5.0 + (captured * 0.5)
                
        #         # Extra turn bonus
        #         if landing_pit == 6:
        #             reward += 6.0
                    
        #     # Blocking opponent's potential moves
        #     for opp_pit in range(6):
        #         if pre_move_board[self.opponent_player][opp_pit] > 0:
        #             opp_landing = (opp_pit + pre_move_board[self.opponent_player][opp_pit]) % 14
        #             if 0 <= opp_landing < 6 and last_action == (5 - opp_landing):
        #                 reward += 2.0
        
        # Terminal rewards

    
if __name__ == "__main__":
    env = MancalaEnv(agent_player='player_1')
    agent = DQNAgent(state_shape=(29,))
    
    # Load the saved model
    try:
        agent.load_model("models/no_mancala_dqn_best.h5")
        print("Successfully loaded saved model!")
        print("Successfully loaded saved model!")
        
        # Reset exploration parameters for continued training
        agent.epsilon = 0.35  # Start with some exploration
        agent.epsilon_min = 0.05  # Start with some exploration
        agent.epsilon_decay = 0.998  # Slower decay
        agent.batch_size = 256  # Larger batch size
        agent.gamma = 0.98  # Slightly higher discount factor
        agent.target_update_freq = 500  # Update every 500 steps (was 1000)
    except Exception as e:
        print(e)
        print("No saved model found, starting fresh training")
    
    print("Starting training...")
    agent.train(env, episodes=8000)
    agent.save_model("mancala_dqn_improved_v3.h5")
    print("Training completed. Model saved.")
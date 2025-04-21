import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
import threading
import multiprocessing
from collections import deque
import os
from dqn import MancalaEnv
from ai.rules import initialize_board, get_valid_moves, make_move, is_terminal

class A3CAgent:
    def __init__(self, state_shape=(29,), action_size=6,  # Tuple format
                 global_model=None, is_global=False):
        self.state_shape = state_shape  # Now properly as tuple (29,)
        self.action_size = action_size
        self.is_global = is_global
        
        # Improved hyperparameters
        self.gamma = 0.95  # Slightly lower discount factor
        self.entropy_coef = 0.1  # Higher entropy for more exploration
        self.value_coef = 0.25  # Lower value coefficient
        self.max_grad_norm = 1.0  # More generous gradient clipping
        self.learning_rate = 0.00025
        
        # Build model
        self.model = self._build_model()
        self.optimizer = RMSprop(learning_rate=self.learning_rate, rho=0.99, epsilon=1e-7)
        
        # Global model reference
        self.global_model = global_model.model if global_model and not is_global else self.model
        
        # Training statistics
        self.episode_count = 0
        self.best_reward = -np.inf

    def _build_model(self):
        inputs = Input(shape=self.state_shape)  # Now gets proper tuple
        
        # Network architecture (unchanged)
        x = Dense(256, activation='elu', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Dense(128, activation='elu', kernel_initializer='he_normal')(x)
        
        # Policy head
        policy = Dense(self.action_size, activation='softmax',
                      kernel_initializer='glorot_uniform')(x)
        
        # Value head
        value = Dense(1, activation='linear',
                     kernel_initializer='glorot_uniform')(x)
        
        return Model(inputs=inputs, outputs=[policy, value])

    def _calculate_reward(self, board, last_action=None, done=False):
        """Enhanced reward function"""
        my_store = board[self.env.agent_player][6]
        opp_store = board[self.env.opponent_player][6]
        store_diff = my_store - opp_store
        
        # Normalized components
        norm_store_diff = store_diff / 48.0
        norm_my_store = my_store / 48.0
        
        reward = 2.0 * norm_store_diff  # Base reward
        
        # Strategic bonuses
        if last_action is not None:
            stones = board[self.env.agent_player][last_action]
            landing_pit = (last_action + stones) % 14
            
            # Capture bonus
            if 0 <= landing_pit < 6 and board[self.env.agent_player][landing_pit] == 1:
                captured = board[self.env.opponent_player][5 - landing_pit]
                reward += 1.5 + (0.25 * captured)
            
            # Extra turn bonus
            if (last_action + stones) % 14 == 6:
                reward += 2.0
            
            # Tempo bonus
            if stones == (6 - last_action):
                reward += 1.0
        
        # Terminal rewards
        if done:
            if my_store > 24:  # Clear win
                reward += 10.0
            elif my_store > opp_store:  # Moderate win
                reward += 5.0
            elif my_store == opp_store:  # Tie
                reward -= 2.0
            else:  # Loss
                reward -= 5.0
        
        return np.clip(reward, -10.0, 15.0)

        
    def get_action(self, state, valid_moves):
        """Get action from policy with proper masking and normalization"""
        # Debugging prints (optional)
        if hasattr(self, 'debug') and self.debug:
            print(f"State shape: {state.shape}")
            print(f"Valid moves: {valid_moves}")
        
        # Ensure proper input shape
        if len(state.shape) == 1:
            state = state[np.newaxis, ...]  # Add batch dimension
        
        # Get policy from model
        policy, _ = self.model.predict(state, verbose=0)
        policy = policy[0]  # Remove batch dimension
        
        # Create boolean mask for valid moves
        valid_mask = np.zeros(self.action_size, dtype=bool)
        for move in valid_moves:
            pit_index = move if move < 6 else move - 7  # Convert to 0-5 index
            if pit_index < self.action_size:  # Safety check
                valid_mask[pit_index] = True
        
        # Apply mask and handle zero probabilities
        masked_policy = np.zeros_like(policy)
        masked_policy[valid_mask] = policy[valid_mask]  # Copy only valid moves
        
        # Handle cases where all valid moves have zero probability
        if np.sum(masked_policy) <= 0:
            # Fall back to uniform distribution over valid moves
            masked_policy[valid_mask] = 1.0 / np.sum(valid_mask)
            if hasattr(self, 'debug') and self.debug:
                print("Warning: All valid moves had zero probability - using uniform fallback")
        else:
            # Normalize probabilities
            masked_policy = masked_policy / np.sum(masked_policy)
        
        # Sample action
        try:
            action = np.random.choice(self.action_size, p=masked_policy)
            # Verify the sampled action is actually valid
            if not valid_mask[action]:
                # If somehow an invalid action was sampled (shouldn't happen with proper masking)
                valid_indices = np.where(valid_mask)[0]
                action = np.random.choice(valid_indices)
            
            return action
        except ValueError as e:
            print(f"Error in action sampling: {e}")
            print(f"Masked policy sum: {np.sum(masked_policy)}")
            print(f"Valid moves: {valid_moves}")
            return np.random.choice(list(valid_moves)) if valid_moves else 0
    
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

    def _rules_based_opponent(self, board, current_player):
        """Simple rules-based opponent for Mancala"""
        if current_player == 'player_1':
            pits = board['player_1'][:6]
            store_index = 6
        else:
            pits = board['player_2'][:6]
            store_index = 6
        
        # Rule 1: Prioritize moves that land in the store
        for i in range(5, -1, -1):
            if pits[i] == (store_index - i):
                return i
        
        # Rule 2: Choose the pit with most stones that can capture opponent's stones
        max_stones = -1
        best_move = 0
        for i in range(6):
            if pits[i] == 0:
                continue
            if pits[i] > max_stones:
                max_stones = pits[i]
                best_move = i
        
        return best_move

    def _random_opponent(self, valid_moves):
        """Random opponent strategy"""
        return np.random.choice(valid_moves)
    
    def train(self, states, actions, rewards, next_states, dones):
        """Improved training with GAE"""
        # Ensure states have proper shape
        if len(states.shape) == 1:
            states = np.expand_dims(states, axis=0)
        if len(next_states.shape) == 1:
            next_states = np.expand_dims(next_states, axis=0)
        
        # Get values and next values
        _, values = self.model.predict(states, verbose=0)
        _, next_values = self.model.predict(next_states, verbose=0)
        
        # Squeeze while preserving batch dimension if needed
        values = tf.squeeze(values, axis=-1)  # Remove last dimension if it's 1
        next_values = tf.squeeze(next_values, axis=-1)
        
        # Compute TD errors
        targets = rewards + self.gamma * next_values * (1 - dones)
        deltas = targets - values
        
        # Compute GAE
        advantages = np.zeros_like(deltas)
        last_advantage = 0
        for t in reversed(range(len(deltas))):
            advantages[t] = last_advantage = deltas[t] + self.gamma * 0.95 * last_advantage
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Convert to tensors
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
        advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        targets_tensor = tf.convert_to_tensor(targets, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Get current predictions
            policies, values = self.model(states_tensor, training=True)
            values = tf.squeeze(values, axis=-1)
            
            # Policy loss
            actions_onehot = tf.one_hot(actions_tensor, self.action_size)
            log_probs = tf.reduce_sum(actions_onehot * tf.math.log(policies + 1e-10), axis=1)
            policy_loss = -tf.reduce_mean(log_probs * advantages_tensor)
            
            # Entropy bonus
            entropy = -tf.reduce_sum(policies * tf.math.log(policies + 1e-10), axis=1)
            entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
            
            # Value loss (MSE between targets and values)
            value_loss = 0.5 * tf.reduce_mean(tf.square(targets_tensor - values)) * self.value_coef
            
            total_loss = policy_loss + entropy_loss + value_loss
        
        # Compute gradients
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        if grads[0] is not None:  # Check if gradients were computed
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_variables))
        
        # Sync models if worker
        if not self.is_global:
            self.model.set_weights(self.global_model.get_weights())
        
        return total_loss.numpy()

class Worker(threading.Thread):
    def __init__(self, env, global_agent, worker_id, max_episodes=2000, t_max=10):
        threading.Thread.__init__(self)
        self.env = env
        self.global_agent = global_agent
        # Fixed: Pass state_shape as tuple
        self.worker_agent = A3CAgent(
            state_shape=(29,),  # Critical fix here
            action_size=6,
            global_model=global_agent
        )
        self.worker_id = worker_id
        self.max_episodes = max_episodes
        self.t_max = t_max  # Reduced from 20 to 10 for faster updates
        
    def run(self):
        for episode in range(self.max_episodes):
            state = self.env.reset()
            state = self.worker_agent.preprocess_state(self.env.board, self.env.current_player)
            
            episode_reward = 0
            done = False
            memory = []
            
            while not done:
                states, actions, rewards, next_states, dones = [], [], [], [], []
                
                for t in range(self.t_max):
                    valid_moves = self.env.get_valid_moves(self.env.current_player)
                    
                    if self.env.current_player == self.env.agent_player:
                        action = self.worker_agent.get_action(state, valid_moves)
                        _, reward, done = self.env.step(action)
                        next_state = self.worker_agent.preprocess_state(self.env.board, self.env.current_player)
                        
                        # Store experience
                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)
                        next_states.append(next_state)
                        dones.append(done)
                        
                        state = next_state
                        episode_reward += reward
                    else:
                        # Mixed opponent strategy (50% rules-based, 50% random)
                        if np.random.rand() < 0.3:
                            action = self.worker_agent._rules_based_opponent(self.env.board, self.env.current_player)
                        else:
                            action = self.worker_agent._random_opponent(valid_moves)
                        
                        _, _, done = self.env.step(action)
                        state = self.worker_agent.preprocess_state(self.env.board, self.env.current_player)
                    
                    if done:
                        break
                
                if len(states) > 0:
                    loss = self.worker_agent.train(
                        np.array(states),
                        np.array(actions),
                        np.array(rewards),
                        np.array(next_states),
                        np.array(dones)
                    )
            
            # Update global statistics
            with self.global_agent.lock:
                self.global_agent.episode_count += 1
                if episode_reward > self.global_agent.best_reward:
                    self.global_agent.best_reward = episode_reward
                    self.global_agent.model.save("mancala_a3c_best.h5")
                
                print(f"Worker {self.worker_id}, Episode {episode}: "
                      f"Reward {episode_reward:.1f}, Loss {loss:.4f}")

def train_a3c(env, num_workers=4, max_episodes=4000):
    """Enhanced training function"""
    global_agent = A3CAgent(is_global=True)
    global_agent.lock = threading.Lock()  # For thread-safe updates
    
    workers = []
    for worker_id in range(num_workers):
        worker_env = MancalaEnv(agent_player='player_1')
        worker = Worker(worker_env, global_agent, worker_id, max_episodes//num_workers)
        workers.append(worker)
    
    for worker in workers:
        worker.start()
    
    for worker in workers:
        worker.join()
    
    global_agent.model.save("mancala_a3c_final.h5")
    return global_agent

if __name__ == "__main__":
    env = MancalaEnv(agent_player='player_1')
    a3c_agent = train_a3c(env, num_workers=multiprocessing.cpu_count())
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from mancala_ai.engine.core import initialize_board, get_valid_moves, make_move, is_terminal
from mancala_ai.agents.minimax import simple_minimax
from mancala_ai.agents.advanced_heuristic import advanced_heuristic_minimax

import os
from datetime import datetime
import math

# ==================== HYPERPARAMETERS ====================
# Training parameters
BATCH_SIZE = 256
GAMMA = 0.8                  # Discount factor
EPS_START = 1              # Initial exploration rate
EPS_END = 0.1               # Minimum exploration rate
EPS_DECAY = 0.000003       # Decay rate for exploration
MEMORY_SIZE = 10000000      # Replay buffer size
LEARNING_RATE = 0.0001      # Learning rate
TARGET_UPDATE_FREQ = 1000   # Steps between target network updates
TAU = 0.015                 # Soft update parameter
ALPHA = 0.6                 # Prioritization exponent
BETA_START = 0.4            # Initial importance sampling weight
BETA_FRAMES = 100000        # Steps to anneal beta to 1.0

USE_DOUBLE_DQN = True
# Network architecture
# HIDDEN_UNITS = [128, 64, 32]     # Layer sizes
HIDDEN_UNITS = [64, 64, 32]     # Layer sizes
DROPOUT_RATE = 0.1               # Dropout rate

# Training reporting
REPORTING_PERIOD = 100          # Episodes between progress reports
SAVE_FREQ = 1000                # Episodes between model saves

# Opponent options
# OPPONENTS = ['random']
OPPONENTS = ['random', 'minimax']

# Reward shaping
WIN_REWARD = 5.0               
CAPTURE_BONUS = 0.8
EXTRA_TURN_BONUS = 1.4
STORE_DIFF_WEIGHT = 0.8
REWARD_SCALE = 0.5
POSITIONAL_ADV = 0.5
POTENTIAL_CAPTURE_BONUS = 0.5
POTENTIAL_EXTRA_TURN_BONUS = 0.5
DEFENSIVE_MOVE_BONUS = 0.5

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_fn = os.path.join("save", f"policy7.pt")
# ==================== HELPER FUNCTIONS ====================
def opponent_player(current_player):
    """Utility function to determine the opponent of the given player."""
    return 'player_2' if current_player == 'player_1' else 'player_1'

class SimpleMinimaxAgent:
    def __init__(self, depth=3):
        self.depth = depth

    def select_action(self, board, player):
        _, move = simple_minimax(board, self.depth, player, player)
        return move

class AdvancedHeuristicAgent:
    def __init__(self, depth=3):
        self.depth = depth

    def select_action(self, board, player):
        _, move = advanced_heuristic_minimax(board, 
                                             self.depth,
                                             alpha=-math.inf,
                                             beta=math.inf, 
                                             current_player=player, 
                                             maximizing_for=player)
        return move

# ==================== PRIORITIZED REPLAY BUFFER ====================
class PrioritizedReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE, alpha=ALPHA):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority
        
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=BETA_START):
        if len(self.buffer) == 0:
            return [], [], []
            
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

# ==================== DQN NETWORK ====================
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, HIDDEN_UNITS[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(DROPOUT_RATE))
        
        # Hidden layers
        for i in range(1, len(HIDDEN_UNITS)):
            self.layers.append(nn.Linear(HIDDEN_UNITS[i-1], HIDDEN_UNITS[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(DROPOUT_RATE))
        
        # Output layer
        self.layers.append(nn.Linear(HIDDEN_UNITS[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ==================== DQN AGENT ====================
class DQNAgent:
    def __init__(self, state_shape=(29,), action_size=6):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer()
        self.gamma = GAMMA
        self.epsilon = EPS_START
        self.epsilon_min = EPS_END
        self.epsilon_decay = EPS_DECAY
        self.batch_size = BATCH_SIZE
        
        # Networks
        self.policy_net = DQN(state_shape[0], action_size).to(device)
        self.target_net = DQN(state_shape[0], action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        # Training tracking
        self.train_step = 0
        self.beta = BETA_START
        self.beta_increment = (1.0 - BETA_START) / BETA_FRAMES
        
        # TensorBoard logging
        self.writer = SummaryWriter()
        self.current_phase = 1
        # Opponent Pool
        self.opponent_pool = []
        for opponent_name in OPPONENTS:
            if opponent_name == 'random':
                self.opponent_pool.append('random')
            elif opponent_name == 'minimax':
                self.opponent_pool.append(SimpleMinimaxAgent(depth=3))
            elif opponent_name == 'advanced_heuristic':
                self.opponent_pool.append(AdvancedHeuristicAgent(depth=3))
        
        self.manual_epsilon_override = False

    def update_target_model(self):
        """Soft update target network weights."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def get_action(self, state, valid_moves, greedy=False):
        """Select an action using epsilon-greedy policy over valid moves."""
        if not greedy and np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        
        # Mask invalid moves
        masked_q = np.full(6, -np.inf)
        for move in valid_moves:
            pit_index = move if move < 6 else move - 7
            masked_q[pit_index] = q_values[pit_index]
        
        return np.argmax(masked_q)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        self.train_step += 1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Sample from prioritized replay buffer
        samples, indices, weights = self.memory.sample(self.batch_size, self.beta)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([sample[0] for sample in samples])).to(device)
        actions = torch.LongTensor(np.array([sample[1] for sample in samples])).to(device)
        rewards = torch.FloatTensor(np.array([sample[2] for sample in samples])).to(device)
        next_states = torch.FloatTensor(np.array([sample[3] for sample in samples])).to(device)
        dones = torch.FloatTensor(np.array([sample[4] for sample in samples])).to(device)
        weights = torch.FloatTensor(weights).to(device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            # Double DQN: use policy net to select actions, target net to evaluate
            if USE_DOUBLE_DQN:
                next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values, reduction='none')
        loss = (weights * loss).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        # Update priorities based on TD errors
        with torch.no_grad():
            td_errors = (expected_q_values - current_q_values).abs().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors)
        
        if self.train_step % TARGET_UPDATE_FREQ == 0:
            self.update_target_model()
            
        return loss.item()

    def preprocess_state(self, board, current_player):
        """Efficient state representation with 29 features."""
        p1_pits = np.array(board['player_1'][:6])
        p1_store = board['player_1'][6]
        p2_pits = np.array(board['player_2'][:6])
        p2_store = board['player_2'][6]
        
        # Normalized features
        features = [
            # Player 1 perspective (normalized)
            *(p1_pits / 24.0),  # 6 features
            p1_store / 48.0,     # 1 feature
            
            # Player 2 perspective (normalized)
            *(p2_pits / 24.0),   # 6 features
            p2_store / 48.0,     # 1 feature
            
            # Relative differences
            *(p1_pits - p2_pits) / 24.0,  # 6 features
            (p1_store - p2_store) / 48.0, # 1 feature
            
            # Turn information
            1.0 if current_player == 'player_1' else 0.0,
            1.0 if current_player == 'player_2' else 0.0,
            
            # Game phase indicator
            min(1.0, (p1_store + p2_store) / 30.0),
            
            # Strategic features
            sum(p1_pits) / 24.0,
            sum(p2_pits) / 24.0,
            float(any(p == 0 for p in p1_pits)),
            float(any(p == 0 for p in p2_pits)),
            1.0  # Bias term
        ]
        
        return np.array(features, dtype=np.float32)

    # ==================== MAIN TRAINING LOOP ====================
    def train(self, env):
        """Continuous training loop with win rate tracking."""
        best_win_rate = -np.inf
        wins_agent = 0
        wins_opponent = 0
        episode = 0
        total_batches_trained = 0  # Track total batches trained
        stored_episodes = 0
        episode_durations = deque(maxlen=REPORTING_PERIOD)
        episode_rewards = deque(maxlen=REPORTING_PERIOD)
        losses = deque(maxlen=REPORTING_PERIOD)
        
        try:
            while True:
                if episode == 8000 and self.current_phase == 0:
                    print("Switching to phase 2")
                    self.current_phase += 1
                    if self.epsilon < 0.5:
                        self.manual_epsilon_override = True
                        print(f"Epsilon manually increased by .2 at episode {episode}")
                    
                if episode == 16000 and self.current_phase == 1:
                    print("Switching to phase 3")
                    self.current_phase += 1
                    if self.epsilon < 0.5:
                        self.manual_epsilon_override = True
                    # self.opponent_pool = []
                    # self.opponent_pool.append('random')
                    # self.opponent_pool.append(SimpleMinimaxAgent(depth=3))
                    
                state = env.reset()
                state = self.preprocess_state(env.board, env.current_player)
                done = False
                total_reward = 0
                steps = 0
                
                while not done:
                    steps += 1
                    if env.current_player == env.agent_player:
                        valid_moves = env.get_valid_moves(env.current_player)
                        action = self.get_action(state, valid_moves)
                        _, reward, done = env.step(action, current_phase=self.current_phase)
                        next_state = self.preprocess_state(env.board, env.current_player)
                        self.remember(state, action, reward, next_state, done)
                        loss = self.replay()
                        if loss is not None:  # Only count if a batch was trained
                            total_batches_trained += 1
                            # Update epsilon based on training steps
                            if not self.manual_epsilon_override:
                                if self.epsilon > self.epsilon_min:
                                    self.epsilon = EPS_END + (EPS_START - EPS_END) * \
                                                    math.exp(-1. * self.train_step * EPS_DECAY)
                            else:
                                # Optional: Gradually decay epsilon again after manual increase
                                if self.epsilon > self.epsilon_min:
                                    self.epsilon = EPS_END + 0.2 + (EPS_START - EPS_END) * \
                                                    math.exp(-1. * self.train_step * EPS_DECAY)
                            losses.append(loss)
                        total_reward += reward
                        state = next_state
                    else:
                        valid_moves = env.get_valid_moves(env.current_player)
                        opponent_agent = random.choice(self.opponent_pool)

                        if opponent_agent == 'random':
                            action = random.choice(valid_moves)
                        else:
                            action = opponent_agent.select_action(env.board, env.current_player)

                        _, _, done = env.step(action, current_phase=self.current_phase)
                        state = self.preprocess_state(env.board, env.current_player)
                # Track metrics
                episode_durations.append(steps)
                episode_rewards.append(total_reward)
                
                # Track wins
                if env.board[env.agent_player][6] > env.board[env.opponent_player][6]:
                    wins_agent += 1
                    stored_episodes += 1
                elif random.random() < 0.1:  # Store some losing episodes
                    stored_episodes += 1
                elif env.board[env.agent_player][6] < env.board[env.opponent_player][6]:
                    wins_opponent += 1
                
                # Report progress periodically
                if episode % REPORTING_PERIOD == 0 and episode > 0:
                    win_rate = wins_agent / REPORTING_PERIOD
                    wins_opponent_rate = wins_opponent / REPORTING_PERIOD
                    avg_reward = np.mean(episode_rewards)
                    avg_duration = np.mean(episode_durations)
                    avg_loss = np.mean(losses) if losses else 0
                    memory_size = len(self.memory)
                    
                    print(f"\nOver the last {REPORTING_PERIOD} episodes:\n Episode {episode} | "
                          f"Stored: {stored_episodes} |"
                          f"Avg Reward: {avg_reward:.2f} | "
                          f"Avg Duration: {avg_duration:.2f} | "
                          f"Epsilon: {self.epsilon:.5f}\n"
                          f"Agent: {win_rate:.2%} | Opponent: {wins_opponent_rate:.2%} | "
                          f"Draw: {1-win_rate-(wins_opponent_rate):.2%} | \n"
                          f"Total Batches Trained: {total_batches_trained} | "
                          f"Replay Size: {memory_size}/{MEMORY_SIZE} | "
                          f"Avg Training Loss: {avg_loss:.4f}")
                    
                    # Save best model
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        self.save_model("save/mancala_dqn_best_7.pt")
                        print(f"New best model saved with win rate: {best_win_rate:.2%}")
                    
                        
                    # Log to TensorBoard
                    self.writer.add_scalar("Exploration rate", self.epsilon, episode)
                    self.writer.add_scalar("Episode duration", avg_duration, episode)
                    self.writer.add_scalar("Win rate", win_rate, episode)
                    self.writer.add_scalar("Reward earned by model",  avg_reward, episode),
                    self.writer.add_scalar("Training loss", avg_loss, episode)
                    
                    # Reset counters for next reporting period
                    wins_agent = 0
                    wins_opponent = 0
                    total_reward = 0
                    episode_durations.clear()  # Reset for next period
                    episode_rewards.clear()   # Reset for next period
                    losses.clear()

                # Save model periodically
                if episode % SAVE_FREQ == 0 and episode > 0:
                    print("Updating target net & saving checkpoint...")
                    if os.path.isfile(model_fn):
                        os.remove(model_fn)
                    self.save_model(model_fn)
                
                episode += 1
                
        except KeyboardInterrupt:
            print("\nTraining stopped by user. Saving final model...")
            self.save_model(model_fn)
            print(f"Best win rate achieved: {best_win_rate:.2%}")
            print(f"Final metrics - "
                  f"Total episodes: {episode} | "
                  f"Stored episodes: {stored_episodes} | "
                  f"Total batches: {total_batches_trained}")

    def save_model(self, filepath):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'beta': self.beta
        }, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.epsilon = checkpoint['epsilon']
        # self.train_step = checkpoint['train_step']
        self.beta = checkpoint['beta']
        self.policy_net.train()  # Set to training mode

# ==================== MANCALA ENVIRONMENT ====================
class MancalaEnv:
    def __init__(self, agent_player='player_1'):
        self.agent_player = agent_player
        self.opponent_player = opponent_player(agent_player)
        self.reward_scale = REWARD_SCALE
        self.reset()
        
    def _get_player_stones(self, player):
        """Returns total stones in the player's store"""
        return self.board[player][-1]  # Assuming last pit is the store

    def reset(self):
        self.board = initialize_board()
        self.current_player = 'player_1'
        return self.board

    def step(self, action, current_phase):
        # --- Snapshot before move ---
        pre_move_board = {
            'player_1': self.board['player_1'].copy(),
            'player_2': self.board['player_2'].copy()
        }

        # --- Execute move ---
        self.board, extra_turn = make_move(self.board, self.current_player, action)

        reward = 0.0
        done = is_terminal(self.board)

        if self.current_player == self.agent_player:
            # --- Initialize turn-based tracking if needed ---
            if not hasattr(self, '_turn_start_store'):
                self._turn_start_store = pre_move_board[self.agent_player][6]

            # --- Calculate reward ---
            reward = self._calculate_reward(action, extra_turn, pre_move_board, current_phase)

            # --- If turn ends, finalize reward and cleanup ---
            if not extra_turn:
                del self._turn_start_store
            else:
                reward += 0.1

        # --- Switch players unless extra turn ---
        if not extra_turn:
            self.current_player = opponent_player(self.current_player)

        return self.board, reward, done


    def get_valid_moves(self, player=None):
        player = player or self.current_player
        return get_valid_moves(self.board, player)

    def _calculate_reward(self, action, extra_turn, pre_move_board, current_phase=1):
        reward = 0.0
        my_pits = self.board[self.agent_player][:6]
        opp_pits = self.board[self.opponent_player][:6]
        my_store = self.board[self.agent_player][6]
        opp_store = self.board[self.opponent_player][6]
    
        # 1. Store gain this turn
        current_store = self.board[self.agent_player][6]
        store_gain = (current_store - self._turn_start_store) /48.0
        reward += store_gain * STORE_DIFF_WEIGHT  # Tunable

        # 2. Extra turn bonus
        if extra_turn:
            reward += EXTRA_TURN_BONUS


        # 4. Capture reward
        stones = pre_move_board[self.agent_player][action]
        landing_index = (action + stones) % 14
        if 0 <= landing_index < 6:
            my_pits_before = pre_move_board[self.agent_player][:6]
            opp_pits_before = pre_move_board[self.opponent_player][:6]
            opp_pit_index = 5 - landing_index
            if (action + stones) < 6:
                # Check if the landing pit was empty before and opponent had stones
                if my_pits_before[landing_index] == 0 and opp_pits_before[opp_pit_index] > 0:
                    captured = opp_pits_before[opp_pit_index]
                    if captured > 0:
                        reward += CAPTURE_BONUS * (captured+1)
            elif (action + stones) > 13:
                if my_pits_before[landing_index] == 0:
                    captured = opp_pits_before[opp_pit_index]
                    reward += CAPTURE_BONUS * (captured+1)
                    
        # 3. Terminal game bonus
        if is_terminal(self.board):
            margin = (my_store - opp_store) / 48.0
            if my_store > opp_store:
                reward += WIN_REWARD + margin
            elif my_store < opp_store:
                reward -= WIN_REWARD - abs(margin)
            else:
                reward -= 1.0  # Draw
                
        game_phase = (my_store + opp_store) / 48.0  # 0=early, 1=late
        
        score_diff = (my_store - opp_store) / 48.0
        
        reward += score_diff * 2.0 * game_phase
        
        if current_phase == 1:
            return reward
            
        # 5. Heuristic: Potential captures
        my_pits = self.board[self.agent_player][:6]
        opp_pits = self.board[self.opponent_player][:6]
        capture_potential = sum(
            opp_pits[5 - i] > 0 and my_pits[i] == 0
            for i in range(6)
        )
        reward += POTENTIAL_CAPTURE_BONUS * capture_potential

        # 6. Heuristic: Potential extra turns
        potential_extra = sum(
            my_pits[i] == (6 - i)
            for i in range(6)
        )
        reward += POTENTIAL_EXTRA_TURN_BONUS * potential_extra
        
        if current_phase == 2:
            return reward
        
        # 7. Seed Conservation (don't empty pits unless beneficial)
        stones_moved = pre_move_board[self.agent_player][action]
        if pre_move_board[self.agent_player][action] == stones_moved:  # Empties the pit
            if landing_index != 6:  # Only penalize if not landing in store
                reward -= 0.2
        
        # 8. Tempo Control (large moves create complex distributions)
        if stones_moved > 10:
            reward += 0.5
        
        vulnerable_pits = 0
        
        # 9. Punish for potential opponent captures
        for i in range(6): 
            if my_pits[i] > 0 and opp_pits[5 - i] == 0:
                for j in range(5-i):
                    if opp_pits[j] + j == 5 - i:
                        vulnerable_pits += 1
                    
        reward -= 0.5 * vulnerable_pits
        
        if current_phase == 3:
            return reward


if __name__ == "__main__":
    env = MancalaEnv(agent_player='player_1')
    agent = DQNAgent(state_shape=(29,))
    
    # Load existing model if available
    try:
        agent.load_model(model_fn)
        print("Loaded existing model weights")
    except Exception as e:
        print(f"Warning: Could not load model due to {e}. Starting fresh.")
    
    print("Starting training... Press Ctrl+C to stop.")
    print(f"Progress will be reported every {REPORTING_PERIOD} episodes")
    print(f"Models will be saved every {SAVE_FREQ} episodes")
    print(f"TensorBoard logs available in 'runs' directory")
    
    agent.train(env)
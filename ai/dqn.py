import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import tensorflow as tf
from ai.rules import initialize_board, get_valid_moves, make_move, is_terminal

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class PrioritizedReplayBuffer:
    def __init__(self, maxlen=5000, alpha=0.6):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= np.max(weights)
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class DQNAgent:
    def __init__(self, state_size=14, action_size=6):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.02
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.tensorboard = TensorBoard(log_dir="./logs")
        self.checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor="loss")
        self.early_stopping = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)

    def _build_model(self):
        """Neural Network with proper input shape and batch normalization"""
        model = Sequential([
            Dense(32, input_shape=(self.state_size,), activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def _preprocess_state(self, board):
        """Add batch dimension and normalize"""
        return np.array(board).reshape(1, -1) / 4.0

    def get_action(self, state, valid_moves):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
        state = self._preprocess_state(state)
        act_values = self.model.predict(state, verbose=0)
        return valid_moves[np.argmax(act_values[0][valid_moves])]

    def replay(self, batch_size=128):
        samples, indices, weights = self.memory.sample(batch_size)
        
        states = []
        targets = []
        
        for state, action, reward, next_state, done in samples:
            state = self._preprocess_state(state)
            next_state = self._preprocess_state(next_state)
            
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                # Double DQN: Use online model to select action, target model to evaluate
                online_action = np.argmax(self.model.predict(next_state, verbose=0)[0])
                t = self.target_model.predict(next_state, verbose=0)
                target[0][action] = reward + self.gamma * t[0][online_action]
            
            states.append(state[0])
            targets.append(target[0])
        
        # Batch training with importance sampling weights
        self.model.fit(
            np.array(states), 
            np.array(targets), 
            sample_weight=np.array(weights),
            epochs=1, 
            verbose=0,
            callbacks=[self.tensorboard, self.checkpoint, self.early_stopping]
        )
        
        # Update priorities
        preds = self.model.predict(np.array(states), verbose=0)
        errors = np.abs(np.array(targets) - preds).mean(axis=1)
        self.memory.update_priorities(indices, errors)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, episodes=1000, batch_size=128):
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Only the agent's moves are stored in memory
                if env.current_player == env.agent_player:
                    valid_moves = env.get_valid_moves()
                    action = self.get_action(state, valid_moves)
                    next_state, reward, done = env.step(action)
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                else:
                    # Let environment handle opponent's move (self-play)
                    next_state, reward, done = env.step(None)
                    state = next_state
                
                if done:
                    print(f"Episode: {e+1}/{episodes}, Reward: {total_reward}, Epsilon: {self.epsilon:.2f}")
                    if len(self.memory.buffer) > batch_size:
                        self.replay(batch_size)
                    break
            
            
class MancalaEnv:
    def __init__(self, agent=None, agent_player="player_1"):
        """
        Initialize the Mancala environment.
        :param agent: The DQN agent instance for self-play
        :param agent_player: The player the agent is controlling ("player_1" or "player_2")
        """
        self.board = initialize_board()
        self.current_player = "player_1"
        self.agent_player = agent_player
        self.opponent_player = "player_2" if agent_player == "player_1" else "player_1"
        self.agent = agent  # Store the agent reference for self-play

    def step(self, action):
        # If it's the agent's turn, use the provided action
        if self.current_player == self.agent_player:
            new_board, extra_turn = make_move(self.board, self.current_player, action)
            self.board = new_board
        else:
            # For opponent's turn, get action from the agent (self-play)
            state = self._get_state()
            valid_moves = get_valid_moves(self.board, self.current_player)
            opponent_action = self.agent.get_action(state, valid_moves)
            new_board, extra_turn = make_move(self.board, self.current_player, opponent_action)
            self.board = new_board

        # Calculate reward from agent's perspective
        reward = self._calculate_reward()
        
        # Switch players if no extra turn
        if not extra_turn:
            self.current_player = self.opponent_player if self.current_player == self.agent_player else self.agent_player

        done = is_terminal(self.board)
        return self._get_state(), reward, done

    def reset(self):
        """Reset the environment and return the initial state."""
        self.board = initialize_board()
        self.current_player = "player_1"
        return self._get_state()

    def _get_state(self):
        """
        Get the current state of the board from the agent's perspective.
        """
        if self.agent_player == "player_1":
            state = (
                self.board["player_1"][:6] +
                [self.board["player_1"][6]] +
                self.board["player_2"][:6] +
                [self.board["player_2"][6]]
            )
        else:
            state = (
                self.board["player_2"][:6] +
                [self.board["player_2"][6]] +
                self.board["player_1"][:6] +
                [self.board["player_1"][6]]
            )
        return np.array(state)

    def get_valid_moves(self):
        """Get valid moves for the current player."""
        return get_valid_moves(self.board, self.current_player)


    
    def _get_captured_stones(self):
        """
        Track the number of stones captured during the last move.
        """
        # Assuming the last move is stored in the environment
        if hasattr(self, "last_move"):
            _, captured_stones = self.last_move  # Extract captured stones from the last move
            return captured_stones
        return 0  # No capture if no move has been made
    
    def _earned_extra_turn(self):
        """
        Check if the agent earned an extra turn in the last move.
        """
        # Assuming the last move is stored in the environment
        if hasattr(self, "last_move"):
            extra_turn = self.last_move[1]  # Extract extra turn flag from the last move
            return extra_turn
        return False  # No extra turn if no move has been made

    def _count_empty_pits(self, pits):
        """
        Count the number of empty pits in a given list of pits.
        :param pits: List of stones in pits (e.g., [3, 0, 5, 0, 2, 0])
        :return: Number of empty pits
        """
        return pits.count(0)  # Count pits with 0 stones
    
    
    def _evaluate_potential_captures(self, agent_pits, opponent_pits):
        """
        Evaluate potential future captures based on the current board state.
        :param agent_pits: List of stones in the agent's pits
        :param opponent_pits: List of stones in the opponent's pits
        :return: Number of potential captures
        """
        potential_captures = 0

        for i in range(len(agent_pits)):
            # Check if the agent can land in an empty pit and capture opponent's stones
            if agent_pits[i] == 0 and opponent_pits[5 - i] > 0:
                potential_captures += opponent_pits[5 - i]

        return potential_captures

    def _calculate_stone_imbalance(self, pits):
        """
        Calculate the standard deviation of stones in the pits to measure balance.
        :param pits: List of stones in pits (e.g., [3, 0, 5, 0, 2, 0])
        :return: Standard deviation of stones in the pits
        """
        if not pits:
            return 0  # No imbalance if no pits

        mean = sum(pits) / len(pits)
        variance = sum((x - mean) ** 2 for x in pits) / len(pits)
        return variance ** 0.5  # Standard deviation

    def _calculate_reward(self):
        """
        Calculate the reward from the agent's perspective using 10 best heuristics.
        """
        # Determine agent and opponent stores
        if self.agent_player == "player_1":
            agent_store = self.board["player_1"][6]
            opponent_store = self.board["player_2"][6]
            agent_pits = self.board["player_1"][:6]
            opponent_pits = self.board["player_2"][:6]
        else:
            agent_store = self.board["player_2"][6]
            opponent_store = self.board["player_1"][6]
            agent_pits = self.board["player_2"][:6]
            opponent_pits = self.board["player_1"][:6]

        # Heuristic 1: Score difference
        score_diff = agent_store - opponent_store

        # Heuristic 2: Stones captured (assuming captures are tracked elsewhere)
        captured_stones = self._get_captured_stones()  # Implement this function
        capture_reward = captured_stones * 1

        # Heuristic 3: Extra turns
        extra_turn_reward = 2 if self._earned_extra_turn() else 0  # Implement this function

        # Heuristic 4: Empty pits (penalize)
        empty_pits = self._count_empty_pits(agent_pits)  # Implement this function
        empty_pit_penalty = -0.01 * empty_pits

        # Heuristic 5: Opponent's empty pits (reward)
        opponent_empty_pits = self._count_empty_pits(opponent_pits)  # Implement this function
        opponent_empty_reward = 0.01 * opponent_empty_pits

        # Heuristic 6: Stones in store
        store_reward = agent_store * 0.04

        # Heuristic 7: Stones in opponent's store (penalize)
        opponent_store_penalty = -opponent_store * 0.01

        # Heuristic 8: Potential captures (reward)
        potential_captures = self._evaluate_potential_captures(agent_pits, opponent_pits)  # Implement this function
        potential_capture_reward = potential_captures * 0.04

        # Heuristic 9: Balance of stones
        stone_imbalance = self._calculate_stone_imbalance(agent_pits)  # Implement this function
        balance_reward = -stone_imbalance * 0.01

        # Heuristic 10: Endgame advantage
        endgame_reward = 0
        if is_terminal(self.board):
            if agent_store > 24:
                endgame_reward = 1  # Win
            elif opponent_store > 24:
                endgame_reward = -1  # Lose
            else:
                endgame_reward = 0.5 if score_diff > 0 else -0.5  # Draw with advantage

        # Combine all rewards
        reward = (
            score_diff * 0.1 +
            capture_reward +
            extra_turn_reward +
            empty_pit_penalty +
            opponent_empty_reward +
            store_reward +
            opponent_store_penalty +
            potential_capture_reward +
            balance_reward +
            endgame_reward
        )

        return reward
        
# Usage
if __name__ == "__main__":
    agent = DQNAgent()
    env = MancalaEnv(agent=agent, agent_player="player_1")  # Pass the agent to the environment
    
    # Train the agent
    agent.train(env, episodes=500)
    
    # Save the trained model
    agent.model.save("mancala_dqn.h5")